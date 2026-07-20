"""Frame / timing integrity checks for compression and upscaling outputs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass

log = logging.getLogger("compression.integrity")

FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    pix_fmt: str


async def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def probe_media(path: str) -> MediaInfo:
    # Avoid -count_frames (full decode) under the 180s validator path —
    # nb_frames / duration×fps is enough for integrity warnings.
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,codec_name,pix_fmt:format=duration",
        "-of",
        "json",
        path,
    ]
    rc, stdout, stderr = await _run(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed: {stderr[-300:]}")
    data = json.loads(stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    duration = float(fmt.get("duration") or 0.0)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    codec = str(stream.get("codec_name") or "")
    pix_fmt = str(stream.get("pix_fmt") or "")
    fps = 30.0
    rate = stream.get("avg_frame_rate") or "30/1"
    if isinstance(rate, str) and "/" in rate:
        num, den = rate.split("/", 1)
        if float(den) != 0:
            fps = float(num) / float(den)
    frames = int(stream.get("nb_frames") or stream.get("nb_read_frames") or 0)
    if frames <= 0 and duration > 0 and fps > 0:
        frames = int(round(duration * fps))
    return MediaInfo(duration, width, height, fps, frames, codec, pix_fmt)


async def validate_output_matches_source(
    source_path: str,
    output_path: str,
    *,
    expect_scale: int | None = None,
    duration_tol: float = 0.35,
    frame_tol: int = 2,
) -> tuple[bool, str]:
    """
    Ensure FPS/duration/frame-count integrity. For upscaling, expect_scale checks resolution.
    """
    src = await probe_media(source_path)
    out = await probe_media(output_path)

    if out.duration <= 0 or out.frame_count <= 0:
        return False, "output has zero duration/frames"
    if abs(out.duration - src.duration) > duration_tol and src.duration > 0:
        return False, f"duration mismatch src={src.duration:.3f} out={out.duration:.3f}"
    if src.frame_count > 0 and abs(out.frame_count - src.frame_count) > frame_tol:
        return False, f"frame count mismatch src={src.frame_count} out={out.frame_count}"
    if src.fps > 0 and out.fps > 0 and abs(out.fps - src.fps) / src.fps > 0.05:
        return False, f"fps mismatch src={src.fps:.3f} out={out.fps:.3f}"
    if expect_scale and src.width > 0 and src.height > 0:
        exp_w = src.width * expect_scale
        exp_h = src.height * expect_scale
        # Allow 1px rounding / even alignment.
        if abs(out.width - exp_w) > 2 or abs(out.height - exp_h) > 2:
            return False, f"resolution mismatch expected {exp_w}x{exp_h} got {out.width}x{out.height}"
    return True, "ok"
