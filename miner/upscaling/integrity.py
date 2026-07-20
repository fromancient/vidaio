"""Frame / resolution integrity for upscaling outputs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass

log = logging.getLogger("upscaling.integrity")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    frame_count: int


async def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def probe_media(path: str) -> MediaInfo:
    cmd = [
        FFPROBE_BIN, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
        "-of", "json", path,
    ]
    rc, stdout, stderr = await _run(cmd)
    if rc != 0:
        raise RuntimeError(stderr[-300:])
    data = json.loads(stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    duration = float(fmt.get("duration") or 0.0)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = 30.0
    rate = stream.get("avg_frame_rate") or "30/1"
    if isinstance(rate, str) and "/" in rate:
        num, den = rate.split("/", 1)
        if float(den) != 0:
            fps = float(num) / float(den)
    frames = int(stream.get("nb_read_frames") or 0)
    if frames <= 0 and duration > 0 and fps > 0:
        frames = int(round(duration * fps))
    return MediaInfo(duration, width, height, fps, frames)


async def validate_upscale_output(
    source_path: str, output_path: str, scale: int
) -> tuple[bool, str]:
    src = await probe_media(source_path)
    out = await probe_media(output_path)
    if out.duration <= 0 or out.frame_count <= 0:
        return False, "empty output"
    if src.duration > 0 and abs(out.duration - src.duration) > 0.5:
        return False, f"duration mismatch {src.duration:.2f} vs {out.duration:.2f}"
    if src.frame_count > 0 and abs(out.frame_count - src.frame_count) > 3:
        return False, f"frame mismatch {src.frame_count} vs {out.frame_count}"
    if src.width > 0 and abs(out.width - src.width * scale) > 2:
        return False, f"width mismatch expected~{src.width * scale} got {out.width}"
    if src.height > 0 and abs(out.height - src.height * scale) > 2:
        return False, f"height mismatch expected~{src.height * scale} got {out.height}"
    return True, "ok"
