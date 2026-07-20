"""Cheap source features for offline-trained / heuristic RD decisions (<~2s)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass

log = logging.getLogger("compression.features")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")


@dataclass
class SourceFeatures:
    duration: float
    width: int
    height: int
    fps: float
    bitrate: float
    codec: str
    bits_per_pixel: float
    pixels: int
    complexity: str  # easy|medium|hard
    scene_density: float  # scenes per minute (approx)
    motion_proxy: float
    detail_proxy: float

    def to_dict(self) -> dict:
        return asdict(self)


async def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


def _complexity(bpp: float, pixels: int, scene_density: float) -> str:
    if bpp < 0.04 and scene_density < 8:
        c = "easy"
    elif bpp < 0.10 and scene_density < 20:
        c = "medium"
    else:
        c = "hard"
    if pixels >= 3840 * 2160 and c == "easy":
        c = "medium"
    if scene_density >= 25:
        c = "hard"
    return c


async def extract_source_features(path: str, *, timeout: float = 3.0) -> SourceFeatures:
    """ffprobe + one short signalstats window — no VMAF."""

    async def _body() -> SourceFeatures:
        cmd = [
            FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,bit_rate,codec_name:format=duration,bit_rate",
            "-of", "json", path,
        ]
        rc, stdout, stderr = await _run(cmd)
        if rc != 0:
            raise RuntimeError(f"ffprobe failed: {stderr[-200:]}")
        data = json.loads(stdout)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        duration = float(fmt.get("duration") or 0.0)
        bitrate = float(stream.get("bit_rate") or fmt.get("bit_rate") or 0.0)
        codec = str(stream.get("codec_name") or "unknown")
        fps = 30.0
        rate = stream.get("avg_frame_rate") or "30/1"
        if isinstance(rate, str) and "/" in rate:
            num, den = rate.split("/", 1)
            if float(den) != 0:
                fps = float(num) / float(den)
        pixels = max(1, width * height)
        bpp = bitrate / (pixels * fps) if bitrate > 0 and fps > 0 else 0.08

        # One mid-clip 0.8s stats window (cheap).
        start = max(0.0, duration * 0.4) if duration > 1 else 0.0
        stats_cmd = [
            FFMPEG_BIN, "-hide_banner", "-ss", f"{start:.2f}", "-t", "0.8",
            "-i", path, "-vf", "scale=160:-2,signalstats,metadata=print:file=-",
            "-f", "null", "-",
        ]
        _, so, se = await _run(stats_cmd)
        text = so + se
        ys = [float(m) for m in re.findall(r"YAVG=([0-9.]+)", text)]
        sats = [float(m) for m in re.findall(r"SATAVG=([0-9.]+)", text)]
        motion = 0.0
        if len(ys) > 1:
            mean = sum(ys) / len(ys)
            motion = sum((y - mean) ** 2 for y in ys) / len(ys)
        yavg = sum(ys) / len(ys) if ys else 128.0
        sat = sum(sats) / len(sats) if sats else 0.0
        detail = sat + abs(yavg - 128.0)

        # Scene density proxy from bpp+motion (full scene detect is too slow for 180s).
        scene_density = 6.0 + min(30.0, motion * 2.0) + (12.0 if bpp > 0.12 else 0.0)
        complexity = _complexity(bpp, pixels, scene_density)

        return SourceFeatures(
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            bitrate=bitrate,
            codec=codec,
            bits_per_pixel=bpp,
            pixels=pixels,
            complexity=complexity,
            scene_density=scene_density,
            motion_proxy=motion,
            detail_proxy=detail,
        )

    return await asyncio.wait_for(_body(), timeout=timeout)
