"""Local VMAF measurement aligned with SN85 validators (VMAF NEG + harmonic mean)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("compression.vmaf")

FFMPEG_VMAF_BIN = os.getenv(
    "FFMPEG_VMAF_BIN",
    str(Path(__file__).resolve().parent / "bin" / "ffmpeg-vmaf"),
)
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
# Validators score compression with VMAF NEG v0.6.1 (see services/scoring/vmaf_metric.py).
VMAF_MODEL = os.getenv("ADAPTIVE_VMAF_MODEL", "version=vmaf_v0.6.1neg")


@dataclass
class VmafResult:
    mean: float
    min_score: float
    p5: float
    harmonic_mean: float
    frame_count: int
    raw_log: str = ""

    @property
    def score_signal(self) -> float:
        """Primary gate/score signal matching validator harmonic_mean preference."""
        return self.harmonic_mean if self.harmonic_mean > 0 else self.mean


async def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


def _harmonic_mean(scores: list[float]) -> float:
    positives = [s for s in scores if s > 1e-9]
    if not positives:
        return 0.0
    return len(positives) / sum(1.0 / s for s in positives)


def _parse_vmaf_json(payload: str) -> VmafResult | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    frames = data.get("frames") or []
    scores = []
    for frame in frames:
        metrics = frame.get("metrics") or {}
        if "vmaf" in metrics:
            scores.append(float(metrics["vmaf"]))

    pooled = (data.get("pooled_metrics") or {}).get("vmaf") or {}
    if scores:
        scores_sorted = sorted(scores)
        p5_idx = max(0, int(len(scores_sorted) * 0.05) - 1)
        mean = float(pooled.get("mean", sum(scores) / len(scores)))
        hmean = float(pooled.get("harmonic_mean", _harmonic_mean(scores)))
        return VmafResult(
            mean=mean,
            min_score=min(scores),
            p5=scores_sorted[p5_idx],
            harmonic_mean=hmean,
            frame_count=len(scores),
        )

    if "mean" in pooled:
        mean = float(pooled["mean"])
        hmean = float(pooled.get("harmonic_mean", mean))
        return VmafResult(
            mean=mean,
            min_score=mean,
            p5=mean,
            harmonic_mean=hmean,
            frame_count=0,
        )
    return None


def _parse_vmaf_log_fallback(text: str) -> VmafResult | None:
    match = re.search(r"VMAF score:\s*([0-9.]+)", text)
    if not match:
        return None
    mean = float(match.group(1))
    return VmafResult(
        mean=mean,
        min_score=mean,
        p5=mean,
        harmonic_mean=mean,
        frame_count=0,
        raw_log=text,
    )


async def calculate_vmaf(
    reference_path: str,
    distorted_path: str,
    *,
    n_threads: int | None = None,
    model: str | None = None,
) -> VmafResult:
    """
    Compute VMAF between reference and distorted videos.
    Defaults to VMAF NEG (validators' compression model).
    """
    vmaf_bin = FFMPEG_VMAF_BIN if os.path.exists(FFMPEG_VMAF_BIN) else FFMPEG_BIN
    threads = n_threads or max(1, (os.cpu_count() or 4) // 2)
    model_spec = model or VMAF_MODEL

    with tempfile.TemporaryDirectory(prefix="vmaf_") as tmp:
        log_path = os.path.join(tmp, "vmaf.json")
        # Distorted first, reference second — matches validator / libvmaf convention.
        filter_complex = (
            f"[0:v]scale2ref=flags=bicubic[dist][ref];"
            f"[dist][ref]libvmaf=log_fmt=json:log_path={log_path}:"
            f"n_threads={threads}:model={model_spec}"
        )
        cmd = [
            vmaf_bin,
            "-hide_banner",
            "-i",
            distorted_path,
            "-i",
            reference_path,
            "-filter_complex",
            filter_complex,
            "-f",
            "null",
            "-",
        ]
        rc, stdout, stderr = await _run(cmd)
        combined = stdout + "\n" + stderr
        if os.path.exists(log_path):
            result = _parse_vmaf_json(Path(log_path).read_text(encoding="utf-8"))
            if result:
                return result

        # Fallback without explicit model (older libvmaf builds).
        if "model" in stderr.lower() or rc != 0:
            filter_fallback = (
                f"[0:v]scale2ref=flags=bicubic[dist][ref];"
                f"[dist][ref]libvmaf=log_fmt=json:log_path={log_path}:n_threads={threads}"
            )
            cmd[cmd.index("-filter_complex") + 1] = filter_fallback
            rc, stdout, stderr = await _run(cmd)
            combined = stdout + "\n" + stderr
            if os.path.exists(log_path):
                result = _parse_vmaf_json(Path(log_path).read_text(encoding="utf-8"))
                if result:
                    log.warning("VMAF NEG model unavailable; fell back to default VMAF model")
                    return result

        fallback = _parse_vmaf_log_fallback(combined)
        if fallback:
            return fallback

        raise RuntimeError(
            f"VMAF calculation failed (rc={rc}). "
            f"Ensure {vmaf_bin} has libvmaf. stderr tail: {stderr[-800:]}"
        )


async def extract_sample_clip(
    input_path: str,
    output_path: str,
    *,
    start_seconds: float,
    duration_seconds: float,
    accurate: bool = False,
) -> None:
    """
    Extract a short clip. accurate=True seeks after -i (frame-accurate) and
    re-encodes so calibration/VMAF pairs stay temporally aligned.
    """
    if accurate:
        cmd = [
            FFMPEG_BIN,
            "-hide_banner",
            "-y",
            "-i",
            input_path,
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-vsync",
            "cfr",
            output_path,
        ]
        rc, _, stderr = await _run(cmd)
        if rc != 0 or not os.path.exists(output_path):
            raise RuntimeError(f"accurate sample extract failed: {stderr[-300:]}")
        return

    cmd_copy = [
        FFMPEG_BIN,
        "-hide_banner",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-sn",
        "-dn",
        output_path,
    ]
    rc, _, _ = await _run(cmd_copy)
    if rc == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return

    cmd_re = [
        FFMPEG_BIN,
        "-hide_banner",
        "-y",
        "-i",
        input_path,
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        output_path,
    ]
    rc2, _, stderr2 = await _run(cmd_re)
    if rc2 != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"sample extract failed: {stderr2[-300:]}")


async def calculate_vmaf_aligned_window(
    reference_path: str,
    distorted_path: str,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> VmafResult:
    """
    Score the same temporal window on both files via trim filters (no keyframe
    seek skew). Used for sample→full calibration.
    """
    with tempfile.TemporaryDirectory(prefix="vmaf_cal_") as tmp:
        ref_clip = os.path.join(tmp, "ref.mp4")
        dist_clip = os.path.join(tmp, "dist.mp4")
        await extract_sample_clip(
            reference_path,
            ref_clip,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            accurate=True,
        )
        await extract_sample_clip(
            distorted_path,
            dist_clip,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            accurate=True,
        )
        return await calculate_vmaf(ref_clip, dist_clip)
