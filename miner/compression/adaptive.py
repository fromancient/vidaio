"""
Deadline-aware rate-distortion optimizer for SN85 compression.

Target: ~18–22× at the VMAF threshold (score ≈ 0.75–0.81), not max CQ / max VMAF.
Strategy: VMAF-NEG harmonic mean, 4 representative samples, ≤4 CQ probes,
dynamic margin, faster probe presets, deadline budgets that prefer a mid score
over a timeout (timeout ≈ EMA zero).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from score_sim import calculate_compression_score
from vmaf_local import calculate_vmaf, extract_sample_clip
from zones import (
    build_shot_zones,
    build_x265_zones_param,
    estimate_shot_complexities,
)

log = logging.getLogger("compression.adaptive")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

ADAPTIVE_ENABLED_DEFAULT = os.getenv("ADAPTIVE_COMPRESSION", "true").lower() in (
    "1",
    "true",
    "yes",
)
# Initial margin until calibration data exists (0.8–1.0 recommended).
ADAPTIVE_VMAF_MARGIN = float(os.getenv("ADAPTIVE_VMAF_MARGIN", "0.8"))
ADAPTIVE_SAMPLE_SECONDS = float(os.getenv("ADAPTIVE_SAMPLE_SECONDS", "3.0"))
ADAPTIVE_MAX_CANDIDATES = int(os.getenv("ADAPTIVE_MAX_CANDIDATES", "4"))
ADAPTIVE_MIN_DURATION_FOR_SEARCH = float(os.getenv("ADAPTIVE_MIN_DURATION_FOR_SEARCH", "3.0"))
ADAPTIVE_SVT_PRESET = os.getenv("ADAPTIVE_SVT_PRESET", "5")
ADAPTIVE_X265_PRESET = os.getenv("ADAPTIVE_X265_PRESET", "medium")
ADAPTIVE_SVT_PROBE_PRESET = os.getenv("ADAPTIVE_SVT_PROBE_PRESET", "8")
ADAPTIVE_X265_PROBE_PRESET = os.getenv("ADAPTIVE_X265_PROBE_PRESET", "veryfast")
ADAPTIVE_SVT_PARAMS = os.getenv(
    "ADAPTIVE_SVT_PARAMS",
    "tune=0:enable-overlays=1:enable-tf=1:scd=1",
)
ADAPTIVE_X265_PARAMS = os.getenv(
    "ADAPTIVE_X265_PARAMS",
    "aq-mode=3:psy-rd=2.0:psy-rdoq=1.0:strong-intra-smoothing=0:rc-lookahead=40",
)
ADAPTIVE_GATE_MODE = os.getenv("ADAPTIVE_GATE_MODE", "mean").strip().lower()
# Unconditional cliff push disabled (0). Kept only for env compatibility.
ADAPTIVE_PUSH_STEPS = int(os.getenv("ADAPTIVE_PUSH_STEPS", "0"))
ADAPTIVE_SCORE_EPS = float(os.getenv("ADAPTIVE_SCORE_EPS", "0.01"))
ADAPTIVE_TARGET_RATIO_LO = float(os.getenv("ADAPTIVE_TARGET_RATIO_LO", "18.0"))
ADAPTIVE_TARGET_RATIO_HI = float(os.getenv("ADAPTIVE_TARGET_RATIO_HI", "22.0"))
ADAPTIVE_TARGET_SCORE = float(os.getenv("ADAPTIVE_TARGET_SCORE", "0.76"))
ADAPTIVE_SEARCH_BUDGET_FRAC = float(os.getenv("ADAPTIVE_SEARCH_BUDGET_FRAC", "0.15"))
ADAPTIVE_ENCODE_BUDGET_FRAC = float(os.getenv("ADAPTIVE_ENCODE_BUDGET_FRAC", "0.65"))
ADAPTIVE_DEFAULT_DEADLINE_SECONDS = float(
    os.getenv(
        "ADAPTIVE_DEADLINE_SECONDS",
        os.getenv("MINER_COMPRESSION_SERVICE_TIMEOUT_SECONDS", "1800"),
    )
)
ADAPTIVE_MARGIN_STATE = os.getenv(
    "ADAPTIVE_MARGIN_STATE",
    "/tmp/organic-proxy/adaptive_margin_state.json",
)
ADAPTIVE_MARGIN_MIN = float(os.getenv("ADAPTIVE_MARGIN_MIN", "0.5"))
ADAPTIVE_MARGIN_MAX = float(os.getenv("ADAPTIVE_MARGIN_MAX", "2.0"))
# Discard absurd calibration errors (misaligned seeks used to poison margin to 60+).
ADAPTIVE_CALIBRATION_MAX_ABS = float(os.getenv("ADAPTIVE_CALIBRATION_MAX_ABS", "6.0"))


class ComplexityClass(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class VideoAnalysis:
    duration: float
    width: int
    height: int
    fps: float
    bitrate: float
    codec: str
    bits_per_pixel: float
    complexity: ComplexityClass
    scene_count: int = 0
    sample_starts: list[float] = field(default_factory=list)
    sample_labels: list[str] = field(default_factory=list)
    scene_starts: list[float] = field(default_factory=list)
    sample_stats: list[dict] = field(default_factory=list)
    shot_complexities: list[float] = field(default_factory=list)


@dataclass
class CandidateResult:
    cq: int
    preset: str
    compression_rate: float
    compression_ratio: float
    vmaf_mean: float
    vmaf_min: float
    vmaf_p5: float
    vmaf_hmean: float
    vmaf_std: float
    score: float
    reason: str
    encoded_bytes: int
    reference_bytes: int
    encode_seconds: float = 0.0

    @property
    def gate_vmaf(self) -> float:
        return self.vmaf_hmean if self.vmaf_hmean > 0 else self.vmaf_mean


@dataclass
class AdaptiveDecision:
    enabled: bool
    cq: int
    preset: str
    analysis: Optional[VideoAnalysis]
    candidates: list[CandidateResult] = field(default_factory=list)
    selected: Optional[CandidateResult] = None
    message: str = ""
    margin_used: float = 0.0
    search_seconds: float = 0.0
    probes: int = 0
    x265_zones: str | None = None
    encode_deadline_seconds: float = 0.0

    def to_log_dict(self) -> dict:
        payload = {
            "enabled": self.enabled,
            "cq": self.cq,
            "preset": self.preset,
            "message": self.message,
            "margin_used": self.margin_used,
            "search_seconds": round(self.search_seconds, 2),
            "probes": self.probes,
            "x265_zones": self.x265_zones,
            "encode_deadline_seconds": round(self.encode_deadline_seconds, 1),
            "candidates": [asdict(c) for c in self.candidates],
        }
        if self.analysis:
            payload["analysis"] = {
                **{k: v for k, v in asdict(self.analysis).items() if k != "complexity"},
                "complexity": self.analysis.complexity.value,
            }
        if self.selected:
            payload["selected"] = asdict(self.selected)
        return payload


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


def _load_margin_state() -> dict:
    path = Path(ADAPTIVE_MARGIN_STATE)
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning(f"margin state load failed: {exc}")
    return {"underpredictions": [], "updated_at": None}


def _save_margin_state(state: dict) -> None:
    path = Path(ADAPTIVE_MARGIN_STATE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = time.time()
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning(f"margin state save failed: {exc}")


def reset_margin_state() -> None:
    """Clear poisoned underprediction history."""
    _save_margin_state({"underpredictions": [], "updated_at": None})


def record_vmaf_calibration(predicted_sample_vmaf: float, measured_full_vmaf: float) -> None:
    """Call after aligned mid-clip validation: underprediction = pred - measured."""
    under = float(predicted_sample_vmaf) - float(measured_full_vmaf)
    if abs(under) > ADAPTIVE_CALIBRATION_MAX_ABS:
        log.warning(
            f"calibration outlier discarded under={under:.2f} "
            f"(pred={predicted_sample_vmaf:.2f} measured={measured_full_vmaf:.2f}); "
            f"max_abs={ADAPTIVE_CALIBRATION_MAX_ABS}"
        )
        return
    state = _load_margin_state()
    hist = [
        float(x)
        for x in (state.get("underpredictions") or [])
        if abs(float(x)) <= ADAPTIVE_CALIBRATION_MAX_ABS
    ]
    hist.append(under)
    state["underpredictions"] = hist[-64:]
    _save_margin_state(state)


def current_dynamic_margin(*, sample_vmaf_std: float = 0.0) -> float:
    """
    margin = max(0.5, rolling p95 underprediction + 0.2), clamped.
    Until enough clean calibration exists, start near ADAPTIVE_VMAF_MARGIN.
    """
    state = _load_margin_state()
    hist = [
        float(x)
        for x in (state.get("underpredictions") or [])
        if abs(float(x)) <= ADAPTIVE_CALIBRATION_MAX_ABS
    ]
    if len(hist) >= 5:
        ordered = sorted(hist)
        p95_idx = min(len(ordered) - 1, max(0, int(math.ceil(0.95 * len(ordered)) - 1)))
        base = max(ADAPTIVE_MARGIN_MIN, ordered[p95_idx] + 0.2)
    else:
        base = ADAPTIVE_VMAF_MARGIN

    uncertainty = min(0.6, max(0.0, sample_vmaf_std) * 0.25)
    margin = base + uncertainty
    return max(ADAPTIVE_MARGIN_MIN, min(ADAPTIVE_MARGIN_MAX, margin))


def _classify_complexity(bpp: float, width: int, height: int, scene_count: int, duration: float) -> ComplexityClass:
    scenes_per_min = scene_count / max(duration / 60.0, 1e-3)
    if bpp < 0.04 and scenes_per_min < 8:
        complexity = ComplexityClass.EASY
    elif bpp < 0.10 and scenes_per_min < 20:
        complexity = ComplexityClass.MEDIUM
    else:
        complexity = ComplexityClass.HARD

    if width * height >= 3840 * 2160 and complexity == ComplexityClass.EASY:
        complexity = ComplexityClass.MEDIUM
    if scenes_per_min >= 25:
        complexity = ComplexityClass.HARD
    return complexity


async def _detect_scene_starts(path: str, duration: float) -> list[float]:
    """Lightweight scene-cut timestamps via ffmpeg scene filter."""
    if duration < 4.0:
        return []
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-i",
        path,
        "-vf",
        "select='gt(scene,0.35)',showinfo",
        "-f",
        "null",
        "-",
    ]
    rc, _, stderr = await _run(cmd)
    if rc != 0 and not stderr:
        return []
    starts = [0.0]
    for match in re.finditer(r"pts_time:([0-9.]+)", stderr):
        t = float(match.group(1))
        if 0.5 < t < duration - 0.5:
            starts.append(t)
    # Deduplicate near-duplicates.
    starts = sorted(set(round(t, 2) for t in starts))
    filtered: list[float] = []
    for t in starts:
        if not filtered or t - filtered[-1] >= 1.0:
            filtered.append(t)
    return filtered[:40]


async def _segment_stats(path: str, start: float, duration: float = 1.0) -> dict[str, float]:
    """Cheap motion / detail / luma proxies for representative sampling."""
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        path,
        "-vf",
        "scale=320:-2,signalstats,metadata=print:file=-",
        "-f",
        "null",
        "-",
    ]
    rc, stdout, stderr = await _run(cmd)
    text = stdout + "\n" + stderr
    ys: list[float] = []
    sats: list[float] = []
    for match in re.finditer(r"YAVG=([0-9.]+)", text):
        ys.append(float(match.group(1)))
    for match in re.finditer(r"SATAVG=([0-9.]+)", text):
        sats.append(float(match.group(1)))
    yavg = sum(ys) / len(ys) if ys else 128.0
    sat = sum(sats) / len(sats) if sats else 0.0
    # Motion proxy: variance of YAVG across frames (temporal change).
    motion = 0.0
    if len(ys) > 1:
        mean = sum(ys) / len(ys)
        motion = sum((y - mean) ** 2 for y in ys) / len(ys)
    # Detail proxy: saturation + deviation from mid-gray (texture/contrast).
    detail = sat + abs(yavg - 128.0)
    darkness = max(0.0, 140.0 - yavg) + max(0.0, 20.0 - sat)
    return {"motion": motion, "detail": detail, "darkness": darkness, "yavg": yavg}


async def _pick_representative_starts(
    path: str,
    duration: float,
    sample_seconds: float,
    scene_starts: list[float],
) -> tuple[list[float], list[str], list[dict]]:
    """Select highest-motion, highest-detail, darkest, and median-complexity shots."""
    candidates: list[float] = []
    for frac in (0.12, 0.28, 0.44, 0.60, 0.76, 0.90):
        candidates.append(max(0.0, min(duration - sample_seconds, duration * frac)))
    for sc in scene_starts:
        start = max(0.0, min(duration - sample_seconds, sc + 0.15))
        candidates.append(start)

    # Unique within 1s.
    uniq: list[float] = []
    for t in sorted(candidates):
        if not uniq or t - uniq[-1] >= 1.0:
            uniq.append(t)
    uniq = uniq[:10]
    if not uniq:
        return [0.0], ["fallback"], [{"start": 0.0, "motion": 0.0, "detail": 0.0, "darkness": 0.0}]

    stats: list[tuple[float, dict[str, float]]] = []
    for start in uniq:
        try:
            s = await _segment_stats(path, start, duration=min(1.2, sample_seconds))
            s["start"] = start
            stats.append((start, s))
        except Exception:
            stats.append((start, {"start": start, "motion": 0.0, "detail": 0.0, "darkness": 0.0, "yavg": 128.0}))

    by_motion = max(stats, key=lambda x: x[1]["motion"])
    by_detail = max(stats, key=lambda x: x[1]["detail"])
    by_dark = max(stats, key=lambda x: x[1]["darkness"])
    # Median complexity = median of motion+detail.
    ranked = sorted(stats, key=lambda x: x[1]["motion"] + x[1]["detail"])
    by_median = ranked[len(ranked) // 2]

    picks: list[tuple[str, float]] = [
        ("motion", by_motion[0]),
        ("detail", by_detail[0]),
        ("dark", by_dark[0]),
        ("median", by_median[0]),
    ]
    # Dedupe starts; refill from ranked list if needed.
    starts: list[float] = []
    labels: list[str] = []
    used: set[float] = set()
    for label, start in picks:
        key = round(start, 1)
        if key in used:
            continue
        used.add(key)
        starts.append(start)
        labels.append(label)

    for start, _ in ranked:
        if len(starts) >= 4:
            break
        key = round(start, 1)
        if key in used:
            continue
        used.add(key)
        starts.append(start)
        labels.append("fill")

    while len(starts) < 4:
        starts.append(starts[-1] if starts else 0.0)
        labels.append("pad")

    return starts[:4], labels[:4], [s for _, s in stats]


async def analyze_video(path: str) -> VideoAnalysis:
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,bit_rate,codec_name:format=duration,bit_rate",
        "-of",
        "json",
        path,
    ]
    rc, stdout, stderr = await _run(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed: {stderr[-400:]}")

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

    bpp = 0.0
    if width > 0 and height > 0 and fps > 0 and bitrate > 0:
        bpp = bitrate / (width * height * fps)

    scene_starts = await _detect_scene_starts(path, duration)
    complexity = _classify_complexity(bpp, width, height, len(scene_starts), duration)
    sample_len = min(ADAPTIVE_SAMPLE_SECONDS, max(1.0, duration * 0.35))
    sample_starts, sample_labels, sample_stats = await _pick_representative_starts(
        path, duration, sample_len, scene_starts
    )
    shot_complexities = estimate_shot_complexities(scene_starts, duration, sample_stats)

    return VideoAnalysis(
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        bitrate=bitrate,
        codec=codec,
        bits_per_pixel=bpp,
        complexity=complexity,
        scene_count=len(scene_starts),
        sample_starts=sample_starts,
        sample_labels=sample_labels,
        scene_starts=scene_starts,
        sample_stats=sample_stats,
        shot_complexities=shot_complexities,
    )


def _initial_probes(encoder: str, complexity: ComplexityClass) -> list[int]:
    """Three CQ anchors; complexity only shifts the bracket."""
    if encoder == "libsvtav1" or "av1" in encoder:
        base = [28, 38, 48]
    elif encoder in {"libx265", "hevc_nvenc"} or "265" in encoder or "hevc" in encoder:
        base = [22, 28, 34]
    elif encoder in {"libx264", "h264_nvenc"} or "264" in encoder:
        base = [20, 26, 32]
    else:
        base = [28, 36, 44]

    shift = 0
    if complexity == ComplexityClass.EASY:
        shift = 4
    elif complexity == ComplexityClass.HARD:
        # Mild downward shift — keep probes in the high-ratio band (was -4, too conservative).
        shift = -2
    return [max(10, min(63, c + shift)) for c in base]


def _select_final_preset(
    encoder: str,
    complexity: ComplexityClass,
    deadline_left: float,
    *,
    duration_seconds: float = 30.0,
    width: int = 1920,
    height: int = 1080,
) -> str:
    """
    Preset by score-per-second / deadline headroom — NOT hard→always-fast.
    H200 software encodes of ~30s 4K finish in tens of seconds; use medium when safe.
    """
    from perf_db import predict_encode_seconds

    if encoder == "libsvtav1" or "av1" in encoder:
        candidates = []
        if deadline_left > 1200 and complexity == ComplexityClass.EASY:
            candidates.append(os.getenv("ADAPTIVE_SVT_PRESET_EASY", "4"))
        candidates.append(ADAPTIVE_SVT_PRESET)  # default 5
        candidates.append(os.getenv("ADAPTIVE_SVT_PRESET_HARD", "6"))
        # Pick slowest preset whose estimate fits in 55% of remaining budget.
        for preset in candidates:
            est = predict_encode_seconds(
                codec="libsvtav1",
                width=width,
                height=height,
                complexity=complexity.value,
                preset=str(preset),
                duration_seconds=max(1.0, duration_seconds),
                default_fps=0.8 if complexity == ComplexityClass.HARD else 1.5,
            )
            if est < deadline_left * 0.55:
                return str(preset)
        return os.getenv("ADAPTIVE_SVT_PRESET_HARD", "6")

    if encoder in {"libx265", "libx264"} or "265" in encoder or "264" in encoder:
        # Prefer higher quality when wall-clock allows (H200 has headroom).
        ordered = []
        if deadline_left > 1200:
            ordered.append("slow")
        elif deadline_left > 900 and complexity != ComplexityClass.HARD:
            ordered.append("slow")
        ordered.extend(["medium", "fast", "veryfast"])
        # Dedupe preserving order
        seen = set()
        presets = []
        for p in ordered:
            if p not in seen:
                seen.add(p)
                presets.append(p)
        for preset in presets:
            est = predict_encode_seconds(
                codec="libx265" if "265" in encoder or encoder == "libx265" else "libx264",
                width=width,
                height=height,
                complexity=complexity.value,
                preset=preset,
                duration_seconds=max(1.0, duration_seconds),
                default_fps=0.9 if complexity == ComplexityClass.HARD else 1.8,
            )
            if est < deadline_left * 0.55:
                return preset
        return "fast"
    return ADAPTIVE_SVT_PRESET


def _probe_preset(encoder: str, request_preset: str) -> str:
    if encoder == "libsvtav1" or "av1" in encoder:
        return ADAPTIVE_SVT_PROBE_PRESET
    if encoder in {"libx265", "libx264"} or "265" in encoder or "264" in encoder:
        return ADAPTIVE_X265_PROBE_PRESET
    return request_preset


def _append_encoder_tuning(cmd: list[str], encoder: str) -> None:
    if encoder == "libsvtav1" and ADAPTIVE_SVT_PARAMS:
        cmd.extend(["-svtav1-params", ADAPTIVE_SVT_PARAMS])
    elif encoder == "libx265" and ADAPTIVE_X265_PARAMS:
        cmd.extend(["-x265-params", ADAPTIVE_X265_PARAMS])


async def _encode_sample(
    sample_path: str,
    output_path: str,
    *,
    encoder: str,
    cq: int,
    encode_preset: str,
    request_preset: str,
    target_width: int | None = None,
    target_height: int | None = None,
) -> None:
    vf = ["setsar=1"]
    if target_width and target_height:
        w = target_width if target_width % 2 == 0 else target_width - 1
        h = target_height if target_height % 2 == 0 else target_height - 1
        vf.insert(0, f"scale={w}:{h}")

    software = encoder in {"libsvtav1", "libx264", "libx265", "libvpx-vp9"}
    cmd = [FFMPEG_BIN, "-hide_banner", "-y"]
    if not software:
        cmd.extend(["-hwaccel", "cuda"])
    cmd.extend(["-i", sample_path, "-map", "0:v:0", "-an", "-c:v", encoder])

    if encoder == "libsvtav1":
        cmd.extend(["-crf", str(cq), "-preset", encode_preset])
        _append_encoder_tuning(cmd, encoder)
    elif encoder in {"libx264", "libx265"}:
        cmd.extend(["-crf", str(cq), "-preset", encode_preset])
        _append_encoder_tuning(cmd, encoder)
    elif encoder == "libvpx-vp9":
        cmd.extend(["-crf", str(cq), "-b:v", "0"])
    else:
        cmd.extend(["-cq", str(cq), "-preset", request_preset])

    cmd.extend(["-vf", ",".join(vf), "-movflags", "+faststart", output_path])
    rc, _, stderr = await _run(cmd)
    if rc == 0 and os.path.exists(output_path):
        return

    fallback = "libsvtav1" if "av1" in encoder or encoder == "libsvtav1" else (
        "libx265" if "hevc" in encoder or "265" in encoder else "libx264"
    )
    fb_preset = ADAPTIVE_SVT_PROBE_PRESET if fallback == "libsvtav1" else ADAPTIVE_X265_PROBE_PRESET
    cmd_cpu = [
        FFMPEG_BIN,
        "-hide_banner",
        "-y",
        "-i",
        sample_path,
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        fallback,
        "-crf",
        str(cq),
        "-preset",
        fb_preset,
    ]
    _append_encoder_tuning(cmd_cpu, fallback)
    cmd_cpu.extend(["-vf", ",".join(vf), output_path])
    rc2, _, stderr2 = await _run(cmd_cpu)
    if rc2 != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"sample encode failed cq={cq}: {stderr[-300:] or stderr2[-300:]}")


async def _evaluate_cq(
    sample_paths: list[str],
    work_dir: str,
    *,
    encoder: str,
    cq: int,
    encode_preset: str,
    final_preset: str,
    request_preset: str,
    vmaf_threshold: float,
    target_width: int | None,
    target_height: int | None,
) -> CandidateResult:
    t0 = time.monotonic()

    async def _one(index: int, sample_path: str) -> tuple[int, int, float, float, float, float]:
        out_path = os.path.join(work_dir, f"cq{cq}_sample{index}.mp4")
        await _encode_sample(
            sample_path,
            out_path,
            encoder=encoder,
            cq=cq,
            encode_preset=encode_preset,
            request_preset=request_preset,
            target_width=target_width,
            target_height=target_height,
        )
        vmaf = await calculate_vmaf(sample_path, out_path)
        return (
            os.path.getsize(out_path),
            os.path.getsize(sample_path),
            vmaf.mean,
            vmaf.min_score,
            vmaf.p5,
            vmaf.score_signal,
        )

    parts = await asyncio.gather(*[_one(i, p) for i, p in enumerate(sample_paths)])
    encoded_sizes = sum(p[0] for p in parts)
    reference_sizes = sum(p[1] for p in parts)
    vmaf_means = [p[2] for p in parts]
    vmaf_mins = [p[3] for p in parts]
    vmaf_p5s = [p[4] for p in parts]
    vmaf_hmeans = [p[5] for p in parts]

    compression_rate = encoded_sizes / max(1, reference_sizes)
    compression_ratio = 1.0 / max(1e-9, compression_rate)
    vmaf_mean = sum(vmaf_means) / len(vmaf_means)
    if ADAPTIVE_GATE_MODE == "min":
        vmaf_hmean = min(vmaf_hmeans)
    else:
        vmaf_hmean = sum(vmaf_hmeans) / len(vmaf_hmeans)
    vmaf_std = 0.0
    if len(vmaf_hmeans) > 1:
        mean_h = sum(vmaf_hmeans) / len(vmaf_hmeans)
        vmaf_std = math.sqrt(sum((v - mean_h) ** 2 for v in vmaf_hmeans) / len(vmaf_hmeans))

    score, _, _, reason = calculate_compression_score(
        vmaf_score=vmaf_hmean,
        compression_rate=compression_rate,
        vmaf_threshold=vmaf_threshold,
    )

    return CandidateResult(
        cq=cq,
        preset=final_preset,
        compression_rate=compression_rate,
        compression_ratio=compression_ratio,
        vmaf_mean=vmaf_mean,
        vmaf_min=min(vmaf_mins),
        vmaf_p5=min(vmaf_p5s),
        vmaf_hmean=vmaf_hmean,
        vmaf_std=vmaf_std,
        score=score,
        reason=reason,
        encoded_bytes=encoded_sizes,
        reference_bytes=reference_sizes,
        encode_seconds=time.monotonic() - t0,
    )


def _ratio_preference(ratio: float) -> float:
    """Soft preference for the 18–22× sweet spot (diminishing returns above ~20×)."""
    mid = 0.5 * (ADAPTIVE_TARGET_RATIO_LO + ADAPTIVE_TARGET_RATIO_HI)
    if ADAPTIVE_TARGET_RATIO_LO <= ratio <= ADAPTIVE_TARGET_RATIO_HI:
        return 1.0
    if ratio < ADAPTIVE_TARGET_RATIO_LO:
        return max(0.0, ratio / ADAPTIVE_TARGET_RATIO_LO)
    # Mild penalty past the plateau — still allow higher ratio if score wins clearly.
    return max(0.55, 1.0 - 0.02 * (ratio - ADAPTIVE_TARGET_RATIO_HI))


def _pick_best(
    candidates: list[CandidateResult],
    *,
    vmaf_threshold: float,
    margin: float,
) -> CandidateResult | None:
    """
    Final pick maximizes SN85 score among candidates that clear the exact VMAF
    threshold. Margin is for search targeting only — do not block a higher-ratio
    winner that still clears threshold (was selecting 3×@0.32 over 9×@0.45).
    """
    clear = [c for c in candidates if c.gate_vmaf >= vmaf_threshold and c.score > 0]
    soft = [c for c in candidates if c.score > 0 and c not in clear]
    pool = clear if clear else soft
    if not pool:
        return None

    def rank_key(c: CandidateResult) -> tuple:
        hits_target = 1 if c.score >= ADAPTIVE_TARGET_SCORE else 0
        # Soft-prefer the 18–22× plateau; still let raw score dominate.
        in_sweet = 1 if ADAPTIVE_TARGET_RATIO_LO <= c.compression_ratio <= ADAPTIVE_TARGET_RATIO_HI else 0
        near_sweet = 1 if c.compression_ratio >= 12.0 else 0
        return (hits_target, c.score, in_sweet, near_sweet, _ratio_preference(c.compression_ratio), c.cq)

    pool.sort(key=rank_key, reverse=True)
    best = pool[0]
    # Among near-tied scores, take higher CQ (more compression) if still clearing threshold.
    for c in pool[1:]:
        if c.cq <= best.cq:
            continue
        if (best.score - c.score) <= ADAPTIVE_SCORE_EPS and c.gate_vmaf >= vmaf_threshold:
            best = c
    # If a safe (margin) candidate ties the winner on score, keep the winner (already max score).
    _ = margin  # retained for API compatibility / logging callers
    return best


def _predict_cq_for_target(
    candidates: list[CandidateResult],
    target_vmaf: float,
) -> int | None:
    """Monotonic local fit CQ → VMAF; predict highest CQ expected to pass target."""
    if len(candidates) < 2:
        return None
    ordered = sorted(candidates, key=lambda c: c.cq)
    # Find bracket where VMAF crosses target (VMAF falls as CQ rises).
    for left, right in zip(ordered, ordered[1:]):
        if left.gate_vmaf >= target_vmaf >= right.gate_vmaf:
            if abs(left.gate_vmaf - right.gate_vmaf) < 1e-6:
                return (left.cq + right.cq) // 2
            t = (left.gate_vmaf - target_vmaf) / (left.gate_vmaf - right.gate_vmaf)
            return int(round(left.cq + t * (right.cq - left.cq)))
    # All pass → push toward highest CQ with linear extrapolation from top two.
    if ordered[-1].gate_vmaf >= target_vmaf:
        a, b = ordered[-2], ordered[-1]
        slope = (b.gate_vmaf - a.gate_vmaf) / max(1, b.cq - a.cq)
        if slope >= -1e-6:
            return min(63, b.cq + 4)
        delta = (b.gate_vmaf - target_vmaf) / abs(slope)
        return max(b.cq, min(63, int(round(b.cq + delta))))
    # All fail → lower CQ.
    a, b = ordered[0], ordered[1]
    slope = (b.gate_vmaf - a.gate_vmaf) / max(1, b.cq - a.cq)
    if slope >= -1e-6:
        return max(10, a.cq - 6)
    delta = (target_vmaf - a.gate_vmaf) / abs(slope)
    return max(10, min(a.cq, int(round(a.cq - delta))))


async def optimize_compression(
    input_path: str,
    *,
    encoder: str,
    vmaf_threshold: float,
    seed_cq: int = 35,
    preset: str = "p4",
    target_width: int | None = None,
    target_height: int | None = None,
    enabled: bool | None = None,
    work_root: str | None = None,
    deadline_seconds: float | None = None,
) -> AdaptiveDecision:
    use_adaptive = ADAPTIVE_ENABLED_DEFAULT if enabled is None else enabled
    if not use_adaptive:
        return AdaptiveDecision(
            enabled=False,
            cq=seed_cq,
            preset=preset,
            analysis=None,
            message="adaptive disabled",
        )

    deadline = float(deadline_seconds or ADAPTIVE_DEFAULT_DEADLINE_SECONDS)
    search_budget = max(45.0, deadline * ADAPTIVE_SEARCH_BUDGET_FRAC)
    search_t0 = time.monotonic()

    analysis = await analyze_video(input_path)
    if analysis.duration < ADAPTIVE_MIN_DURATION_FOR_SEARCH:
        final_preset = _select_final_preset(
            encoder,
            analysis.complexity,
            deadline,
            duration_seconds=analysis.duration,
            width=analysis.width,
            height=analysis.height,
        )
        return AdaptiveDecision(
            enabled=True,
            cq=seed_cq,
            preset=final_preset,
            analysis=analysis,
            message="video too short for adaptive search; using seed CQ",
        )

    final_preset = _select_final_preset(
        encoder,
        analysis.complexity,
        deadline - (time.monotonic() - search_t0),
        duration_seconds=analysis.duration,
        width=analysis.width,
        height=analysis.height,
    )
    probe_preset = _probe_preset(encoder, preset)

    work_dir = work_root or os.path.join(
        os.path.dirname(input_path), f"adaptive_{os.path.basename(input_path)}"
    )
    os.makedirs(work_dir, exist_ok=True)
    samples_dir = os.path.join(work_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    try:
        sample_paths: list[str] = []
        sample_len = min(ADAPTIVE_SAMPLE_SECONDS, max(1.0, analysis.duration * 0.35))
        for index, start in enumerate(analysis.sample_starts):
            sample_path = os.path.join(samples_dir, f"sample_{index}.mp4")
            await extract_sample_clip(
                input_path,
                sample_path,
                start_seconds=start,
                duration_seconds=sample_len,
            )
            sample_paths.append(sample_path)

        candidates: list[CandidateResult] = []
        tried: set[int] = set()
        margin = current_dynamic_margin()
        floor = vmaf_threshold + margin

        def _budget_left() -> float:
            return search_budget - (time.monotonic() - search_t0)

        async def _try(cq: int) -> CandidateResult | None:
            nonlocal margin, floor
            cq = max(10, min(63, int(cq)))
            if cq in tried:
                return next((c for c in candidates if c.cq == cq), None)
            if len(tried) >= ADAPTIVE_MAX_CANDIDATES:
                return None
            if _budget_left() < 8.0 and candidates:
                log.info(
                    f"adaptive search budget exhausted ({search_budget:.0f}s); "
                    f"stopping before cq={cq}"
                )
                return None
            tried.add(cq)
            try:
                result = await _evaluate_cq(
                    sample_paths,
                    work_dir,
                    encoder=encoder,
                    cq=cq,
                    encode_preset=probe_preset,
                    final_preset=final_preset,
                    request_preset=preset,
                    vmaf_threshold=vmaf_threshold,
                    target_width=target_width,
                    target_height=target_height,
                )
                candidates.append(result)
                margin = current_dynamic_margin(sample_vmaf_std=result.vmaf_std)
                floor = vmaf_threshold + margin
                log.info(
                    f"adaptive cq={cq} score={result.score:.4f} "
                    f"ratio={result.compression_ratio:.2f}x "
                    f"vmaf_h={result.vmaf_hmean:.2f}±{result.vmaf_std:.2f} "
                    f"floor={floor:.2f} t={result.encode_seconds:.1f}s "
                    f"({result.reason})"
                )
                return result
            except Exception as exc:
                log.warning(f"adaptive cq={cq} failed: {exc}")
                return None

        # A/B/C: three initial probes from codec+complexity bracket.
        for probe in _initial_probes(encoder, analysis.complexity):
            if _budget_left() < 8.0 and candidates:
                break
            await _try(probe)

        # D: one correction probe — aim near exact threshold (not fat margin) to unlock ratio.
        if len(tried) < ADAPTIVE_MAX_CANDIDATES and candidates and _budget_left() >= 8.0:
            # Search uses margin; correction targets a tighter gate so we climb CQ toward 18–22×.
            correction_target = vmaf_threshold + max(0.4, min(margin, 1.2) * 0.45)
            predicted = _predict_cq_for_target(candidates, correction_target)
            if predicted is not None:
                clear_now = [c for c in candidates if c.gate_vmaf >= vmaf_threshold]
                if clear_now:
                    # If best clearer still has lots of VMAF headroom and low ratio, push CQ up.
                    best_clear = max(clear_now, key=lambda c: (c.score, c.cq))
                    if (
                        best_clear.gate_vmaf >= vmaf_threshold + 3.0
                        and best_clear.compression_ratio < ADAPTIVE_TARGET_RATIO_LO
                    ):
                        predicted = max(predicted, best_clear.cq + 3)
                    else:
                        predicted = max(predicted, max(c.cq for c in clear_now))
                if predicted not in tried:
                    await _try(predicted)

        best = _pick_best(candidates, vmaf_threshold=vmaf_threshold, margin=margin)
        search_seconds = time.monotonic() - search_t0
        encode_deadline = max(60.0, deadline * ADAPTIVE_ENCODE_BUDGET_FRAC - search_seconds)

        # Re-pick encode preset with remaining budget (H200: prefer medium over fast).
        final_preset = _select_final_preset(
            encoder,
            analysis.complexity,
            encode_deadline,
            duration_seconds=analysis.duration,
            width=analysis.width,
            height=analysis.height,
        )
        if best:
            best.preset = final_preset

        chosen_cq = best.cq if best else seed_cq
        x265_zones = None
        if encoder in {"libx265", "libx264"} or "265" in encoder:
            shots = build_shot_zones(
                analysis.scene_starts,
                analysis.duration,
                complexities=analysis.shot_complexities,
            )
            x265_zones = build_x265_zones_param(shots, fps=analysis.fps, base_cq=chosen_cq)

        if not best:
            return AdaptiveDecision(
                enabled=True,
                cq=seed_cq,
                preset=final_preset,
                analysis=analysis,
                candidates=candidates,
                margin_used=margin,
                search_seconds=search_seconds,
                probes=len(tried),
                x265_zones=x265_zones,
                encode_deadline_seconds=encode_deadline,
                message="no viable candidate; falling back to seed CQ",
            )

        return AdaptiveDecision(
            enabled=True,
            cq=best.cq,
            preset=final_preset,
            analysis=analysis,
            candidates=candidates,
            selected=best,
            margin_used=margin,
            search_seconds=search_seconds,
            probes=len(tried),
            x265_zones=x265_zones,
            encode_deadline_seconds=encode_deadline,
            message=(
                f"selected cq={best.cq} score={best.score:.4f} "
                f"ratio={best.compression_ratio:.2f}x "
                f"vmaf_h={best.vmaf_hmean:.2f} floor={floor:.2f} "
                f"margin={margin:.2f} preset={final_preset} "
                f"probes={len(tried)} search={search_seconds:.1f}s "
                f"encode_budget={encode_deadline:.0f}s "
                f"zones={'yes' if x265_zones else 'no'} "
                f"complexity={analysis.complexity.value} "
                f"samples={analysis.sample_labels}"
            ),
        )
    finally:
        if os.getenv("ADAPTIVE_KEEP_WORK", "false").lower() not in ("1", "true", "yes"):
            shutil.rmtree(work_dir, ignore_errors=True)
