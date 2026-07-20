"""
Fast rate-distortion planner for SN85 under a 165s batch deadline.

Aligned to validator scoring (services/scoring/scoring_function.py):
  - 70% compression ratio + 30% VMAF quality
  - Hard zero if VMAF < threshold-5 or ratio < 1.25×
  - Only top-5 compression accumulate_score get on-chain weights

No online multi-probe VMAF. Prefer clearing the VMAF threshold with
aggressive CQ so ratio lands in the competitive band (~8–15× on typical
~25–30 Mbps 4K sources), not the previous ~2× fat encodes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from source_features import SourceFeatures

log = logging.getLogger("compression.rd_predictor")

# Prefer finishing over slow quality — presets must encode ~30s 4K in <<30s wall
# under concurrent load (validator dendrite timeout=180s).
DEFAULT_SVT_PRESET = os.getenv("FAST_SVT_PRESET", "9")
DEFAULT_X265_PRESET = os.getenv("FAST_X265_PRESET", "ultrafast")

# Aim slightly under validator VBR target so mux/audio never trips +10% cap.
VBR_TARGET_FRAC = float(os.getenv("VBR_TARGET_FRAC", "0.92"))


@dataclass
class EncodePlan:
    cq: int
    preset: str
    codec_mode: str  # CRF | VBR
    target_bitrate_bps: int | None
    margin: float
    expected_ratio_lo: float
    expected_ratio_hi: float
    expected_score_lo: float
    expected_score_hi: float
    reason: str


def _threshold_bucket(vmaf_threshold: float) -> int:
    if vmaf_threshold >= 91:
        return 93
    if vmaf_threshold >= 87:
        return 89
    return 85


def _av1_cq(threshold: int, complexity: str) -> int:
    """
    SVT CQ (higher = smaller). Tuned for ultrafast/preset-9 on 4K:
    prior 36/32/30 hard left files too large; push for ratio while keeping
    headroom above hard cutoff (threshold-5).
    """
    table = {
        85: {"easy": 52, "medium": 48, "hard": 44},
        89: {"easy": 48, "medium": 44, "hard": 40},
        93: {"easy": 42, "medium": 38, "hard": 34},
    }
    return table[threshold][complexity]


def _hevc_crf(threshold: int, complexity: str) -> int:
    """
    x265 CRF (higher = smaller). Prior hard@89 CQ22 → ~10–16 Mbps (~2×) and
    score ~0.25 — not top-5 viable. Target ~8–14× on ~27 Mbps sources.
    """
    table = {
        85: {"easy": 44, "medium": 40, "hard": 36},
        89: {"easy": 40, "medium": 36, "hard": 32},
        93: {"easy": 32, "medium": 28, "hard": 25},
    }
    return table[threshold][complexity]


def _h264_crf(threshold: int, complexity: str) -> int:
    table = {
        85: {"easy": 38, "medium": 34, "hard": 30},
        89: {"easy": 34, "medium": 30, "hard": 26},
        93: {"easy": 30, "medium": 26, "hard": 22},
    }
    return table[threshold][complexity]


def _expected_band(threshold: int, complexity: str) -> tuple[float, float, float, float]:
    """(ratio_lo, ratio_hi, score_lo, score_hi) if VMAF clears threshold."""
    bands = {
        85: {
            "easy": (14.0, 22.0, 0.60, 0.80),
            "medium": (11.0, 18.0, 0.52, 0.72),
            "hard": (9.0, 15.0, 0.45, 0.68),
        },
        89: {
            "easy": (12.0, 18.0, 0.55, 0.75),
            "medium": (9.0, 15.0, 0.45, 0.68),
            "hard": (7.0, 12.0, 0.40, 0.60),
        },
        93: {
            "easy": (7.0, 12.0, 0.40, 0.55),
            "medium": (5.5, 9.0, 0.35, 0.48),
            "hard": (4.5, 8.0, 0.30, 0.45),
        },
    }
    return bands[threshold][complexity]


def _dynamic_margin(complexity: str) -> float:
    base = {"easy": 0.5, "medium": 0.7, "hard": 0.9}.get(complexity, 0.7)
    try:
        from adaptive import current_dynamic_margin

        m = current_dynamic_margin()
        return max(0.4, min(1.2, m))
    except Exception:
        return base


def select_plan(
    features: SourceFeatures,
    *,
    encoder: str,
    vmaf_threshold: float,
    codec_mode: str = "CRF",
    target_bitrate_bps: int | None = None,
    remaining_seconds: float = 30.0,
) -> EncodePlan:
    thr = _threshold_bucket(vmaf_threshold)
    complexity = features.complexity
    margin = _dynamic_margin(complexity)
    rlo, rhi, slo, shi = _expected_band(thr, complexity)
    mode = (codec_mode or "CRF").upper()
    is_svt = encoder == "libsvtav1" or "av1" in encoder
    is_hevc = encoder in {"libx265"} or "265" in encoder or "hevc" in encoder

    if is_svt:
        cq = _av1_cq(thr, complexity)
        if remaining_seconds < 20:
            preset = "9"
        else:
            preset = DEFAULT_SVT_PRESET
        try:
            preset = str(min(9, int(preset)))
        except ValueError:
            preset = "9"
        cq_cap = 55 if thr <= 85 else (50 if thr <= 89 else 45)
    elif is_hevc:
        cq = _hevc_crf(thr, complexity)
        preset = "ultrafast" if remaining_seconds < 20 else DEFAULT_X265_PRESET
        cq_cap = 48 if thr <= 85 else (44 if thr <= 89 else 38)
    else:
        cq = _h264_crf(thr, complexity)
        preset = "ultrafast" if remaining_seconds < 20 else DEFAULT_X265_PRESET
        cq_cap = 45 if thr <= 85 else (40 if thr <= 89 else 35)


    # Ratio-first nudges (never lower CQ for motion — that caused ~2× encodes).
    br = float(features.bitrate or 0.0)
    if br >= 20_000_000:
        cq += 2
    if br >= 35_000_000:
        cq += 2
    if complexity == "easy":
        cq += 2
    # VMAF93 hard cutoff is 88 — keep a small quality cushion on hardest clips only.
    if thr >= 93 and complexity == "hard" and (features.motion_proxy > 12 or features.detail_proxy > 50):
        cq = max(10, cq - 1)

    cq = min(cq_cap, max(10, cq))

    if mode == "VBR" and target_bitrate_bps:
        # Pass raw target; vbr_plan.py applies VBR_TARGET_FRAC once.
        video_bps = max(50_000, int(target_bitrate_bps))
        return EncodePlan(
            cq=cq,
            preset=preset,
            codec_mode="VBR",
            target_bitrate_bps=video_bps,
            margin=margin,
            expected_ratio_lo=rlo,
            expected_ratio_hi=rhi,
            expected_score_lo=slo,
            expected_score_hi=shi,
            reason=(
                f"vbr thr={thr} c={complexity} preset={preset} "
                f"br_raw={video_bps} (frac applied in vbr_plan)"
            ),
        )

    return EncodePlan(
        cq=cq,
        preset=preset,
        codec_mode="CRF",
        target_bitrate_bps=None,
        margin=margin,
        expected_ratio_lo=rlo,
        expected_ratio_hi=rhi,
        expected_score_lo=slo,
        expected_score_hi=shi,
        reason=(
            f"crf thr={thr} c={complexity} cq={cq} preset={preset} "
            f"expect≈{rlo:.0f}-{rhi:.0f}x score≈{slo:.2f}-{shi:.2f}"
        ),
    )
