"""Per-shot complexity → x265 zone CRF offsets (single continuous encode)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("compression.zones")


@dataclass
class ShotZone:
    start_seconds: float
    end_seconds: float
    complexity: float  # higher = harder → lower CRF (more bits)
    label: str = ""


def build_shot_zones(
    scene_starts: list[float],
    duration: float,
    *,
    complexities: list[float] | None = None,
) -> list[ShotZone]:
    """Build contiguous shot intervals from scene cuts."""
    cuts = sorted(t for t in scene_starts if 0.0 < t < duration)
    bounds = [0.0] + cuts + [duration]
    shots: list[ShotZone] = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        if end - start < 0.25:
            continue
        complexity = 1.0
        if complexities and i < len(complexities):
            complexity = max(0.1, float(complexities[i]))
        shots.append(ShotZone(start_seconds=start, end_seconds=end, complexity=complexity, label=f"shot{i}"))
    if not shots:
        shots = [ShotZone(0.0, duration, 1.0, "full")]
    return shots


def complexity_to_crf_delta(complexity: float, mean_complexity: float) -> int:
    """
    Harder shots get negative CRF delta (more bits); easy shots get positive.
    Clamp to ±4 so the global CQ search remains valid.
    """
    if mean_complexity <= 1e-9:
        return 0
    ratio = complexity / mean_complexity
    if ratio >= 1.45:
        return -3
    if ratio >= 1.20:
        return -2
    if ratio >= 1.08:
        return -1
    if ratio <= 0.55:
        return 3
    if ratio <= 0.75:
        return 2
    if ratio <= 0.90:
        return 1
    return 0


def build_x265_zones_param(
    shots: list[ShotZone],
    *,
    fps: float,
    base_cq: int,
) -> str | None:
    """
    x265 zones=startFrame,endFrame,crf=N/...
    Returns None when zones would be a no-op (single shot / zero deltas).
    """
    if len(shots) < 2 or fps <= 0:
        return None

    mean_c = sum(s.complexity for s in shots) / len(shots)
    parts: list[str] = []
    any_delta = False
    for shot in shots:
        delta = complexity_to_crf_delta(shot.complexity, mean_c)
        if delta:
            any_delta = True
        crf = max(10, min(51, int(base_cq) + delta))
        start_f = max(0, int(shot.start_seconds * fps))
        end_f = max(start_f + 1, int(shot.end_seconds * fps))
        parts.append(f"{start_f},{end_f},crf={crf}")

    if not any_delta:
        return None
    zones = "/".join(parts)
    log.info(f"x265 zones ({len(parts)} shots, base_cq={base_cq}): {zones[:240]}")
    return zones


def estimate_shot_complexities(
    scene_starts: list[float],
    duration: float,
    sample_stats: list[dict[str, float]],
) -> list[float]:
    """Map sparse sample stats onto shot intervals (motion+detail+darkness)."""
    cuts = sorted(t for t in scene_starts if 0.0 < t < duration)
    bounds = [0.0] + cuts + [duration]
    complexities: list[float] = []
    for i in range(len(bounds) - 1):
        mid = 0.5 * (bounds[i] + bounds[i + 1])
        if sample_stats:
            nearest = min(
                sample_stats,
                key=lambda s: abs(float(s.get("start", 0.0)) - mid),
            )
            score = (
                float(nearest.get("motion", 0.0))
                + float(nearest.get("detail", 0.0))
                + 0.5 * float(nearest.get("darkness", 0.0))
            )
            complexities.append(max(0.1, score))
        else:
            complexities.append(1.0)
    return complexities
