"""Subnet compression score simulator — mirrors validator scoring_function.py."""

from __future__ import annotations

import math


def calculate_compression_score(
    vmaf_score: float,
    compression_rate: float,
    vmaf_threshold: float,
    compression_weight: float = 0.70,
    quality_weight: float = 0.30,
    soft_threshold_margin: float = 5.0,
) -> tuple[float, float, float, str]:
    """
    Returns (final_score, compression_component, quality_component, reason).
    compression_rate = compressed_size / original_size (lower is better).
    """
    if abs(compression_weight + quality_weight - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {compression_weight + quality_weight}")

    hard_cutoff = vmaf_threshold - soft_threshold_margin

    if compression_rate >= 0.80:
        compression_ratio = 1 / compression_rate if compression_rate > 0 else 1.0
        return (
            0.0,
            0.0,
            0.0,
            f"No meaningful compression (ratio: {compression_ratio:.2f}x). Minimum 1.25x required.",
        )

    if vmaf_score < hard_cutoff:
        return 0.0, 0.0, 0.0, f"VMAF {vmaf_score:.2f} below hard cutoff ({hard_cutoff:.2f})"

    normalization_factor = 1.12
    compression_ratio = 1 / compression_rate if compression_rate > 0 else 1.0

    if vmaf_score < vmaf_threshold:
        soft_zone_position = (vmaf_score - hard_cutoff) / soft_threshold_margin
        quality_factor = 0.7 * (soft_zone_position**2)
        if compression_ratio <= 20:
            compression_component = ((compression_ratio - 1) / 19) ** 1.5
        else:
            compression_component = 1.0 + 0.3 * math.log(compression_ratio / 20)
        final_score = (compression_component * quality_factor) / normalization_factor
        return (
            min(1.0, final_score),
            compression_component,
            quality_factor,
            f"VMAF {vmaf_score:.2f} in soft zone",
        )

    vmaf_excess = vmaf_score - vmaf_threshold
    max_vmaf_excess = max(1e-6, 100 - vmaf_threshold)
    quality_component = 0.7 + 0.3 * min(1.0, vmaf_excess / max_vmaf_excess)

    if compression_ratio <= 20:
        compression_component = ((compression_ratio - 1.25) / 18.75) ** 0.9
    else:
        compression_component = 1.0 + 0.1 * math.log(compression_ratio / 20)

    final_score = (
        compression_weight * compression_component + quality_weight * quality_component
    ) / normalization_factor
    return min(1.0, final_score), compression_component, quality_component, "success"
