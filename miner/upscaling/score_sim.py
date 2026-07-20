"""Local upscaling score helpers mirroring validator upscaling_scoring.py."""

from __future__ import annotations

import math


def calculate_length_score(content_length: float) -> float:
    return math.log(1 + content_length) / math.log(1 + 320)


def calculate_preliminary_score(
    quality_score: float,
    length_score: float,
    quality_weight: float = 0.5,
    length_weight: float = 0.5,
) -> float:
    return (quality_score * quality_weight) + (length_score * length_weight)


def calculate_final_score(s_pre: float) -> float:
    return 0.1 * math.exp(6.979 * (s_pre - 0.5))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def calculate_quality_score(pieapp_score: float) -> float:
    sigmoid_normalized_score = sigmoid(pieapp_score)
    original_at_zero = (1 - (math.log10(sigmoid(0) + 1) / math.log10(3.5))) ** 2.5
    original_at_two = (1 - (math.log10(sigmoid(2.0) + 1) / math.log10(3.5))) ** 2.5
    original_value = (1 - (math.log10(sigmoid_normalized_score + 1) / math.log10(3.5))) ** 2.5
    return 1 - ((original_value - original_at_zero) / (original_at_two - original_at_zero))


# Approximate length contribution at equal quality (for planning).
LENGTH_SCORE_5S = calculate_length_score(5)
LENGTH_SCORE_10S = calculate_length_score(10)
