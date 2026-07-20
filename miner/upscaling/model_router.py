"""
Model router for upscaling candidates.

Currently ships Real-ESRGAN via Video2X. BasicVSR++ / RealBasicVSR hooks are
ready for when subnet fine-tuned weights are installed under UPSCALE_MODEL_DIR.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("upscaling.router")

MODEL_DIR = os.getenv("UPSCALE_MODEL_DIR", "/models/upscaling")


class Route(str, Enum):
    REALESRGAN = "realesrgan"
    BASICVSRPP = "basicvsrpp"
    REALBASICVSR = "realbasicvsr"
    ANIME = "anime"


@dataclass
class RouteDecision:
    route: Route
    model_name: str
    reason: str


def _weights_exist(name: str) -> bool:
    path = os.path.join(MODEL_DIR, name)
    return os.path.isdir(path) or os.path.isfile(path)


def select_route(
    *,
    hint: str | None = None,
    severe_artifacts: bool = False,
    anime: bool = False,
) -> RouteDecision:
    if anime:
        return RouteDecision(Route.ANIME, "realesr-animevideov3", "anime/line-art hint")

    if severe_artifacts and _weights_exist("realbasicvsr"):
        return RouteDecision(Route.REALBASICVSR, "realbasicvsr", "severe artifacts + weights present")

    if _weights_exist("basicvsrpp") and (hint or "").lower() in {"basicvsr", "basicvsrpp", "temporal"}:
        return RouteDecision(Route.BASICVSRPP, "basicvsrpp", "temporal model requested + weights present")

    # Default stock path — competitive baseline until fine-tunes are available.
    model = os.getenv("UPSCALE_DEFAULT_REALESRGAN_MODEL", "realesr-animevideov3")
    return RouteDecision(Route.REALESRGAN, model, "default Real-ESRGAN via Video2X")
