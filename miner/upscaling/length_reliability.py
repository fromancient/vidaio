"""Advertise longest upscaling length only when p95 completion is safe."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

log = logging.getLogger("upscaling.length")

STATE_PATH = os.getenv(
    "UPSCALE_LENGTH_STATE",
    "/tmp/organic-proxy/upscale_length_state.json",
)
# Require p95 wall time comfortably under validator deadline.
SAFE_P95_FRACTION = float(os.getenv("UPSCALE_LENGTH_SAFE_P95_FRAC", "0.70"))
DEFAULT_DEADLINE = float(os.getenv("UPSCALE_REQUEST_DEADLINE_SECONDS", "900"))


def _load() -> dict:
    path = Path(STATE_PATH)
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning(f"length state load failed: {exc}")
    return {"samples_10s": [], "samples_5s": [], "recommended": 10}


def _save(state: dict) -> None:
    path = Path(STATE_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = time.time()
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning(f"length state save failed: {exc}")


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(0.95 * len(ordered)) - 1)))
    return ordered[idx]


def record_upscale_runtime(content_length: int, wall_seconds: float, success: bool) -> None:
    state = _load()
    key = "samples_10s" if content_length >= 10 else "samples_5s"
    samples = list(state.get(key) or [])
    if success:
        samples.append(float(wall_seconds))
    else:
        # Failures count as deadline blows for conservatism.
        samples.append(float(DEFAULT_DEADLINE))
    state[key] = samples[-40:]
    state["recommended"] = recommend_max_content_length(state=state)
    _save(state)


def recommend_max_content_length(state: dict | None = None, deadline: float | None = None) -> int:
    """
    Advertise 10s only when enough successful samples show p95 under safe budget.
    Otherwise fall back to 5s.
    """
    state = state or _load()
    deadline = float(deadline or DEFAULT_DEADLINE)
    safe = deadline * SAFE_P95_FRACTION
    samples = [float(x) for x in (state.get("samples_10s") or [])]
    if len(samples) < 5:
        # Not enough evidence — keep configured preference if never measured.
        return int(state.get("recommended") or 10)
    if _p95(samples) <= safe:
        return 10
    return 5


def get_recommended_length() -> int:
    return int(_load().get("recommended") or recommend_max_content_length())
