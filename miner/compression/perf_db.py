"""Codec-specific encode performance database for deadline planning."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("compression.perf_db")

PERF_DB_PATH = os.getenv(
    "ADAPTIVE_PERF_DB",
    "/tmp/organic-proxy/codec_perf_db.json",
)


def _key(codec: str, width: int, height: int, complexity: str, preset: str) -> str:
    # Bucket resolution to keep the table small.
    pixels = max(1, width * height)
    if pixels >= 3840 * 2160:
        res = "4k"
    elif pixels >= 1920 * 1080:
        res = "1080p"
    elif pixels >= 1280 * 720:
        res = "720p"
    else:
        res = "sd"
    return f"{codec}|{res}|{complexity}|{preset}"


def _load() -> dict[str, Any]:
    path = Path(PERF_DB_PATH)
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning(f"perf db load failed: {exc}")
    return {"entries": {}, "updated_at": None}


def _save(data: dict[str, Any]) -> None:
    path = Path(PERF_DB_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data["updated_at"] = time.time()
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning(f"perf db save failed: {exc}")


def record_encode(
    *,
    codec: str,
    width: int,
    height: int,
    complexity: str,
    preset: str,
    duration_seconds: float,
    wall_seconds: float,
    cq: int | None = None,
    compression_ratio: float | None = None,
) -> None:
    if duration_seconds <= 0 or wall_seconds <= 0:
        return
    fps = duration_seconds / wall_seconds
    data = _load()
    entries = data.setdefault("entries", {})
    k = _key(codec, width, height, complexity, preset)
    prev = entries.get(k) or {"samples": 0, "encode_fps_ema": fps}
    n = int(prev.get("samples", 0))
    alpha = 0.25 if n else 1.0
    ema = (1 - alpha) * float(prev.get("encode_fps_ema", fps)) + alpha * fps
    entry = {
        "samples": n + 1,
        "encode_fps_ema": ema,
        "last_cq": cq,
        "last_ratio": compression_ratio,
        "last_wall": wall_seconds,
    }
    entries[k] = entry
    _save(data)


def predict_encode_seconds(
    *,
    codec: str,
    width: int,
    height: int,
    complexity: str,
    preset: str,
    duration_seconds: float,
    default_fps: float = 2.0,
) -> float:
    """Wall-clock estimate for a full encode."""
    data = _load()
    entry = (data.get("entries") or {}).get(_key(codec, width, height, complexity, preset))
    fps = float(entry.get("encode_fps_ema", default_fps)) if entry else default_fps
    fps = max(0.05, fps)
    # Hard content / slow presets: slightly pessimistic.
    if complexity == "hard":
        fps *= 0.85
    return duration_seconds / fps


def can_finish_two_pass(estimate_one_pass: float, deadline_left: float) -> bool:
    """Two-pass ≈ 1.7× one-pass cost; require 35% headroom for 5-video batches."""
    return (estimate_one_pass * 1.7) < (deadline_left * 0.65)


def suggest_fallback_preset(encoder: str, current: str) -> str:
    if encoder == "libsvtav1" or "av1" in encoder:
        # 4K RA cannot go above 9 on SVT-AV1 v4.1.
        try:
            return str(min(9, int(current) + 1))
        except ValueError:
            return "9"
    order = ["veryslow", "slower", "slow", "medium", "fast", "faster", "veryfast", "superfast", "ultrafast"]
    cur = (current or "medium").lower()
    if cur in order:
        idx = order.index(cur)
        return order[min(len(order) - 1, idx + 2)]
    return "veryfast"
