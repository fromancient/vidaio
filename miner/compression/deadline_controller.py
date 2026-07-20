"""Hard wall-clock budgets for SN85 compression (validator dendrite timeout=180s)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass


# Validator waits ~180s for the full 5-video synapse response.
VALIDATOR_DEADLINE_SECONDS = float(os.getenv("SN85_VALIDATOR_DEADLINE_SECONDS", "180"))
# Internal hard stop — leave headroom for axon serialization.
INTERNAL_DEADLINE_SECONDS = float(os.getenv("SN85_INTERNAL_DEADLINE_SECONDS", "165"))
DOWNLOAD_BUDGET = float(os.getenv("SN85_DOWNLOAD_BUDGET_SECONDS", "8"))
ANALYZE_BUDGET = float(os.getenv("SN85_ANALYZE_BUDGET_SECONDS", "3"))
ENCODE_BUDGET = float(os.getenv("SN85_ENCODE_BUDGET_SECONDS", "130"))
UPLOAD_BUDGET = float(os.getenv("SN85_UPLOAD_BUDGET_SECONDS", "19"))


@dataclass
class DeadlineClock:
    """Monotonic deadline for one validator compression round."""

    start: float
    hard_deadline: float

    @classmethod
    def start_round(cls, seconds: float | None = None) -> "DeadlineClock":
        now = time.monotonic()
        limit = float(seconds if seconds is not None else INTERNAL_DEADLINE_SECONDS)
        return cls(start=now, hard_deadline=now + limit)

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def remaining(self) -> float:
        return max(0.0, self.hard_deadline - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def slice(self, max_seconds: float) -> float:
        """Timeout for the next step: min(requested, remaining)."""
        return max(0.05, min(max_seconds, self.remaining()))

    def encode_timeout_per_video(self, n_videos: int, concurrency: int) -> float:
        """
        Wall budget for encodes. With 5 videos and concurrency 5, each gets ~encode_budget.
        With concurrency 2, waves of ceil(n/c) stretch the same wall.
        """
        waves = max(1, (max(1, n_videos) + max(1, concurrency) - 1) // max(1, concurrency))
        # Leave upload/validate cushion from remaining time.
        usable = max(5.0, self.remaining() - UPLOAD_BUDGET - 3.0)
        per_wave = usable / waves
        # Cap so a single stuck encode cannot eat the whole round.
        return max(8.0, min(per_wave, ENCODE_BUDGET / max(1, waves)))
