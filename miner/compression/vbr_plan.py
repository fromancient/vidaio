"""VBR planner: hit requested bitrate accurately; two-pass when deadline allows."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from perf_db import can_finish_two_pass, predict_encode_seconds

log = logging.getLogger("compression.vbr")


@dataclass
class VbrPlan:
    target_bitrate_bps: int
    two_pass: bool
    preset: str
    bufsize_bps: int
    maxrate_bps: int
    reason: str


def plan_vbr(
    *,
    encoder: str,
    target_bitrate_bps: int,
    duration_seconds: float,
    width: int,
    height: int,
    complexity: str,
    preset: str,
    deadline_left: float,
) -> VbrPlan:
    """
    Validators allow bitrate <= target + 10%. Aim slightly under so mux/audio
    never trips the upper bound. Two-pass is disabled on the SN85 180s path —
    5 concurrent clips cannot afford ~1.7× encode cost.
    """
    # Single application of target fraction (rd_predictor passes raw target).
    video_bps = max(50_000, int(target_bitrate_bps * float(os.getenv("VBR_TARGET_FRAC", "0.92"))))
    # Constrain peaks so average stays stable (x264/x265 only; SVT ignores maxrate).
    maxrate = int(video_bps * 1.08)
    bufsize = int(video_bps * 2.0)

    estimate = predict_encode_seconds(
        codec=encoder,
        width=width,
        height=height,
        complexity=complexity,
        preset=preset,
        duration_seconds=max(1.0, duration_seconds),
    )
    force_one_pass = os.getenv("SN85_FORCE_ONE_PASS_VBR", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    two_pass = False
    reason = "one-pass constrained VBR"
    # Under validator deadlines, always one-pass. Two-pass only for long organic jobs.
    if (
        not force_one_pass
        and deadline_left >= 90.0
        and encoder in {"libx264", "libx265"}
        and can_finish_two_pass(estimate, deadline_left)
    ):
        two_pass = True
        reason = f"two-pass VBR (est_1pass={estimate:.0f}s, left={deadline_left:.0f}s)"
    elif deadline_left < estimate * 1.15:
        reason = f"one-pass rushed (est={estimate:.0f}s, left={deadline_left:.0f}s)"
        log.warning(reason)
    elif force_one_pass:
        reason = f"one-pass VBR (SN85 fast path, est={estimate:.0f}s)"

    return VbrPlan(
        target_bitrate_bps=video_bps,
        two_pass=two_pass,
        preset=preset,
        bufsize_bps=bufsize,
        maxrate_bps=maxrate,
        reason=reason,
    )
