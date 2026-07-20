#!/usr/bin/env python3
"""
Generate validator-like synthetic upscaling training pairs.

Mirrors the public idea: start from high-res video, apply downscale + compression
+ blur + noise combinations, write (lq, hq) pairs for BasicVSR++/RealBasicVSR fine-tunes.

Usage:
  python scripts/generate_upscale_degradations.py --input /path/to/hq.mp4 --out /data/pairs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path


FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")


DEGRADATIONS = [
    {"name": "down2_bicubic", "vf": "scale=iw/2:ih/2:flags=bicubic"},
    {"name": "down2_bilinear", "vf": "scale=iw/2:ih/2:flags=bilinear"},
    {"name": "down2_blur", "vf": "scale=iw/2:ih/2:flags=bicubic,gblur=sigma=1.2"},
    {"name": "down2_noise", "vf": "scale=iw/2:ih/2:flags=bicubic,noise=alls=8:allf=t"},
    {
        "name": "down2_compress",
        "vf": "scale=iw/2:ih/2:flags=bicubic",
        "encode": ["-c:v", "libx264", "-crf", "32", "-preset", "veryfast"],
    },
    {
        "name": "down4_mixed",
        "vf": "scale=iw/4:ih/4:flags=bicubic,gblur=sigma=0.8,noise=alls=5:allf=t",
        "encode": ["-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
    },
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="High-quality source video")
    ap.add_argument("--out", required=True, help="Output directory for pairs")
    ap.add_argument("--max-seconds", type=float, default=12.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hq = out / "hq.mp4"
    run(
        [
            FFMPEG,
            "-y",
            "-i",
            args.input,
            "-t",
            str(args.max_seconds),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-an",
            str(hq),
        ]
    )

    manifest = []
    for deg in DEGRADATIONS:
        lq = out / f"lq_{deg['name']}.mp4"
        cmd = [FFMPEG, "-y", "-i", str(hq), "-vf", deg["vf"]]
        cmd.extend(deg.get("encode") or ["-c:v", "libx264", "-crf", "23", "-preset", "veryfast"])
        cmd.extend(["-an", str(lq)])
        run(cmd)
        manifest.append({"hq": str(hq), "lq": str(lq), "degradation": deg["name"]})

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} pairs → {out} (id={uuid.uuid4().hex[:8]})")


if __name__ == "__main__":
    main()
