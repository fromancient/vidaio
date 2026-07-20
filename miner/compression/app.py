"""
Compression microservice — wraps the ffmpeg binary installed in this container.

Accepts a video path (on shared volume) OR a URL, codec, and quality settings,
runs GPU-accelerated ffmpeg compression, returns the output path or S3 URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import Optional

import boto3
import httpx
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from adaptive import ADAPTIVE_ENABLED_DEFAULT, optimize_compression, record_vmaf_calibration
from deadline_controller import ENCODE_BUDGET, INTERNAL_DEADLINE_SECONDS
from integrity import validate_output_matches_source
from perf_db import record_encode, suggest_fallback_preset
from rd_predictor import select_plan
from source_features import extract_source_features
from vbr_plan import plan_vbr
from vmaf_local import calculate_vmaf_aligned_window

# Fast RD is the production path — full multi-probe VMAF search blows the 180s validator timeout.
FAST_RD_MODE = os.getenv("FAST_RD_MODE", "true").lower() in ("1", "true", "yes")
ADAPTIVE_FULL_SEARCH = os.getenv("ADAPTIVE_FULL_SEARCH", "false").lower() in ("1", "true", "yes")
# Per-clip encode wall when 5 videos run concurrently under a 165s round.
DEFAULT_PER_CLIP_ENCODE_TIMEOUT = float(os.getenv("FAST_ENCODE_TIMEOUT_SECONDS", "55"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("compression")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _detect_nvenc_available()
    cleanup_task = asyncio.create_task(_cleanup_worker())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(title="Video Compression Service", lifespan=lifespan)

SHARED_VOLUME_PATH = os.getenv("SHARED_VOLUME_PATH", "/tmp/organic-proxy")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_COMPRESSION", "3"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE_COMPRESSION") or os.getenv("MAX_QUEUE_SIZE", "5"))
DISABLE_REMOTE_IO = os.getenv("DISABLE_REMOTE_IO", "false").lower() in ("1", "true", "yes")

# Storage provider label only. Uploads use one S3-compatible code path.
STORAGE_PROVIDER = os.getenv("MINER_STORAGE_PROVIDER", "s3").lower()
S3_REGION = os.getenv("MINER_STORAGE_S3_REGION", "us-east-1").strip() or "us-east-1"
S3_BUCKET = os.getenv("MINER_STORAGE_S3_BUCKET_NAME", "").strip()
S3_ACCESS_KEY_ID = os.getenv("MINER_STORAGE_S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("MINER_STORAGE_S3_SECRET_ACCESS_KEY", "").strip()
S3_ENDPOINT_URL = (
    os.getenv("MINER_STORAGE_S3_ENDPOINT_URL")
    or os.getenv("MINER_STORAGE_S3_ENDPOINT")
    or ""
).strip().rstrip("+").rstrip("/")
S3_PRESIGNED_EXPIRY = int(
    os.getenv("MINER_STORAGE_S3_PRESIGNED_EXPIRY")
    or os.getenv("S3_PRESIGNED_EXPIRY")
    or "3600"
)
PRESIGNED_URL_CLEANUP_GRACE_SECONDS = int(os.getenv("PRESIGNED_URL_CLEANUP_GRACE_SECONDS", "600"))
TEMP_FILE_TTL_SECONDS = int(
    os.getenv("MINER_TEMP_FILE_TTL_SECONDS")
    or os.getenv("TEMP_FILE_TTL_SECONDS")
    or str(min(S3_PRESIGNED_EXPIRY, 604800) + PRESIGNED_URL_CLEANUP_GRACE_SECONDS)
)
CLEANUP_INTERVAL_SECONDS = int(
    os.getenv("MINER_CLEANUP_INTERVAL_SECONDS") or os.getenv("CLEANUP_INTERVAL_SECONDS") or "300"
)
CLEANUP_MAX_VOLUME_BYTES = int(
    os.getenv("MINER_CLEANUP_MAX_VOLUME_BYTES") or os.getenv("CLEANUP_MAX_VOLUME_BYTES") or "9000000000"
)
CLEANUP_MIN_FILE_AGE_SECONDS = int(
    os.getenv("MINER_CLEANUP_MIN_FILE_AGE_SECONDS") or os.getenv("CLEANUP_MIN_FILE_AGE_SECONDS") or "60"
)
CLEANUP_ENABLED = os.getenv("MINER_CLEANUP_ENABLED", os.getenv("CLEANUP_ENABLED", "true")).lower() in (
    "1",
    "true",
    "yes",
)
STORAGE_CLEANUP_ENABLED = os.getenv(
    "MINER_STORAGE_CLEANUP_ENABLED", os.getenv("STORAGE_CLEANUP_ENABLED", "true")
).lower() in ("1", "true", "yes")
STORAGE_CLEANUP_PREFIXES = [
    prefix.strip()
    for prefix in os.getenv("MINER_STORAGE_CLEANUP_PREFIXES", "processing/,upscaling/").split(",")
    if prefix.strip()
]
STORAGE_OBJECT_TTL_SECONDS = int(
    os.getenv("MINER_STORAGE_OBJECT_TTL_SECONDS")
    or os.getenv("STORAGE_OBJECT_TTL_SECONDS")
    or str(min(S3_PRESIGNED_EXPIRY, 604800) + PRESIGNED_URL_CLEANUP_GRACE_SECONDS)
)
COMPRESSION_CHUNKING_ENABLED = os.getenv("COMPRESSION_CHUNKING_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
COMPRESSION_CHUNK_MIN_DURATION_SECONDS = int(os.getenv("COMPRESSION_CHUNK_MIN_DURATION_SECONDS", "1200"))
COMPRESSION_CHUNK_TARGET_SECONDS = int(os.getenv("COMPRESSION_CHUNK_TARGET_SECONDS", "600"))
COMPRESSION_CHUNK_PARALLELISM = max(1, int(os.getenv("COMPRESSION_CHUNK_PARALLELISM", "2")))
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
# Default final-encode presets (adaptive may override per-request via req.preset).
SVT_PRESET = os.getenv("ADAPTIVE_SVT_PRESET", os.getenv("FAST_SVT_PRESET", "9"))
X265_PRESET = os.getenv("ADAPTIVE_X265_PRESET", os.getenv("FAST_X265_PRESET", "ultrafast"))
SVT_PARAMS = os.getenv(
    "ADAPTIVE_SVT_PARAMS",
    "tune=0",  # keep light — overlays/tf cost wall-clock under 180s deadline
)
X265_PARAMS = os.getenv(
    "ADAPTIVE_X265_PARAMS",
    "aq-mode=3:rc-lookahead=5:frame-threads=4",
)
# Cap encoder threads so concurrent clips don't oversubscribe 24-core H200.
def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


_SVT_LP = _env_int("SVT_LP", max(2, (os.cpu_count() or 8) // max(1, MAX_CONCURRENT)))
_X265_POOLS = _env_int("X265_POOLS", max(2, (os.cpu_count() or 8) // max(1, MAX_CONCURRENT)))
COMPRESSION_DEADLINE_SECONDS = float(
    os.getenv("MINER_COMPRESSION_SERVICE_TIMEOUT_SECONDS", str(DEFAULT_PER_CLIP_ENCODE_TIMEOUT + 20))
)
_X265_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
}

_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue_size = 0
_active_count = 0
_active_file_paths: set[str] = set()
_lock = asyncio.Lock()

# Codec name → preferred encoder mapping. H100/H200 datacenter GPUs often
# lack NVENC; software fallbacks are selected automatically at startup.
CODEC_MAP = {
    "AV1": "av1_nvenc",
    "H264": "h264_nvenc",
    "H.264": "h264_nvenc",
    "HEVC": "hevc_nvenc",
    "H265": "hevc_nvenc",
    "H.265": "hevc_nvenc",
    "VP9": "libvpx-vp9",
}

SOFTWARE_CODEC_MAP = {
    "AV1": "libsvtav1",
    "H264": "libx264",
    "H.264": "libx264",
    "HEVC": "libx265",
    "H265": "libx265",
    "H.265": "libx265",
    "VP9": "libvpx-vp9",
}

_nvenc_available: bool | None = None


async def _detect_nvenc_available() -> bool:
    global _nvenc_available
    if _nvenc_available is not None:
        return _nvenc_available
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:d=0.1",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    returncode, _, stderr, run_error = await _run_process(cmd, "nvenc-probe", "probe")
    ok = returncode == 0 and not run_error
    if not ok:
        log.warning(
            "NVENC unavailable on this GPU/runtime; using software encoders "
            f"(libsvtav1/libx265/libx264). detail={(run_error or stderr.decode(errors='replace'))[-200:]}"
        )
    _nvenc_available = ok
    return ok


def _resolve_encoder(codec: str, prefer_nvenc: bool) -> str:
    key = codec.upper()
    if prefer_nvenc:
        return CODEC_MAP.get(key, "av1_nvenc")
    return SOFTWARE_CODEC_MAP.get(key, "libsvtav1")


def _is_software_encoder(encoder: str) -> bool:
    return encoder in {"libsvtav1", "libx264", "libx265", "libvpx-vp9"}


def _software_preset(encoder: str, req_preset: str) -> str:
    """Prefer adaptive-selected preset when it looks like a real encoder preset."""
    if encoder == "libsvtav1":
        if req_preset and str(req_preset).isdigit():
            return str(req_preset)
        return SVT_PRESET
    if encoder in {"libx264", "libx265"}:
        if req_preset and str(req_preset).lower() in _X265_PRESETS:
            return str(req_preset).lower()
        return X265_PRESET
    return req_preset or SVT_PRESET


def _svt_preset_for_source(req_preset: str, *, width: int = 0, height: int = 0) -> str:
    """Clamp SVT preset for 4K — v4.1 RA mode rejects >9 at UHD."""
    raw = _software_preset("libsvtav1", req_preset)
    try:
        level = int(raw)
    except ValueError:
        level = int(SVT_PRESET) if str(SVT_PRESET).isdigit() else 9
    # Challenge clips are almost always 4K; keep safe default even if probe skipped.
    uhd = (width * height) >= (3840 * 2160) or width == 0
    if uhd:
        level = min(level, 9)
    return str(max(0, min(12, level)))


def _merge_encoder_params(base: str, extra: str) -> str:
    parts = [p for p in (base or "").split(":") if p]
    seen = {p.split("=", 1)[0] for p in parts}
    for p in (extra or "").split(":"):
        if not p:
            continue
        key = p.split("=", 1)[0]
        if key not in seen:
            parts.append(p)
            seen.add(key)
    return ":".join(parts)


def _svt_params_with_threads(color: dict[str, str] | None = None) -> str:
    """Size SVT logical processors to nproc/concurrency; tag color for validator gates."""
    params = _merge_encoder_params(SVT_PARAMS, f"lp={_SVT_LP}")
    if color:
        mapping = {
            "color_primaries": "color-primaries",
            "color_transfer": "transfer-characteristics",
            "color_space": "matrix-coefficients",
        }
        extras = []
        for src_key, dst_key in mapping.items():
            val = (color.get(src_key) or "").strip()
            if val and val not in {"unknown", "unspecified"}:
                extras.append(f"{dst_key}={val}")
        if extras:
            params = _merge_encoder_params(params, ":".join(extras))
    return params


def _x265_params_with_threads(
    zones: str | None = None,
    color: dict[str, str] | None = None,
) -> str:
    params = _merge_encoder_params(
        X265_PARAMS, f"pools={_X265_POOLS}:frame-threads={min(4, _X265_POOLS)}"
    )
    if color:
        mapping = {
            "color_primaries": "colorprim",
            "color_transfer": "transfer",
            "color_space": "colormatrix",
        }
        extras = []
        for src_key, dst_key in mapping.items():
            val = (color.get(src_key) or "").strip()
            if val and val not in {"unknown", "unspecified"}:
                extras.append(f"{dst_key}={val}")
        if extras:
            params = _merge_encoder_params(params, ":".join(extras))
    if zones:
        params = _merge_encoder_params(params, f"zones={zones}")
    return params


async def _probe_color_tags(path: str) -> dict[str, str]:
    """Read source color tags so outputs pass validator colorspace gates."""
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=color_space,color_primaries,color_transfer",
        "-of",
        "json",
        path,
    ]
    try:
        rc, stdout, stderr, err = await _run_process(cmd, "color-probe", "probe", timeout=3.0)
        if rc != 0 or err:
            return {}
        data = json.loads(stdout.decode(errors="replace") or "{}")
        stream = (data.get("streams") or [{}])[0]
        out: dict[str, str] = {}
        for key in ("color_space", "color_primaries", "color_transfer"):
            val = stream.get(key)
            if val and str(val) not in {"unknown", "unspecified", "none"}:
                out[key] = str(val)
        return out
    except Exception as exc:
        log.debug(f"color probe failed: {exc}")
        return {}


def _build_ffmpeg_args(
    local_input: str,
    output_path: str,
    req: "CompressRequest",
    encoder: str,
    *,
    x265_zones: str | None = None,
    vbr_bitrate_bps: int | None = None,
    vbr_maxrate_bps: int | None = None,
    vbr_bufsize_bps: int | None = None,
    pass_num: int | None = None,
    passlogfile: str | None = None,
    color: dict[str, str] | None = None,
    width: int = 0,
    height: int = 0,
) -> list[str]:
    ffmpeg_args = [
        FFMPEG_BIN,
        "-y",
        "-threads",
        str(max(2, _SVT_LP)),
    ]
    if not _is_software_encoder(encoder):
        ffmpeg_args.extend(["-hwaccel", "cuda"])

    ffmpeg_args.extend([
        "-i", local_input,
        "-map", "0:v:0",
    ])
    if pass_num != 1:
        ffmpeg_args.extend(["-map", "0:a?"])
    ffmpeg_args.extend(["-c:v", encoder])

    bitrate = vbr_bitrate_bps or req.target_bitrate
    if req.codec_mode == "VBR" and bitrate:
        ffmpeg_args.extend(["-b:v", str(int(bitrate))])
        # SVT-AV1: -maxrate/-bufsize map to mbr which is CRF-only → hard fail in VBR.
        if encoder != "libsvtav1":
            if vbr_maxrate_bps:
                ffmpeg_args.extend(["-maxrate", str(int(vbr_maxrate_bps))])
            if vbr_bufsize_bps:
                ffmpeg_args.extend(["-bufsize", str(int(vbr_bufsize_bps))])
        if encoder == "libsvtav1":
            ffmpeg_args.extend([
                "-preset",
                _svt_preset_for_source(req.preset, width=width, height=height),
            ])
            params = _svt_params_with_threads(color)
            if params:
                ffmpeg_args.extend(["-svtav1-params", params])
        elif encoder in {"libx264", "libx265"}:
            ffmpeg_args.extend(["-preset", _software_preset(encoder, req.preset)])
            if pass_num is not None:
                ffmpeg_args.extend(["-pass", str(pass_num)])
                if passlogfile:
                    ffmpeg_args.extend(["-passlogfile", passlogfile])
            if encoder == "libx265":
                params = _x265_params_with_threads(x265_zones, color)
                if params:
                    ffmpeg_args.extend(["-x265-params", params])
        elif not _is_software_encoder(encoder):
            ffmpeg_args.extend(["-preset", req.preset])
    elif encoder == "libsvtav1":
        ffmpeg_args.extend([
            "-crf",
            str(req.cq),
            "-preset",
            _svt_preset_for_source(req.preset, width=width, height=height),
        ])
        params = _svt_params_with_threads(color)
        if params:
            ffmpeg_args.extend(["-svtav1-params", params])
    elif encoder in {"libx264", "libx265"}:
        ffmpeg_args.extend(["-crf", str(req.cq), "-preset", _software_preset(encoder, req.preset)])
        if encoder == "libx265":
            params = _x265_params_with_threads(x265_zones, color)
            if params:
                ffmpeg_args.extend(["-x265-params", params])
    elif encoder == "libvpx-vp9":
        ffmpeg_args.extend(["-crf", str(req.cq), "-b:v", "0"])
    else:
        ffmpeg_args.extend(["-cq", str(req.cq), "-preset", req.preset])

    # Validator hard-requires yuv420p + 1:1 SAR; color tags must match source.
    video_filters = []
    if req.target_width and req.target_height:
        w = req.target_width if req.target_width % 2 == 0 else req.target_width - 1
        h = req.target_height if req.target_height % 2 == 0 else req.target_height - 1
        video_filters.append(f"scale={w}:{h}")

    video_filters.append("setsar=1")
    video_filters.append("format=yuv420p")
    ffmpeg_args.extend(["-vf", ",".join(video_filters)])
    ffmpeg_args.extend(["-pix_fmt", "yuv420p"])

    if color:
        if color.get("color_space"):
            ffmpeg_args.extend(["-colorspace", color["color_space"]])
        if color.get("color_primaries"):
            ffmpeg_args.extend(["-color_primaries", color["color_primaries"]])
        if color.get("color_transfer"):
            ffmpeg_args.extend(["-color_trc", color["color_transfer"]])

    if pass_num == 1:
        ffmpeg_args.extend(["-an", "-f", "null", os.devnull])
    else:
        ffmpeg_args.extend(["-c:a", "copy", "-sn", "-dn", "-movflags", "+faststart", output_path])
    return ffmpeg_args


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _get_s3_client():
    client_kwargs = {
        "region_name": S3_REGION,
        "aws_access_key_id": S3_ACCESS_KEY_ID or None,
        "aws_secret_access_key": S3_SECRET_ACCESS_KEY or None,
        "config": Config(signature_version="s3v4"),
    }
    if S3_ENDPOINT_URL:
        client_kwargs["endpoint_url"] = S3_ENDPOINT_URL
    return boto3.client("s3", **client_kwargs)


def _storage_config_status() -> dict[str, object]:
    return {
        "provider": STORAGE_PROVIDER,
        "region": S3_REGION,
        "bucket_configured": bool(S3_BUCKET),
        "access_key_configured": bool(S3_ACCESS_KEY_ID),
        "secret_key_configured": bool(S3_SECRET_ACCESS_KEY),
        "endpoint_configured": bool(S3_ENDPOINT_URL),
    }


def _compression_config_status() -> dict[str, object]:
    return {
        "chunking_enabled": COMPRESSION_CHUNKING_ENABLED,
        "chunk_min_duration_seconds": COMPRESSION_CHUNK_MIN_DURATION_SECONDS,
        "chunk_target_seconds": COMPRESSION_CHUNK_TARGET_SECONDS,
        "chunk_parallelism": COMPRESSION_CHUNK_PARALLELISM,
        "adaptive_enabled_default": ADAPTIVE_ENABLED_DEFAULT,
        "fast_rd_mode": FAST_RD_MODE,
        "adaptive_full_search": ADAPTIVE_FULL_SEARCH,
        "adaptive_vmaf_margin": float(os.getenv("ADAPTIVE_VMAF_MARGIN", "0.8")),
        "per_clip_encode_timeout": DEFAULT_PER_CLIP_ENCODE_TIMEOUT,
        "internal_round_deadline": INTERNAL_DEADLINE_SECONDS,
        "encode_budget": ENCODE_BUDGET,
        "deadline_seconds": COMPRESSION_DEADLINE_SECONDS,
        "svt_preset": SVT_PRESET,
        "x265_preset": X265_PRESET,
        "svt_lp": _SVT_LP,
        "x265_pools": _X265_POOLS,
        "force_one_pass_vbr": os.getenv("SN85_FORCE_ONE_PASS_VBR", "true").lower() in ("1", "true", "yes"),
        "vmaf_model": os.getenv("ADAPTIVE_VMAF_MODEL", "version=vmaf_v0.6.1neg"),
    }


def _validate_s3_config():
    missing = []
    if not S3_BUCKET:
        missing.append("MINER_STORAGE_S3_BUCKET_NAME")
    if not S3_ACCESS_KEY_ID:
        missing.append("MINER_STORAGE_S3_ACCESS_KEY_ID")
    if not S3_SECRET_ACCESS_KEY:
        missing.append("MINER_STORAGE_S3_SECRET_ACCESS_KEY")

    if missing:
        raise RuntimeError(f"Missing storage configuration: {', '.join(missing)}")


async def _download_url(url: str, dest: str):
    log.info(f"Downloading {url[:80]}... → {dest}")
    async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
    log.info(f"Downloaded {os.path.getsize(dest) / (1024*1024):.1f} MB → {dest}")


def _upload_to_s3(local_path: str, key: str) -> str:
    _validate_s3_config()
    client = _get_s3_client()
    client.upload_file(local_path, S3_BUCKET, key)
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=min(S3_PRESIGNED_EXPIRY, 604800),
    )
    log.info(f"Uploaded to {STORAGE_PROVIDER} storage: s3://{S3_BUCKET}/{key}")
    return url


def _cleanup(*paths: str):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _cleanup_tree(path: str):
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _format_process_output(stdout: bytes, stderr: bytes) -> str:
    parts = []
    stdout_msg = stdout.decode(errors="replace").strip()
    stderr_msg = stderr.decode(errors="replace").strip()
    if stdout_msg:
        parts.append(f"stdout:\n{stdout_msg}")
    if stderr_msg:
        parts.append(f"stderr:\n{stderr_msg}")
    return "\n\n".join(parts)


async def _run_process(
    cmd: list[str],
    task_label: str,
    step: str,
    *,
    timeout: float | None = None,
) -> tuple[int | None, bytes, bytes, str]:
    proc = None
    try:
        log.info(f"[{task_label}] {step}: {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if timeout and timeout > 0:
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                log.error(f"[{task_label}] {step} timed out after {timeout:.0f}s — killing")
                with suppress(ProcessLookupError):
                    proc.kill()
                with suppress(Exception):
                    await proc.communicate()
                return None, b"", b"", f"timeout after {timeout:.0f}s"
        else:
            stdout, stderr = await proc.communicate()
        return proc.returncode, stdout, stderr, ""
    except FileNotFoundError:
        return None, b"", b"", f"{cmd[0]} binary not found"
    except OSError as e:
        return None, b"", b"", f"Failed to start {cmd[0]}: {e}"


async def _probe_duration_seconds(path: str, task_label: str) -> float | None:
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    returncode, stdout, stderr, run_error = await _run_process(cmd, task_label, "ffprobe duration")
    if returncode != 0 or run_error:
        detail = run_error or stderr.decode(errors="replace").strip()
        log.warning(f"[{task_label}] Failed to probe duration: {detail}")
        return None
    try:
        return float(stdout.decode().strip())
    except ValueError:
        log.warning(f"[{task_label}] ffprobe returned invalid duration: {stdout!r}")
        return None


async def _probe_segment_duration_seconds(path: str, task_label: str) -> float:
    duration = await _probe_duration_seconds(path, task_label)
    if duration is None:
        raise RuntimeError(f"Unable to probe segment duration: {path}")
    return duration


def _format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


async def _log_chunk_seams(segments: list[str], task_label: str) -> list[float]:
    durations = [
        await _probe_segment_duration_seconds(segment, task_label)
        for segment in segments
    ]
    seams: list[float] = []
    elapsed = 0.0
    for duration in durations[:-1]:
        elapsed += duration
        seams.append(elapsed)

    log.info(
        f"[{task_label}] chunk_seams "
        + json.dumps(
            {
                "seam_seconds": [round(seam, 3) for seam in seams],
                "seam_timestamps": [_format_timestamp(seam) for seam in seams],
                "segment_durations_seconds": [round(duration, 3) for duration in durations],
            }
        )
    )
    return seams


def _should_chunk(req: "CompressRequest", duration_seconds: float | None) -> bool:
    if req.chunked is not None:
        return req.chunked
    return (
        COMPRESSION_CHUNKING_ENABLED
        and duration_seconds is not None
        and duration_seconds >= COMPRESSION_CHUNK_MIN_DURATION_SECONDS
    )


def _concat_file_line(path: str) -> str:
    escaped = path.replace("'", "'\\''")
    return f"file '{escaped}'\n"


async def _split_at_keyframes(
    local_input: str,
    segments_dir: str,
    task_label: str,
    chunk_duration_seconds: int,
) -> list[str]:
    os.makedirs(segments_dir, exist_ok=True)
    segment_pattern = os.path.join(segments_dir, "input_%05d.mp4")
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-y",
        "-i", local_input,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c", "copy",
        "-sn",
        "-dn",
        "-f", "segment",
        "-segment_time", str(chunk_duration_seconds),
        "-reset_timestamps", "1",
        "-segment_format", "mp4",
        segment_pattern,
    ]
    returncode, stdout, stderr, run_error = await _run_process(cmd, task_label, "split segments")
    if returncode != 0 or run_error:
        detail = run_error or _format_process_output(stdout, stderr) or "segment split failed"
        raise RuntimeError(detail)

    segments = sorted(
        os.path.join(segments_dir, filename)
        for filename in os.listdir(segments_dir)
        if filename.startswith("input_") and filename.endswith(".mp4")
    )
    if not segments:
        raise RuntimeError("segment split produced no files")
    return segments


async def _compress_chunked(
    local_input: str,
    output_path: str,
    req: "CompressRequest",
    encoder: str,
    task_label: str,
) -> None:
    chunk_duration_seconds = req.chunk_duration_seconds or COMPRESSION_CHUNK_TARGET_SECONDS
    parallelism = max(1, req.chunk_parallelism or COMPRESSION_CHUNK_PARALLELISM)
    work_dir = os.path.join(SHARED_VOLUME_PATH, f"{task_label}_chunks")
    encoded_dir = os.path.join(work_dir, "encoded")
    tracked_paths: list[str] = []

    try:
        input_segments = await _split_at_keyframes(local_input, work_dir, task_label, chunk_duration_seconds)
        if len(input_segments) < 2:
            raise RuntimeError("segment split produced one chunk; falling back to single-pass compression")
        await _log_chunk_seams(input_segments, task_label)

        os.makedirs(encoded_dir, exist_ok=True)
        encoded_segments = [
            os.path.join(encoded_dir, f"encoded_{index:05d}.mp4")
            for index, _ in enumerate(input_segments)
        ]
        tracked_paths = [*input_segments, *encoded_segments]
        await _track_temp_files(*tracked_paths)

        log.info(
            f"[{task_label}] Compressing {len(input_segments)} chunks "
            f"(target={chunk_duration_seconds}s, parallelism={parallelism})"
        )
        semaphore = asyncio.Semaphore(parallelism)

        async def _compress_one(index: int, segment_input: str, segment_output: str):
            async with semaphore:
                cmd = _build_ffmpeg_args(segment_input, segment_output, req, encoder)
                returncode, stdout, stderr, run_error = await _run_process(
                    cmd,
                    task_label,
                    f"compress chunk {index + 1}/{len(input_segments)}",
                )
                if returncode != 0 or run_error:
                    detail = run_error or _format_process_output(stdout, stderr) or "chunk compression failed"
                    raise RuntimeError(f"chunk {index + 1} failed: {detail}")
                if not os.path.exists(segment_output):
                    raise RuntimeError(f"chunk {index + 1} output missing: {segment_output}")

        chunk_results = await asyncio.gather(
            *[
                _compress_one(index, segment_input, encoded_segments[index])
                for index, segment_input in enumerate(input_segments)
            ],
            return_exceptions=True,
        )
        chunk_errors = [result for result in chunk_results if isinstance(result, Exception)]
        if chunk_errors:
            raise RuntimeError(str(chunk_errors[0]))

        concat_list = os.path.join(work_dir, "concat.txt")
        with open(concat_list, "w", encoding="utf-8") as file:
            for segment in encoded_segments:
                file.write(_concat_file_line(segment))

        cmd = [
            FFMPEG_BIN,
            "-hide_banner",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        returncode, stdout, stderr, run_error = await _run_process(cmd, task_label, "merge chunks")
        if returncode != 0 or run_error:
            detail = run_error or _format_process_output(stdout, stderr) or "chunk merge failed"
            raise RuntimeError(detail)
    finally:
        await _untrack_temp_files(*tracked_paths)
        _cleanup_tree(work_dir)


def _cleanup_config_status() -> dict[str, object]:
    return {
        "enabled": CLEANUP_ENABLED,
        "interval_seconds": CLEANUP_INTERVAL_SECONDS,
        "temp_file_ttl_seconds": TEMP_FILE_TTL_SECONDS,
        "max_volume_bytes": CLEANUP_MAX_VOLUME_BYTES,
        "min_file_age_seconds": CLEANUP_MIN_FILE_AGE_SECONDS,
        "presigned_url_expiry_seconds": min(S3_PRESIGNED_EXPIRY, 604800),
        "presigned_url_cleanup_grace_seconds": PRESIGNED_URL_CLEANUP_GRACE_SECONDS,
        "storage_cleanup_enabled": STORAGE_CLEANUP_ENABLED,
        "storage_object_ttl_seconds": STORAGE_OBJECT_TTL_SECONDS,
        "storage_cleanup_prefixes": STORAGE_CLEANUP_PREFIXES,
    }


def _shared_root() -> str:
    return os.path.abspath(SHARED_VOLUME_PATH)


def _normalize_path(path: str) -> str:
    return os.path.abspath(path)


def _is_shared_path(path: str) -> bool:
    try:
        return os.path.commonpath([_shared_root(), _normalize_path(path)]) == _shared_root()
    except ValueError:
        return False


async def _track_temp_files(*paths: str):
    async with _lock:
        for path in paths:
            if path and _is_shared_path(path):
                _active_file_paths.add(_normalize_path(path))


async def _untrack_temp_files(*paths: str):
    async with _lock:
        for path in paths:
            if path:
                _active_file_paths.discard(_normalize_path(path))


async def _protected_paths_snapshot() -> set[str]:
    async with _lock:
        return set(_active_file_paths)


def _remove_stale_file(path: str, reason: str) -> int:
    try:
        size = os.path.getsize(path)
        os.remove(path)
        log.info(f"Removed {reason} temp file: {path} ({size} bytes)")
        return size
    except FileNotFoundError:
        return 0
    except OSError as e:
        log.warning(f"Failed to remove temp file {path}: {e}")
        return 0


async def _cleanup_shared_volume_once():
    if not CLEANUP_ENABLED or not os.path.isdir(SHARED_VOLUME_PATH):
        return

    now = time.time()
    protected_paths = await _protected_paths_snapshot()
    total_bytes = 0
    candidates: list[tuple[float, str, int]] = []

    for root, _, files in os.walk(SHARED_VOLUME_PATH):
        for filename in files:
            path = os.path.abspath(os.path.join(root, filename))
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                continue

            total_bytes += stat.st_size
            if path in protected_paths:
                continue

            candidates.append((stat.st_mtime, path, stat.st_size))
            if now - stat.st_mtime >= TEMP_FILE_TTL_SECONDS:
                total_bytes -= _remove_stale_file(path, "expired")

    if CLEANUP_MAX_VOLUME_BYTES <= 0 or total_bytes <= CLEANUP_MAX_VOLUME_BYTES:
        return

    for _, path, size in sorted(candidates):
        if total_bytes <= CLEANUP_MAX_VOLUME_BYTES:
            break
        if path in protected_paths or not os.path.exists(path):
            continue
        try:
            file_age = now - os.path.getmtime(path)
        except FileNotFoundError:
            continue
        if file_age < CLEANUP_MIN_FILE_AGE_SECONDS:
            continue
        total_bytes -= _remove_stale_file(path, "over-quota")


def _storage_cleanup_ready() -> bool:
    return (
        STORAGE_CLEANUP_ENABLED
        and STORAGE_OBJECT_TTL_SECONDS > 0
        and bool(STORAGE_CLEANUP_PREFIXES)
        and bool(S3_BUCKET)
        and bool(S3_ACCESS_KEY_ID)
        and bool(S3_SECRET_ACCESS_KEY)
    )


def _cleanup_expired_storage_objects_once() -> int:
    if not _storage_cleanup_ready():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=STORAGE_OBJECT_TTL_SECONDS)
    client = _get_s3_client()
    deleted = 0

    for prefix in STORAGE_CLEANUP_PREFIXES:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            expired_objects = []
            for obj in page.get("Contents", []):
                last_modified = obj.get("LastModified")
                if last_modified is None:
                    continue
                if last_modified.tzinfo is None:
                    last_modified = last_modified.replace(tzinfo=timezone.utc)
                if last_modified < cutoff:
                    expired_objects.append({"Key": obj["Key"]})

            for index in range(0, len(expired_objects), 1000):
                batch = expired_objects[index : index + 1000]
                if not batch:
                    continue
                response = client.delete_objects(
                    Bucket=S3_BUCKET,
                    Delete={"Objects": batch, "Quiet": True},
                )
                deleted += len(batch) - len(response.get("Errors", []))

    if deleted:
        log.info(f"Deleted {deleted} expired {STORAGE_PROVIDER} object(s)")
    return deleted


async def _cleanup_worker():
    if not CLEANUP_ENABLED and not _storage_cleanup_ready():
        log.info("Cleanup worker disabled")
        return

    log.info(
        "Starting cleanup worker "
        f"(path={SHARED_VOLUME_PATH}, ttl={TEMP_FILE_TTL_SECONDS}s, "
        f"interval={CLEANUP_INTERVAL_SECONDS}s, max_bytes={CLEANUP_MAX_VOLUME_BYTES}, "
        f"storage_ttl={STORAGE_OBJECT_TTL_SECONDS}s)"
    )

    while True:
        try:
            if CLEANUP_ENABLED:
                await _cleanup_shared_volume_once()
            if _storage_cleanup_ready():
                await asyncio.to_thread(_cleanup_expired_storage_objects_once)
        except Exception as e:
            log.warning(f"Cleanup pass failed: {e}")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)


def _queue_capacity() -> int:
    return MAX_CONCURRENT + MAX_QUEUE_SIZE


def _queue_snapshot_locked() -> dict[str, int]:
    queued_tasks = max(0, _queue_size - _active_count)
    return {
        "max_concurrent": MAX_CONCURRENT,
        "max_queue_size": MAX_QUEUE_SIZE,
        "active_tasks": _active_count,
        "queued_tasks": queued_tasks,
        "total_pending": _queue_size,
        "queue_capacity_remaining": max(0, _queue_capacity() - _queue_size),
    }


async def _queue_snapshot() -> dict[str, int]:
    async with _lock:
        return _queue_snapshot_locked()


@asynccontextmanager
async def _queued_task(task_label: str):
    global _queue_size

    async with _lock:
        if _queue_size >= _queue_capacity():
            snapshot = _queue_snapshot_locked()
            detail = (
                f"Compression queue full: {snapshot['queued_tasks']}/{MAX_QUEUE_SIZE} queued, "
                f"{snapshot['active_tasks']}/{MAX_CONCURRENT} active"
            )
            log.warning(f"[{task_label}] {detail}")
            raise HTTPException(status_code=429, detail=detail)

        _queue_size += 1
        queue_position = max(0, _queue_size - MAX_CONCURRENT)
        snapshot = _queue_snapshot_locked()

    try:
        yield queue_position, snapshot
    finally:
        async with _lock:
            _queue_size = max(0, _queue_size - 1)


@asynccontextmanager
async def _running_task():
    global _active_count

    async with _semaphore:
        async with _lock:
            _active_count += 1
            snapshot = _queue_snapshot_locked()

        try:
            yield snapshot
        finally:
            async with _lock:
                _active_count = max(0, _active_count - 1)


class CompressRequest(BaseModel):
    video_paths: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Input video path or URL list, up to 5 items",
    )
    task_id: str = Field("", description="Task ID for logging")
    codec: str = Field("AV1", description="Target codec: AV1, H264, HEVC, VP9")
    codec_mode: str = Field("CRF", description="Rate control mode: CRF or VBR")
    cq: int = Field(35, description="Seed CQ / fallback (lower = higher quality)")
    preset: str = Field("p4", description="Encoder preset")
    target_bitrate: Optional[int] = Field(None, description="Target bitrate in bps (for VBR mode)")
    target_width: Optional[int] = Field(None, description="Target width for downscaling")
    target_height: Optional[int] = Field(None, description="Target height for downscaling")
    chunked: Optional[bool] = Field(None, description="Override automatic long-video chunking")
    chunk_duration_seconds: Optional[int] = Field(None, description="Target chunk duration")
    chunk_parallelism: Optional[int] = Field(None, description="Parallel chunk encodes per request")
    vmaf_threshold: Optional[float] = Field(
        None,
        description="Validator VMAF threshold used by adaptive CQ search (e.g. 85/89/93)",
    )
    adaptive: Optional[bool] = Field(
        None,
        description="Enable adaptive CQ search. Defaults to ADAPTIVE_COMPRESSION env.",
    )
    encode_timeout_seconds: Optional[float] = Field(
        None,
        description="Hard encode wall-clock for this clip (SN85 batch deadline slice)",
    )


class CompressResponse(BaseModel):
    output_paths: list[str] = Field(default_factory=list, description="Per-input local output paths")
    output_urls: list[str] = Field(default_factory=list, description="Per-input S3 presigned URLs")
    errors: list[Optional[str]] = Field(default_factory=list, description="Per-input errors")
    success: bool
    active_tasks: Optional[int] = None
    queued_tasks: Optional[int] = None


@app.get("/health")
async def health():
    snapshot = await _queue_snapshot()
    return {
        "status": "ok",
        **snapshot,
        "storage": _storage_config_status(),
        "cleanup": _cleanup_config_status(),
        "compression": _compression_config_status(),
    }


@app.get("/queue")
async def queue_status():
    return await _queue_snapshot()


async def _calibrate_sample_vs_output(
    source_path: str,
    output_path: str,
    predicted_vmaf: float | None,
    task_label: str,
) -> None:
    """Update dynamic margin from sample-predicted vs aligned mid-clip VMAF."""
    if predicted_vmaf is None or predicted_vmaf <= 0:
        return
    try:
        duration = await _probe_duration_seconds(source_path, task_label) or 0.0
        start = max(0.0, duration * 0.40)
        sample_len = min(2.5, max(1.0, duration * 0.08)) if duration > 0 else 2.0
        measured = await calculate_vmaf_aligned_window(
            source_path,
            output_path,
            start_seconds=start,
            duration_seconds=sample_len,
        )
        under = predicted_vmaf - measured.score_signal
        record_vmaf_calibration(predicted_vmaf, measured.score_signal)
        log.info(
            f"[{task_label}] calibration pred={predicted_vmaf:.2f} "
            f"measured={measured.score_signal:.2f} under={under:.2f}"
        )
    except Exception as exc:
        log.warning(f"[{task_label}] calibration skipped: {exc}")


async def _encode_with_deadline(
    local_input: str,
    output_path: str,
    req: "CompressRequest",
    encoder: str,
    task_label: str,
    *,
    encode_timeout: float,
    x265_zones: str | None = None,
    duration_seconds: float | None = None,
    complexity: str = "medium",
    width: int = 0,
    height: int = 0,
    color: dict[str, str] | None = None,
) -> tuple[int | None, bytes, bytes, str]:
    """Final encode with hard timeout and one faster fallback before failure."""
    encode_t0 = time.monotonic()

    def _args(active_req, zones=None, **kwargs):
        return _build_ffmpeg_args(
            local_input,
            output_path,
            active_req,
            encoder,
            x265_zones=zones,
            color=color,
            width=width,
            height=height,
            **kwargs,
        )

    async def _once(active_req: "CompressRequest", zones: str | None, label: str, timeout: float):
        if active_req.codec_mode.upper() == "VBR" and active_req.target_bitrate:
            plan = plan_vbr(
                encoder=encoder,
                target_bitrate_bps=int(active_req.target_bitrate),
                duration_seconds=float(duration_seconds or 30.0),
                width=width or 1920,
                height=height or 1080,
                complexity=complexity,
                preset=_software_preset(encoder, active_req.preset),
                deadline_left=timeout,
            )
            log.info(f"[{task_label}] vbr_plan {plan.reason} bitrate={plan.target_bitrate_bps}")
            active_req = active_req.model_copy(update={"preset": plan.preset})
            if plan.two_pass and encoder in {"libx264", "libx265"}:
                passlog = os.path.join(SHARED_VOLUME_PATH, f"{task_label}_ffmpeg2pass")
                cmd1 = _args(
                    active_req,
                    vbr_bitrate_bps=plan.target_bitrate_bps,
                    vbr_maxrate_bps=plan.maxrate_bps,
                    vbr_bufsize_bps=plan.bufsize_bps,
                    pass_num=1,
                    passlogfile=passlog,
                )
                half = max(30.0, timeout * 0.45)
                rc1, so1, se1, err1 = await _run_process(cmd1, task_label, f"{label} pass1", timeout=half)
                if rc1 != 0 or err1:
                    return rc1, so1, se1, err1 or "vbr pass1 failed"
                cmd2 = _args(
                    active_req,
                    zones,
                    vbr_bitrate_bps=plan.target_bitrate_bps,
                    vbr_maxrate_bps=plan.maxrate_bps,
                    vbr_bufsize_bps=plan.bufsize_bps,
                    pass_num=2,
                    passlogfile=passlog,
                )
                return await _run_process(
                    cmd2, task_label, f"{label} pass2", timeout=max(30.0, timeout - (time.monotonic() - encode_t0))
                )
            cmd = _args(
                active_req,
                zones,
                vbr_bitrate_bps=plan.target_bitrate_bps,
                vbr_maxrate_bps=plan.maxrate_bps,
                vbr_bufsize_bps=plan.bufsize_bps,
            )
            return await _run_process(cmd, task_label, label, timeout=timeout)

        cmd = _args(active_req, zones)
        return await _run_process(cmd, task_label, label, timeout=timeout)

    # Reserve a real fallback window up front — leftover after a full timeout is useless (~4s).
    primary_budget = max(8.0, encode_timeout * 0.78)
    fallback_budget = max(0.0, encode_timeout - primary_budget)
    rc, stdout, stderr, err = await _once(req, x265_zones, "final encode", primary_budget)
    if err.startswith("timeout") or (rc is not None and rc != 0):
        if fallback_budget >= 10.0:
            fb_preset = suggest_fallback_preset(encoder, req.preset)
            fb_cq = max(10, int(req.cq) - 1) if req.codec_mode.upper() != "VBR" else req.cq
            fb_req = req.model_copy(update={"preset": fb_preset, "cq": fb_cq})
            log.warning(
                f"[{task_label}] encode fallback preset={fb_preset} cq={fb_cq} "
                f"timeout={fallback_budget:.0f}s (prior: {err or rc})"
            )
            _cleanup(output_path)
            rc, stdout, stderr, err = await _once(fb_req, None, "fallback encode", fallback_budget)
        else:
            log.error(
                f"[{task_label}] encode failed and fallback budget too small "
                f"({fallback_budget:.0f}s); prior={err or rc}"
            )

    wall = time.monotonic() - encode_t0
    if rc == 0 and duration_seconds:
        record_encode(
            codec=encoder,
            width=width or 1920,
            height=height or 1080,
            complexity=complexity,
            preset=str(req.preset),
            duration_seconds=float(duration_seconds),
            wall_seconds=wall,
            cq=req.cq,
        )
    return rc, stdout, stderr, err


async def _compress_one(req: CompressRequest, input_video: str, task_label: str) -> CompressResponse:
    remote_mode = _is_url(input_video)
    prefer_nvenc = await _detect_nvenc_available()
    encoder = _resolve_encoder(req.codec, prefer_nvenc)

    if remote_mode and DISABLE_REMOTE_IO:
        return CompressResponse(success=False, errors=["Remote URL input is disabled for this local-only service"])

    local_input = ""
    output_path = ""
    output_filename = ""
    returncode: int | None = None
    stdout = b""
    stderr = b""
    run_error = ""

    try:
        async with _queued_task(task_label) as (queue_position, snapshot):
            log.info(
                f"[{task_label}] Queued compression "
                f"(codec={encoder}, cq={req.cq}, position={queue_position}, "
                f"waiting={snapshot['queued_tasks']}, remote={remote_mode})"
            )

            os.makedirs(SHARED_VOLUME_PATH, exist_ok=True)

            # --- Resolve input to local path ---
            if remote_mode:
                local_input = os.path.join(SHARED_VOLUME_PATH, f"{task_label}_input.mp4")
                await _track_temp_files(local_input)
                try:
                    await _download_url(input_video, local_input)
                except Exception as e:
                    _cleanup(local_input)
                    return CompressResponse(success=False, errors=[f"Failed to download input: {e}"])
            else:
                local_input = input_video
                if not os.path.exists(local_input):
                    raise HTTPException(status_code=400, detail=f"Input file not found: {local_input}")
                await _track_temp_files(local_input)

            basename = os.path.splitext(os.path.basename(local_input))[0]
            output_filename = f"{basename}_compressed.mp4"
            output_path = os.path.join(SHARED_VOLUME_PATH, output_filename)
            await _track_temp_files(output_path)

            duration_seconds = await _probe_duration_seconds(local_input, task_label)
            use_chunked = _should_chunk(req, duration_seconds)
            color_tags = await _probe_color_tags(local_input)
            if color_tags:
                log.info(f"[{task_label}] source color tags: {color_tags}")

            # Hold the concurrency slot for adaptive search + final encode.
            # Running adaptive outside the semaphore let all 5 validator clips
            # probe CQ in parallel and blow past the miner HTTP timeout.
            async with _running_task() as running_snapshot:
                request_t0 = time.monotonic()
                effective_req = req
                x265_zones: str | None = None
                predicted_vmaf: float | None = None
                complexity = "medium"
                width = 0
                height = 0
                encode_timeout = float(
                    req.encode_timeout_seconds
                    if req.encode_timeout_seconds is not None
                    else DEFAULT_PER_CLIP_ENCODE_TIMEOUT
                )
                vmaf_threshold = float(req.vmaf_threshold) if req.vmaf_threshold is not None else 85.0

                # --- Fast RD path (default): features → plan → encode. No multi-probe VMAF. ---
                use_full_search = ADAPTIVE_FULL_SEARCH and (
                    ADAPTIVE_ENABLED_DEFAULT if req.adaptive is None else req.adaptive
                )
                if FAST_RD_MODE or not use_full_search:
                    try:
                        features = await extract_source_features(local_input, timeout=2.5)
                        complexity = features.complexity
                        width = features.width
                        height = features.height
                        remaining = max(5.0, encode_timeout - (time.monotonic() - request_t0))
                        bitrate_bps = None
                        if req.codec_mode.upper() == "VBR" and req.target_bitrate:
                            # Validator target_bitrate is Mbps in miner payload; service may get bps.
                            raw = int(req.target_bitrate)
                            bitrate_bps = raw if raw > 100_000 else raw * 1_000_000
                        plan = select_plan(
                            features,
                            encoder=encoder,
                            vmaf_threshold=vmaf_threshold,
                            codec_mode=req.codec_mode,
                            target_bitrate_bps=bitrate_bps,
                            remaining_seconds=remaining,
                        )
                        updates = {"cq": plan.cq, "preset": plan.preset}
                        if plan.codec_mode == "VBR" and plan.target_bitrate_bps:
                            updates["target_bitrate"] = plan.target_bitrate_bps
                            updates["codec_mode"] = "VBR"
                        effective_req = req.model_copy(update=updates)
                        log.info(
                            f"[{task_label}] fast_rd {plan.reason} "
                            f"expect_score≈{plan.expected_score_lo:.2f}-{plan.expected_score_hi:.2f} "
                            f"timeout={encode_timeout:.0f}s"
                        )
                    except Exception as e:
                        log.warning(f"[{task_label}] fast_rd failed, seed cq={req.cq}: {e}")
                        # Emergency: fastest preset
                        fb = "9" if encoder == "libsvtav1" else "ultrafast"
                        effective_req = req.model_copy(update={"preset": fb})
                elif use_full_search and req.codec_mode.upper() != "VBR":
                    try:
                        decision = await optimize_compression(
                            local_input,
                            encoder=encoder,
                            vmaf_threshold=vmaf_threshold,
                            seed_cq=req.cq,
                            preset=req.preset,
                            target_width=req.target_width,
                            target_height=req.target_height,
                            enabled=True,
                            work_root=os.path.join(SHARED_VOLUME_PATH, f"{task_label}_adaptive"),
                            deadline_seconds=encode_timeout,
                        )
                        log.info(f"[{task_label}] adaptive_decision {json.dumps(decision.to_log_dict())}")
                        if decision.cq != req.cq or decision.preset != req.preset:
                            effective_req = req.model_copy(update={"cq": decision.cq, "preset": decision.preset})
                        x265_zones = decision.x265_zones
                        if decision.selected:
                            predicted_vmaf = decision.selected.vmaf_hmean
                        if decision.analysis:
                            complexity = decision.analysis.complexity.value
                            width = decision.analysis.width
                            height = decision.analysis.height
                    except Exception as e:
                        log.warning(f"[{task_label}] adaptive search failed, using seed cq={req.cq}: {e}")

                # Never let encode exceed remaining per-clip budget.
                elapsed = time.monotonic() - request_t0
                encode_timeout = max(5.0, encode_timeout - elapsed)

                # Disable chunking under the 180s validator path — too much overhead.
                use_chunked = False

                log.info(
                    f"[{task_label}] Starting compression "
                    f"(active={running_snapshot['active_tasks']}/{MAX_CONCURRENT}, "
                    f"queued={running_snapshot['queued_tasks']}, "
                    f"duration={duration_seconds}, chunked={use_chunked}, "
                    f"cq={effective_req.cq}, preset={effective_req.preset}, "
                    f"encode_timeout={encode_timeout:.0f}s, zones={'yes' if x265_zones else 'no'})"
                )

                if use_chunked:
                    try:
                        await _compress_chunked(
                            local_input, output_path, effective_req, encoder, task_label
                        )
                        returncode = 0
                    except RuntimeError as e:
                        if "falling back to single-pass compression" in str(e):
                            log.warning(f"[{task_label}] {e}")
                            returncode, stdout, stderr, run_error = await _encode_with_deadline(
                                local_input,
                                output_path,
                                effective_req,
                                encoder,
                                task_label,
                                encode_timeout=encode_timeout,
                                x265_zones=x265_zones,
                                duration_seconds=duration_seconds,
                                complexity=complexity,
                                width=width,
                                height=height,
                                color=color_tags,
                            )
                        else:
                            run_error = str(e)
                            log.error(f"[{task_label}] Chunked compression failed: {run_error}")
                else:
                    returncode, stdout, stderr, run_error = await _encode_with_deadline(
                        local_input,
                        output_path,
                        effective_req,
                        encoder,
                        task_label,
                        encode_timeout=encode_timeout,
                        x265_zones=x265_zones,
                        duration_seconds=duration_seconds,
                        complexity=complexity,
                        width=width,
                        height=height,
                        color=color_tags,
                    )

                if returncode == 0 and os.path.exists(output_path):
                    # Ratio push: if output still too large and budget remains, one CQ+ bump.
                    # Competitive scores need ~8–18×; <4× rarely reaches top-5.
                    try:
                        src_sz = os.path.getsize(local_input)
                        out_sz = os.path.getsize(output_path)
                        ratio = (src_sz / out_sz) if out_sz > 0 else 0.0
                        elapsed = time.monotonic() - request_t0
                        left = encode_timeout - elapsed
                        if (
                            ratio > 0
                            and ratio < 4.0
                            and left >= 18.0
                            and effective_req.codec_mode.upper() != "VBR"
                        ):
                            push_cq = min(55, int(effective_req.cq) + 4)
                            push_req = effective_req.model_copy(update={"cq": push_cq})
                            log.info(
                                f"[{task_label}] ratio_push {ratio:.2f}x→cq {effective_req.cq}->{push_cq} "
                                f"(left={left:.0f}s)"
                            )
                            _cleanup(output_path)
                            rc2, so2, se2, err2 = await _encode_with_deadline(
                                local_input,
                                output_path,
                                push_req,
                                encoder,
                                task_label,
                                encode_timeout=left,
                                duration_seconds=duration_seconds,
                                complexity=complexity,
                                width=width,
                                height=height,
                                color=color_tags,
                            )
                            if rc2 == 0 and os.path.exists(output_path):
                                returncode, stdout, stderr, run_error = rc2, so2, se2, err2
                                effective_req = push_req
                                out_sz2 = os.path.getsize(output_path)
                                ratio2 = (src_sz / out_sz2) if out_sz2 > 0 else 0.0
                                log.info(f"[{task_label}] ratio_push result {ratio:.2f}x → {ratio2:.2f}x")
                            else:
                                log.warning(f"[{task_label}] ratio_push failed, keeping first encode")
                    except Exception as exc:
                        log.warning(f"[{task_label}] ratio_push skipped: {exc}")

                    # Skip expensive integrity/calibration when time is tight.
                    if encode_timeout > 8:
                        try:
                            ok, reason = await validate_output_matches_source(local_input, output_path)
                            if not ok:
                                log.warning(f"[{task_label}] integrity warning: {reason}")
                        except Exception as exc:
                            log.warning(f"[{task_label}] integrity skipped: {exc}")
                    if predicted_vmaf and (time.monotonic() - request_t0) < encode_timeout:
                        await _calibrate_sample_vs_output(
                            local_input, output_path, predicted_vmaf, task_label
                        )

        snapshot = await _queue_snapshot()
        stats = dict(active_tasks=snapshot["active_tasks"], queued_tasks=snapshot["queued_tasks"])

        if returncode is None:
            _cleanup(output_path)
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(success=False, errors=[run_error or "compression did not start"], **stats)

        if returncode != 0:
            err_msg = run_error or _format_process_output(stdout, stderr) or "compression failed without output"
            log.error(f"[{task_label}] ffmpeg failed (rc={returncode}): {err_msg}")
            _cleanup(output_path)
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(success=False, errors=[err_msg], **stats)

        if not os.path.exists(output_path):
            log.error(f"[{task_label}] Output file not found: {output_path}")
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(success=False, errors=["Output file not created"], **stats)

        # --- Remote mode: upload result to S3, return URL ---
        if remote_mode:
            try:
                s3_key = f"processing/{task_label}/{output_filename}"
                output_url = _upload_to_s3(output_path, s3_key)
                log.info(f"[{task_label}] Compression complete (remote): {output_url[:80]}...")
                return CompressResponse(output_urls=[output_url], errors=[None], success=True, **stats)
            except Exception as e:
                log.error(f"[{task_label}] S3 upload failed: {e}")
                return CompressResponse(success=False, errors=[f"S3 upload failed: {e}"], **stats)
            finally:
                _cleanup(local_input, output_path)

        log.info(f"[{task_label}] Compression complete (local): {output_path}")
        return CompressResponse(output_paths=[output_path], errors=[None], success=True, **stats)
    finally:
        await _untrack_temp_files(local_input, output_path)


def _combine_compress_responses(responses: list[CompressResponse]) -> CompressResponse:
    output_paths = [response.output_paths[0] if response.output_paths else "" for response in responses]
    output_urls = [response.output_urls[0] if response.output_urls else "" for response in responses]
    errors = [response.errors[0] if response.errors else None for response in responses]
    success = all(response.success for response in responses)
    latest = responses[-1] if responses else None

    return CompressResponse(
        output_paths=output_paths,
        output_urls=output_urls,
        errors=errors,
        success=success,
        active_tasks=latest.active_tasks if latest else None,
        queued_tasks=latest.queued_tasks if latest else None,
    )


@app.post("/compress", response_model=CompressResponse)
async def compress(req: CompressRequest):
    input_videos = [video_path.strip() for video_path in req.video_paths]
    if any(not video_path for video_path in input_videos):
        raise HTTPException(status_code=400, detail="video_paths entries are required")

    base_task_id = req.task_id or uuid.uuid4().hex[:8]
    responses = await asyncio.gather(
        *[
            _compress_one(
                req,
                input_video,
                base_task_id if len(input_videos) == 1 else f"{base_task_id}-{index + 1}",
            )
            for index, input_video in enumerate(input_videos)
        ]
    )

    return responses[0] if len(responses) == 1 else _combine_compress_responses(responses)
