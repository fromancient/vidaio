#!/usr/bin/env bash
# Build and start miner processing containers for the chosen task profile.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/miner"

PROFILE="${1:-compression}"  # compression | upscaling-video2x | upscaling-ffmpeg | both
set -a
# shellcheck disable=SC1091
source "${REPO_ROOT}/miner/.env"
set +a

mkdir -p "${MINER_SHARED_DIR:-/tmp/vidaio-miner-video-tmp}"
chmod 777 "${MINER_SHARED_DIR:-/tmp/vidaio-miner-video-tmp}" || true

start_compression() {
  echo "Building/starting compression service on :8004 ..."
  docker compose up -d --build compression
}

start_upscaling_video2x() {
  echo "Building/starting Video2X upscaling on :8003 ..."
  docker compose --profile upscaling-video2x up -d --build upscaling-video2x
}

start_upscaling_ffmpeg() {
  echo "Building/starting FFmpeg upscaling on :8005 ..."
  docker compose --profile upscaling-ffmpeg up -d --build upscaling-ffmpeg
}

case "${PROFILE}" in
  compression)
    start_compression
    ;;
  upscaling-video2x|upscaling)
    start_upscaling_video2x
    ;;
  upscaling-ffmpeg)
    start_upscaling_ffmpeg
    ;;
  both)
    start_compression
    start_upscaling_video2x
    ;;
  *)
    echo "Usage: $0 [compression|upscaling-video2x|upscaling-ffmpeg|both]"
    exit 1
    ;;
esac

echo
docker compose ps
echo
echo "Health checks:"
curl -sf http://127.0.0.1:8004/health && echo " compression :8004 OK" || echo " compression :8004 not ready"
curl -sf http://127.0.0.1:8003/health && echo " upscaling-video2x :8003 OK" || echo " upscaling-video2x :8003 not ready"
curl -sf http://127.0.0.1:8005/health && echo " upscaling-ffmpeg :8005 OK" || echo " upscaling-ffmpeg :8005 not ready"
