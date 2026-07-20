#!/usr/bin/env bash
# Readiness checklist for Vidaio SN85 mining.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0
WARN=0

ok() { echo "[OK]  $*"; PASS=$((PASS + 1)); }
bad() { echo "[FAIL] $*"; FAIL=$((FAIL + 1)); }
warn() { echo "[WARN] $*"; WARN=$((WARN + 1)); }

echo "=== Vidaio SN85 miner readiness ==="
echo

# GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU=$(nvidia-smi -L 2>/dev/null | head -1 || true)
  ok "GPU: ${GPU}"
else
  bad "nvidia-smi not found"
fi

# Docker + nvidia runtime
if docker info >/dev/null 2>&1; then
  ok "Docker daemon reachable"
else
  bad "Docker daemon not reachable"
fi
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  ok "Docker GPU passthrough works"
else
  warn "Docker GPU test failed (nvidia-container-toolkit may be missing)"
fi

# Python / bittensor
if [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
  if python -c "import bittensor as bt; print(bt.__version__)" >/tmp/bt_ver 2>/dev/null; then
    ok "bittensor import OK ($(cat /tmp/bt_ver))"
  else
    bad "bittensor import failed — run: pip uninstall scalecodec cyscale -y; pip install 'async-substrate-interface>=1.5.15,<2'"
  fi
else
  bad "venv missing — create with python3 -m venv venv && pip install -e ."
fi

# Env
if [[ -f miner/.env ]]; then
  ok "miner/.env present"
  # shellcheck disable=SC1091
  set -a; source miner/.env; set +a
  if [[ -z "${MINER_STORAGE_S3_ACCESS_KEY_ID:-}" || "${MINER_STORAGE_S3_ACCESS_KEY_ID}" == "your-access-key" ]]; then
    bad "S3 credentials still placeholders in miner/.env"
  else
    ok "S3 credentials look filled"
  fi
  ok "Backend=${MINER_PROCESSING_BACKEND:-unset} warrant=${MINER_WARRANT_TASK:-unset}"
else
  bad "miner/.env missing — cp miner/.env.template miner/.env"
fi

# Shared volume
SHARED="${MINER_SHARED_DIR:-/tmp/vidaio-miner-video-tmp}"
if [[ -d "${SHARED}" ]]; then
  ok "Shared volume dir exists: ${SHARED}"
else
  bad "Shared volume missing: ${SHARED}"
fi

# Health endpoints
for pair in "8004:compression" "8003:upscaling-video2x" "8005:upscaling-ffmpeg"; do
  port="${pair%%:*}"
  name="${pair##*:}"
  if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    ok "${name} healthy on :${port}"
  else
    warn "${name} not healthy on :${port} (start only the profile you mine)"
  fi
done

# Wallet
if [[ -d "${HOME}/.bittensor/wallets" ]] && ls "${HOME}/.bittensor/wallets" >/dev/null 2>&1; then
  ok "Wallets directory present (~/.bittensor/wallets)"
  ls -1 "${HOME}/.bittensor/wallets" | sed 's/^/       - /'
else
  bad "No wallets found — create with btcli wallet create / new_hotkey"
fi

echo
echo "=== Summary: ${PASS} ok, ${WARN} warn, ${FAIL} fail ==="
if [[ "${FAIL}" -gt 0 ]]; then
  echo "Not ready to mine until FAIL items are fixed."
  exit 1
fi
echo "Core checks passed. Register on netuid 85 if needed, then:"
echo "  WALLET_NAME=... HOTKEY_NAME=... AXON_PORT=8091 ./scripts/start_miner.sh"
exit 0
