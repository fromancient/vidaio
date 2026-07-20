#!/usr/bin/env bash
# Start Vidaio SN85 miner process (processing containers must already be healthy).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "${REPO_ROOT}/venv/bin/activate"
set -a
# shellcheck disable=SC1091
source "${REPO_ROOT}/miner/.env"
set +a

# Required for bittensor>=10.5 to honor --wallet.name / --netuid CLI flags.
export BT_NO_PARSE_CLI_ARGS=false

WALLET_NAME="${WALLET_NAME:-ai}"
HOTKEY_NAME="${HOTKEY_NAME:-aiu20}"
AXON_PORT="${AXON_PORT:-8091}"
NETWORK="${NETWORK:-finney}"
NETUID="${NETUID:-85}"
USE_PM2="${USE_PM2:-1}"

if [[ -z "${WALLET_NAME}" || -z "${HOTKEY_NAME}" ]]; then
  echo "Usage: WALLET_NAME=... HOTKEY_NAME=... [AXON_PORT=8091] $0"
  echo "Optional: NETWORK=finney NETUID=85 USE_PM2=0"
  exit 1
fi

if [[ -z "${MINER_STORAGE_S3_ACCESS_KEY_ID:-}" || "${MINER_STORAGE_S3_ACCESS_KEY_ID}" == "your-access-key" ]]; then
  echo "ERROR: Fill real S3/Backblaze credentials in miner/.env before mining."
  exit 1
fi

# Prefer absolute python from venv so PM2 does not lose PATH.
PYTHON_BIN="${REPO_ROOT}/venv/bin/python3"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

CMD=(
  "${PYTHON_BIN}" neurons/miner.py
  --wallet.name "${WALLET_NAME}"
  --wallet.hotkey "${HOTKEY_NAME}"
  --subtensor.network "${NETWORK}"
  --netuid "${NETUID}"
  --axon.port "${AXON_PORT}"
  --logging.debug
)

echo "Warrant task: ${MINER_WARRANT_TASK:-unset}"
echo "Backend: ${MINER_PROCESSING_BACKEND:-unset}"
echo "Wallet: ${WALLET_NAME}/${HOTKEY_NAME}"
echo "Starting miner on axon port ${AXON_PORT} (netuid ${NETUID}, ${NETWORK})"

if [[ "${USE_PM2}" == "1" ]] && command -v pm2 >/dev/null 2>&1; then
  pm2 delete vidaio-miner >/dev/null 2>&1 || true
  # Pass argv as separate pm2 args (avoid bash -c string that drops env).
  pm2 start "${PYTHON_BIN}" \
    --name vidaio-miner \
    --cwd "${REPO_ROOT}" \
    --update-env \
    --interpreter none \
    -- \
    neurons/miner.py \
    --wallet.name "${WALLET_NAME}" \
    --wallet.hotkey "${HOTKEY_NAME}" \
    --subtensor.network "${NETWORK}" \
    --netuid "${NETUID}" \
    --axon.port "${AXON_PORT}" \
    --logging.debug
  pm2 save || true
  sleep 3
  pm2 logs vidaio-miner --lines 40 --nostream
else
  exec "${CMD[@]}"
fi
