#!/usr/bin/env bash
# =============================================================================
# Run ScreenSuite against a CLOUD / server-side OpenAI-compatible endpoint
# (e.g. https://voice-staging.cyberbot.vn/v1) or a local vLLM server.
#
# VTCC box proxy notes (READ before running):
#   - The box routes HuggingFace through a proxy (set_s5_proxy/remove_proxy in
#     your ~/.bashrc). The MODEL API must NEVER go through that proxy (it MITMs
#     TLS with a self-signed cert). Keep NO_PROXY set to the API host at all times.
#   - After datasets are cached once, run with HF_DATASETS_OFFLINE=1 to go fully
#     offline (see the "offline" command).
#
# Usage:
#   ./run_cloud.sh env                       # print the env vars you need to set
#   ./run_cloud.sh smoke                     # 20-sample sanity check
#   ./run_cloud.sh run                       # ~300 samples/benchmark (default)
#   ./run_cloud.sh full                      # FULL test sets (see time notes below)
#   ./run_cloud.sh local                     # same as run but against local vLLM :8000
#   ./run_cloud.sh offline                   # run from HF cache only (no network)
#   ./run_cloud.sh results                   # side-by-side table
#
# Env overrides: API_BASE, API_KEY, MODEL_ID, WORKERS, SAMPLES, RUN_NAME
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

API_BASE="${API_BASE:-https://voice-staging.cyberbot.vn/v1}"
MODEL_ID="${MODEL_ID:-Qwen3-VL-4B-Instruct}"
API_KEY="${API_KEY:-your-key-here}"
WORKERS="${WORKERS:-8}"
SAMPLES="${SAMPLES:-300}"
RUN_NAME="${RUN_NAME:-cloud}"
API_HOST="${API_HOST:-voice-staging.cyberbot.vn}"

# Load bashrc helpers (set_s5_proxy / remove_proxy) if present
if ! declare -F remove_proxy >/dev/null 2>&1 && [ -f "$HOME/.bashrc" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.bashrc" 2>/dev/null || true
fi

PY="uv run python examples/run_qwen3_vl.py"

proxy_off() {
    if declare -F remove_proxy >/dev/null 2>&1; then remove_proxy; else
        unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy || true
    fi
    # Model API always bypasses the proxy (proxy MITMs TLS with a self-signed cert)
    export NO_PROXY="$API_HOST"
    export no_proxy="$API_HOST"
}

env_help() {
    echo "Set these before running (or pass overrides):"
    echo "  export API_BASE=https://voice-staging.cyberbot.vn/v1   # or your endpoint"
    echo "  export API_KEY=<your-key>"
    echo "  export MODEL_ID=Qwen3-VL-4B-Instruct"
    echo "  export WORKERS=8"
    echo "  export NO_PROXY=$API_HOST                              # required on the VTCC box"
    echo "  export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1          # optional: offline HF cache"
}

smoke() {
    proxy_off
    $PY --api-base "$API_BASE" --api-key "$API_KEY" --model-id "$MODEL_ID" \
        --smoke --benchmarks screenqa_short
}

run() {
    proxy_off
    $PY --api-base "$API_BASE" --api-key "$API_KEY" --model-id "$MODEL_ID" \
        --n-samples "$SAMPLES" --workers "$WORKERS" --run-name "$RUN_NAME"
}

full() {
    proxy_off
    # NOTE: full test sets = ~86k API calls total; websrc_dev alone is 52k.
    # Time estimate at ~8 workers: screenqa_short 8.4k ~20-40 min, android_control hours.
    $PY --api-base "$API_BASE" --api-key "$API_KEY" --model-id "$MODEL_ID" \
        --load-full --workers "$WORKERS" --run-name "$RUN_NAME"
}

local() {
    proxy_off
    $PY --api-base http://localhost:8000/v1 --api-key EMPTY --model-id "$MODEL_ID" \
        --n-samples "$SAMPLES" --workers "$WORKERS" --run-name "$RUN_NAME"
}

offline() {
    proxy_off
    export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
    $PY --api-base "$API_BASE" --api-key "$API_KEY" --model-id "$MODEL_ID" \
        --n-samples "$SAMPLES" --workers "$WORKERS" --run-name "$RUN_NAME"
}

results() {
    uv run python examples/summarize_results.py --compare --metric f1
}

case "${1:-run}" in
  env)     env_help ;;
  smoke)   smoke ;;
  run)     run ;;
  full)    full ;;
  local)   local ;;
  offline) offline ;;
  results) results ;;
  *)
    echo "usage: $0 {env|smoke|run|full|local|offline|results}"
    exit 1
    ;;
esac