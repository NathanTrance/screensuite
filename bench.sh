#!/usr/bin/env bash
# Helper for running ScreenSuite benchmarks with the VTCC proxy setup.
#
#   ./bench.sh download [bench...]        # pass 1: proxy on, cache full datasets, no API calls
#   ./bench.sh run [bench...]             # pass 2: proxy off, offline HF, evaluate (full test sets)
#   ./bench.sh all                        # download + run everything
#   ./bench.sh parallel [bench...]        # run benchmarks in parallel (one process each), then summarize
#   ./bench.sh results                    # summarize all runs
#   ./bench.sh smoke [bench...]           # quick smoke test
#
# Env overrides: API_HOST, API_BASE, MODEL_ID, WORKERS, RUN_NAME
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

API_HOST="${API_HOST:-voice-staging.cyberbot.vn}"
API_BASE="${API_BASE:-https://voice-staging.cyberbot.vn/v1}"
MODEL_ID="${MODEL_ID:-Qwen3-VL-4B-Instruct}"
WORKERS="${WORKERS:-16}"

# Load bashrc helpers (set_s5_proxy / remove_proxy) if not already available
if ! declare -F set_s5_proxy >/dev/null 2>&1 || ! declare -F remove_proxy >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$HOME/.bashrc" 2>/dev/null || true
fi

proxy_on() {
    if declare -F set_s5_proxy >/dev/null 2>&1; then set_s5_proxy; else
        echo "warning: set_s5_proxy not found in bashrc; assuming proxy is already active"
    fi
    # Never tunnel the model API through the proxy (its TLS cert is MITM'd by the proxy)
    export NO_PROXY="$API_HOST"
    export no_proxy="$API_HOST"
}

proxy_off() {
    if declare -F remove_proxy >/dev/null 2>&1; then remove_proxy; else
        echo "warning: remove_proxy not found in bashrc; unsetting proxy vars manually"
        unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy || true
    fi
    export NO_PROXY="$API_HOST"
    export no_proxy="$API_HOST"
}

run_one() {  # bench_name
    uv run python examples/run_qwen3_vl.py \
        --api-base "$API_BASE" --model-id "$MODEL_ID" \
        --load-full --workers "$WORKERS" \
        --run-name "${RUN_NAME:-full_$1}" --benchmarks "$1"
}

cmd_download() {  # [bench...]
    proxy_on
    if [ "$#" -eq 0 ]; then set -- screenqa_short screenqa_complex screenspot-v2-click-prompt screenspot-v2-bounding-box-prompt screenspot-pro-click-prompt screenspot-pro-bounding-box-prompt websrc_dev visualwebbench showdown_clicks mmind2web android_control; fi
    for b in "$@"; do
        echo ">>> Downloading dataset: $b"
        uv run python examples/run_qwen3_vl.py --api-base "$API_BASE" --model-id "$MODEL_ID" \
            --load-full --download-only --benchmarks "$b"
    done
    proxy_off
    echo ">>> Datasets cached. Now run: ./bench.sh run"
}

cmd_run() {  # [bench...]
    proxy_off
    export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
    if [ "$#" -eq 0 ]; then
        uv run python examples/run_qwen3_vl.py --api-base "$API_BASE" --model-id "$MODEL_ID" \
            --load-full --workers "$WORKERS" --run-name "${RUN_NAME:-full_${MODEL_ID}}"
    else
        for b in "$@"; do
            echo ">>> Running benchmark: $b"
            run_one "$b"
        done
    fi
}

cmd_parallel() {  # [bench...]
    proxy_off
    export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
    if [ "$#" -eq 0 ]; then set -- screenqa_short screenqa_complex screenspot-v2-click-prompt screenspot-v2-bounding-box-prompt screenspot-pro-click-prompt screenspot-pro-bounding-box-prompt websrc_dev visualwebbench showdown_clicks mmind2web android_control; fi
    pids=()
    for b in "$@"; do
        echo ">>> Launching benchmark: $b (log: output/log_${b}.log)"
        ( run_one "$b" > "output/log_${b}.log" 2>&1 ) &
        pids+=("$!")
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    if [ "$fail" -ne 0 ]; then echo ">>> Some benchmarks failed - check output/log_*.log"; fi
    ./bench.sh results
}

cmd_smoke() {  # [bench...]
    proxy_off
    export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
    uv run python examples/run_qwen3_vl.py --api-base "$API_BASE" --model-id "$MODEL_ID" \
        --smoke --benchmarks "$@"
}

cmd_results() {
    uv run python examples/summarize_results.py --compare
}

usage() {
    sed -n '2,9p' "$0"
    exit 1
}

case "${1:-}" in
    download) shift; cmd_download "$@" ;;
    run)      shift; cmd_run "$@" ;;
    all)      shift; cmd_download; cmd_run ;;
    parallel) shift; cmd_parallel "$@" ;;
    smoke)    shift; cmd_smoke "$@" ;;
    results)  shift; cmd_results "$@" ;;
    *) usage ;;
esac
