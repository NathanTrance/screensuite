#!/usr/bin/env bash
# =============================================================================
# Run the 3 mobile benchmarks against the ON-DEVICE Qwen3-VL-4B server
# (Qualcomm NPU, OpenAI-compatible API on device port 18181).
#
# Prerequisites:
#   - Phone connected via adb, qcom_llm_server app running (port 18181 listening)
#   - `uv sync --python 3.11` already done in this repo
#
# Usage:
#   ./run_on_device.sh forward                # set up adb port forward
#   ./run_on_device.sh smoke                  # 20-sample sanity check (workers=1)
#   ./run_on_device.sh run                    # ~500 samples/benchmark (default, ~15-25 min each)
#   ./run_on_device.sh full                   # FULL test sets (screenqa_short 8.4k, complex 11.9k,
#                                             #   android_control 3k traces) -> 5-10 HOURS each
#   ./run_on_device.sh results                # side-by-side table of all runs
#
# Env overrides: SAMPLES (default 500), RUN_NAME (default qcom),
#                BENCHS (space-separated, default = all three)
#
# IMPORTANT:
#   - Always use --workers 1: the NPU server is single-concurrency; parallel
#     requests get "503 model busy" and inflate proportion_missing.
#   - avg_latency_s per benchmark is in the results JSONL (measured per call,
#     typically ~1.5 s/image on the NPU).
#   - answers.jsonl has ONE line per model call; android_control makes one call
#     per action step per trace, so line count > sample count (normal).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL_ID:-Qwen3-VL-4B-Instruct}"
API="http://localhost:18181/v1"
API_KEY="EMPTY"
RUN_NAME="${RUN_NAME:-qcom}"
SAMPLES="${SAMPLES:-500}"
BENCHS="${BENCHS:-screenqa_short screenqa_complex android_control}"

PY="uv run python examples/run_qwen3_vl.py"

forward() {
  adb forward tcp:18181 tcp:18181
  adb forward --list
}

# Re-establish the adb forward if it has dropped (adb server restarts / USB hiccups
# clear forwards, which manifests as APIConnectionError even though the phone is fine).
ensure_forward() {
  if ! adb forward --list 2>/dev/null | grep -q "tcp:18181"; then
    echo ">>> adb forward missing, re-adding..."
    forward
  fi
}

# Background watchdog: keep the forward alive for the whole (possibly multi-hour) run.
watch_forward() {
  while true; do
    sleep 20
    if ! adb forward --list 2>/dev/null | grep -q "tcp:18181"; then
      echo ">>> [watchdog] forward dropped, re-adding..."
      adb forward tcp:18181 tcp:18181 2>/dev/null || true
    fi
  done
}

start_watchdog() {
  watch_forward &
  WATCH_PID=$!
  trap 'kill "$WATCH_PID" 2>/dev/null || true' EXIT
}

stop_watchdog() {
  kill "$WATCH_PID" 2>/dev/null || true
  trap - EXIT
}

smoke() {
  forward
  $PY --api-base "$API" --api-key "$API_KEY" --model-id "$MODEL" \
      --smoke --benchmarks screenqa_short
}

run() {
  forward
  start_watchdog
  # shellcheck disable=SC2086
  $PY --api-base "$API" --api-key "$API_KEY" --model-id "$MODEL" \
      --n-samples "$SAMPLES" --workers 1 --run-name "$RUN_NAME" --benchmarks $BENCHS
  stop_watchdog
}

full() {
  forward
  start_watchdog
  # shellcheck disable=SC2086
  $PY --api-base "$API" --api-key "$API_KEY" --model-id "$MODEL" \
      --load-full --workers 1 --run-name "$RUN_NAME" --benchmarks $BENCHS
  stop_watchdog
}

results() {
  uv run python examples/summarize_results.py --compare --metric f1
}

case "${1:-run}" in
  forward) forward ;;
  smoke)   smoke ;;
  run)     run ;;
  full)    full ;;
  results) results ;;
  *)
    echo "usage: $0 {forward|smoke|run|full|results}"
    exit 1
    ;;
esac