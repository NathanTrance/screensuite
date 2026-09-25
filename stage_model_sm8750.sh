#!/usr/bin/env bash
# =============================================================================
# Stage the Qwen3-VL-4B-Instruct GenieX/QAIRT bundle for SM8750 (Snapdragon 8
# Elite, "sun", DSP v79) onto the test phone, replacing the missing
# /storage/emulated/0/Download/Qwen3-VL-4B-Instruct-V79 directory.
#
# Device (verified): SM8750, board sun, Android 15 (SDK 35).
# Bundles: ~3.03 GB zip, ~4.4 GB extracted. Needs ~8 GB free on the PC.
#
# Usage:
#   ./stage_model_sm8750.sh              # latest (v0.62.0)
#   VERSION=v0.59.0 ./stage_model_sm8750.sh   # baseline from repo docs
#   SERIAL=68a063cd ./stage_model_sm8750.sh
#
# After staging: the app should load the model and bind :18181 within ~15-30 s.
# =============================================================================
set -euo pipefail

VERSION="${VERSION:-v0.62.0}"
SERIAL="${SERIAL:-68a063cd}"
DEVICE_DIR="/storage/emulated/0/Download/Qwen3-VL-4B-Instruct-V79"
WORKDIR="${WORKDIR:-$HOME/qwen3vl_stage}"

BASE="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/qwen3_vl_4b_instruct/releases/$VERSION"
ZIP_NAME="qwen3_vl_4b_instruct-geniex_qairt-w4a16-qualcomm_snapdragon_8_elite.zip"
URL="$BASE/$ZIP_NAME"

echo ">>> version: $VERSION"
echo ">>> device : $SERIAL"

# 0. sanity: device present + target feature check
adb -s "$SERIAL" get-state >/dev/null
SOC=$(adb -s "$SERIAL" shell getprop ro.soc.model | tr -d '\r')
if [ "$SOC" != "SM8750" ]; then
    echo "WARNING: ro.soc.model = '$SOC' (expected SM8750). Continue? [y/N]"
    read -r ans; [ "$ans" = "y" ] || exit 1
fi

# 1. download
mkdir -p "$WORKDIR"
cd "$WORKDIR"
if [ ! -f "$ZIP_NAME" ]; then
    echo ">>> downloading ~3 GB..."
    curl -L --fail -o "$ZIP_NAME" "$URL"
fi
ls -lh "$ZIP_NAME"

# 2. extract (find the directory containing part1_of_4.bin — the file the app loads;
#    note: current AI Hub bundles no longer ship geniex.json, the app doesn't need it)
echo ">>> extracting..."
rm -rf extracted && mkdir extracted
unzip -q -o "$ZIP_NAME" -d extracted
BUNDLE_DIR=$(dirname "$(find extracted -name part1_of_4.bin -print -quit)")
if [ -z "$BUNDLE_DIR" ] || [ ! -f "$BUNDLE_DIR/part1_of_4.bin" ] || [ ! -f "$BUNDLE_DIR/genie_config.json" ]; then
    echo "ERROR: part1_of_4.bin / genie_config.json not found in extracted zip"; exit 1
fi
echo ">>> bundle dir: $BUNDLE_DIR"
ls "$BUNDLE_DIR" | head -8

# 3. push to the phone (contents directly under the expected directory)
adb -s "$SERIAL" root >/dev/null 2>&1 || true
adb -s "$SERIAL" wait-for-device
sleep 2
adb -s "$SERIAL" shell "mkdir -p $DEVICE_DIR"
echo ">>> pushing ~4.4 GB (this takes a while)..."
adb -s "$SERIAL" push "$BUNDLE_DIR/." "$DEVICE_DIR/"

# 4. verify on device
adb -s "$SERIAL" shell "ls $DEVICE_DIR | head -8; echo ...; ls $DEVICE_DIR | wc -l"

# 5. relaunch the server and wait for the listener
adb -s "$SERIAL" shell am force-stop com.vai.qcom_llm_server
sleep 2
adb -s "$SERIAL" shell am start -n com.vai.qcom_llm_server/.MainActivity
adb -s "$SERIAL" forward tcp:18181 tcp:18181

echo ">>> waiting for model load + listener (up to 180 s)..."
for i in $(seq 1 18); do
    sleep 10
    code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://localhost:18181/v1/models || true)
    echo "   t=$((i*10))s  /v1/models -> $code"
    [ "$code" = "200" ] && break
done
curl -s -m 5 http://localhost:18181/v1/models || true
echo
echo ">>> done. If 200: server is back. Run the repro:"
echo "    uv run python examples/repro_18777.py --timeout 60"
