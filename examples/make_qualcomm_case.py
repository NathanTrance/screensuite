#!/usr/bin/env python
"""Build the Qualcomm support-case attachment package.

Creates output/qualcomm_case/ with, for each problematic sample + control:
  - the exact resized PNG sent to the model
  - the exact JSON request body (curl -d @file)
  - the server logcat captured during the hang (copied)
  - a README with reproduction commands

Usage:
  HF_DATASETS_OFFLINE=1 uv run python examples/make_qualcomm_case.py
"""
import base64
import io
import json
import os
import shutil

from datasets import load_dataset

from screensuite import EvaluationConfig, ImageResizeConfig, get_registry

CASE_DIR = os.path.join("output", "qualcomm_case")
SAMPLES = [
    ("18777", "HANG"),
    ("5362", "HANG"),
    ("10165", "CONTROL_OK"),
]


def main():
    os.makedirs(CASE_DIR, exist_ok=True)
    registry = get_registry()
    bench = next(b for b in registry.list_all() if b.name == "screenqa_complex_500")
    cfg = EvaluationConfig(
        test_mode=False, parallel_workers=1, max_samples_to_test=None,
        run_name=None, image_resize_config=ImageResizeConfig(),
    )

    stream = load_dataset("nathantrance/screenqa-complex-500", split="train", streaming=True)
    rows = {}
    for r in stream:
        sid = str(r["screen_id"])
        if sid in dict((s, 1) for s, _ in SAMPLES):
            rows[sid] = r
        if len(rows) == len(SAMPLES):
            break

    for sid, tag in SAMPLES:
        row = rows[sid]
        ann = bench._get_annotated_input_from_sample(row, cfg)

        # exact payload as sent by the client (PIL -> PNG -> base64 data URL)
        serialized = []
        for msg in ann.messages:
            content = []
            for part in msg["content"]:
                if part["type"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image":
                    buf = io.BytesIO()
                    part["image"].save(buf, "PNG")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()},
                    })
            serialized.append({"role": msg["role"], "content": content})

        image_path = os.path.join(CASE_DIR, f"image_{sid}.png")
        for part in ann.messages[0]["content"]:
            if part["type"] == "image":
                part["image"].save(image_path, "PNG")

        payload = {
            "model": "Qwen3-VL-4B-Instruct-V79",
            "max_tokens": 4096,
            "temperature": 0,
            "stream": False,
            "messages": serialized,
        }
        payload_path = os.path.join(CASE_DIR, f"payload_{sid}.json")
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"{sid} [{tag}]: {image_path} ({os.path.getsize(image_path)//1024} KB), "
              f"{payload_path} ({os.path.getsize(payload_path)//1024} KB)")

    # copy logs if present
    for src, dst in [
        ("output/server_log_18777.txt", "logcat_hang_18777.txt"),
        ("output/server_log_10165.txt", "logcat_control_10165.txt"),
    ]:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(CASE_DIR, dst))
            print(f"copied {src} -> {os.path.join(CASE_DIR, dst)}")

    readme = """# Qualcomm support case — attachments

Silent HTP/CDSP graph hang on Qwen3-VL-4B-Instruct-V79 (QAIRT W4A16), dtype `Qwen3-VL-4B-Instruct-V79`, DSP v79 (`soc_model: 69`).

## Reproduction (host side, server on device:18181)

```bash
adb forward tcp:18181 tcp:18181

# Sample 18777 — HANGS (no response, no first token, ever)
curl -s -m 60 -X POST http://localhost:18181/v1/chat/completions \\
  -H 'Content-Type: application/json' -d @payload_18777.json

# Sample 5362 — HANGS (same)
curl -s -m 60 -X POST http://localhost:18181/v1/chat/completions \\
  -H 'Content-Type: application/json' -d @payload_5362.json

# Sample 10165 — CONTROL: returns in ~2.6 s ("5 days"), same prompt template + max_tokens
curl -s -m 60 -X POST http://localhost:18181/v1/chat/completions \\
  -H 'Content-Type: application/json' -d @payload_10165.json
```

- `image_18777.png`, `image_5362.png`, `image_10165.png` — the exact images in the payloads (532x952, 1092x1932, 532x952)
- `payload_*.json` — exact request bodies (image embedded as base64 data URL), `max_tokens=4096`, `temperature=0`

## Observed behavior for the hanging requests

- No response at any timeout (verified 5 s / 25 s / 60 s / 120 s / 300 s)
- Streaming mode: 0 SSE chunks in 100+ s
- `logcat` (`GenieXSdk` V): `geniex_vlm_generate` logs `VlmGenerateInput(...)` and then **nothing** —
  no `VlmGenerateOutput`, no `ErrorCode`, no callback, ever (see `logcat_hang_18777.txt`)
- App CPU during hang: **0%** (main thread blocked, work is on CDSP); no crash, no tombstone
- `dmesg`: clean; all remoteprocs `running`/`attached` (CDSP not crashed)
- After the hang: inference mutex held → all other requests get `503 model busy`; eventually
  `/v1/models` also stops responding until the app process is restarted
- REGRESSION: the same 18777 request completed in 2.8 s on 2026-09-08 with the same app/bundle

## Content analysis (input-dependent trigger)

- Trigger = specific image + text prompt > ~300 chars (bisect: 256 chars OK, 336 chars hang, same image)
- 18777 image = most edge-dense in the 500-sample set (z=+3.3); 5362 image = darkest decile (z=-1.9)
- Not size-related: all requests ~360 prompt tokens (256 image + ~100 text), context cap 4096
- Image tokens are fixed 256/image (`image_features: [256, 2560]`); pixels do not change token count

## Bundle facts (from /storage/emulated/0/Download/Qwen3-VL-4B-Instruct-V79/)

- `genie_config.json`: `context.size 4096`
- `text-generator.json`: `enable-graph-switching: false`, QnnHtp, kv-dim 128
- `metadata.json`: `/genie/context_lengths [4096]`, vision output `[256, 2560]`, preprocess 512x512, patch 16, merge 2
- `img-enc-htp.json`: vision-param 32x32
- `htp_backend_ext_config.json`: `dsp_arch: v79`, `perf_profile: burst`

## Hypothesis

Out-of-range fixed-point values (attention/normalization statistics) for these
extreme pixel distributions cause NaN/denormal propagation, and the DSP graph
never terminates instead of reporting an error (stall, not fault). Requested:
QnnProfiler/HTP-side analysis of the stuck graph for these inputs.
"""
    with open(os.path.join(CASE_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"\ncase package ready: {CASE_DIR}/")


if __name__ == "__main__":
    main()
