# Bug Report: Silent NPU (HTP/CDSP) hang on sample 18777 (ScreenQA-Complex)

**Status:** Open — needs QAIRT/Qualcomm-side investigation
**Report date:** 2026-09-09
**Reported by:** nhatth (testing)

---

## Summary

A single ScreenQA-Complex sample (`screen_id=18777`) deterministically hangs the
on-device VLM server: the request never completes, no token is ever generated,
the inference mutex is held forever (all subsequent requests get `503 model busy`),
and after enough blocked workers the whole HTTP layer stops responding
(including `GET /v1/models`) until the app is restarted.

The hang is a **silent HTP (Hexagon Tensor Processor) graph-execution hang on the
CDSP**: no crash, no error code, no SDK timeout — the native `geniex_vlm_generate`
call simply never returns. It is triggered by this specific image combined with a
text prompt longer than ~300 characters. It is a regression: the exact same
request completed in 2.8 s on 2026-09-08 and hangs on 2026-09-09 (V79 build).

---

## Environment

| Item | Value |
|---|---|
| Device | Snapdragon 8 Elite-class, serial `68a063cd` (V79 DSP, `dsp_arch: v79`, `soc_model: 69`) |
| Server app | `com.vai.qcom_llm_server` (android-qcom-llm-server) |
| Model bundle | `Qwen3-VL-4B-Instruct-V79` (QAIRT, W4A16), staged at `/storage/emulated/0/Download/Qwen3-VL-4B-Instruct-V79` |
| Server | NanoHTTPD on `0.0.0.0:18181`, inference serialized via `inferenceLock` |
| Client | ScreenSuite benchmark harness (`examples/debug_samples.py`) over `adb forward tcp:18181 tcp:18181` |
| First observed | 2026-09-08 (intermittent 90 s stalls in full runs) |
| Deterministic repro | 2026-09-09 (hang even on a freshly restarted, warm server) |

---

## Reproduction

```bash
# From the screensuite repo (client side; server-side repro = repro_18777.py):
adb forward tcp:18181 tcp:18181
uv run python examples/repro_18777.py --timeout 60
```

- Input image: `output/repro_18777.png` (532×952 RGB PNG, 358 KB, base64 466 KB)
  - raw: 540×960 JPEG → resized to 532×952 with Qwen smart-resize (factor 28)
- Prompt: the standard ScreenQA template (~407 chars, 406 logged as `text_chars`),
  question: *"How many more people are available for the top 1000 subscription than
  the top 100?"* (ground truth: `982`)
- Request: `model=Qwen3-VL-4B-Instruct-V79, max_tokens=4096, temperature=0, stream=false`

**Expected:** answer in ~0.5-3 s (like control sample 10165).
**Actual:** never completes. Verified at 5 s, 25 s, 60 s, 120 s and 300 s client
timeouts — no response, no partial tokens (streaming mode: 0 chunks).

---

## Evidence

### 1. Native SDK log (logcat tag `GenieXSdk`, V level)

Hanging request (2026-09-09 11:44:02.817):

```
GenieXSdk: [src/vlm.cpp:95:geniex_vlm_generate] VlmGenerateInput(prompt_utf8: <|im_start|>system
... (prompt + image_paths: api_img_*.png, image_count: 1, image_max_length: 0,
     GenerationConfig(max_tokens: 4096, SamplerConfig(temperature: 0, top_p: 0, top_k: 0 ...)))
```

…and **no further log line ever** — no `VlmGenerateOutput`, no error, no profile.
Working control (10165, same day) logs:

```
GenieXSdk: [src/vlm.cpp:118:geniex_vlm_generate] ErrorCode[0](Success): VlmGenerateOutput(
  full_text: 5 days, ttft: 345795 us, decode_time: 81117 us, prompt_tokens: 362,
  generated_tokens: 3, stop_reason: eos)
```

### 2. Server process state during the hang

- **App CPU: 0.0%** — the app thread is blocked on a native call (FastRPC wait), not computing.
- **Memory stable**: 1.6 GB native heap, no OOM.
- **No crash**: no tombstones, no `SIGSEGV`/`SIGABRT`, no `FATAL` in logcat.

### 3. DSP / remoteproc state during the hang (with adb root)

- `dmesg`: clean — no CDSP/adsprpc/fastrpc/hexagon errors.
- All remoteprocs up: `remoteproc0-2,4: running`, `remoteproc3: attached`
  (CDSP = `32300000.remoteproc-cdsp`).
- `/dev/fastrpc-cdsp(-secure)` present.

⇒ The DSP did **not** crash; the graph execution is **stuck in-flight** (the FastRPC
call never returns).

### 4. Failure cascade

1. Hung request holds `inferenceLock` forever → every concurrent request gets
   `503 {"error":{"message":"model busy"}}` (verified: bursts of 42 consecutive
   instant-503s after each stuck request in a 500-sample run).
2. Blocked NanoHTTPD worker threads accumulate → eventually even
   `GET /v1/models` stops responding (curl `000`).
3. Only recovery: restart the app (`am force-stop` + `am start`), or the hung
   request finishing on its own (in earlier runs, stuck requests self-terminated
   after ~90-120 s — consistent with a 4096-token generation at ~37 tok/s hitting
   the context ceiling).

---

## Root-cause isolation (experiments)

Same server state, same serialization pipeline (PIL → 532×952 PNG → base64 data URL),
`temperature=0`:

| # | Image | Text prompt | Length (chars) | Result |
|---|---|---|---|---|
| 1 | 18777 | Full ScreenQA template + its question | 407 | **HANG** (0 tokens, 100 s+) |
| 2 | 18777 | Full template minus the `<no answer>` sentence | 256 | OK, 2.7 s |
| 3 | 18777 | "Please be careful and think step by step. " ×8 | 336 | **HANG** |
| 4 | 18777 | Short ("Answer... Question... Output:") | ~120 | OK, 2.6 s |
| 5 | 10165 | Full ScreenQA template + its question | 407 | OK, 2.6 s |
| 6 | 10165 | Short | ~120 | OK, 2.5 s |

Conclusions:

- **Not** the image alone (rows 2, 4 pass with the same image).
- **Not** the question alone (row 6 passes with 18777's question).
- **Not** request size in any ordinary sense: ~360 prompt tokens total, 466 KB
  payload — both identical to working controls. Bundle limits: context 4096,
  image = fixed 256 tokens (`image_features: [256, 2560]`), 512×512 preprocessing.
  The hanging requests are **11× below the compiled context cap**.
- **Trigger = this image + text prompt longer than ~300 chars.** The length
  boundary sits between 256 (OK) and 336 (hang) chars for this image, while the
  same 407-char prompt is fine with a different image → input-dependent
  (image content × text length) prefill hang.
- **Regression**: the exact full-prompt request completed in 2.8 s on 2026-09-08
  (`usage: prompt_tokens 359, completion_tokens 2`) and hangs on 2026-09-09.
  The served bundle/app changed in between (V79 deployment) — the server repo
  was last updated 2026-08-25.

---

## Bundle facts relevant to the hang (read from the device)

From `/storage/emulated/0/Download/Qwen3-VL-4B-Instruct-V79/`:

| File | Value |
|---|---|
| `genie_config.json` | `context.size: 4096`, sampler `temp 0.8, top-k 40, top-p 0.95` |
| `text-generator.json` | context 4096; QnnHtp: `enable-graph-switching: false`, `kv-dim: 128`, `cpu-mask: 0xe0`, `n-threads: 3` |
| `metadata.json` | `/genie/context_lengths: [4096]`; vision output `image_features: [256, 2560]`; preprocessing `512×512, patch 16, merge 2` |
| `img-enc-htp.json` | vision-param `height 32, width 32` (32×32 patch grid) |
| masks | `full/window_attention_mask.raw` = 1024×1024 float32 (4 MB) |
| `htp_backend_ext_config.json` | `dsp_arch: v79`, `perf_profile: burst` |

`enable-graph-switching: false` means one static HTP graph — when an input hits
an edge case, there is no fallback.

---

## What this means / hypothesis for the server author

- The **error is a silent HTP graph-execution hang on the CDSP** for a specific
  (image content × text length) input: no fault, no error return, no timeout.
  The FastRPC call blocks forever; the app is a bystander.
- It is **not** a context/token overflow at this scale (~360/4096 tokens) and
  **not** a request-size issue — it is an input-dependent NPU graph bug,
  introduced/regressed in the V79 QAIRT build.
- To go deeper requires Qualcomm-side tooling that the app cannot expose:
  QnnProfiler / QNN SaND / HTP performance counters on the CDSP, and repro
  natively via `geniex serve` with the same PNG + prompt.

---

## Suggested next steps (author side)

1. Reproduce natively: `geniex serve` + `output/repro_18777.png` + the full
   ScreenQA prompt (406 chars). Confirm hang without the HTTP layer.
2. If confirmed, run under QnnProfiler to see the DSP-side graph state (which
   node/context is stuck, HTP cycle counters).
3. Check whether the vision-encoder graph input (32×32 grid + mrope position
   ids) has a static-shape edge case for this image's preprocessed pixels
   combined with longer text sequences.
4. Consider `enable-graph-switching: true` and/or a prefill watchdog/timeout in
   the SDK wrapper so a stuck graph can be aborted instead of blocking forever.
5. Consider enforcing `max_tokens` clamp so prompt+generation never exceeds the
   compiled context (the app currently forwards `max_tokens=4096` from clients).

## Client-side workarounds (current)

- Keep text prompts < ~250 chars when sending images (dodges the trigger).
- Cap `max_tokens` at ≤1024 (author's own evals use 32).
- Watchdog: restart the app when the endpoint goes silent (pending).
- Skip/handle `screen_id=18777` in ScreenQA runs until fixed.

## Repro artifacts

| Artifact | Path |
|---|---|
| Exact request repro | `examples/repro_18777.py` (screensuite repo) |
| Image sent to the model | `output/repro_18777.png` (+ base64 in `output/repro_18777_b64.txt`) |
| Debug runner (timings + server ping) | `examples/debug_samples.py` |
| Server logs captured during hang | `output/server_log_18777.txt`, `output/server_log_10165.txt` (logcat `QcomLLMServer` + `GenieXSdk` filtered) |
