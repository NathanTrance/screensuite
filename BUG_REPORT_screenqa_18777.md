# Bug Report: Silent NPU (HTP/CDSP) hang on ScreenQA-Complex samples 18777 & 5362

**Status:** Open — needs QAIRT/Qualcomm-side investigation
**Report date:** 2026-09-09 (updated with second trigger + cascade analysis)
**Reported by:** nhatth (testing)

---

## Summary

Two ScreenQA-Complex samples (`screen_id=18777`, `screen_id=5362`) deterministically
hang the on-device VLM server: the request never completes, no token is ever
generated, the inference mutex is held forever (all subsequent requests get
`503 model busy`), and after enough blocked workers the whole HTTP layer stops
responding (including `GET /v1/models`) until the app is restarted.

The hang is a **silent HTP (Hexagon Tensor Processor) graph-execution hang on the
CDSP**: no crash, no error code, no SDK timeout — the native `geniex_vlm_generate`
call simply never returns. Each trigger is a specific image combined with a text
prompt longer than ~300 characters. It is a regression: the exact same
request completed in 2.8 s on 2026-09-08 and hangs on 2026-09-09 (V79 build).

**Working hypothesis for the mechanism:** a value (pixel statistics / attention
scale / position encoding) going out of the HTP's fixed-point range for these
inputs produces NaN/denormal propagation, and the DSP graph never terminates
instead of failing — consistent with: no fault, no error return, 0% app CPU,
clean dmesg, DSP remoteproc still "running".

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

## Second trigger + full-run cascade analysis (500-sample debug run)

`output/debug_samples.jsonl` (screenqa_complex_500, workers=1, `--api-timeout 60`):

### Bursts

| Burst | Trigger idx | Trigger id | Trigger latency | Follow-on failures |
|---|---|---|---|---|
| 1 | 335 | **18777** | 62.6 s (client timeout; request stuck server-side) | 336-376 (41 instant 503s, 0.4-0.8 s each) |
| 2 | 392 | **5362** | 62.5 s (same) | 393-433 (41 instant 503s) |

All 84 failures are `503 model busy`. Only the two burst-first samples are real
triggers; the other 82 are cascade victims (their content is fine — verified by
the 14 other "how many more" questions that passed, incl. `42238`, `12163`,
`7630`...).

Timing: trigger stuck for ~62 s (client) + 41 × ~0.7 s ≈ 29 s of busy-rejections
≈ **~90 s total stuck duration** before the lock was released — i.e. the hung
request self-terminated after ~90 s in this run (but hung >300 s in later
isolated repros — duration varies with server state).

### Trigger sample contents

| | 18777 | 5362 |
|---|---|---|
| Question | "How many more people are available for the top 1000 subscription than the top 100?" | "How many more folders are there than albums?" |
| Ground truth | `982` | `4` |
| Prompt length | 406 chars | 368 chars |
| Raw image | 540×960 | 1080×1920 |
| Resized (sent) | 532×952 | 1092×1932 |
| PNG / base64 | 350 KB / 466 KB | 463 KB / 618 KB |
| Mean brightness (0-255) | 219.6 (rank 472/500, z=+1.0) | **26.8 (rank 9/500, z=-1.9 — darkest decile)** |
| Edge density (>30 diff) | **8.5% (rank 494/500, z=+3.3 — 2nd most detailed in the set)** | 2.1% (median) |

Both questions are the same type ("how many more X than Y" — counting +
subtraction). 14 other same-type questions passed, so the question alone is not
the trigger. Image stats: 18777 is an extreme outlier in content density;
5362 is an extreme outlier in darkness (both directions of "out-of-range"
inputs — consistent with the NaN/out-of-range mechanism hypothesis).

### What does NOT correlate

- Text length alone (non-triggers span 345-448 chars, triggers 368/406)
- Question type (see above)
- Payload bytes (largest payload in the run = 2962 KB, passed)
- Image size class (both 532×952 and 1092×1932 classes contain passing samples)
- Token count (all samples ≈ 256 image + 60-120 text tokens)

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
- **Mechanism candidate (to verify with QnnProfiler):** the two triggers are
  extreme outliers in opposite directions (densest image / darkest image). A
  value going out of the HTP's fixed-point range (e.g. attention logits,
  normalization statistics, or the merged position ids for these pixel
  distributions) could yield NaN/denormal values inside the graph; the DSP then
  fails to terminate instead of reporting an error. Symptoms match a stall
  (0% CPU, no fault) rather than a fault.
- To go deeper requires Qualcomm-side tooling that the app cannot expose:
  QnnProfiler / QNN SaND / HTP performance counters on the CDSP, and repro
  natively via `geniex serve` with the same PNG + prompt.

---

## Suggested next steps (author side)

1. Reproduce natively: `geniex serve` + `output/repro_18777.png` (and
   `output/trigger_5362.png`) + the full ScreenQA prompt (406 chars). Confirm
   hang without the HTTP layer.
2. If confirmed, run under QnnProfiler to see the DSP-side graph state (which
   node/context is stuck, HTP cycle counters, NaN/denormal flags if exposed).
3. Check whether the vision-encoder graph input (32×32 grid + mrope position
   ids) has a static-shape edge case for these images' preprocessed pixels
   (extreme brightness/density outliers) combined with longer text sequences.
4. Investigate fixed-point range violations: feed the two trigger PNGs through
   the QAIRT preprocessing offline and inspect intermediate values (pixel
   normalization, attention logits) for out-of-range/NaN before the HTP graph.
5. Consider `enable-graph-switching: true` and/or a prefill watchdog/timeout in
   the SDK wrapper so a stuck graph can be aborted instead of blocking forever.
6. Consider enforcing `max_tokens` clamp so prompt+generation never exceeds the
   compiled context (the app currently forwards `max_tokens=4096` from clients).

## Client-side workarounds (current)

- Keep text prompts < ~250 chars when sending images (dodges the trigger).
- Cap `max_tokens` at ≤1024 (author's own evals use 32).
- Watchdog: restart the app when the endpoint goes silent (pending).
- Skip/handle `screen_id=18777` and `screen_id=5362` in ScreenQA runs until fixed.

## Repro artifacts

| Artifact | Path |
|---|---|
| Exact request repro (18777) | `examples/repro_18777.py` (screensuite repo) |
| Trigger images (resized, what the model sees) | `output/repro_18777.png`, `output/trigger_5362.png` (+ base64 in `output/repro_18777_b64.txt`) |
| Debug runner (timings + server ping) | `examples/debug_samples.py` |
| Full-run evidence (500 samples, 2 bursts) | `output/debug_samples.jsonl` |
| Server logs captured during hang | `output/server_log_18777.txt`, `output/server_log_10165.txt` (logcat `QcomLLMServer` + `GenieXSdk` filtered) |
