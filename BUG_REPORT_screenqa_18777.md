# Bug Report: Repetition-loop decoding wedges the single-concurrency VLM server

**Samples:** ScreenQA-Complex `screen_id=18777` and `screen_id=5362` (Qwen3-VL-4B-Instruct-V79)
**Status:** Root cause confirmed by Qualcomm CE (2026-09-25), verified locally
**Reported by:** nhatth (testing) · **Server author:** Lâm

---

## Summary (corrected root cause)

The model **enters a repetition loop** on these inputs and emits the same token until
`max_tokens` is reached:

| Sample | Output (greedy, temp=0) | Repetition |
|---|---|---|
| 18777 ("How many more people...") | `999999999999999999...` | `9` |
| 5362 ("How many more folders than albums?") | `1111111111111111...` | `1` |
| 10165 (control) | `5 days` then **EOS** | – |

Because the server serves **one inference at a time** (mutex) and generates
non-streaming, a looping request holds the lock for the full generation
(~4096 tokens ÷ ~25 tok/s ≈ **160 s**) while all other requests get
`503 model busy`. A client-side timeout (curl `--max-time`) only disconnects the
client — the native generation keeps running and keeps holding the mutex.

This fully explains the observed "hang" and the ~90-120 s bursts of 42 consecutive
503s in the 500-sample run. It is **not** an NPU/CDSP hang, not a NaN/fixed-point
issue, and not an input-size problem.

**Secondary server-side gaps** (as suggested by Qualcomm CE):
- No server-side generation deadline/timeout.
- `vlmWrapper.stopStream()` is not called when the client disconnects.
- No repetition penalty available (request schema only has `max_tokens`/`temperature`).

---

## Verification (2026-09-25, latest bundle v0.62.0, SM8750 device `68a063cd`)

Requests = the exact filed payloads (`output/qualcomm_case/payload_*.json`):

| Request | max_tokens | Time | Result |
|---|---|---|---|
| 18777 | 16 | 1.11 s | `9999999999999999`, `finish_reason=length` |
| 18777 | 256 | 10.62 s | 256× `9`, length, decode 25.2 tok/s |
| 18777 | 4096 | no response within 90 s | mutex held (≈160 s expected) |
| 5362 | 16 | 1.11 s | `1111111111111111`, length |
| 5362 | 4096 | no response within 90 s | mutex held |
| 10165 (control) | 4096 | 0.55 s | `5 days`, **stop_reason=eos** |

The issue **reproduces on the latest release (v0.62.0, 2026-09-10)** — it is not
fixed by updating the bundle.

---

## Cascade mechanics (why one request kills the server)

1. Looping request takes the inference mutex and generates up to `max_tokens`
   (4096 → ~160 s at ~25 tok/s on the NPU).
2. Every concurrent request → `503 {"error":"model busy"}` (correct mutex behavior),
   e.g. 42 consecutive instant-503s per stuck request in the debug run.
3. Client timeouts do not cancel server-side generation.
4. Recovery only when the loop hits `max_tokens` (self-recovers after ~90-160 s),
   or the app is restarted.

## Earlier misdiagnosis (for the record)

Initial client-side evidence suggested a "silent NPU hang" (no first token, 0 SSE
chunks, 0% app CPU, clean dmesg). That was wrong:
- the "0 SSE chunks" run hit an **already-wedged server** (a previous looping
  request held the mutex) and received a 503 JSON body instead of SSE;
- `GenieXSdk VlmGenerateOutput` only logs at completion — a 160 s generation had
  simply not finished when logs were captured;
- 0% app CPU is expected during DSP-side decoding.

---

## Fixes / mitigations

**Client (applied in this benchmark harness):**
- FIXED 2026-09-25: `examples/run_qwen3_vl.py` set `max_tokens=4096` on the smolagents
  model constructor, which (per smolagents' documented priority) overrides per-call
  kwargs — silently overriding the benchmark configs' limits and re-enabling the full
  4096-token loops. Now `max_tokens` is only set when explicitly requested; ScreenQA
  runs use the config's 256. Clean 500-sample run after the fix: f1=0.621,
  exact_match=0.582, proportion_missing=0.0, avg latency 0.85 s; both loop samples
  completed in ~11 s each without wedging the server.
- Cap `max_tokens` for QA-style benchmarks at **≤256** (answers are short;
  caps loop damage to ~10 s). ScreenQA config now uses 256.
- Keep per-request client timeouts but do not rely on them to release the mutex.

**Server (recommended to the author):**
- Enforce a generation deadline (`max_tokens` and/or wall-clock) server-side.
- Call `vlmWrapper.stopStream()` on client disconnect / deadline.
- Expose `repetition_penalty` (and optionally `top_p`) to reduce loops.
- Consider re-calibrating/quantizing the model with in-domain data (Qualcomm CE
  suggestion) to reduce repetitive outputs.

---

## Repro artifacts (this repo)

| Artifact | Path |
|---|---|
| Exact payloads (18777 / 5362 / control) | `output/qualcomm_case/payload_*.json` |
| Images sent | `output/qualcomm_case/image_*.png` |
| Reproducer | `examples/repro_18777.py` |
| Per-sample debug runner | `examples/debug_samples.py` |
| Run evidence (2 bursts of 42) | `output/debug_samples.jsonl` |
| Updated logs (v0.62.0, hang + control) | `output/qualcomm_case/logcat_hang_*_v062.txt` |

## Environment

- Device: SM8750 (Snapdragon 8 Elite, `sun`, DSP v79), Android 15 (SDK 35)
- App: `com.vai.qcom_llm_server` (GenieX SDK + NanoHTTPD, serialized inference)
- Bundle: Qwen3-VL-4B-Instruct-V79 QAIRT W4A16, context 4096, `enable-graph-switching: false`
- Sampling: `temperature=0` (greedy) - the loop occurs even at temp 0
