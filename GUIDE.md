# GUIDE — Benchmarking Qwen3-VL-4B with ScreenSuite

This guide explains how to run ScreenSuite's **offline** benchmarks (perception, grounding, single-step actions) against `Qwen3-VL-4B-Instruct` served from either:

1. **A remote OpenAI-compatible endpoint** (e.g. `https://voice-staging.cyberbot.vn/v1`)
2. **A local vLLM server** (GPU machine)

and how to compare the numbers with the [ScreenSuite blog post](https://huggingface.co/blog/screensuite).

---

## 1. What we run (and what we skip)

The blog's chart covers two groups:

| Group | Benchmarks | Where it runs |
|---|---|---|
| Perception/Grounding | ScreenQA-short/complex, ScreenSpot v1/v2/pro (click + bbox), WebSRC, VisualWebBench | Anywhere (just API calls) |
| Single-step actions | Showdown-Clicks, Multimodal-Mind2Web, AndroidControl | Anywhere (just API calls) |
| Multi-step agents | OSWorld, AndroidWorld, GAIA-Web, Mind2Web-Live, BrowseComp | **Skipped** — requires Docker + KVM on bare-metal Linux |

The run script automatically skips multi-step benchmarks, so you get the 10 offline scores that are directly comparable to the blog chart.

---

## 2. Prerequisites

- Python >= 3.11 (the repo pins 3.11; `uv` installs it automatically)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- For local serving: a Linux box (or WSL2) with a GPU (~10 GB VRAM at bf16) — vLLM does not run natively on Windows
- An API key for the remote endpoint (if using it)

---

## 3. Install ScreenSuite

```bash
git clone --recurse-submodules git@github.com:huggingface/screensuite.git
cd screensuite

pip install uv
uv sync --extra submodules --python 3.11
```

Create your config file:

```bash
cp .env.example .env
```

Edit `.env` and set one of these two blocks:

**Remote endpoint:**
```bash
OPENAI_API_BASE=https://voice-staging.cyberbot.vn/v1
OPENAI_API_KEY=your-key-here
MODEL_ID=Qwen3-VL-4B-Instruct
```

**Local vLLM (comment out the block above):**
```bash
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY
MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
```

> Note: the `.env` file is optional — every setting can be passed as a CLI flag (see below).

---

## 4. Serve the model locally (optional, if not using the remote endpoint)

On the GPU machine:

```bash
pip install vllm
vllm serve Qwen/Qwen3-VL-4B-Instruct --max-model-len 16384 --gpu-memory-utilization 0.9
```

This starts an OpenAI-compatible server at `http://localhost:8000/v1`.

> Why vLLM (not Ollama/Transformers)? Qwen3-VL's image processor smart-resize must run server-side; vLLM implements it exactly like the blog's setup, keeping numbers comparable. It also gives the same tokenizer/chat-template behavior.

---

## 5. Run the benchmarks

All commands run from the repo root.

### 5.1 Smoke test first (recommended)

20 samples per benchmark, 1 worker — a few minutes, catches endpoint/model-name/auth issues early:

```bash
uv run python examples/run_qwen3_vl.py --smoke
```

### 5.2 Full run — remote endpoint

```bash
uv run python examples/run_qwen3_vl.py --n-samples 300 --workers 8
```

### 5.3 Full run — local vLLM

```bash
uv run python examples/run_qwen3_vl.py \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --n-samples 300 --workers 8
```

### 5.4 Options

| Flag | Default | Meaning |
|---|---|---|
| `--api-base` | env `OPENAI_API_BASE` | Endpoint base URL |
| `--api-key` | env `OPENAI_API_KEY` / `EMPTY` | API key |
| `--model-id` | env `MODEL_ID` / `Qwen3-VL-4B-Instruct` | Model name as served |
| `--n-samples` | `300` | Max samples per benchmark |
| `--workers` | `4` | Parallel workers (keep low for remote endpoints) |
| `--max-tokens` | `4096` | Max output tokens |
| `--run-name` | auto (`<model>_<date>`) | Output file name; resume with the same name |
| `--smoke` | off | 20 samples, 1 worker, `test_mode=True` |
| `--no-resize` | off | Send images at original resolution (default: Qwen smart-resize, matching the blog) |
| `--benchmarks` | all offline | e.g. `--benchmarks screenqa_short screenspot-v2-click-prompt` |

> Tips:
> - **Workers:** the perception benchmarks are pure API calls and parallelize well; the agentic ones (mmind2web, android_control) call the model in a loop — a few workers is plenty.
> - **Resume:** re-running with the same `--run-name` skips benchmarks already in the result file.

---

## 6. Results

Results are appended to `output/<run-name>.jsonl`, one JSON line per benchmark:

```json
{"benchmark_name": "screenqa_short", "metrics": {"accuracy": 0.55}}
```

### View a single run

```bash
uv run python examples/summarize_results.py
```

### Compare multiple runs side by side

```bash
# one column per run file, pick the key metric
uv run python examples/summarize_results.py --compare --metric accuracy
```

---

## 7. Interpreting the numbers

Compare your rows against the blog's chart (`Qwen2.5-VL-3B` and `Qwen2.5-VL-7B` are the closest baselines to Qwen3-VL-4B).

Caveats to keep in mind when reading the scores:

- ScreenSuite is **vision-only** — no DOM/accessibility tree. Scores on Mind2Web, OSWorld etc. are lower than the original papers' numbers on purpose.
- Grounding scores (ScreenSpot click/bbox) depend on the image resize config; keep the default unless you know your serving stack processes images differently.
- `temperature=0` is set in the script for reproducibility (matching the blog runs).
- Sample counts in the blog: ScreenSpot ~1.3k, ScreenQA ~8–12k, VisualWebBench ~1.5k... you're running a subset (`--n-samples`), so expect noise on small sample counts. Use `--n-samples 500` for grounding benchmarks if you want stable numbers.

---

## 8. Internet / network requirements

The benchmark machine needs network for **three** things. Two of them are covered if the box can reach Hugging Face Hub (datasets and model weights download straight onto the box — no pre-staging needed):

| Need | When | What | Covered by HF access? |
|---|---|---|---|
| PyPI packages (`uv sync`: torch, smolagents, transformers, datasets...) | Install (one-time) | ~4 GB of wheels | ❌ No — separate from HF Hub |
| Benchmark datasets (10 HF repos, ~10–25 GB images) | First run of each benchmark | `load_dataset()` caches to `~/.cache/huggingface` | ✅ Yes |
| Model weights (Qwen3-VL-4B ~9 GB) | Local vLLM serving | vLLM pulls from HF Hub on first `serve` | ✅ Yes |
| Model endpoint | Every request | Remote: `https://voice-staging.cyberbot.vn/v1`; local: `localhost:8000` | ❌ Remote endpoint is a separate host |

### If the box cannot reach PyPI

Option A — **venv copy**: run `uv sync` on any internet machine **with the same OS/Python**, then copy the whole `.venv` directory (and the repo) to the target.

Option B — **wheel staging** (any OS):

```bash
# on an internet machine, target = linux x86_64 + python 3.11:
pip download -r requirements.txt --only-binary=:all: \
  --platform manylinux2014_x86_64 --python-version 311 \
  -d wheels/
# copy wheels/ to the target, then:
pip install --no-index --find-links=wheels/ -r requirements.txt
```

Option C — check whether the box has a PyPI mirror configured (`pip config list`) or `PIP_INDEX_URL` set — many offline clusters do.

### Datasets cache (portable)

The HF cache is plain files (parquet + images), OS-agnostic. To re-locate or back it up:

```bash
export HF_HOME=/data/hf_cache   # point anywhere before first run
```

If the box has HF access, this is only useful to move the cache to a bigger disk.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `uv: command not found` | `pip install uv` |
| `Provider is required` / auth errors | Check `OPENAI_API_KEY` and that the base URL ends with `/v1` |
| `Error running benchmark ... 404` | The endpoint doesn't serve `--model-id` under that exact name — check the served model name |
| Timeouts on big screenshots | Re-run with the default image resize (don't pass `--no-resize`) |
| `evdev` build error during install | On Linux: `sudo apt-get install build-essential` |
| Multi-step benchmarks silently skipped | Expected — they need Docker + KVM on bare-metal Linux |
| Zero/low scores everywhere | Check a single prompt by running `--smoke --benchmarks screenqa_short` and look at the model's raw answers |
