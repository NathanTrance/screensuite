# SUMMARY — ScreenSuite + Qwen3-VL-4B benchmarking

Everything needed to run ScreenSuite's offline benchmarks against Qwen3-VL-4B-Instruct from an OpenAI-compatible endpoint, including the fixes needed for the VTCC box (proxy, TLS, offline HF).

## Repo

- Fork: `https://github.com/NathanTrance/screensuite` (push target: `fork`)
- Upstream: `https://github.com/huggingface/screensuite` (`origin`)
- Main entry: `examples/run_qwen3_vl.py` · results viewer: `examples/summarize_results.py` · config template: `.env.example`

## What runs and what doesn't

| Group | Benchmarks | Runs on this box? |
|---|---|---|
| Perception/Grounding | ScreenQA-short/complex (mobile), ScreenSpot v2/pro (desktop/Android UI), WebSRC, VisualWebBench | ✅ offline, just API calls |
| Single-step | Showdown-Clicks, Multimodal-Mind2Web, AndroidControl | ✅ offline |
| Multi-step | OSWorld, AndroidWorld (+MiniWob), GAIA-Web, Mind2Web-Live, BrowseComp | ❌ needs bare-metal Linux + Docker + KVM |

ScreenSpot-v2 is 100% desktop (`pc_*.png`); ScreenSpot-Pro is Android UI + AutoCAD. Mobile coverage offline = ScreenQA + AndroidControl.

## Install (on a fresh machine)

```bash
git clone --recurse-submodules https://github.com/NathanTrance/screensuite.git
cd screensuite
pip install uv
uv sync --extra submodules --python 3.11
cp .env.example .env   # set OPENAI_API_BASE / OPENAI_API_KEY / MODEL_ID
```

Notes for the VTCC box:
- `osworld` submodule uses an SSH URL — if the clone hangs: `git config --global url."https://github.com/".insteadOf "git@github.com:"` then `git submodule update --init --recursive --force`
- TLS errors from uv → `export UV_NATIVE_TLS=true` (rustls doesn't trust the corp CA; git works because it uses the system store)
- `uv sync` fails building `android-world`/`osworld` (pkg_resources removed in setuptools 82+) → root `pyproject.toml` already pins `build-constraint-dependencies = ["setuptools<82"]`; if your local copy is older, patch the submodules directly:
  `sed -i 's/requires = \["setuptools>=61.0"\]/requires = ["setuptools>=61,<82"]/' android_world/pyproject.toml osworld/pyproject.toml`
  then `rm -rf ~/.cache/uv/builds-v0 .venv && uv sync --extra submodules --python 3.11`

## Network / proxy (VTCC box)

The proxy exists for HuggingFace only. The model endpoint must go **direct** — the proxy MITMs TLS with a self-signed cert, which breaks the OpenAI client.

```bash
# bashrc helpers: set_s5_proxy / remove_proxy (you already have them)
export NO_PROXY=voice-staging.cyberbot.vn        # ALWAYS — API bypasses the proxy
export no_proxy=voice-staging.cyberbot.vn
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1    # offline pass: cache-only HF
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt   # alternative to --insecure
```

If the API still fails TLS: `--insecure` flag on the runner (skips verification for that endpoint).

## Run

```bash
# smoke (20 samples, 1 worker) — always first
uv run python examples/run_qwen3_vl.py --smoke --benchmarks screenqa_short

# full test set, one benchmark (e.g. ScreenQA-Short, 8,427 samples)
uv run python examples/run_qwen3_vl.py --load-full --benchmarks screenqa_short --workers 16 --run-name full-screenqa-short

# full test sets, all offline benchmarks (≈86k API calls total; websrc_dev alone is 52k)
uv run python examples/run_qwen3_vl.py --load-full --workers 16 --run-name full-all

# results
uv run python examples/summarize_results.py --compare --metric f1
uv run python examples/summarize_results.py --compare --metric accuracy
```

Key flags: `--n-samples N` (default 300) · `--load-full` (whole split; also sets max_samples=None) · `--workers N` · `--run-name X` (resumable: skips benchmarks already in `output/<run>.jsonl`) · `--insecure` · `--no-resize` (send original resolution; default = Qwen smart-resize, matches the blog) · `--benchmarks b1 b2`

Results: `output/<run-name>.jsonl` (one JSON line per benchmark) + per-sample answers in `output/<benchmark>/<run>/answers.jsonl`.

## Fixes baked into this fork (in case you diff against upstream)

1. **Lazy Docker client** — `DockerProvider` no longer connects at import time; importing `screensuite` works without Docker.
2. **smolagents compat** — `HfApiModel` → `InferenceClientModel` fallback (removed in newer smolagents); `OpenAIModel`/`OpenAIServerModel` handled in the runner.
3. **setuptools<82** build constraint for the submodule packages.
4. **Split slicing at load** (`test[:N]`) with fallback to full-split + `select` — avoids materializing 69k-row train sets for small runs; `max_samples` supported in all offline `load()` overrides.
5. **Bounded in-flight window** in `get_model_responses` (workers×8) instead of submitting all samples at once.
6. **Free images after generation** — retained responses keep only `{width, height}` instead of multi-MB PIL images (fixes OOM freeze at ~3000 samples on full splits).
7. **Concise per-sample error logs** (tracebacks at debug only) — no more thousands-of-lines flood when the API errors.
8. **`--insecure` + temperature=0** in the runner.

## Comparison targets

Compare against the blog chart: Qwen2.5-VL-3B / 7B are the closest baselines for Qwen3-VL-4B. ScreenSuite is vision-only (no DOM/accessibility tree), so scores on Mind2Web etc. are lower than the original papers on purpose.

## Troubleshooting quick table

| Symptom | Fix |
|---|---|
| uv TLS error | `UV_NATIVE_TLS=true` |
| clone hangs on submodule | insteadOf rewrite + `git submodule update --init --recursive --force` |
| pkg_resources build error | setuptools<82 (already in pyproject) + `rm -rf ~/.cache/uv/builds-v0 .venv` |
| `load()` got an unexpected keyword argument 'max_samples' | stale fork — `git pull` |
| API: SSL CERTIFICATE_VERIFY_FAILED | `NO_PROXY=voice-staging.cyberbot.vn` or `--insecure` |
| dataset generation dies mid-way | disk space (`df -h`) + `rm -rf ~/.cache/huggingface/datasets` |
| machine freezes on big runs | free the fixed `0484748` — re-pull |
| 'set' object has no attribute 'append' | free the fixed `679681b` — re-pull |
| multistep benchmarks silently skipped | expected without Docker + KVM |
