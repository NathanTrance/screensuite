#!/usr/bin/env python
"""Identify samples that crash the on-device/remote server.

Runs one benchmark sample-by-sample (workers=1), logging per-sample request
metadata (image dims, est. payload, est. tokens) plus a server health ping after
EACH request. When the server dies, the sample whose request was in flight is the
crash candidate; the log then shows the recovery pattern (errors until server is
back), letting you compare the crashing sample's payload vs. normal ones.

Usage:
  uv run python examples/debug_samples.py \
    --benchmark screenqa_short_500 --n 100 \
    --api-base http://localhost:18181/v1 --model-id Qwen3-VL-4B-Instruct-V79 \
    --out output/debug_samples.jsonl

Output columns (one JSON per sample step):
  sample_idx, step_idx, sample_id, text_chars, img_w, img_h, img_png_kb,
  est_img_tokens, latency_s, ok, error, server_alive_after
Plus lines like:  >>>>>> SERVER DIED on sample 37 step 0 (recovered in 12.3s)
"""
import argparse
import io
import json
import math
import os
import time
from collections.abc import Generator
from datetime import datetime

import httpx

from screensuite import EvaluationConfig, ImageResizeConfig, get_registry
from screensuite.response_generation import select_examples_to_test

try:
    from smolagents import OpenAIModel
    SERVER_MODEL_CLS = OpenAIModel
except ImportError:
    from smolagents import OpenAIServerModel
    SERVER_MODEL_CLS = OpenAIServerModel


def ping_server(api_base: str, timeout: float = 3.0) -> tuple[bool, float]:
    t0 = time.monotonic()
    try:
        r = httpx.get(f"{api_base}/models", timeout=timeout)
        return r.status_code == 200, time.monotonic() - t0
    except Exception:
        return False, time.monotonic() - t0


def extract_metadata(messages: list[dict]) -> dict:
    text_chars = 0
    img_w = img_h = img_png_kb = None
    n_images = 0
    for message in messages:
        for content in message.get("content", []):
            if isinstance(content, dict):
                if content.get("type") == "text":
                    text_chars += len(content.get("text", ""))
                elif content.get("type") == "image":
                    n_images += 1
                    img = content.get("image")
                    if img is not None and hasattr(img, "width"):
                        img_w, img_h = img.width, img.height
                        buf = io.BytesIO()
                        img.save(buf, "PNG")
                        img_png_kb = round(len(buf.getvalue()) / 1024, 1)
    est_img_tokens = None
    if img_w and img_h:
        est_img_tokens = math.ceil(img_w / 28) * math.ceil(img_h / 28) * 256
    return {
        "text_chars": text_chars,
        "n_images": n_images,
        "img_w": img_w,
        "img_h": img_h,
        "img_png_kb": img_png_kb,
        "est_img_tokens": est_img_tokens,
        "est_total_tokens": (est_img_tokens or 0) + text_chars // 4,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="screenqa_short_500")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--api-base", default="http://localhost:18181/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-id", default="Qwen3-VL-4B-Instruct-V79")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--ping-interval", type=float, default=2.0)
    parser.add_argument("--max-recovery-wait", type=float, default=600.0,
                        help="Max seconds to wait for the server to recover before giving up")
    parser.add_argument("--out", default="output/debug_samples.jsonl")
    parser.add_argument("--from-index", type=int, default=0, help="Skip to sample index")
    args = parser.parse_args()

    registry = get_registry()
    bench = next(b for b in registry.list_all() if b.name == args.benchmark)
    bench.load(max_samples=args.n)

    evaluation_config = EvaluationConfig(
        test_mode=False,
        parallel_workers=1,
        max_samples_to_test=args.n,
        run_name=None,
        image_resize_config=ImageResizeConfig(),
    )
    samples = select_examples_to_test(bench.dataset, evaluation_config)

    model = SERVER_MODEL_CLS(
        model_id=args.model_id,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=0,
        client_kwargs={"max_retries": 1, "timeout": args.api_timeout},
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    logf = open(args.out, "a", encoding="utf-8")
    print(f"# debug run {datetime.now().isoformat()} benchmark={args.benchmark} n={args.n}")

    crash_candidates: list[dict] = []
    for sample_idx, sample in enumerate(samples):
        if sample_idx < args.from_index:
            continue
        sample_id = None
        for key in ("screen_id", "file_name", "episode_id"):
            if isinstance(sample, dict) and key in sample:
                sample_id = sample[key]
                break

        annotated = bench._get_annotated_input_from_sample(sample, evaluation_config)
        if isinstance(annotated, Generator):
            steps = annotated
        else:
            steps = [annotated]

        for step_idx, annotated_input in enumerate(steps):
            meta = extract_metadata(annotated_input.messages)
            t0 = time.monotonic()
            ok = True
            error = None
            try:
                model.generate(annotated_input.messages)
            except Exception as e:
                ok = False
                error = f"{type(e).__name__}: {e}"
            latency = round(time.monotonic() - t0, 2)

            alive, ping_t = ping_server(args.api_base)
            record = {
                "ts": datetime.now().isoformat(),
                "sample_idx": sample_idx,
                "step_idx": step_idx,
                "sample_id": sample_id,
                "latency_s": latency,
                "ok": ok,
                "error": error,
                "server_alive_after": alive,
                **meta,
            }
            logf.write(json.dumps(record, default=str) + "\n")
            logf.flush()

            if not alive:
                crash_candidates.append(record)
                print(f">>>>>>>> SERVER DIED on sample {sample_idx} step {step_idx} "
                      f"(latency {latency}s, ok={ok}, err={str(error)[:80]})")
                print(f"         meta: {meta}")
                # Wait for recovery, logging how long it takes
                t_wait = time.monotonic()
                while time.monotonic() - t_wait < args.max_recovery_wait:
                    time.sleep(args.ping_interval)
                    alive_now, _ = ping_server(args.api_base)
                    if alive_now:
                        print(f"         recovered after {time.monotonic() - t_wait:.1f}s")
                        break
                else:
                    print("         server did not recover within max-recovery-wait; aborting")
                    logf.close()
                    return
                alive = True

    logf.close()
    print(f"\n# done. crash candidates: {len(crash_candidates)}")
    for c in crash_candidates:
        print(f"#   sample {c['sample_idx']} step {c['step_idx']} id={c['sample_id']} "
              f"tokens_est={c['est_total_tokens']} img={c['img_w']}x{c['img_h']} png_kb={c['img_png_kb']}")
    if crash_candidates:
        all_tokens = [c["est_total_tokens"] for c in crash_candidates]
        print(f"# crash tokens: {all_tokens}")


if __name__ == "__main__":
    main()
