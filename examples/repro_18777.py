#!/usr/bin/env python
"""Minimal repro for the server author: sample 18777 hangs geniex_vlm_generate.

Usage:
  uv run python examples/repro_18777.py [--timeout 60] [--api-base http://localhost:18181/v1]

Evidence so far:
  - Control sample 10165 completes in ~0.45 s (ttft 345 ms, stop_reason=eos).
  - Sample 18777: geniex_vlm_generate is called (visible in logcat, V level) and
    NEVER returns - no VlmGenerateOutput, no streamed tokens, no profile.
  - The request is small: 358 KB PNG (532x952, resized), ~360 prompt tokens.
  - After the hang, the whole HTTP layer dies (/v1/models included) until the
    app is restarted (am force-stop + am start).
"""
import argparse
import base64
import os
import time

import httpx

PROMPT_TEMPLATE = (
    "Answer the question based on the screenshot only. Do not use any other sources of information. "
    'The answer should be succinct and as short as possible.\n'
    'If the answer is a text from the image, provide it exactly without rephrasing or augmenting. '
    'If there is no answer on the image, output "<no answer>".\n\n'
    "\n\nQuestion:\nHow many more people are available for the top 1000 subscription than the top 100?\n\n"
    "Output:\n"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="output/repro_18777.png")
    parser.add_argument("--api-base", default="http://localhost:18181/v1")
    parser.add_argument("--model", default="Qwen3-VL-4B-Instruct-V79")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    b64 = base64.b64encode(open(args.image, "rb").read()).decode()
    payload = {
        "model": args.model,
        "max_tokens": 4096,
        "temperature": 0,
        "stream": False,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": PROMPT_TEMPLATE},
            ],
        }],
    }

    print(f"POST {args.api_base}/chat/completions (image={os.path.basename(args.image)}, "
          f"{len(b64)//1024} KB b64, timeout={args.timeout}s)")
    t0 = time.monotonic()
    try:
        r = httpx.post(f"{args.api_base}/chat/completions", json=payload, timeout=args.timeout)
        dt = time.monotonic() - t0
        if r.status_code == 200:
            j = r.json()
            print(f"OK in {dt:.1f}s: {j['choices'][0]['message']['content'][:100]!r}")
            print("timings:", j.get("timings"))
            print("usage:", j.get("usage"))
        else:
            print(f"HTTP {r.status_code} in {dt:.1f}s: {r.text[:200]}")
    except Exception as e:
        print(f"FAILED after {time.monotonic() - t0:.1f}s: {type(e).__name__}: {e}")
        print("-> If this times out: the server is hung on this sample. Check logcat:")
        print("   adb logcat -d -s QcomLLMServer:V | tail -20")
        print("   (the last line should be VlmGenerateInput(...) with no VlmGenerateOutput)")


if __name__ == "__main__":
    main()
