#!/usr/bin/env python
"""Run ScreenSuite offline benchmarks against Qwen3-VL-4B (or any OpenAI-compatible endpoint).

Works with:
  - a remote OpenAI-compatible endpoint (e.g. https://voice-staging.cyberbot.vn/v1)
  - a local vLLM / SGLang / Ollama server (OPENAI_API_BASE=http://localhost:8000/v1)

Usage:
  python examples/run_qwen3_vl.py --smoke                          # 20 samples, 1 worker, sanity check
  python examples/run_qwen3_vl.py --n-samples 300 --workers 8      # real run vs remote endpoint
  python examples/run_qwen3_vl.py --api-base http://localhost:8000/v1 \
      --api-key EMPTY --model-id Qwen/Qwen3-VL-4B-Instruct         # local vLLM

Config can also come from .env: OPENAI_API_BASE, OPENAI_API_KEY, MODEL_ID.
Results are appended to output/{run_name}.jsonl (one line per benchmark).
"""
import argparse
import copy
import json
import os
from datetime import datetime

from dotenv import load_dotenv
import httpx

try:
    from smolagents import OpenAIModel
    SERVER_MODEL_CLS = OpenAIModel
except ImportError:
    from smolagents import OpenAIServerModel
    SERVER_MODEL_CLS = OpenAIServerModel

from screensuite import EvaluationConfig, ImageResizeConfig, get_registry

load_dotenv()

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(RESULTS_DIR, mode=0o775, exist_ok=True)


def launch_test(model, benchmarks, original_evaluation_config, load_samples):
    evaluation_config = copy.deepcopy(original_evaluation_config)  # NOTE: important!
    if evaluation_config.run_name is None:
        model_name = model.model_id.replace("/", "-")
        evaluation_config.run_name = f"{model_name}_{datetime.now().strftime('%Y-%m-%d')}"

    print(f"===== Running evaluation under name: {evaluation_config.run_name} =====")

    output_results_file = os.path.join(RESULTS_DIR, f"{evaluation_config.run_name}.jsonl")
    processed_benchmarks = set()
    try:
        with open(output_results_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "benchmark_name" in data:
                        processed_benchmarks.add(data["benchmark_name"])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass

    print("-> Found these processed benchmarks: ", processed_benchmarks)

    for benchmark in sorted(benchmarks, key=lambda b: b.name):
        if benchmark.name in processed_benchmarks:
            print(f"Skipping already processed benchmark: {benchmark.name}")
            continue
        if "multistep" in benchmark.tags:
            continue
        print("=" * 100)
        print(f"Running benchmark: {benchmark.name}")
        try:
            benchmark.load(max_samples=load_samples)
            results = benchmark.evaluate(model, evaluation_config)
            print(f"Results for {benchmark.name}: {results}")
            metrics_entry = {"benchmark_name": benchmark.name, "metrics": results._metrics}
            with open(output_results_file, "a") as f:
                f.write(json.dumps(metrics_entry) + "\n")
        except Exception as e:
            print(f"Error running benchmark {benchmark.name}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-VL against ScreenSuite offline benchmarks")
    parser.add_argument("--api-base", type=str, default=os.environ.get("OPENAI_API_BASE"),
                        help="OpenAI-compatible base URL (env: OPENAI_API_BASE)")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        help="API key (env: OPENAI_API_KEY)")
    parser.add_argument("--model-id", type=str, default=os.environ.get("MODEL_ID", "Qwen3-VL-4B-Instruct"),
                        help="Model name as served by the endpoint")
    parser.add_argument("--n-samples", type=int, default=300, help="Max samples per benchmark")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Quick sanity check: 20 samples, 1 worker")
    parser.add_argument("--load-full", action="store_true",
                        help="Load the full dataset instead of slicing to the number of samples "
                             "(matches the blog protocol exactly; much slower first run)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max retries for API calls (default: 2)")
    parser.add_argument("--api-timeout", type=float, default=60.0,
                        help="API request timeout in seconds (default: 60)")
    parser.add_argument("--insecure", action="store_true",
                        help="Skip TLS certificate verification for the model endpoint "
                             "(use only on trusted networks / corporate proxies with self-signed certs)")
    parser.add_argument("--no-resize", action="store_true",
                        help="Send images at original resolution (default: Qwen smart-resize, same as the blog run)")
    parser.add_argument("--benchmarks", type=str, nargs="*", default=None,
                        help="Explicit benchmark names (default: all 'to_evaluate' offline benchmarks)")
    args = parser.parse_args()

    if not args.api_base:
        raise SystemExit("No API base provided. Pass --api-base or set OPENAI_API_BASE in .env.")

    registry = get_registry()
    if args.benchmarks:
        benchmarks = [b for b in registry.list_all() if b.name in args.benchmarks]
    else:
        benchmarks = registry.get_by_tags(tags=["to_evaluate"], match_all=False)

    n_samples = 20 if args.smoke else args.n_samples
    workers = 1 if args.smoke else args.workers
    if args.load_full:
        n_samples = None  # evaluate the entire split
    load_samples = None if args.load_full else n_samples

    print(f"Endpoint : {args.api_base}")
    print(f"Model    : {args.model_id}")
    print(f"Samples  : {n_samples} (per benchmark) | Workers: {workers}")
    print(f"Benchmarks selected ({len([b for b in benchmarks if 'multistep' not in b.tags])} offline):")
    for b in benchmarks:
        if "multistep" not in b.tags:
            print(f"  - {b.name}")

    model_kwargs = dict(
        model_id=args.model_id,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=0,
    )
    # More retries: on-device / flaky endpoints intermittently drop connections;
    client_kwargs: dict = {"max_retries": args.max_retries}
    if args.api_timeout:
        client_kwargs["timeout"] = args.api_timeout
    if args.insecure:
        print("WARNING: skipping TLS certificate verification for the model endpoint")
        client_kwargs["http_client"] = httpx.Client(verify=False)
    model_kwargs["client_kwargs"] = client_kwargs
    try:
        model = SERVER_MODEL_CLS(**model_kwargs)
    except TypeError:
        if "client_kwargs" in model_kwargs:
            model_kwargs.pop("client_kwargs")
            model = SERVER_MODEL_CLS(**model_kwargs)
        else:
            raise

    if args.smoke:
        # test_mode selects `parallel_workers` samples; max_samples_to_test must not be set in test mode
        evaluation_config = EvaluationConfig(
            test_mode=True,
            parallel_workers=20,
            max_samples_to_test=None,
            run_name=args.run_name,
            image_resize_config=None if args.no_resize else ImageResizeConfig(),
        )
    else:
        evaluation_config = EvaluationConfig(
            test_mode=False,
            parallel_workers=workers,
            max_samples_to_test=n_samples,
            run_name=args.run_name,
            image_resize_config=None if args.no_resize else ImageResizeConfig(),
        )

    launch_test(model, benchmarks, evaluation_config, load_samples)


if __name__ == "__main__":
    main()
