#!/usr/bin/env python
"""Print a score table from ScreenSuite result files.

Usage:
  python examples/summarize_results.py                       # all runs in output/
  python examples/summarize_results.py --run Qwen3-VL-4B-Instruct_2025-01-01
  python examples/summarize_results.py --compare             # one column per run
"""
import argparse
import glob
import json
import os
import sys


def load_results(path):
    runs = {}
    for file in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        run_name = os.path.basename(file).replace(".jsonl", "")
        runs[run_name] = {}
        with open(file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "benchmark_name" in entry:
                        runs[run_name][entry["benchmark_name"]] = entry.get("metrics", {})
                except json.JSONDecodeError:
                    continue
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output"))
    parser.add_argument("--run", type=str, default=None, help="Only show this run")
    parser.add_argument("--compare", action="store_true", help="Table with one column per run")
    parser.add_argument("--metric", type=str, default=None, help="Pick one metric key instead of printing all")
    args = parser.parse_args()

    runs = load_results(args.results_dir)
    if args.run:
        runs = {args.run: runs[args.run]} if args.run in runs else sys.exit(f"Run '{args.run}' not found in {args.results_dir}")
    if not runs:
        sys.exit(f"No result files (*.jsonl) found in {args.results_dir}")

    bench_names = sorted({b for run in runs.values() for b in run})
    metric_keys = sorted({k for run in runs.values() for b in run.values() for k in b})

    if not args.compare:
        for run, benchmarks in runs.items():
            print(f"\n=== {run} ===")
            for bench in bench_names:
                if bench not in benchmarks:
                    print(f"  {bench:40s} (no results)")
                    continue
                metrics = benchmarks[bench]
                parts = ", ".join(f"{k}={v}" for k, v in metrics.items() if args.metric is None or k == args.metric)
                print(f"  {bench:40s} {parts}")
    else:
        metric = args.metric or (metric_keys[0] if metric_keys else None)
        if metric is None:
            sys.exit("No metrics found in results")
        print(f"metric: {metric}")
        header = f"{'benchmark':40s}" + "".join(f"{r[:22]:24s}" for r in runs)
        print(header)
        for bench in bench_names:
            row = f"{bench:40s}"
            for run in runs:
                m = runs[run].get(bench, {})
                row += f"{str(m.get(metric, '-')):24s}"
            print(row)


if __name__ == "__main__":
    main()
