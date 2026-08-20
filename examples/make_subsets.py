#!/usr/bin/env python
"""Build small (N-sample) random subsets of the ScreenSuite test splits and push them to HF.

Usage:
  uv run python examples/make_subsets.py --n 500 --seed 42 --owner NathanTrance

Creates/pushes three datasets (public):
  <owner>/screenqa-short-500
  <owner>/screenqa-complex-500
  <owner>/android-control-500

Subsets are drawn with a deterministic reservoir sample (uniform across the split,
seeded) from the FULL test split, keeping the exact same columns/schema so the
existing benchmarks work unchanged when pointed at them.
"""
import argparse
import random

from datasets import Dataset, Sequence, Value, load_dataset
from huggingface_hub import create_repo

BENCHMARKS = [
    # (name, source repo, source split, target suffix)
    ("screenqa_short", "rootsautomation/RICO-ScreenQA-Short", "test", "screenqa-short-500"),
    ("screenqa_complex", "rootsautomation/RICO-ScreenQA-Complex", "test", "screenqa-complex-500"),
    ("android_control", "smolagents/android-control", "test", "android-control-500"),
]


def reservoir_sample(stream, n: int, seed: int) -> list[dict]:
    """Uniform random n-sample from a stream of rows, deterministic per seed."""
    rng = random.Random(seed)
    reservoir: list[dict] = []
    for i, row in enumerate(stream):
        if i < n:
            reservoir.append(row)
        else:
            j = rng.randrange(i + 1)
            if j < n:
                reservoir[j] = row
    print(f"  scanned {i + 1} rows, kept {len(reservoir)}")
    return reservoir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--owner", default="NathanTrance")
    parser.add_argument("--only", choices=[b[0] for b in BENCHMARKS], default=None)
    args = parser.parse_args()

    for name, src_repo, split, target in BENCHMARKS:
        if args.only and name != args.only:
            continue
        print(f"=== {name}: {src_repo} ({split}) -> {args.owner}/{target} ===")
        stream = load_dataset(src_repo, split=split, streaming=True)
        rows = reservoir_sample(stream, args.n, args.seed)
        ds = Dataset.from_list(rows)
        # Large screenshot lists (android_control's screenshots_b64) blow past the 2GB
        # arrow list<string> limit when serialized; cast string lists to large_string.
        for col in ds.column_names:
            feat = ds.features[col]
            if isinstance(feat, Sequence) and isinstance(feat.feature, Value) and feat.feature.dtype == "string":
                print(f"  casting {col}: list<string> -> list<large_string>")
                ds = ds.cast_column(col, Sequence(Value("large_string")))
        create_repo(f"{args.owner}/{target}", repo_type="dataset", exist_ok=True)
        ds.push_to_hub(f"{args.owner}/{target}", private=False)
        print(f"  pushed {len(ds)} rows to {args.owner}/{target}")


if __name__ == "__main__":
    main()
