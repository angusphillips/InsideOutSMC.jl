#!/usr/bin/env python3
"""Summarise sPCE / NMC bounds across all *_over_training_seeds.csv result files.

Each such CSV has one row per training seed: `seed,mean_bound,std`, where
mean_bound is that seed's reported bound (averaged over the estimator's MC runs)
and std is the within-seed MC std. This script aggregates the *reported bounds*
(column 2) across seeds, reporting the mean and standard error
(SE = sample_std / sqrt(n_seeds)).

Usage:
  scripts/summarize_bounds.py                 # scan experiments/**/data
  scripts/summarize_bounds.py path [path...]  # scan given dirs/files/globs
  scripts/summarize_bounds.py --csv out.csv   # also write a tidy summary CSV
"""
import csv
import glob
import math
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUFFIXES = ("_spce_over_training_seeds.csv", "_nmc_over_training_seeds.csv")


def find_files(args):
    files = set()
    if not args:
        args = [os.path.join(REPO_ROOT, "experiments")]
    for a in args:
        if os.path.isdir(a):
            for suf in SUFFIXES:
                files.update(glob.glob(os.path.join(a, "**", "*" + suf), recursive=True))
        elif os.path.isfile(a):
            files.add(a)
        else:  # treat as a glob
            files.update(glob.glob(a, recursive=True))
    return sorted(f for f in files if f.endswith(SUFFIXES))


def read_bounds(path):
    """Return the list of per-seed reported bounds (column 2)."""
    bounds = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            row = [c for c in row if c.strip() != ""]
            if len(row) < 2:
                continue
            try:
                # skip a header row if present
                float(row[0])
            except ValueError:
                continue
            bounds.append(float(row[1]))
    return bounds


def classify(path):
    rel = os.path.relpath(path, REPO_ROOT)
    kind = "sPCE" if path.endswith("_spce_over_training_seeds.csv") else "NMC"
    # label = directory holding the csv, relative to repo (identifies exp + tag)
    label = os.path.dirname(rel)
    return label, kind


def main():
    args = sys.argv[1:]
    out_csv = None
    if "--csv" in args:
        i = args.index("--csv")
        out_csv = args[i + 1]
        del args[i : i + 2]

    files = find_files(args)
    if not files:
        print("No *_over_training_seeds.csv files found.", file=sys.stderr)
        return 1

    rows = []
    for path in files:
        bounds = read_bounds(path)
        n = len(bounds)
        if n == 0:
            continue
        mean = statistics.fmean(bounds)
        sd = statistics.stdev(bounds) if n > 1 else float("nan")
        se = sd / math.sqrt(n) if n > 1 else float("nan")
        label, kind = classify(path)
        rows.append((label, kind, n, mean, se))

    rows.sort(key=lambda r: (r[0], r[1]))
    w = max((len(r[0]) for r in rows), default=20)
    print(f"{'result':<{w}}  {'bound':<5}  {'n':>2}  {'mean':>10}  {'stderr':>9}")
    print("-" * (w + 34))
    for label, kind, n, mean, se in rows:
        se_s = f"{se:9.4f}" if se == se else "      n/a"  # nan check
        print(f"{label:<{w}}  {kind:<5}  {n:>2}  {mean:>10.4f}  {se_s}")

    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["result", "bound", "n_seeds", "mean", "stderr"])
            for r in rows:
                writer.writerow(r)
        print(f"\nWrote summary to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
