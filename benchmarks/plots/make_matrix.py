#!/usr/bin/env python
"""Write a synthetic computeMatrix (reference-point) file for plotting benchmarks.

Usage:
    python benchmarks/plots/make_matrix.py REGIONS SAMPLES BINS OUT.gz [--seed 0]

Presets used by run.py: small = 2000 4 200, mid = 5000 8 500,
large = 10911 10 880 (the size of the documented real benchmark). Values are
lognormal peak shapes plus gamma noise with 1% NaNs, so percentiles, geometric
means and confidence intervals behave like real coverage.
"""

import argparse
import gzip
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("regions", type=int)
    parser.add_argument("samples", type=int)
    parser.add_argument("bins", type=int)
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    bins, samples = args.bins, args.samples
    header = {
        "upstream": bins * 25, "downstream": bins * 25, "body": 0, "bin size": 50,
        "ref point": ["TSS"] * samples, "verbose": False, "bin avg type": "mean",
        "missing data as zero": False, "min threshold": None, "max threshold": None,
        "scale": 1, "skip zeros": False, "nan after end": False, "proc number": 1,
        "sort regions": "keep", "sort using": "mean",
        "unscaled 5 prime": 0, "unscaled 3 prime": 0,
        "group_labels": ["genes"], "group_boundaries": [0, args.regions],
        "sample_labels": [f"sample {i + 1} with a descriptive title" for i in range(samples)],
        "sample_boundaries": [i * bins for i in range(samples + 1)],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    shape = np.exp(-((np.arange(bins) - bins / 2) / (bins / 6)) ** 2)
    with gzip.open(args.out, "wt", compresslevel=1) as handle:
        handle.write("@" + json.dumps(header) + "\n")
        for start in range(0, args.regions, 500):
            count = min(500, args.regions - start)
            values = (rng.lognormal(0, 1, (count, samples, 1)) * shape
                      + rng.gamma(1, 0.3, (count, samples, bins))).reshape(count, -1)
            values[rng.random(values.shape) < 0.01] = np.nan
            for offset, row in enumerate(values):
                region = start + offset
                text = "\t".join("nan" if v != v else f"{v:.4f}" for v in row)
                handle.write(f"chr1\t{region * 10000}\t{region * 10000 + 1}\tg{region}\t.\t+\t{text}\n")


if __name__ == "__main__":
    main()
