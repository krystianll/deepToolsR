#!/usr/bin/env python
"""Benchmark bamCoverageR (native) against stock bamCoverage, if installed.

Usage:
    python benchmarks/bamcoverage.py INPUT.bam [--binSize 50] [--threads 8] [--repeat 3]

Meaningful numbers need a real (large, indexed) BAM; the bundled test BAMs are
tiny and only exercise correctness, not throughput. Times the native backend at
1 and N threads and, if a `bamCoverage` executable is on PATH, the stock tool.
"""

import argparse
import shutil
import statistics
import subprocess
import tempfile
import time


def _time(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bam")
    ap.add_argument("--binSize", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    from deeptoolsr import _coverage
    tmp = tempfile.mkdtemp()

    def native(threads):
        return lambda: _coverage.bam_coverage_bigwig(
            args.bam, tmp + "/n.bw", bin_size=args.binSize,
            aggregation="mean", filter_mode="deeptools", threads=threads)

    print(f"BAM: {args.bam}  binSize={args.binSize}  repeat={args.repeat}\n")
    lo, med = _time(native(1), args.repeat)
    print(f"native  1 thread   : {med:.3f}s (best {lo:.3f}s)")
    base = med
    lo, med = _time(native(args.threads), args.repeat)
    print(f"native  {args.threads:<2} threads  : {med:.3f}s (best {lo:.3f}s)  "
          f"speedup x{base / med:.1f}")

    stock = shutil.which("bamCoverage")
    if stock:
        def run_stock():
            subprocess.run([stock, "-b", args.bam, "-o", tmp + "/s.bw",
                            "-bs", str(args.binSize), "--normalizeUsing", "None",
                            "-p", str(args.threads)],
                           check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        lo, med = _time(run_stock, args.repeat)
        print(f"stock   {args.threads:<2} threads  : {med:.3f}s (best {lo:.3f}s)  "
              f"native is x{med / base:.1f} vs stock-1-thread-equivalent")
    else:
        print("stock bamCoverage not on PATH; skipping comparison.")


if __name__ == "__main__":
    main()
