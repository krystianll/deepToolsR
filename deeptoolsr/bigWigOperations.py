#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bigWigOperationsR -- fast operations on bigWig files.

Subcommands:
  info   print header info (like Kent bigWigInfo); optionally dump chrom.sizes.
  scale  multiply every value in a bigWig by a factor (e.g. -1 to flip a track).
  merge  combine multiple bigWigs (mean/sum).
"""

import argparse
import sys

from deeptoolsr.cli_errors import concise_cli_errors
from deeptoolsr.path_validation import atomic_output_path, validate_input_output_paths
from deeptoolsr.numeric_validation import (
    compression_level,
    float32_finite,
    nonnegative_int,
    positive_uint32,
)

from deeptoolsr import _bigwig, parserCommon, options as run_options


def _cmd_info(args):
    for path in args.bigwig:
        hdr = _bigwig.bigwig_info(path, not bool(args.chromSizes))
        chroms = hdr['chroms']
        if not args.chromSizes:
            print("file:          %s" % path)
            print("  version:       %d" % hdr["version"])
            print("  zoomLevels:    %d" % hdr["nLevels"])
            print("  chromCount:    %d" % len(chroms))
            print("  basesCovered:  %d" % hdr["nBasesCovered"])
            print("  mean:          %g" % hdr["mean"])
            print("  min:           %g" % hdr["minVal"])
            print("  max:           %g" % hdr["maxVal"])
            print("  std:           %g" % hdr["std"])
    if args.chromSizes:
        # Use the first bigWig's chrom list.
        chroms = _bigwig.bigwig_info(args.bigwig[0], False)['chroms']
        lines = ["%s\t%d" % (c, length) for c, length in chroms.items()]
        if args.chromSizes == "-":
            sys.stdout.write("\n".join(lines) + "\n")
        else:
            with atomic_output_path(args.chromSizes, suffix='.chromsizes.tmp') as temporary:
                with open(temporary, "w", newline="", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
            sys.stderr.write("Wrote %s\n" % args.chromSizes)


def _cmd_scale(args):
    with atomic_output_path(args.outFileName, suffix='.bigwig.tmp') as temporary:
        _bigwig.bigwig_scale(args.bigwig, temporary, args.scaleFactor,
                             max_zooms=args.zoomLevels,
                             compression_level=args.compressionLevel)
    sys.stderr.write("Wrote %s\n" % args.outFileName)


def _cmd_merge(args):
    if len(args.bigwig) < 1:
        sys.exit("merge needs at least one input bigWig.")
    threads = args.run_options.threads
    out_format = args.outFileFormat
    if out_format == "bedgraph" and args.outFileName.endswith(".gz"):
        out_format = "bedgraph.gz"
    with atomic_output_path(args.outFileName, suffix='.bigwig.tmp') as temporary:
        _bigwig.bigwig_merge(args.bigwig, temporary, out_format, args.operation,
                             args.binSize, args.scale, threads,
                             args.zoomLevels, args.compressionLevel)
    sys.stderr.write("Merged %d bigWigs -> %s\n" % (len(args.bigwig), args.outFileName))


def parse_arguments():
    p = argparse.ArgumentParser(prog="bigWigOperationsR",
                                description="Operations on bigWig files.")
    parserCommon.add_version(p, 'bigWigOperationsR')
    run_options.add_config_option(p)
    sub = p.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="Print bigWig header info / chrom.sizes.")
    run_options.add_config_option(info, default=argparse.SUPPRESS)
    info.add_argument("bigwig", nargs="+", help="Input bigWig file(s).")
    info.add_argument("--chromSizes", "-c", default=None, metavar="FILE",
                      help="Write chrom.sizes (name<TAB>length) to FILE ('-' = stdout) "
                           "instead of printing header info.")
    info.set_defaults(func=_cmd_info)

    scale = sub.add_parser("scale", help="Scale every value in a bigWig by a factor.")
    run_options.add_config_option(scale, default=argparse.SUPPRESS)
    scale.add_argument("--bigwig", "-b", required=True, help="Input bigWig.")
    scale.add_argument("--outFileName", "-o", required=True, help="Output bigWig.")
    scale.add_argument("--scaleFactor", "-s", type=float32_finite, required=True,
                       help="Multiply every value by this (e.g. -1 to flip).")
    scale.add_argument("--zoomLevels", type=nonnegative_int, default=10)
    scale.add_argument("--compressionLevel", type=compression_level, default=-1)
    scale.set_defaults(func=_cmd_scale)

    merge = sub.add_parser("merge", help="Merge multiple bigWigs (sum/mean).")
    run_options.add_config_option(merge, default=argparse.SUPPRESS)
    merge.add_argument("--bigwig", "-b", nargs="+", required=True,
                       help="Input bigWigs. Chroms are the union of inputs (a chrom "
                            "missing from a sample contributes 0); shared chroms must "
                            "agree on length.")
    merge.add_argument("--outFileName", "-o", required=True, help="Output.")
    merge.add_argument("--operation", choices=["sum", "mean"], default="sum",
                       help="Combine inputs by sum (default) or mean.")
    merge.add_argument("--binSize", "-bs", type=positive_uint32, default=1,
                       help="Output bin size (default 1 = per-base, exact).")
    merge.add_argument("--scale", type=float32_finite, default=1.0,
                       help="Scale the merged output.")
    run_options.add_processor_option(merge)
    merge.add_argument("--outFileFormat", "-of", choices=["bigwig", "bedgraph"],
                       default="bigwig",
                       help="bigwig (default); bedgraph, with .gz name = gzipped.")
    merge.add_argument("--zoomLevels", type=nonnegative_int, default=10)
    merge.add_argument("--compressionLevel", type=compression_level, default=-1)
    merge.set_defaults(func=_cmd_merge)
    return p


@concise_cli_errors("bigWigOperationsR")
def main(args=None):
    parser = parse_arguments()
    parsed = parser.parse_args(args)
    parsed.run_options = run_options.resolve_run_options(parsed)
    outputs = []
    if parsed.command == "info":
        outputs = [parsed.chromSizes]
    else:
        outputs = [parsed.outFileName]
    inputs = parsed.bigwig if isinstance(parsed.bigwig, list) else [parsed.bigwig]
    try:
        validate_input_output_paths(inputs, outputs)
    except ValueError as error:
        parser.error(str(error))
    parsed.func(parsed)


if __name__ == "__main__":
    main()
