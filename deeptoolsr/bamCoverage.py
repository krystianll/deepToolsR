#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bamCoverageR -- fast native BAM -> bigWig coverage.

A multicore htslib/libBigWig backend (see BAMCOVERAGE_ROADMAP.md). Coverage is
provided solely by the native ``_coverage`` extension; there is no pure-Python
fallback for this tool.
"""

import argparse
from contextlib import ExitStack
import os
import sys

from deeptoolsr import _coverage, parserCommon, options as run_options
from deeptoolsr.cli_errors import concise_cli_errors
from deeptoolsr.path_validation import atomic_output_path, validate_input_output_paths
from deeptoolsr.numeric_validation import (
    CompatibilityRule,
    compression_level,
    extend_reads as extend_reads_value,
    float32_finite,
    min_max_error,
    nonnegative_int,
    positive_int,
    positive_uint32,
    option_was_supplied,
    specified_options,
    uint16_int,
    uint8_int,
    validate_compatibility_rules,
)


# --- helpers ---------------------------------------------------------------

def _insert_suffix(path, suffix):
    """Insert ``suffix`` before the extension (``a.bw`` + ``.plus`` -> ``a.plus.bw``;
    ``a.bedgraph.gz`` -> ``a.plus.bedgraph.gz``)."""
    gz = path.endswith(".gz")
    base = path[:-3] if gz else path
    root, ext = os.path.splitext(base)
    return root + suffix + ext + (".gz" if gz else "")


def _resolve_bam_chromosome(chrom, chroms):
    """Resolve stock-deepTools chr/no-chr aliases, preferring an exact name."""
    if chrom in chroms:
        return chrom
    if chrom.startswith("chr"):
        alternate = chrom[3:]
        if alternate == "M":
            alternate = "MT"
    else:
        alternate = "chrM" if chrom == "MT" else "chr" + chrom
    return alternate if alternate in chroms else None


def _warn_unresolved_chromosomes(source, names):
    names = sorted(set(names))
    if names:
        preview = ", ".join(names[:8])
        if len(names) > 8:
            preview += " (+{} more)".format(len(names) - 8)
        sys.stderr.write(
            "WARNING: omitted unresolved chromosome name{} from {}: {}\n".format(
                "s" if len(names) != 1 else "", source,
                preview))


def _parse_blacklist(path, chroms):
    """Read a BED into ``{chrom: [(start, end, strand), ...]}`` (strand +1/-1/0)."""
    regions = {}
    unresolved = []
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t") if "\t" in line else line.split()
            if len(f) < 3:
                continue
            original_chrom = f[0]
            chrom = _resolve_bam_chromosome(original_chrom, chroms)
            if chrom is None:
                unresolved.append(original_chrom)
                continue
            strand = {"+": 1, "-": -1}.get(f[5].strip(), 0) if len(f) >= 6 else 0
            regions.setdefault(chrom, []).append((int(f[1]), int(f[2]), strand))
    _warn_unresolved_chromosomes(path, unresolved)
    return regions


def _estimate_fragment_length(bam_path, cov, sample=10000):
    """Median fragment length: proper-pair |TLEN| if paired, else median read
    length. Delegates to the native coverage backend (`cov`) so no pysam (and
    thus no htslib) is required."""
    return cov.estimate_fragment_length(bam_path, sample)


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _subtract_intervals(intervals, excluded):
    """Subtract sorted, merged half-open intervals from another merged set."""
    result = []
    j = 0
    for start, end in intervals:
        cursor = start
        while j < len(excluded) and excluded[j][1] <= cursor:
            j += 1
        k = j
        while k < len(excluded) and excluded[k][0] < end:
            ex_start, ex_end = excluded[k]
            if ex_start > cursor:
                result.append((cursor, min(ex_start, end)))
            cursor = max(cursor, ex_end)
            if cursor >= end:
                break
            k += 1
        if cursor < end:
            result.append((cursor, end))
    return result


def _parse_probe_bed(path, chroms=None):
    """Read a stranded BED (>=6 columns) into [(chrom, start, end, +1/-1), ...]."""
    by_chromosome = {}
    unresolved = []
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t") if "\t" in line else line.split()
            if len(f) < 6:
                continue
            strand = {"+": 1, "-": -1}.get(f[5].strip())
            if strand is None:
                continue
            chrom = f[0]
            if chroms is not None:
                original_chrom = chrom
                chrom = _resolve_bam_chromosome(original_chrom, chroms)
                if chrom is None:
                    unresolved.append(original_chrom)
                    continue
            start, end = int(f[1]), int(f[2])
            if start < 0 or end <= start:
                continue
            by_chromosome.setdefault(chrom, {1: [], -1: []})[strand].append(
                (start, end))

    _warn_unresolved_chromosomes(path, unresolved)
    # Duplicate/overlapping same-strand genes must not multiply the evidence.
    # Bases annotated on both strands are ambiguous and contribute to neither.
    regions = []
    for chrom, strands in by_chromosome.items():
        plus = _merge_intervals(strands[1])
        minus = _merge_intervals(strands[-1])
        regions.extend((chrom, start, end, 1)
                       for start, end in _subtract_intervals(plus, minus))
        regions.extend((chrom, start, end, -1)
                       for start, end in _subtract_intervals(minus, plus))
    regions.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return regions


def _infer_strandedness(bam, bed, min_mapq, coverage_module,
                        ignore_duplicates=False, sam_flag_include=0,
                        sam_flag_exclude=0):
    """Infer library strandedness from known-strand probe genes (RSeQC-style)."""
    chroms = {target["name"]
              for target in coverage_module.bam_index_stats(bam)["targets"]}
    regions = _parse_probe_bed(bed, chroms)
    if not regions:
        sys.exit("--strandedness file=%s has no stranded (>=6-column) entries." % bed)
    r = coverage_module.infer_strandedness(
        bam, regions, min_mapq, True, ignore_duplicates,
        sam_flag_include, sam_flag_exclude)
    frac, total = r["forward_fraction"], r["total"]
    if total < 200:
        sys.stderr.write("WARNING: only %d independent probe observations; "
                         "strandedness call may be "
                         "unreliable.\n" % total)
    if total == 0:
        call = "none"
    elif frac >= 0.8:
        call = "forward"
    elif frac <= 0.2:
        call = "reverse"
    else:
        call = "none"
    sys.stderr.write("Inferred strandedness: %s (forward-fraction %.3f over %d "
                     "sampled independent observations in %d non-ambiguous probe "
                     "intervals).\n" % (call, frac, total, r.get("sampled_probes", len(regions))))
    return call


_STRANDEDNESS = {
    "none": "none", "unstranded": "none",
    "forward": "forward", "second": "forward", "fr": "forward",
    "reverse": "reverse", "first": "reverse", "rf": "reverse",
}
_COVERAGE_MODES = {
    "coverage-mean": ("mean", "none"),
    "coverage-sum": ("sum", "none"),
    "read-count": ("count", "none"),
    "cpm": ("count", "cpm"),
    "rpkm": ("count", "rpkm"),
    "bpm": ("count", "bpm"),
}
_COLLAPSE = {"none": "none", "5prime": "5prime", "5": "5prime",
             "3prime": "3prime", "3": "3prime", "center": "center"}


def _normalization_metric(value):
    """Validate case-insensitively while retaining the user's spelling."""
    if value.lower() not in _COVERAGE_MODES:
        raise argparse.ArgumentTypeError(
            "choose coverage-mean, coverage-sum, read-count, CPM, RPKM, or BPM")
    return value


def parse_arguments():
    p = argparse.ArgumentParser(
        prog="bamCoverageR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Native (htslib) BAM -> bigWig coverage.")
    parserCommon.add_version(p, 'bamCoverageR')
    run_options.add_config_option(p)

    req = p.add_argument_group("Required")
    req.add_argument("--bam", "-b", required=True, help="Coordinate-sorted, indexed BAM.")
    req.add_argument("--outFileName", "-o", required=True, help="Output bigWig.")

    cov = p.add_argument_group("Coverage")
    cov.add_argument("--binSize", "-bs", type=positive_uint32, default=50,
                     help="Bin size in bp (default 50).")
    run_options.add_processor_option(cov)
    cov.add_argument("--outFileFormat", "-of", choices=["bigwig", "bedgraph"],
                     default="bigwig",
                     help="Output format (default bigwig). For bedgraph, a .gz output "
                          "filename produces gzipped bedGraph.")
    cov.add_argument("--compressionLevel", type=compression_level, default=-1,
                     help="libdeflate level for bigWig blocks: 1 (fast) .. 12 (small, "
                          "slow); -1 = default (6). Lower speeds up the writer; "
                          "decompression/analysis is unaffected.")
    cov.add_argument("--zoomLevels", type=nonnegative_int, default=10,
                     help="bigWig zoom levels for genome-browser overviews (default "
                          "10). Set 0 for analysis-only tracks (e.g. fed to "
                          "computeMatrix): much faster to write and far less memory at "
                          "small bin sizes, since zoom construction re-reads the whole "
                          "track. Browsers show no zoomed-out overview with 0.")

    shape = p.add_argument_group("Read model")
    shape.add_argument("--extendReads", "-e", nargs="?", const=-1,
                       type=extend_reads_value, default=0,
                       help="Extend to the whole fragment. Paired: real mate span "
                            "(counted once by depth metrics; each eligible mate is "
                            "counted by read-count metrics); single-end: extend to this length, or the "
                            "inferred median with no value.")
    shape.add_argument("--collapse", choices=list(_COLLAPSE), default="none",
                       help="Collapse each fragment to its 5'/3' end or centre "
                            "(strand/strandedness-aware), clamped to the fragment.")
    shape.add_argument("--collapsedLength", type=positive_int, default=1,
                       help="Width of the collapsed feature (default 1; only with --collapse).")

    filt = p.add_argument_group("Filters")
    filt.add_argument("--minMappingQuality", type=uint8_int, default=0)
    filt.add_argument("--samFlagInclude", type=uint16_int, default=0)
    filt.add_argument("--samFlagExclude", type=uint16_int, default=0)
    filt.add_argument("--ignoreDuplicates", action="store_true",
                      help="Drop reads with the duplicate FLAG (honours upstream markdup).")
    filt.add_argument("--minFragmentLength", type=nonnegative_int, default=0,
                      help="Minimum |TLEN|, inclusive. With TLEN zero, use aligned "
                           "CIGAR length including deletions but excluding skipped introns.")
    filt.add_argument("--maxFragmentLength", type=nonnegative_int, default=0,
                      help="Maximum length by the same rule as --minFragmentLength; "
                           "0 means no upper limit.")
    filt.add_argument("--filterMode", choices=["primary", "deeptools"], default="primary",
                      help="'primary' (default) excludes secondary+supplementary; "
                           "'deeptools' skips only unmapped (stock behaviour).")
    filt.add_argument("--blackListFileName", "-bl", default=None,
                      help="BED of regions to zero out (coverage-mask).")
    filt.add_argument("--whiteListFileName", "-wl", default=None,
                      help="BED of regions to keep; coverage elsewhere is zeroed "
                           "(coverage-mask).")
    filt.add_argument("--featureTouchStrand", choices=["ignore", "sense", "antisense"],
                      default="ignore",
                      help="With --filterByOverlap, orient the touch test by strand "
                           "(needs --strandedness): 'sense' = a fragment touches a "
                           "feature only if on the same strand (e.g. a + intron is only "
                           "touched by + fragments, removing antisense contamination); "
                           "'antisense' = opposite; 'ignore' = positional only (default).")
    filt.add_argument("--filterByOverlap", action="store_true",
                      help="Filter whole reads/pairs by whether they *touch* the region "
                           "set (splice-aware, on aligned blocks), instead of masking "
                           "bins: keep only fragments touching --whiteListFileName and/or "
                           "drop fragments touching --blackListFileName. Pair-aware "
                           "(decision on the union of eligible mates after all "
                           "per-alignment filters; a rejected mate neither drives "
                           "the decision nor gains coverage). E.g. intron BED + "
                           "--whiteListFileName --filterByOverlap = nascent-RNA coverage.")
    filt.add_argument("--filterByOverlapHalo", type=nonnegative_int, default=None,
                      help="Optional extra query padding in bases (default: automatic "
                           "10 kb mate prefetch). The effective prefetch is always "
                           "at least --maxFragmentLength when that limit is set. "
                           "Pair-aware filtering is exact without padding; this "
                           "legacy tuning option does not change the result.")

    strand = p.add_argument_group("Strand")
    strand.add_argument("--strandedness", default="none",
                        help="Library type: none | forward (=second/fr) | reverse "
                             "(=first/rf, dUTP; Illumina TruSeq Stranded default). "
                             "forward: read1 is sense (featureCounts -s1, salmon ISF, "
                             "HISAT FR). reverse: read2 is sense (featureCounts -s2, "
                             "salmon ISR, HISAT RF). file=BED infers the library "
                             "type from user-supplied stranded genes.")
    strand.add_argument("--filterRNAstrand", choices=["none", "forward", "reverse", "split"],
                        default="none",
                        help="Output the forward/reverse transcript strand, or 'split' "
                             "to write both in one pass (needs --strandedness).")
    strand.add_argument("--suffix", nargs=2, default=[".plus", ".minus"],
                        metavar=("PLUS", "MINUS"),
                        help="Suffixes for the two --filterRNAstrand split outputs "
                             "(default .plus .minus).")

    norm = p.add_argument_group("Normalization")
    norm.add_argument(
        "--normalizeUsing", default="coverage-mean", type=_normalization_metric,
        help="Output metric: coverage-mean (default; mean per-base depth), "
             "coverage-sum (depth-bases per bin), read-count (reads overlapping "
             "each bin), CPM, RPKM, or BPM. Normalized metrics always use "
             "read counts; each individually eligible alignment contributes once, "
             "including each retained mate of a proper pair.")
    norm.add_argument("--normalizationDenominator", choices=["library", "filtered"],
                      default="library",
                      help="For CPM and RPKM, 'library' (default) uses library size "
                           "from the BAM index, "
                           "independent of filters and shared across strand outputs "
                           "(the correct, comparable normalization). 'filtered': "
                           "post-filter read count (legacy, strand-dependent). "
                           "BPM instead uses its pooled placed-bin sum.")
    norm.add_argument("--exactScaling", action="store_true",
                      help="For CPM and RPKM with the library denominator, count the "
                           "library exactly (primary reads) instead of using index stats.")
    norm.add_argument("--scaleFactor", type=float32_finite, default=1.0)
    norm.add_argument("--scaleFactorMinus", type=float32_finite, default=1.0,
                      help="With --filterRNAstrand split, apply this additional scale "
                           "factor to the minus (reverse) track on top of --scaleFactor "
                           "(e.g. -1 to flip it below the axis; default 1).")
    norm.add_argument("--ignoreForNormalization", "-ignore", nargs="+", default=None,
                      metavar="CHROM", help="For CPM, RPKM, and BPM, chromosomes to "
                      "exclude from the denominator.")
    return p


@concise_cli_errors("bamCoverageR")
def main(args=None):
    parser = parse_arguments()
    raw_args = list(args) if args is not None else sys.argv[1:]
    args = parser.parse_args(raw_args)
    supplied = specified_options(raw_args)

    validate_compatibility_rules(parser, args, supplied, [
        CompatibilityRule(
            lambda ns, _opts: ns.extendReads != 0 and ns.collapse != 'none',
            '--extendReads and --collapse cannot be used together'),
        CompatibilityRule(
            lambda ns, opts: ns.collapse == 'none' and
            option_was_supplied(opts, '--collapsedLength'),
            '--collapsedLength is unused without --collapse', 'warning'),
        CompatibilityRule(
            lambda ns, opts: ns.filterRNAstrand != 'split' and
            option_was_supplied(opts, '--scaleFactorMinus'),
            '--scaleFactorMinus is unused unless --filterRNAstrand split',
            'warning'),
        CompatibilityRule(
            lambda ns, opts: ns.filterRNAstrand != 'split' and
            option_was_supplied(opts, '--suffix'),
            '--suffix is unused unless --filterRNAstrand split', 'warning'),
        CompatibilityRule(
            lambda ns, opts: ns.normalizeUsing.lower() not in ('cpm', 'rpkm') and
            option_was_supplied(opts, '--normalizationDenominator'),
            '--normalizationDenominator is unused unless --normalizeUsing is CPM or RPKM',
            'warning'),
        CompatibilityRule(
            lambda ns, opts: ns.normalizeUsing.lower() not in ('cpm', 'rpkm') and
            option_was_supplied(opts, '--exactScaling'),
            '--exactScaling is unused unless --normalizeUsing is CPM or RPKM',
            'warning'),
        CompatibilityRule(
            lambda ns, opts: ns.normalizeUsing.lower() in ('cpm', 'rpkm') and
            ns.normalizationDenominator == 'filtered' and
            option_was_supplied(opts, '--exactScaling'),
            '--exactScaling is redundant with --normalizationDenominator filtered',
            'warning'),
        CompatibilityRule(
            lambda ns, opts: ns.normalizeUsing.lower() not in ('cpm', 'rpkm', 'bpm') and
            option_was_supplied(opts, '--ignoreForNormalization', '-ignore'),
            '--ignoreForNormalization is unused for unnormalized coverage metrics',
            'warning'),
    ])

    range_error = min_max_error(
        args.minFragmentLength,
        args.maxFragmentLength if args.maxFragmentLength > 0 else None,
        "--minFragmentLength", "--maxFragmentLength")
    if range_error:
        parser.error(range_error)

    # Resolve every output before reading data or loading the native backend.
    out_forward = args.outFileName
    out_reverse = ""
    if args.filterRNAstrand == "split":
        if any(not suffix.strip() for suffix in args.suffix):
            parser.error("--suffix values must not be empty")
        out_forward = _insert_suffix(args.outFileName, args.suffix[0])
        out_reverse = _insert_suffix(args.outFileName, args.suffix[1])
    annotation_inputs = [args.blackListFileName, args.whiteListFileName,
                         _coverage.resolve_bam_index_path(args.bam)]
    if args.strandedness.lower().startswith("file="):
        annotation_inputs.append(args.strandedness[5:])
    try:
        validate_input_output_paths([args.bam] + annotation_inputs,
                                    [out_forward, out_reverse])
    except ValueError as error:
        parser.error(str(error))

    # --- translate options ---
    if args.strandedness.lower().startswith("file="):
        strandedness = _infer_strandedness(args.bam, args.strandedness[5:],
                                           args.minMappingQuality, _coverage,
                                           args.ignoreDuplicates,
                                           args.samFlagInclude,
                                           args.samFlagExclude)
    else:
        strandedness = _STRANDEDNESS.get(args.strandedness.lower())
    if strandedness is None:
        sys.exit("--strandedness must be none|forward|reverse|file=BED "
                 "(aliases first/second/fr/rf).")

    coverage_mode = _COVERAGE_MODES.get(args.normalizeUsing.lower())
    if coverage_mode is None:  # parser validation is repeated for programmatic safety
        sys.exit("--normalizeUsing must be coverage-mean|coverage-sum|read-count|"
                 "CPM|RPKM|BPM.")
    aggregation, normalization = coverage_mode

    if args.filterRNAstrand in ("forward", "reverse", "split") and strandedness == "none":
        sys.exit("--filterRNAstrand %s needs --strandedness forward|reverse."
                 % args.filterRNAstrand)

    if args.filterByOverlap and not (args.blackListFileName or args.whiteListFileName):
        sys.exit("--filterByOverlap needs --whiteListFileName and/or --blackListFileName.")
    if args.filterByOverlapHalo is not None and not args.filterByOverlap:
        sys.exit("--filterByOverlapHalo requires --filterByOverlap.")
    if args.featureTouchStrand != "ignore" and not args.filterByOverlap:
        sys.exit("--featureTouchStrand requires --filterByOverlap.")
    if args.featureTouchStrand != "ignore" and strandedness == "none":
        sys.exit("--featureTouchStrand needs --strandedness forward|reverse.")

    threads = run_options.resolve_run_options(args).threads

    # extendReads: -1 sentinel (flag with no value) -> infer median fragment length.
    extend_reads = args.extendReads
    if extend_reads == -1:
        extend_reads = int(_estimate_fragment_length(args.bam, _coverage))

    blacklist = whitelist = None
    resolved_ignore = args.ignoreForNormalization
    if (args.blackListFileName or args.whiteListFileName or
            args.ignoreForNormalization):
        chroms = {t["name"] for t in _coverage.bam_index_stats(args.bam)["targets"]}
        if args.blackListFileName:
            blacklist = _parse_blacklist(args.blackListFileName, chroms)
        if args.whiteListFileName:
            whitelist = _parse_blacklist(args.whiteListFileName, chroms)
        if args.ignoreForNormalization:
            resolved_ignore = []
            unresolved = []
            for name in args.ignoreForNormalization:
                resolved = _resolve_bam_chromosome(name, chroms)
                if resolved is None:
                    unresolved.append(name)
                elif resolved not in resolved_ignore:
                    resolved_ignore.append(resolved)
            _warn_unresolved_chromosomes(
                '--ignoreForNormalization', unresolved)

    out_format = args.outFileFormat
    if out_format == "bedgraph" and args.outFileName.endswith(".gz"):
        out_format = "bedgraph.gz"

    # Produce every split artifact successfully before replacing any destination.
    with ExitStack() as outputs:
        temp_forward = outputs.enter_context(
            atomic_output_path(out_forward, suffix='.coverage.tmp'))
        temp_reverse = ''
        if args.filterRNAstrand == "split":
            temp_reverse = outputs.enter_context(
                atomic_output_path(out_reverse, suffix='.coverage.tmp'))
        _coverage.bam_coverage_bigwig(
            args.bam, temp_forward, temp_reverse, out_format,
            bin_size=args.binSize,
            aggregation=aggregation,
            min_mapping_quality=args.minMappingQuality,
            sam_flag_include=args.samFlagInclude,
            sam_flag_exclude=args.samFlagExclude,
            filter_mode=args.filterMode,
            ignore_duplicates=args.ignoreDuplicates,
            min_fragment_length=args.minFragmentLength,
            max_fragment_length=args.maxFragmentLength,
            extend_reads=extend_reads,
            strandedness=strandedness,
            filter_rna_strand=args.filterRNAstrand,
            collapse=_COLLAPSE[args.collapse],
            collapsed_length=args.collapsedLength,
            blacklist=blacklist,
            whitelist=whitelist,
            threads=threads,
            normalization=normalization,
            normalization_denominator=args.normalizationDenominator,
            exact_scaling=args.exactScaling,
            scale_factor=args.scaleFactor,
            scale_factor_minus=args.scaleFactorMinus,
            ignore_for_normalization=resolved_ignore,
            max_zooms=args.zoomLevels,
            compression_level=args.compressionLevel,
            filter_by_overlap=args.filterByOverlap,
            feature_touch_strand=args.featureTouchStrand,
            filter_by_overlap_halo=(-1 if args.filterByOverlapHalo is None
                                    else args.filterByOverlapHalo))

    if args.filterRNAstrand == "split":
        sys.stderr.write("Wrote %s and %s\n" % (out_forward, out_reverse))
    else:
        sys.stderr.write("Wrote %s\n" % out_forward)


if __name__ == "__main__":
    main()
