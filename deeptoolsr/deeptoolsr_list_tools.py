#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import sys

from deeptoolsr import config, options as run_options, _statistics
from deeptoolsr.parserCommon import add_version
from deeptoolsr.numeric_validation import bounded_int


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        prog='deeptoolsr',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="\n".join([
            "deepToolsR is a C++-accelerated toolkit for analysing and plotting high-throughput sequencing data, such as ChIP-seq, RNA-seq or MNase-seq.",
            "It is based on deepTools 3.5.6: its commands carry an R suffix and can be installed beside the originals.",
            "",
            "Each tool should be called by its own name as in the following example:",
            "",
            "plotHeatmapR -m matrix.gz -o image.png",
            "",
            "deepToolsR builds on deepTools; if you use it in your research, please cite deepTools:",
            "",
            "Ramírez, Fidel, Devon P. Ryan, Björn Grüning, Vivek Bhardwaj, Fabian Kilpert, Andreas S. Richter, Steffen Heyne, Friederike Dündar, and Thomas Manke. 2016. \"deepTools2: A next Generation Web Server for Deep-Sequencing Data Analysis.\" Nucleic Acids Research, April. doi:10.1093/nar/gkw257.",
            "",
            "",
            "[ Tools for BAM and bigWig file processing ]",
            # "    multiBamSummaryR         - compute read coverages over bam files. Output used for plotCorrelation or plotPCA",
            # "    multiBigwigSummaryR      - extract scores from bigwig files. Output used for plotCorrelation or plotPCA",
            # "    correctGCBiasR           - corrects GC bias from bam file. Don't use it with ChIP data",
            "    bamCoverageR             - fast native (htslib) BAM -> bigWig coverage, multicore",
            "    bigWigOperationsR        - operations on bigWig files (info, scale, merge)",
            # "    bamCompareR              - computes log2 ratio and other operations of read coverage of two samples per bins or regions",
            # "    bigwigCompareR           - computes log2 ratio and other operations from bigwig scores of two samples per bins or regions",
            # "    bigwigAverageR           - computes average from bigwig scores of multiple samples per bins or regions",
            "    computeMatrixR           - prepares the data from bigwig scores for plotting with plotHeatmap(R) or plotProfile(R)",
            # "    alignmentSieveR          - filters BAM alignments according to specified parameters, optionally producing a BEDPE file",
            "",
            # "[ Tools for QC ]",
            # "    plotCorrelationR         - plots heatmaps or scatterplots of data correlation",
            # "    plotPCAR                 - plots PCA",
            # "    plotFingerprintR         - plots the distribution of enriched regions",
            # "    bamPEFragmentSizeR       - returns the read length and paired-end distance from a bam file",
            # "    computeGCBiasR           - computes and plots the GC bias of a sample",
            # "    plotCoverageR            - plots a histogram of read coverage",
            # "    estimateReadFiltering    - estimates the number of reads that will be filtered from a BAM file or files given certain criteria",
            # "",
            "[Heatmaps and summary plots]",
            "    plotHeatmapR             - plots one or multiple heatmaps of user selected regions over different genomic scores",
            "    plotProfileR             - plots the average profile of user selected regions over different genomic scores",
            "    plotMatrixR              - plots profiles, heatmaps, or both from a matrix",
            # "    plotEnrichmentR          - plots the read/fragment coverage of one or more sets of regions",
            "",
            "[Miscellaneous]",
            "    computeMatrixOperationsR - modifies the output of computeMatrix in a variety of ways.",
            "",
            "For more information visit: http://deeptools.readthedocs.org",
        ]))

    add_version(parser, 'deeptoolsr')
    run_options.add_config_option(parser)

    subparsers = parser.add_subparsers(dest='command')
    options = subparsers.add_parser(
        'options', help='Show or change persistent deepToolsR options.')
    options.add_argument(
        '--rasterFilter',
        choices=['triangle', 'mitchell', 'catmullrom', 'cubic', 'box', 'point'],
        help='Resize filter for the native heatmap rasteriser.')
    options.add_argument(
        '--rasterBitDepth', type=int, choices=[8, 16],
        help='Intermediate precision for the native heatmap rasteriser.')
    run_options.add_config_option(options, default=argparse.SUPPRESS)
    run_options.add_processor_option(options, default=None)
    options.add_argument(
        '--fontFamily', nargs='+', metavar='NAME',
        help='Font family for all plot text, with optional fallbacks '
             '(e.g. --fontFamily Arial Helvetica sans-serif). '
             'Pass "default" to restore the default. Individual font roles and '
             'label-layout policies and gaps remain editable only in options.txt.')
    options.add_argument(
        '--fontMultiplier', metavar='FLOAT',
        help='Scale factor applied to every font size at draw time (base sizes '
             'on disk are unchanged). Pass "default" to restore 1.0.')

    describe = subparsers.add_parser(
        'describe', help='Describe plot groups, series and colour slots as JSON.')
    describe.add_argument('tool', choices=['plotHeatmapR', 'plotProfileR', 'plotMatrixR'])
    tool_args = describe.add_argument('tool_args', nargs=argparse.REMAINDER)
    # Python < 3.12 marks REMAINDER positionals required; make it explicit so
    # behaviour and the CLI contract snapshot match on every version.
    tool_args.required = False

    serve = subparsers.add_parser(
        'serve', help='Run a long-lived JSON-lines plotting worker.')
    run_options.add_config_option(serve, default=argparse.SUPPRESS)
    run_options.add_processor_option(serve, default=None)
    serve.add_argument('--cacheBytes', type=bounded_int(minimum=0),
                       metavar='BYTES', default=None,
                       help='Maximum bytes for the session caches.')

    return parser


def _process_options(args):
    path = run_options.config_path(args.config)
    options = config.load_options(path)
    if args.rasterFilter is not None:
        options['raster_filter'] = args.rasterFilter
    if args.rasterBitDepth is not None:
        options['raster_bit_depth'] = args.rasterBitDepth
    if args.numberOfProcessors is not None:
        options['number_of_processors'] = args.numberOfProcessors

    font_family = None
    if args.fontFamily is not None:
        if len(args.fontFamily) == 1 and args.fontFamily[0].lower() == 'default':
            font_family = list(config.default_font_family())
        else:
            font_family = args.fontFamily
    font_multiplier = None
    if args.fontMultiplier is not None:
        if args.fontMultiplier.lower() == 'default':
            font_multiplier = config.default_font_multiplier()
        else:
            try:
                font_multiplier = float(args.fontMultiplier)
            except ValueError:
                raise SystemExit(
                    'Font multiplier must be a positive number or "default".')
            if not (math.isfinite(font_multiplier) and font_multiplier > 0):
                raise SystemExit(
                    'Font multiplier must be a positive number or "default".')
    # deeptoolsr options is the sole writer: it always rewrites the full
    # expanded schema (defaults overlaid with the user's file and any
    # requested changes) so the on-disk options.txt is self-documenting.
    persisted = config.build_persisted_options(
        options, font_family=font_family, font_multiplier=font_multiplier,
        path=path)
    path = config.save_options(persisted, path)
    print('Saved options to {}'.format(path))
    print('Heatmap raster filter: {}'.format(options['raster_filter']))
    print('Heatmap raster bit depth: {}'.format(options['raster_bit_depth']))
    resolved = run_options.resolve_run_options(args)
    print('Number of processors: {}'.format(
        resolved.threads))
    print('Native pool workers: {}'.format(_statistics.pool_workers()))
    print('Ward distance budget: {} bytes'.format(
        resolved.ward_distance_budget_bytes))
    style, _ = config.resolve_style(persisted)
    print('Font family: {}'.format(', '.join(style.font_family)))
    print('Font size multiplier: {}'.format(style.font_multiplier))
    if persisted.get(config.INVALID_KEY):
        print('Note: some malformed settings in options.txt were ignored '
              '(see the "_invalid" list in the file).')


def process_args(args=None):
    raw_args = list(sys.argv[1:] if args is None else args)
    args = parse_arguments().parse_args(raw_args)
    if args.command == 'describe':
        from deeptoolsr.describe import describe_cli
        tool_args = args.tool_args
        if args.config != 'auto' and '--config' not in tool_args:
            tool_args = ['--config', args.config, *tool_args]
        describe_cli(args.tool, tool_args)
    if args.command == 'serve':
        from deeptoolsr.serve import serve_main
        from deeptoolsr.session import explicit_run_options
        explicit = explicit_run_options(raw_args)
        status = serve_main(config=args.config if 'config' in explicit else None,
                            threads=args.numberOfProcessors,
                            cache_bytes=args.cacheBytes)
        if status:
            raise SystemExit(status)
    if args.command == 'options':
        _process_options(args)
    return args


def main(args=None):
    if args is None and len(sys.argv) == 1:
        args = ["--help"]
    process_args(args)
