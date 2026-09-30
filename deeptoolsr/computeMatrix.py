#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from contextlib import ExitStack
from dataclasses import replace
import sys
from deeptoolsr.parserCommon import writableFile
from deeptoolsr import parserCommon
from deeptoolsr.compute import ComputeMatrixBuilder, save_matrix_values
from deeptoolsr.matrix import (Labels, MatrixHeader, RowLayout, save, save_bed,
                               sort as sort_layout)
from deeptoolsr import options as run_options
from deeptoolsr.stats import STATISTIC_CHOICES
from deeptoolsr.cli_errors import concise_cli_errors
from deeptoolsr.path_validation import atomic_output_path, validate_input_output_paths
from deeptoolsr.matrix_file import matrix_output_is_compressed
import deeptoolsr.computeMatrixOperations as cmo
from deeptoolsr.numeric_validation import (
    CompatibilityRule,
    float32_finite,
    lower_bound_float,
    min_max_error,
    nonnegative_int,
    option_was_supplied,
    positive_int,
    specified_options,
    upper_bound_float,
    validate_compatibility_rules,
)


def parse_arguments(args=None):
    parser = \
        argparse.ArgumentParser(
            prog='computeMatrixR',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="""

This tool calculates scores per genome regions and prepares an intermediate file that can be used with ``plotHeatmapR`` and ``plotProfileR``.
Typically, the genome regions are genes, but any other regions defined in a BED file can be used.
computeMatrixR accepts multiple score files (bigWig format) and multiple regions files (BED format).
This tool can also be used to filter and sort regions according
to their score.

To learn more about the specific parameters, type:

$ computeMatrixR reference-point --help or

$ computeMatrixR scale-regions --help

""",
            epilog='An example usage is:\n  computeMatrixR reference-point -S '
            '<bigwig file(s)> -R <bed file(s)> -b 1000\n \n')

    parserCommon.add_version(parser, 'computeMatrixR')
    run_options.add_config_option(parser)

    subparsers = parser.add_subparsers(
        title='Commands',
        dest='command',
        metavar='')

    # scale-regions mode options
    subparsers.add_parser(
        'scale-regions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[
            computeMatrixRequiredArgs(),
            computeMatrixOutputArgs(),
            computeMatrixOptArgs(case='scale-regions'),
            parserCommon.gtf_options()
        ],
        help="In the scale-regions mode, all regions in the BED file are "
        "stretched or shrunken to the length (in bases) indicated by the user.",
        usage='An example usage is:\n  computeMatrixR scale-regions -S '
        '<biwig file(s)> -R <bed file> -b 1000\n\n')

    # reference point arguments
    subparsers.add_parser(
        'reference-point',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[computeMatrixRequiredArgs(),
                 computeMatrixOutputArgs(),
                 computeMatrixOptArgs(case='reference-point'),
                 parserCommon.gtf_options()
                 ],
        help="Reference-point refers to a position within a BED region "
        "(e.g., the starting point). In this mode, only those genomic"
        "positions before (upstream) and/or after (downstream) of the "
        "reference point will be plotted.",
        usage='An example usage is:\n  computeMatrixR reference-point -S '
        '<biwig file(s)> -R <bed file> -a 3000 -b 3000\n\n')

    return parser


def computeMatrixRequiredArgs(args=None):
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('Required arguments')
    required.add_argument('--regionsFileName', '-R',
                          metavar='File',
                          help='File name or names, in BED or GTF format, containing '
                               'the regions to plot. If multiple bed files are given, each one is considered a '
                               'group that can be plotted separately. Also, adding a "#" symbol in the bed file '
                               'causes all the regions until the previous "#" to be considered one group.',
                          nargs='+',
                          required=True)
    required.add_argument('--scoreFileName', '-S',
                          help='bigWig file(s) containing '
                          'the scores to be plotted. Multiple files should be separated by spaced. BigWig '
                          'files can be obtained by using the bamCoverage '
                          'or bamCompare tools. More information about '
                          'the bigWig file format can be found at '
                          'http://genome.ucsc.edu/goldenPath/help/bigWig.html ',
                          metavar='File',
                          nargs='+',
                          required=True)
    required.add_argument('--scoreFileNameMinus', '-s',
                          help='Optional minus DNA-strand bigWig files paired by '
                               'position with --scoreFileName. For features on '
                               'the minus strand, coding-strand values are read '
                               'from the corresponding file in this list. The '
                               'two lists must contain the same number of files.',
                          metavar='File', nargs='+')
    return parser


def computeMatrixOutputArgs(args=None):
    parser = argparse.ArgumentParser(add_help=False)
    output = parser.add_argument_group('Output options')
    output.add_argument('--outFileName', '-out', '-o',
                        help='File name to save the matrix needed by the '
                        '"plotHeatmapR" and "plotProfileR" tools. A .gz '
                        'suffix selects gzip compression; other suffixes write '
                        'plain text.',
                        type=writableFile,
                        required=True)

    output.add_argument('--outFileNameMatrix',
                        help='If this option is given, then the matrix '
                        'of values underlying the heatmap will be saved '
                        'using the indicated name, e.g. IndividualValues.tab.'
                        'This matrix can easily be loaded into R or '
                        'other programs.',
                        metavar='FILE',
                        type=writableFile)
    output.add_argument('--outFileSortedRegions',
                        help='File name in which the regions are saved '
                        'after skiping zeros or min/max threshold values. The '
                        'order of the regions in the file follows the sorting '
                        'order selected. This is useful, for example, to '
                        'generate other heatmaps keeping the sorting of the '
                        'first heatmap. Example: Heatmap1sortedRegions.bed',
                        metavar='BED file',
                        type=writableFile)
    return parser


def computeMatrixOptArgs(case=['scale-regions', 'reference-point'][0]):

    parser = argparse.ArgumentParser(add_help=False)
    optional = parser.add_argument_group('Optional arguments')
    parserCommon.add_version(optional, 'computeMatrixR')

    if case == 'scale-regions':
        optional.add_argument('--regionBodyLength', '-m',
                              default=1000,
                              type=positive_int,
                              help='Distance in bases to which all regions will '
                              'be fit. (Default: %(default)s)')
        optional.add_argument('--beforeRegionStartLength', '-b', '--upstream',
                              default=0,
                              type=nonnegative_int,
                              help='Distance upstream of the start site of '
                              'the regions defined in the region file. If the '
                              'regions are genes, this would be the distance '
                              'upstream of the transcription start site. (Default: %(default)s)')
        optional.add_argument('--afterRegionStartLength', '-a', '--downstream',
                              default=0,
                              type=nonnegative_int,
                              help='Distance downstream of the end site '
                              'of the given regions. If the '
                              'regions are genes, this would be the distance '
                              'downstream of the transcription end site. (Default: %(default)s)')
        optional.add_argument("--unscaled5prime",
                              default=0,
                              type=nonnegative_int,
                              help='Number of bases at the 5-prime end of the '
                              'region to exclude from scaling. By default, '
                              'each region is scaled to a given length (see the --regionBodyLength option). In some cases it is useful to look at unscaled signals around region boundaries, so this setting specifies the number of unscaled bases on the 5-prime end of each boundary. (Default: %(default)s)')
        optional.add_argument("--unscaled3prime",
                              default=0,
                              type=nonnegative_int,
                              help='Like --unscaled5prime, but for the 3-prime '
                              'end. (Default: %(default)s)')

    elif case == 'reference-point':
        optional.add_argument('--referencePoint',
                              default='TSS',
                              choices=['TSS', 'TES', 'center'],
                              help='The reference point for the plotting '
                              'could be either the region start (TSS), the '
                              'region end (TES) or the center of the region. '
                              'Note that regardless of what you specify, '
                              'plotHeatmapR/plotProfileR will default to using "TSS" as the '
                              'label. (Default: %(default)s)')

        # set region body length to zero for reference point mode
        optional.add_argument('--regionBodyLength', help=argparse.SUPPRESS,
                              default=0, type=nonnegative_int)
        optional.add_argument('--unscaled5prime', default=0, type=nonnegative_int, help=argparse.SUPPRESS)
        optional.add_argument('--unscaled3prime', default=0, type=nonnegative_int, help=argparse.SUPPRESS)
        optional.add_argument('--beforeRegionStartLength', '-b', '--upstream',
                              default=500,
                              type=nonnegative_int,
                              metavar='INT bp',
                              help='Distance upstream of the reference-point '
                              'selected. (Default: %(default)s)')
        optional.add_argument('--afterRegionStartLength', '-a', '--downstream',
                              default=1500,
                              metavar='INT bp',
                              type=nonnegative_int,
                              help='Distance downstream of the '
                              'reference-point selected. (Default: %(default)s)')
        optional.add_argument('--nanAfterEnd',
                              action='store_true',
                              help='If set, any values after the region end '
                              'are discarded. This is useful to visualize '
                              'the region end when not using the '
                              'scale-regions mode and when the reference-'
                              'point is set to the TSS.')

    optional.add_argument('--binSize', '-bs',
                          help='Length, in bases, of the non-overlapping '
                          'bins for averaging the score over the '
                          'regions length. (Default: %(default)s)',
                          type=positive_int,
                          default=10)

    optional.add_argument('--sortRegions',
                          help='Whether the output file should present the '
                          'regions sorted. The default is to not sort the regions. '
                          'Note that this is only useful if you plan to plot '
                          'the results yourself and not, for example, with '
                          'plotHeatmapR, which will override this. Note also that '
                          'unsorted output will be in whatever order the regions '
                          'happen to be processed in and not match the order in '
                          'the input files. If you require the output order to '
                          'match that of the input regions, then either specify '
                          '"keep" or use computeMatrixOperationsR to resort the '
                          'results file. (Default: %(default)s)',
                          choices=["descend", "ascend", "no", "keep"],
                          default='keep')

    optional.add_argument('--sortUsing',
                          help='Indicate which method should be used for '
                          'sorting. The value is computed for each row.'
                          'Note that the region_length option will lead '
                          'to a dotted line within the heatmap that indicates '
                          'the end of the regions. (Default: %(default)s)',
                          choices=["mean", "median", "max", "min", "sum",
                                   "region_length", "score"],
                          default='mean')

    optional.add_argument('--sortUsingSamples',
                          help='List of sample numbers (order as in matrix), '
                          'that are used for sorting by --sortUsing, '
                          'no value uses all samples, '
                          'example: --sortUsingSamples 1 3',
                          type=positive_int, nargs='+')

    optional.add_argument('--quantileSortedRegions', '--quantiles',
                          help='After ascending or descending sorting, '
                          'regions can be divided into '
                          'quantiles. The number of quantiles is specified '
                          'by this option. For example, if you specify 4, '
                          'then the regions will be divided into quartiles. '
                          'This is useful to visualize the distribution of the '
                          'signal across the regions. The default is %(default)s '
                          'which means no grouping. Setting > 1 will '
                          'override any existing region groups.',
                          type=positive_int, default=1)

    optional.add_argument('--averageTypeBins',
                          default='mean',
                          choices=STATISTIC_CHOICES['bin'],
                          help='Define the type of statistic that should be '
                          'used over the bin size range. The '
                          'options are: "mean", "median", "min", "max", "sum" '
                          'The default is "mean". (Default: %(default)s)')

    optional.add_argument('--missingDataAsZero',
                          help='If set, missing data (NAs) will be treated as zeros. '
                          'The default is to ignore such cases, which will be depicted as black areas in '
                          'a heatmap. (see the --missingDataColor argument '
                          'of the plotHeatmapR command for additional options).',
                          action='store_true')

    optional.add_argument('--skipZeros',
                          help='Whether regions with only scores of zero '
                          'should be included or not. Default is to include '
                          'them.',
                          action='store_true')

    optional.add_argument('--minThreshold',
                          default=None,
                          type=lower_bound_float,
                          help='Numeric value. Any region containing a '
                          'value that is less than or equal to this '
                          'will be skipped. This is useful to skip, '
                          'for example, genes where the read count is zero '
                          'for any of the bins. This could be the result of '
                          'unmappable areas and can bias the overall results. (Default: %(default)s)')

    optional.add_argument('--maxThreshold',
                          default=None,
                          type=upper_bound_float,
                          help='Numeric value. Any region containing a value '
                          'greater than or equal to this '
                          'will be skipped. The maxThreshold is useful to '
                          'skip those few regions with very high read counts '
                          '(e.g. micro satellites) that may bias the average '
                          'values. (Default: %(default)s)')

    optional.add_argument('--blackListFileName', '-bl',
                          help="A BED file containing regions that should be excluded from all analyses. Currently this works by rejecting genomic chunks that happen to overlap an entry. Consequently, for BAM files, if a read partially overlaps a blacklisted region or a fragment spans over it, then the read/fragment might still be considered.",
                          metavar="BED file",
                          required=False)

    optional.add_argument('--samplesLabel',
                          help='Labels for the samples. This will then be passed to plotHeatmapR and plotProfileR. The '
                          'default is to use the file name of the '
                          'sample. The sample labels should be separated '
                          'by spaces and quoted if a label itself'
                          'contains a space E.g. --samplesLabel label-1 "label 2"  ',
                          nargs='+')

    # in contrast to other tools,
    # computeMatrix by default outputs
    # messages and the --quiet flag supresses them
    optional.add_argument('--quiet', '-q',
                          help='Set to remove any warning or processing '
                          'messages.',
                          action='store_true')

    optional.add_argument('--verbose',
                          help='Being VERY verbose in the status messages. --quiet will disable this.',
                          action='store_true')

    optional.add_argument('--scale',
                          help='Final multiplier applied to the completed binned '
                               'matrix values, after the bin statistic and any '
                               'paired-track combination. (Default: %(default)s)',
                          type=float32_finite,
                          default=1)
    optional.add_argument('--scalePlus',
                          help='Multiplier applied after binning to values derived '
                               'from the plus DNA-strand --scoreFileName bigWigs, '
                               'before an optional sum with the minus bins.',
                          type=float32_finite, default=1)
    optional.add_argument('--scaleMinus',
                          help='Multiplier applied after binning to values derived '
                               'from the paired minus DNA-strand bigWigs, before '
                               'an optional sum with the plus bins.',
                          type=float32_finite, default=1)
    optional.add_argument('--scaleAntisense',
                          help='Additional final multiplier applied only to output '
                               'generated as antisense groups or samples, after '
                               'the bin statistic and paired-track combination. '
                               '(Default: %(default)s)',
                          type=float32_finite, default=1)
    optional.add_argument('--antisense',
                          choices=('skip', 'as_groups', 'as_samples'),
                          default='skip',
                          help='How to output signal opposite to the annotated '
                               'feature strand. Requires paired minus bigWigs. '
                               '(Default: %(default)s)')
    optional.add_argument('--unstranded',
                          choices=('as_plus', 'as_minus', 'sum_strands'),
                          default='sum_strands',
                          help='How unstranded features (e.g. ".", "*") are handled when '
                               'paired minus bigWigs are supplied. '
                               '(Default: %(default)s)')
    run_options.add_config_option(parser, default=argparse.SUPPRESS)
    run_options.add_processor_option(optional, allow_default=True)
    return parser


def process_args(args=None):
    parser = parse_arguments()
    raw_args = list(args) if args is not None else sys.argv[1:]
    args = parser.parse_args(raw_args)
    supplied = specified_options(raw_args)
    threshold_error = min_max_error(
        args.minThreshold, args.maxThreshold,
        '--minThreshold', '--maxThreshold')
    if threshold_error:
        parser.error(threshold_error)
    validate_compatibility_rules(parser, args, supplied, [
        CompatibilityRule(
            lambda ns, _opts: ns.quantileSortedRegions > 1 and
            ns.sortRegions not in ('ascend', 'descend'),
            '--quantiles requires --sortRegions ascend or descend'),
        CompatibilityRule(
            lambda ns, _opts: ns.scoreFileNameMinus is None and
            ns.antisense != 'skip',
            '--antisense {} requires --scoreFileNameMinus'.format(
                args.antisense)),
        CompatibilityRule(
            lambda ns, _opts: ns.scoreFileNameMinus is None and
            ns.unstranded in ('as_plus', 'as_minus'),
            '--unstranded {} requires --scoreFileNameMinus'.format(
                args.unstranded)),
        CompatibilityRule(
            lambda ns, _opts: ns.scoreFileNameMinus is None and
            ns.scaleMinus != 1,
            '--scaleMinus requires --scoreFileNameMinus'),
        CompatibilityRule(
            lambda ns, opts: ns.scoreFileNameMinus is None and
            ns.scaleMinus == 1 and option_was_supplied(opts, '--scaleMinus'),
            '--scaleMinus is unused without --scoreFileNameMinus', 'warning'),
        CompatibilityRule(
            lambda ns, _opts: ns.scaleAntisense != 1 and
            ns.antisense == 'skip',
            '--scaleAntisense requires --antisense as_groups or as_samples'),
        CompatibilityRule(
            lambda ns, opts: ns.scaleAntisense == 1 and
            ns.antisense == 'skip' and
            option_was_supplied(opts, '--scaleAntisense'),
            '--scaleAntisense is unused with --antisense skip', 'warning'),
        CompatibilityRule(
            lambda ns, opts: ns.scoreFileNameMinus is None and
            ns.unstranded == 'sum_strands' and
            option_was_supplied(opts, '--unstranded'),
            '--unstranded is unused without --scoreFileNameMinus', 'warning'),
        CompatibilityRule(
            lambda ns, opts: ns.sortRegions not in ('ascend', 'descend') and
            option_was_supplied(opts, '--sortUsing'),
            '--sortUsing is unused unless --sortRegions is ascend or descend',
            'warning'),
        CompatibilityRule(
            lambda ns, opts: ns.sortRegions not in ('ascend', 'descend') and
            option_was_supplied(opts, '--sortUsingSamples'),
            '--sortUsingSamples is unused unless --sortRegions is ascend or descend',
            'warning'),
        CompatibilityRule(
            lambda ns, opts: ns.sortRegions not in ('ascend', 'descend') and
            ns.quantileSortedRegions == 1 and
            option_was_supplied(opts, '--quantileSortedRegions', '--quantiles'),
            '--quantiles 1 is unused unless --sortRegions is ascend or descend',
            'warning'),
    ])

    if not raw_args:
        parse_arguments().print_help()
        sys.exit()

    if args.quiet is True:
        args.verbose = False

    if (args.scoreFileNameMinus is not None and
            len(args.scoreFileNameMinus) != len(args.scoreFileName)):
        parse_arguments().error(
            '--scoreFileName and --scoreFileNameMinus must contain the same number of files')
    if args.command == 'scale-regions':
        args.nanAfterEnd = False
        args.referencePoint = None
    elif args.command == 'reference-point':
        if args.beforeRegionStartLength == 0 and \
                args.afterRegionStartLength == 0:
            sys.exit("\nUpstrean and downstream regions are both "
                     "set to 0. Nothing to output. Maybe you want to "
                     "use the scale-regions mode?\n")

    return args


@concise_cli_errors("computeMatrixR")
def main(args=None):

    args = process_args(args)
    args.numberOfProcessors = run_options.resolve_run_options(args).threads

    inputs = (list(args.scoreFileName) + list(args.regionsFileName) +
              list(args.scoreFileNameMinus or []) + [args.blackListFileName])
    outputs = [args.outFileName, args.outFileNameMatrix,
               args.outFileSortedRegions]
    try:
        validate_input_output_paths(inputs, outputs)
    except ValueError as error:
        raise SystemExit('Error: {}'.format(error))

    parameters = {'upstream': args.beforeRegionStartLength,
                  'downstream': args.afterRegionStartLength,
                  'body': args.regionBodyLength,
                  'bin size': args.binSize,
                  'ref point': args.referencePoint,
                  'verbose': args.verbose,
                  'bin avg type': args.averageTypeBins,
                  'missing data as zero': args.missingDataAsZero,
                  'min threshold': args.minThreshold,
                  'max threshold': args.maxThreshold,
                  'scale': args.scale,
                  'skip zeros': args.skipZeros,
                  'nan after end': args.nanAfterEnd,
                  'proc number': args.numberOfProcessors,
                  'sort regions': args.sortRegions,
                  'sort using': args.sortUsing,
                  'unscaled 5 prime': args.unscaled5prime,
                  'unscaled 3 prime': args.unscaled3prime
                  }
    if args.scoreFileNameMinus is not None:
        parameters['score files minus'] = args.scoreFileNameMinus
    # Paired-track metadata is omitted for the classic one-track mode. This
    # keeps otherwise unchanged computeMatrix files compatible with deepTools
    # readers and existing reference matrices.
    if args.scoreFileNameMinus is not None:
        parameters['antisense'] = args.antisense
        parameters['unstranded'] = args.unstranded
    if args.scalePlus != 1:
        parameters['scale plus'] = args.scalePlus
    if args.scaleMinus != 1:
        parameters['scale minus'] = args.scaleMinus
    if args.scaleAntisense != 1:
        parameters['scale antisense'] = args.scaleAntisense

    builder = ComputeMatrixBuilder()
    scores_file_list = args.scoreFileName
    matrix = builder.computeMatrix(
        scores_file_list, args.regionsFileName, parameters,
        blackListFileName=args.blackListFileName,
        verbose=args.verbose, allArgs=args, threads=args.numberOfProcessors)
    layout = RowLayout.identity(matrix)
    if args.sortRegions not in ['no', 'keep']:
        sortUsingSamples = []
        if args.sortUsingSamples is not None:
            for i in args.sortUsingSamples:
                if (i > 0 and i <= len(matrix.header.sample_labels)):
                    sortUsingSamples.append(i - 1)
                else:
                    exit("The value {0} for --sortUsingSamples is not valid. Only values from 1 to {1} are allowed.".format(args.sortUsingSamples, len(matrix.header.sample_labels)))
            print('Samples used for ordering within each group: ', sortUsingSamples)
        columns = ([col for sample in sortUsingSamples
                    for col in range(*matrix.header.sample_boundaries[sample:sample + 2])]
                   if sortUsingSamples else None)
        layout = sort_layout(
            matrix, layout, using=args.sortUsing, method=args.sortRegions,
            cols=columns, quantiles=args.quantileSortedRegions,
            threads=args.numberOfProcessors)
    elif args.sortRegions == 'keep':
        order = cmo.sortMatrix(
            matrix, args.regionsFileName, args.transcriptID,
            args.transcript_id_designator, verbose=not args.quiet,
            allow_missing_groups=True)
        # The legacy keep path inserted group metadata before sample metadata.
        # Keep that JSON key order, which is part of the pinned gzip bytes.
        ordered = dict(matrix.header.parameters)
        for key in ('group_labels', 'group_boundaries', 'sample_labels',
                    'sample_boundaries'):
            ordered.pop(key)
        ordered['group_labels'] = list(matrix.header.group_labels)
        ordered['group_boundaries'] = list(matrix.header.group_boundaries)
        ordered['sample_labels'] = list(matrix.header.sample_labels)
        ordered['sample_boundaries'] = list(matrix.header.sample_boundaries)
        matrix.header = MatrixHeader.from_parameters(ordered)
        layout = replace(RowLayout.identity(matrix), rows=order)
    group_names = tuple(
        layout.base_names[origin.source] +
        (f'_Q{origin.quantile}' if origin.quantile is not None else '')
        for origin in layout.origins)
    labels = Labels(group_names, group_names, matrix.header.sample_labels)

    # Produce every requested artifact before atomically replacing any final
    # destination. This preserves an earlier complete result if a later writer
    # fails or the command is interrupted.
    with ExitStack() as outputs:
        matrix_path = outputs.enter_context(
            atomic_output_path(args.outFileName, suffix='.matrix.tmp'))
        values_path = (outputs.enter_context(atomic_output_path(
            args.outFileNameMatrix, suffix='.values.tmp'))
            if args.outFileNameMatrix else None)
        regions_path = (outputs.enter_context(atomic_output_path(
            args.outFileSortedRegions, suffix='.regions.tmp'))
            if args.outFileSortedRegions else None)
        save(matrix, layout, labels, matrix_path,
             compressed=matrix_output_is_compressed(args.outFileName),
             threads=args.numberOfProcessors)
        if values_path:
            save_matrix_values(matrix, layout, labels, values_path)
        if regions_path:
            with open(regions_path, 'w', newline='', encoding='utf-8') as handle:
                save_bed(matrix, layout, labels, handle)
