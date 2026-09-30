#!/usr/bin/env python
from deeptoolsr import stats as kernels, _transform, _compute_matrix_stream, parserCommon
from deeptoolsr.matrix import (Block, Matrix, MatrixFormatError,
                               MatrixHeader, OwnedMatrix, SPECIAL_PARAMS,
                               read_header, serialize_header)
import deeptoolsintervals.parse as dti
import numpy as np
import argparse
from contextlib import ExitStack
from itertools import chain
import sys
import os
import csv
import copy
import re
import gzip
import json
import shutil

from deeptoolsr import options as run_options
from deeptoolsr.region_provenance import (
    antisense_sources, update_antisense_sources,
)
from deeptoolsr.cli_errors import concise_cli_errors
from deeptoolsr.numeric_validation import (
    CompatibilityRule,
    finite_float,
    float32_finite,
    lower_bound_float,
    min_max_error,
    upper_bound_float,
    option_was_supplied,
    specified_options,
    validate_compatibility_rules,
)
from deeptoolsr.path_validation import (atomic_output_path, temporary_path_for,
                                        validate_input_output_paths)
from deeptoolsr.matrix_file import (matrix_output_is_compressed,
                                    open_matrix_text, validated_matrix_rows)
from deeptoolsr.matrix_validation import sample_parameter_values
from deeptoolsr import _statistics


class TransformAction(argparse.Action):
    """Record transform operations in their command-line order."""

    def __call__(self, parser, namespace, values, option_string=None):
        pipeline = getattr(namespace, self.dest, None) or []
        pipeline.append((self.const, values))
        setattr(namespace, self.dest, pipeline)


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='computeMatrixOperationsR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
This tool performs a variety of operations on files produced by computeMatrixR.

detailed help:

  computeMatrixOperationsR info -h

or

  computeMatrixOperationsR relabel -h

or

  computeMatrixOperationsR subset -h

or

  computeMatrixOperationsR filterStrand -h

or

  computeMatrixOperationsR filterValues -h

or

  computeMatrixOperationsR rbind -h

or

  computeMatrixOperationsR cbind -h

or
  computeMatrixOperationsR sort -h

or
  computeMatrixOperationsR dataRange -h

or
  computeMatrixOperationsR transform -h

""",
        epilog='example usages:\n'
               'computeMatrixOperationsR subset -m input.mat.gz -o output.mat.gz --group "group 1" "group 2" --samples "sample 3" "sample 10"\n\n'
               ' \n\n')

    subparsers = parser.add_subparsers(
        title='Commands',
        dest='command',
        metavar='')

    # info
    info_parser = subparsers.add_parser(
        'info',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[infoArgs()],
        help="Print group and sample information",
        usage='An example usage is:\n  computeMatrixOperationsR info -m input.mat.gz\n\n')
    info_parser.add_argument('--json', action='store_true',
                             help='Print the matrix header as JSON.')

    # relabel
    subparsers.add_parser(
        'relabel',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[infoArgs(), relabelArgs()],
        help="Change sample and/or group label information",
        usage='An example usage is:\n  computeMatrixOperationsR relabel -m input.mat.gz -o output.mat.gz --sampleLabels "sample 1" "sample 2"\n\n')

    # subset/reorder
    for command in ('subset', 'reorder'):
        subparsers.add_parser(
            command,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            parents=[infoArgs(), subsetArgs()],
            help="Subset and/or reorder the matrix. 'reorder' is an alias for 'subset'.",
            usage='An example usage is:\n  computeMatrixOperationsR {} -m '
            'input.mat.gz -o output.mat.gz --groups "group 1" "group 2" '
            '--samples "sample 3" "sample 10"\n\n'.format(command))

    # filterStrand
    subparsers.add_parser(
        'filterStrand',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[infoArgs(), filterStrandArgs()],
        help="Filter entries by strand.",
        usage='Example usage:\n  computeMatrixOperationsR filterStrand -m '
        'input.mat.gz -o output.mat.gz --strand +\n\n')

    # filterValues
    subparsers.add_parser(
        'filterValues',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[infoArgs(), filterValuesArgs()],
        help="Filter entries by min/max value.",
        usage='Example usage:\n  computeMatrixOperationsR filterValues -m '
        'input.mat.gz -o output.mat.gz --min 10 --max 1000\n\n')

    # rbind
    subparsers.add_parser(
        'rbind',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[bindArgs(include_group_policy=True)],
        help="merge multiple matrices by concatenating them head to tail. This assumes that the same samples are present in each in the same order.",
        usage='Example usage:\n  computeMatrixOperationsR rbind -m '
        'input1.mat.gz input2.mat.gz -o output.mat.gz\n\n')

    # cbind
    subparsers.add_parser(
        'cbind',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[bindArgs()],
        help="Merge matrices left to right. The first matrix defines the output "
             "groups, rows, and row order. Later rows or groups absent from the "
             "first matrix are discarded; first-matrix rows absent later receive "
             "NA. Validated mode joins by a unique complete stored BED row "
             "(chromosome, blocks, name, score, and strand) within each unique, "
             "non-empty group. --blind instead binds rows by position.",
        usage='Example usage:\n  computeMatrixOperationsR cbind -m '
        'input1.mat.gz input2.mat.gz -o output.mat.gz\n\n')

    # sort
    subparsers.add_parser(
        'sort',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[sortArgs()],
        help='Sort a matrix file to correspond to the order of entries in the desired input file(s). The groups of regions designated by the files must be present in the order found in the output of computeMatrix (otherwise, use the subset command first). Note that this subcommand can also be used to remove unwanted regions, since regions not present in the input file(s) will be omitted from the output.',
        usage='Example usage:\n  computeMatrixOperationsR sort -m input.mat.gz -R regions1.bed regions2.bed regions3.gtf -o input.sorted.mat.gz\n\n')

    # dataRange
    subparsers.add_parser(
        'dataRange',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[infoArgs()],
        help='Returns the min, max, median, 10th and 90th percentile of the matrix values per sample.',
        usage='Example usage:\n  computeMatrixOperationsR dataRange -m input.mat.gz\n\n')

    subparsers.add_parser(
        'transform',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[transformArgs()],
        help='Apply an ordered arithmetic pipeline to one or more matrices.',
        usage='Example usage:\n  computeMatrixOperationsR transform -m input1.gz input2.gz '
              '--add 5 --scale 2 1 --log2FC --scale -1 -o output.gz\n\n')

    parserCommon.add_version(parser, 'computeMatrixOperationsR')
    run_options.add_config_option(parser)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for command, subparser in action.choices.items():
                run_options.add_config_option(
                    subparser, default=argparse.SUPPRESS)
                if command not in ('info', 'relabel'):
                    run_options.add_processor_option(subparser)

    return parser


def bindArgs(include_group_policy=False):
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('Required arguments')

    required.add_argument('--matrixFile', '-m',
                          help='Matrix files from the computeMatrixR tool.',
                          nargs='+',
                          required=True)

    required.add_argument('--outFileName', '-o',
                          help='Output file name',
                          required=True)

    if include_group_policy:
        parser.add_argument('--sameGroupLabels',
                            choices=['merge', 'separate'],
                            default='merge',
                            help='How rbind handles groups with the same label. '
                                 'merge combines matching labels; '
                                 'separate always retains each group.')

    parser.add_argument('--blind', action='store_true',
                        help='Skip content checks, keeping only the shape check '
                             'that keeps the output parseable. For rbind: skip '
                             'sample labels, boundaries, and all per-sample '
                             'geometry checks; still requires the same number of columns. '
                             'For cbind: skip the region-name join and concatenate '
                             'columns in row order, streaming with O(1) memory; '
                             'still requires the same number of rows.')

    return parser


def infoArgs():
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('Required arguments')

    required.add_argument('--matrixFile', '-m',
                          help='Matrix file from the computeMatrixR tool.',
                          required=True)

    return parser


def transformArgs():
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('Required arguments')
    required.add_argument('--matrixFile', '-m', nargs='+', required=True,
                          help='Input computeMatrixR files.')
    required.add_argument('--outFileName', '-o', nargs='+', required=True,
                          help='Output matrix file(s). The number must equal '
                               'the number of matrices remaining after the pipeline.')
    parser.add_argument(
        '--blind', action='store_true',
        help='For operations that combine matrices, skip BED/group matching and '
             'combine rows by position. The matrices must then contain the same '
             'number of rows. Sample geometry, including reference-point '
             'anchors, must match in both modes. Without this option, the first matrix owns the '
             'output groups and rows and later matrices are matched by complete '
             'BED identity within each group.')

    operations = parser.add_argument_group(
        'Ordered operations (executed from left to right)')
    for option, name, help_text in (
            ('--add', 'add', 'Add value(s) to the current matrix or matrices.'),
            ('--subtract', 'subtract_value', 'Subtract value(s) from the current matrix or matrices.'),
            ('--scale', 'scale', 'Multiply by numeric factor(s), or divide each row by row-sum, row-max, row-min, or row-mean.'),
    ):
        operations.add_argument(option, dest='pipeline', action=TransformAction,
                                const=name, nargs='+', metavar='VALUE',
                                help=help_text)
    for option, name, help_text in (
            ('--log2', 'log2', 'Apply log2 separately to every current matrix.'),
            ('--sum', 'sum', 'Sum two or more current matrices.'),
            ('--mean', 'mean', 'Average two or more current matrices.'),
            ('--difference', 'difference', 'Subtract the second matrix from the first; requires exactly two.'),
            ('--ratio', 'ratio', 'Divide the first matrix by the second; requires exactly two.'),
            ('--log2FC', 'log2fc', 'Calculate log2 of first/second; requires exactly two.'),
    ):
        operations.add_argument(option, dest='pipeline', action=TransformAction,
                                const=name, nargs=0, help=help_text)

    settings = parser.add_argument_group('Transform settings')
    settings.add_argument('--pseudocount', default='0',
                          help='Pseudocount for ratio and log2FC: a number or '
                               'min/max-numerator/denominator, optionally divided '
                               'by a number (for example min-denominator/2).')
    settings.add_argument('--scaleUsing',
                          choices=['all', 'upstream', 'body', 'downstream'],
                          default='all',
                          help='Bins used to calculate row-based scale divisors.')
    settings.add_argument('--missingDataAsZero', action='store_true',
                          help='Convert masked and NaN input bins to zero '
                               'before running the transformation pipeline.')
    settings.add_argument('--zeroHandling',
                          choices=['keep', 'any', 'both'], default='keep',
                          help='For two-matrix operations, keep normal results, '
                               'mask a bin when either input is zero, or mask it '
                               'only when both inputs are zero. Zeroes are '
                               'checked after --missingDataAsZero and before '
                               'adding a pseudocount.')
    return parser


def relabelArgs():
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('Required arguments')

    required.add_argument('--outFileName', '-o',
                          help='Output file name',
                          required=True)

    optional = parser.add_argument_group('Optional arguments')

    optional.add_argument('--groupLabels',
                          nargs='+',
                          help="Groups labels. If none are specified then the current labels will be kept.")

    optional.add_argument('--sampleLabels',
                          nargs='+',
                          help="Sample labels. If none are specified then the current labels will be kept.")

    optional.add_argument('--setGroupLabel', nargs=2, action='append',
                          metavar=('SELECTOR', 'LABEL'), default=None,
                          help="Relabel a single group, leaving the rest unchanged. "
                               "SELECTOR is a 1-based index or an exact current "
                               "label (a name relabels every group with that name). "
                               "Repeatable, e.g. --setGroupLabel 2 new --setGroupLabel old other.")

    optional.add_argument('--setSampleLabel', nargs=2, action='append',
                          metavar=('SELECTOR', 'LABEL'), default=None,
                          help="Relabel a single sample; see --setGroupLabel.")

    return parser


def subsetArgs():
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('Required arguments')

    required.add_argument('--outFileName', '-o',
                          help='Output file name',
                          required=True)

    optional = parser.add_argument_group('Optional arguments')

    optional.add_argument('--groups',
                          nargs='+',
                          help="Groups to include, by exact name or 1-based index. If none are specified then all will be included.")

    optional.add_argument('--samples',
                          nargs='+',
                          help="Samples to include, by exact name or 1-based index. If none are specified then all will be included.")

    optional.add_argument('--mergeSameNamedGroups', action='store_true',
                          help='Merge groups with identical labels before '
                               'selecting or reordering groups and samples.')

    return parser


def filterStrandArgs():
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('Required arguments')

    required.add_argument('--outFileName', '-o',
                          help='Output file name',
                          required=True)

    required.add_argument('--strand', '-s',
                          help='Strand',
                          choices=['+', '-', '.'],
                          required=True)

    return parser


def filterValuesArgs():
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('Required arguments')

    required.add_argument('--outFileName', '-o',
                          help='Output file name',
                          required=True)

    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('--min',
                          help='Minimum value. Any row having a single entry less than this will be excluded. The default is no minimum.',
                          type=lower_bound_float,
                          default=None)

    optional.add_argument('--max',
                          help='Maximum value. Any row having a single entry more than this will be excluded. The default is no maximum.',
                          type=upper_bound_float,
                          default=None)

    optional.add_argument('--filterUsingSamples',
                          nargs='+',
                          default=None,
                          help='Samples (exact names or 1-based indices) whose '
                               'values gate the filter. The default is every sample.')

    optional.add_argument('--filterUsingStatistic',
                          choices=list(_FILTER_STATISTICS),
                          default='perBin',
                          help="How to reduce a sample's bins in a row before "
                               "comparing to --min/--max. 'perBin' (default) tests "
                               "every individual bin (the classic behaviour); the "
                               "others test one per-sample statistic per row.")

    optional.add_argument('--filterNans',
                          choices=list(_FILTER_NAN_MODES),
                          default='keep',
                          help="A second filter, on missing data, combined with "
                               "--min/--max (a region/sample fails if either "
                               "filter fails; with the default -inf/+inf bounds "
                               "this runs alone). 'keep' (default) never filters "
                               "on NaNs; 'any_bin' fails a sample with any NaN "
                               "bin; 'any_sample' fails a sample whose bins are "
                               "all NaN; 'all_bins' fails a region whose gated "
                               "bins are all NaN (the whole row, by default).")

    optional.add_argument('--onFilterFail',
                          choices=['removeRegion', 'maskSample'],
                          default='removeRegion',
                          help="What to do when a gating sample is out of range: "
                               "'removeRegion' (default) drops the whole region; "
                               "'maskSample' keeps the region but sets only the "
                               "failing sample's bins to NaN.")

    return parser


def sortArgs():
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('Required arguments')

    required.add_argument('--matrixFile', '-m',
                          help='Matrix file from the computeMatrixR tool.',
                          required=True)

    required.add_argument('--outFileName', '-o',
                          help='Output file name',
                          required=True)

    required.add_argument('--regionsFileName', '-R',
                          help='File name(s), in BED or GTF format, containing the regions. '
                               'If multiple bed files are given, each one is '
                               'considered a group that can be plotted separately. '
                               'Also, adding a "#" symbol in the bed file causes all '
                               'the regions until the previous "#" to be considered '
                               'one group. Alternatively for BED files, putting '
                               'deepTools_group in the header can be used to indicate a '
                               'column with group labels. Note that these should be '
                               'sorted such that all group entries are together.',
                          required=True,
                          nargs='+')

    optional = parser.add_argument_group('Optional arguments')

    optional.add_argument('--transcriptID',
                          default='transcript',
                          help='When a GTF file is used to provide regions, only '
                          'entries with this value as their feature (column 3) '
                          'will be processed as transcripts. (Default: %(default)s)')

    optional.add_argument('--transcript_id_designator',
                          default='transcript_id',
                          help='Each region has an ID (e.g., ACTB) assigned to it, '
                          'which for BED files is either column 4 (if it exists) '
                          'or the interval bounds. For GTF files this is instead '
                          'stored in the last column as a key:value pair (e.g., as '
                          '\'transcript_id "ACTB"\', for a key of transcript_id '
                          'and a value of ACTB). In some cases it can be '
                          'convenient to use a different identifier. To do so, set '
                          'this to the desired key. (Default: %(default)s)')

    return parser


def info_record(path):
    """Return the public header summary without reading matrix values."""
    try:
        header = read_header(path, normalize=True)
    except ValueError as error:
        raise MatrixFormatError(str(error)) from error
    return {
        'protocol': 1,
        'samples': list(header.sample_labels),
        'groups': [
            {'label': label,
             'regions': header.group_boundaries[index + 1] -
             header.group_boundaries[index]}
            for index, label in enumerate(header.group_labels)
        ],
        'bins_per_sample': [
            header.sample_boundaries[index + 1] -
            header.sample_boundaries[index]
            for index in range(len(header.sample_labels))
        ],
        'parameters': dict(header.parameters),
    }


def _run_info_or_range(args):
    if args.command == 'info' and args.json:
        print(json.dumps(info_record(args.matrixFile), ensure_ascii=False))
        return
    matrix = Matrix.load(args.matrixFile, args.threads)
    if args.command == 'info':
        printInfo(matrix)
    else:
        printDataRange(matrix, threads=args.threads)


def printInfo(matrix):
    """
    Print a summary of the matrix: sample/region counts, per-sample geometry
    (bins, up/body/down, unscaled ends, reference point), region-group counts,
    sort state, scaling, and any recorded transformations.
    """
    header = matrix.header
    params = header.parameters
    sample_labels = list(header.sample_labels)
    group_labels = list(header.group_labels)
    group_bounds = header.group_boundaries
    sample_bounds = header.sample_boundaries

    def per_sample(key, idx):
        value = params.get(key)
        if isinstance(value, list):
            if not value:
                return None
            return value[idx] if idx < len(value) else value[-1]
        return value

    print("Samples: {0}".format(len(sample_labels)))
    print("Regions: {0}".format(group_bounds[-1]))

    print("\nRegion groups:")
    for idx, label in enumerate(group_labels):
        print("\t{0}: {1}".format(label, group_bounds[idx + 1] - group_bounds[idx]))

    print("\nSamples:")
    for idx, label in enumerate(sample_labels):
        print("\t{0}:".format(label))
        print("\t\tbins: {0}".format(sample_bounds[idx + 1] - sample_bounds[idx]))
        geometry = []
        for key, name in (('upstream', 'upstream'), ('body', 'body'),
                          ('downstream', 'downstream'),
                          ('unscaled 5 prime', "unscaled 5'"),
                          ('unscaled 3 prime', "unscaled 3'")):
            value = per_sample(key, idx)
            if value:            # omit zero / absent segments
                geometry.append("{0} {1}".format(name, value))
        if geometry:
            print("\t\t{0}".format(", ".join(geometry)))
        ref = per_sample('ref point', idx)
        print("\t\treference point: {0}".format(ref if ref else "none (scaled regions)"))
        binsize = per_sample('bin size', idx)
        if binsize:
            print("\t\tbin size: {0}".format(binsize))

    sort_regions = params.get('sort regions')
    sort_using = params.get('sort using')
    if sort_regions in (None, 'no', 'keep'):
        suffix = " (input order preserved)" if sort_regions == 'keep' else ""
        print("\nSorted: no{0}".format(suffix))
    else:
        print("\nSorted: {0} by {1}".format(sort_regions, sort_using))

    if params.get('scale') is not None:
        print("Scale: {0}".format(params['scale']))
    for key in ('scale plus', 'scale minus', 'antisense', 'scale antisense', 'unstranded'):
        if params.get(key) is not None:
            print("{0}: {1}".format(key, params[key]))

    transforms = params.get('transform operations')
    if transforms:
        print("Transformations:")
        for operation, value in transforms:
            print("\t{0}: {1}".format(operation, value))


def printDataRange(matrix, *, threads):
    print("Samples\tMin\tMax\tMedian\t10th\t90th")
    for i, sample in enumerate(matrix.header.sample_labels):
        start, end = matrix.header.sample_boundaries[i:i + 2]
        block = Block(matrix.values, None, (0, matrix.values.shape[0]),
                      None, (start, end))
        values = kernels.quantiles(block,
                                   [0, 100, 50, 10, 90], exact=True,
                                   threads=threads)
        print("\t".join([sample] + [str(value) for value in values]))


def readTransformHeader(path):
    """Read only the JSON header of a computeMatrix file."""
    return dict(read_header(
        path, scan=True,
        missing_message="Input matrix '{}' has no valid computeMatrix header"
        .format(path)).parameters)


def _stream_header(path):
    """Select the legacy stream error text from the shared header reader."""
    return dict(read_header(
        path, normalize=True,
        missing_message="{} has no computeMatrix JSON header".format(path)
    ).parameters)


def _header_sample_values(header, name):
    sample_count = len(header['sample_boundaries']) - 1
    return sample_parameter_values(header.get(name, 0), name, sample_count)


def validateTransformHeaders(headers, paths, pipeline, outputs, scale_using,
                             blind=False):
    """Preflight an ordered transform using computeMatrix headers only."""
    current = list(zip(headers, paths))
    combining = {'sum', 'mean', 'difference', 'ratio', 'log2fc'}
    locus_fields = ('upstream', 'body', 'downstream',
                    'unscaled 5 prime', 'unscaled 3 prime', 'bin size', 'ref point')

    def require_compatible(items, operation):
        first, first_path = items[0]
        first_sample_bins = np.diff(first['sample_boundaries']).tolist()
        first_regions = first['group_boundaries'][-1]
        first_loci = {name: _header_sample_values(first, name)
                      for name in locus_fields}
        problems = []
        for header, path in items[1:]:
            differences = []
            if len(header['sample_boundaries']) != len(first['sample_boundaries']):
                differences.append('number of samples')
            elif np.diff(header['sample_boundaries']).tolist() != first_sample_bins:
                differences.append('bins per sample')
            if blind and header['group_boundaries'][-1] != first_regions:
                differences.append('number of regions')
            for name in locus_fields:
                if _header_sample_values(header, name) != first_loci[name]:
                    differences.append(name)
            if differences:
                problems.append("'{}': {}".format(path, ', '.join(differences)))
        if problems:
            raise ValueError(
                '--{} requires compatible matrix subdimensions and locus sizes; '
                "'{}' differs from {}".format(
                    'log2FC' if operation == 'log2fc' else operation,
                    first_path, '; '.join(problems)))

    for operation, values in pipeline:
        count = len(current)
        if operation in ('add', 'subtract_value'):
            _values_for_matrices(values, count, '--' + operation, numeric=True)
        elif operation == 'scale':
            factors = _values_for_matrices(values, count, '--scale')
            for (header, path), factor in zip(current, factors):
                if factor in ('row-sum', 'row-max', 'row-min', 'row-mean'):
                    if scale_using != 'all':
                        segment = _header_sample_values(header, scale_using)
                        if any(value is None or value <= 0 for value in segment):
                            raise ValueError(
                                "--scale {} --scaleUsing {} requires that segment "
                                "in every sample of '{}'; found {}".format(
                                    factor, scale_using, path, segment))
                else:
                    try:
                        finite_float(factor)
                    except argparse.ArgumentTypeError:
                        raise ValueError(
                            "Invalid --scale value '{}'; expected a finite number "
                            "or row statistic".format(factor))
        elif operation in combining:
            required = 2 if operation in ('difference', 'ratio', 'log2fc') else None
            if required is not None and count != required:
                raise ValueError('--{} requires exactly two current matrices; {} remain'.format(
                    'log2FC' if operation == 'log2fc' else operation, count))
            if required is None and count < 2:
                raise ValueError('--{} requires at least two current matrices'.format(operation))
            require_compatible(current, operation)
            current = [current[0]]

    if len(outputs) != len(current):
        raise ValueError(
            'The pipeline leaves {} matrix/matrices, but {} output file(s) were supplied'.format(
                len(current), len(outputs)))


def _region_bed_identity(region):
    """Return the complete BED-like identity stored in a matrix region."""
    return (
        str(region[0]),
        tuple((int(start), int(end)) for start, end in region[1]),
        str(region[2]), str(region[5]), str(region[4]))


def _group_row_index(matrix_holder, path, matched_groups=None):
    """Index complete BED rows within unique, non-empty named groups.

    This is the shared validated matching contract for cbind and transform.
    The returned row indices always refer to ``matrix_holder``.
    """
    def repeated(values):
        seen = set()
        duplicates = set()
        for value in values:
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        return duplicates

    labels = list(matrix_holder.header.group_labels)
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError(
            'validated matrix matching requires non-empty group labels in {}'
            .format(path))
    duplicate_labels = sorted(repeated(labels))
    if duplicate_labels:
        raise ValueError(
            'validated matrix matching requires unique group labels; {} repeats '
            'in {}. Use --blind for positional matching'.format(
                ', '.join(repr(label) for label in duplicate_labels), path))

    result = {}
    for group_index, group in enumerate(labels):
        if matched_groups is not None and group not in matched_groups:
            continue
        start = int(matrix_holder.header.group_boundaries[group_index])
        end = int(matrix_holder.header.group_boundaries[group_index + 1])
        rows = matrix_holder.regions[start:end]
        if any(not isinstance(region[2], str) or not region[2].strip()
               for region in rows):
            raise ValueError(
                'validated matrix matching requires non-empty region names in '
                'group {!r} of {}'.format(group, path))
        identities = [_region_bed_identity(region) for region in rows]
        duplicate_rows = repeated(identities)
        if duplicate_rows:
            names = sorted({identity[2] for identity in duplicate_rows})
            raise ValueError(
                'validated matrix matching requires unique complete BED rows '
                'within group {!r}; {} repeats in {}. Use --blind for positional '
                'matching'.format(
                    group, ', '.join(repr(name) for name in names), path))
        result[group] = {
            identity: start + offset
            for offset, identity in enumerate(identities)}
    return result


def _align_transform_matrices(matrices, paths):
    """Align later matrices to the first matrix's group/row universe."""
    first = matrices[0]
    first_index = _group_row_index(first, paths[0])
    first_rows = first.values.shape[0]
    first_groups = list(first.header.group_labels)
    for matrix_holder, path in zip(matrices[1:], paths[1:]):
        other_index = _group_row_index(
            matrix_holder, path, matched_groups=set(first_groups))
        aligned = np.full((first_rows, matrix_holder.values.shape[1]),
                          np.nan, dtype=np.float32)
        for group in first_groups:
            available = other_index.get(group, {})
            for identity, output_row in first_index[group].items():
                source_row = available.get(identity)
                if source_row is not None:
                    aligned[output_row, :] = matrix_holder.values[source_row, :]
        matrix_holder.values = aligned
        matrix_holder.regions = list(first.regions)
        parameters = dict(matrix_holder.header.parameters)
        parameters['group_labels'] = list(first.header.group_labels)
        parameters['group_boundaries'] = list(first.header.group_boundaries)
        matrix_holder.header = MatrixHeader.from_parameters(parameters)


def _values_for_matrices(values, matrix_count, operation, numeric=False):
    if len(values) == 1:
        values = values * matrix_count
    elif len(values) != matrix_count:
        raise ValueError(
            '{} expects one value or one value per current matrix ({}), got {}'.format(
                operation, matrix_count, len(values)))
    if numeric:
        try:
            return [float32_finite(value) for value in values]
        except argparse.ArgumentTypeError as error:
            raise ValueError('{} values must be finite: {}'.format(operation, error))
    return values


_DATA_PSEUDOCOUNT = re.compile(
    r'(min|max)-(numerator|denominator)(?:/(.+))?', re.IGNORECASE)


def validate_pseudocount_specification(specification):
    """Validate a literal or data-derived pseudocount before reading inputs."""
    try:
        return ('literal', float32_finite(specification), None)
    except argparse.ArgumentTypeError:
        pass
    match = _DATA_PSEUDOCOUNT.fullmatch(str(specification))
    if not match:
        raise ValueError(
            "Invalid pseudocount '{}'; use a number or, for example, "
            "min-denominator/2".format(specification))
    extreme, source, divisor_text = match.groups()
    try:
        divisor = finite_float(divisor_text) if divisor_text is not None else 1.0
    except argparse.ArgumentTypeError as error:
        raise ValueError(str(error))
    if divisor == 0:
        raise ValueError('The pseudocount divisor cannot be zero')
    return (extreme.lower(), source.lower(), divisor)


def _sample_parameter(matrix, name, sample_index):
    value = matrix.header.parameters.get(name, 0)
    if isinstance(value, list):
        return value[sample_index]
    return value


def _scale_columns(matrix, sample_index, scale_using):
    start = matrix.header.sample_boundaries[sample_index]
    end = matrix.header.sample_boundaries[sample_index + 1]
    if scale_using == 'all':
        return start, end
    bin_size = _sample_parameter(matrix, 'bin size', sample_index)
    upstream = _sample_parameter(matrix, 'upstream', sample_index) // bin_size
    body = _sample_parameter(matrix, 'body', sample_index) // bin_size
    unscaled5 = _sample_parameter(matrix, 'unscaled 5 prime', sample_index) // bin_size
    downstream = _sample_parameter(matrix, 'downstream', sample_index) // bin_size
    if scale_using == 'upstream':
        selected = start, start + upstream
    elif scale_using == 'body':
        body_start = start + upstream + unscaled5
        selected = body_start, body_start + body
    else:
        selected = end - downstream, end
    if selected[0] >= selected[1]:
        raise ValueError("--scaleUsing {} selected no bins for sample '{}'".format(
            scale_using, matrix.header.sample_labels[sample_index]))
    return selected


def _to_float32_nan(matrix, threads):
    data = np.asarray(matrix)
    if (data.dtype == np.float32
            and data.flags.c_contiguous and data.flags.writeable and data.flags.aligned):
        _transform.nonfinite_to_nan(data, threads)
        return data
    return kernels.copy_values(matrix, threads=threads)


_TRANSFORM_ROW_STATS = {'row-sum': 0, 'row-max': 1, 'row-min': 2, 'row-mean': 3}


def _native_transform():
    return _transform


def scaleRows(matrix, statistic, scale_using='all', *, threads):
    """Divide every sample/region row by a statistic over selected bins, in place."""
    values = matrix.values  # already float32/NaN; mutated in place
    sample_bounds = [int(x) for x in matrix.header.sample_boundaries]
    selected_starts, selected_ends = [], []
    for sample_index in range(len(matrix.header.sample_labels)):
        # _scale_columns validates the segment (raises if it selects no bins).
        start, end = _scale_columns(matrix, sample_index, scale_using)
        selected_starts.append(int(start))
        selected_ends.append(int(end))

    native = _native_transform()
    native.scale_rows(values, sample_bounds, selected_starts, selected_ends,
                      _TRANSFORM_ROW_STATS[statistic], threads)
    return


def resolvePseudocounts(specification, numerator, denominator,
                        sample_boundaries):
    """Resolve one pseudocount for each sample-width block.

    A literal value applies uniformly to every sample.  Data-derived forms
    (for example ``min-denominator/2``) are calculated independently within
    each sample's columns, excluding masked, non-finite, and zero values.
    """
    sample_count = len(sample_boundaries) - 1
    extreme, source, divisor = validate_pseudocount_specification(specification)
    if extreme == 'literal':
        return [source] * sample_count
    mat = numerator if source == 'numerator' else denominator
    ans = []
    for start, end in zip(sample_boundaries[:-1], sample_boundaries[1:]):
        data = kernels.buffers(mat)
        value = kernels._statistics.nonzero_extreme(
            data, extreme == 'max', col_range=(start, end))
        try:
            ans.append(float32_finite(value / divisor))
        except argparse.ArgumentTypeError as error:
            raise ValueError(str(error))
    return ans


def transformMatrices(args):
    pipeline = args.pipeline or []
    headers = [readTransformHeader(path) for path in args.matrixFile]
    validateTransformHeaders(headers, args.matrixFile, pipeline,
                             args.outFileName, args.scaleUsing, args.blind)

    native = _native_transform()
    procs = args.threads

    matrices = []
    for path in args.matrixFile:
        owned = OwnedMatrix.load(path, procs)
        # One conversion to a mutable float32/NaN buffer; every op below is
        # destructive on it (no plotting downstream), so no full-matrix copies.
        owned.values = _to_float32_nan(owned.values, procs)
        matrices.append(owned)

    combining = {'sum', 'mean', 'difference', 'ratio', 'log2fc'}
    if any(operation in combining for operation, _values in pipeline) and not args.blind:
        _align_transform_matrices(matrices, args.matrixFile)
    if args.missingDataAsZero:
        for owned in matrices:
            native.nonfinite_to_zero(owned.values, procs)

    unary = {'add', 'subtract_value', 'scale', 'log2'}
    reductions = {'sum', 'mean'}
    binary = {'difference', 'ratio', 'log2fc'}
    history = []
    for operation, values in pipeline:
        history.append([operation, values])
        if operation in unary:
            if operation == 'log2':
                for owned in matrices:
                    data = owned.values
                    native.log2_inplace(data, procs)
                continue
            if operation == 'scale':
                factors = _values_for_matrices(
                    values, len(matrices), '--scale')
                for owned, factor in zip(matrices, factors):
                    if factor in ('row-sum', 'row-max', 'row-min', 'row-mean'):
                        scaleRows(owned, factor, args.scaleUsing, threads=procs)
                    else:
                        try:
                            value = finite_float(factor)
                        except argparse.ArgumentTypeError:
                            raise ValueError(
                                "Invalid --scale value '{}'; expected a finite "
                                "number or row statistic".format(factor))
                        native.scale_scalar(owned.values, value, procs)
                continue
            numbers = _values_for_matrices(
                values, len(matrices), '--' + operation, numeric=True)
            for owned, number in zip(matrices, numbers):
                step = number if operation == 'add' else -number
                native.add_scalar(owned.values, step, procs)
            continue

        if operation in reductions:
            if len(matrices) < 2:
                raise ValueError('--{} requires at least two current matrices'.format(operation))
            native.combine([owned.values for owned in matrices],
                           0 if operation == 'sum' else 1, procs)
            matrices = [matrices[0]]
            continue

        if operation in binary:
            if len(matrices) != 2:
                raise ValueError('--{} requires exactly two current matrices; {} remain'.format(
                    'log2FC' if operation == 'log2fc' else operation,
                    len(matrices)))
            numerator = matrices[0].values    # becomes the result in place
            denominator = matrices[1].values
            sample_boundaries = matrices[0].header.sample_boundaries
            zero_mode = {'any': 1, 'both': 2}.get(args.zeroHandling, 0)

            if operation == 'difference':
                pseudocounts = []
            else:
                pseudocounts = resolvePseudocounts(
                    args.pseudocount, numerator, denominator, sample_boundaries)
                if len(pseudocounts) != len(sample_boundaries) - 1:
                    raise ValueError(
                        "Pseudocounts must be specified for each sample; "
                        "expected {} values, got {}".format(
                            len(sample_boundaries) - 1, len(pseudocounts)))

            op_code = {'difference': 0, 'ratio': 1, 'log2fc': 2}[operation]
            native.binary_combine(
                numerator, denominator, [int(x) for x in sample_boundaries],
                [float(pc) for pc in pseudocounts], op_code, zero_mode, procs)
            matrices = [matrices[0]]

    if len(args.outFileName) != len(matrices):
        raise ValueError(
            'The pipeline leaves {} matrix/matrices, but {} output file(s) were supplied'.format(
                len(matrices), len(args.outFileName)))
    # Complete every matrix before replacing any member of the output set.
    with ExitStack() as outputs:
        temporary_outputs = [outputs.enter_context(
            atomic_output_path(output, suffix='.matrix.tmp'))
            for output in args.outFileName]
        for owned, output, final_output in zip(
                matrices, temporary_outputs, args.outFileName):
            parameters = dict(owned.header.parameters)
            parameters['transform operations'] = copy.deepcopy(history)
            owned.header = MatrixHeader.from_parameters(parameters)
            # The buffer is already plain float32 with NaN sentinels; save directly.
            owned.save(
                output,
                compressed=matrix_output_is_compressed(final_output), threads=procs)


def resolveLabels(values, labels, kind):
    """Resolve exact labels or 1-based indices, preserving requested order."""
    if values is None:
        return list(range(len(labels)))
    resolved = []
    for value in values:
        if value in labels:
            # A name selects all identically named entries. Use an index when
            # only one of several duplicate labels is desired.
            resolved.extend(idx for idx, label in enumerate(labels)
                            if label == value)
            continue
        try:
            idx = int(value) - 1
        except ValueError:
            idx = -1
        if idx < 0 or idx >= len(labels):
            sys.exit("Error: '{}' is not a valid {} name or 1-based index\n".format(
                value, kind))
        resolved.append(idx)
    return resolved


def mergeSameNamedGroups(matrix):
    """Merge duplicate group labels in first-occurrence order."""
    labels = matrix.header.group_labels
    if len(labels) == len(set(labels)):
        return
    old_bounds = matrix.header.group_boundaries
    merged_labels = list(dict.fromkeys(labels))
    row_indices = []
    merged_bounds = [0]
    for wanted in merged_labels:
        for idx, label in enumerate(labels):
            if label == wanted:
                row_indices.extend(range(old_bounds[idx], old_bounds[idx + 1]))
        merged_bounds.append(len(row_indices))
    matrix.values = matrix.values[row_indices, :]
    matrix.regions = subsetRegions(matrix, row_indices)
    _update_header(matrix, group_labels=merged_labels,
                   group_boundaries=merged_bounds)


def _update_header(matrix, **changes):
    parameters = dict(matrix.header.parameters)
    parameters.update(changes)
    matrix.header = MatrixHeader.from_parameters(parameters)


def getGroupBounds(args, matrix):
    """
    Given the group labels, return an indexing array and the resulting boundaries
    """
    bounds = matrix.header.group_boundaries
    indices = resolveLabels(args.groups, matrix.header.group_labels, 'group')
    o = list()
    obounds = [0]
    for idx in indices:
        o.extend(range(bounds[idx], bounds[idx + 1]))
        obounds.append(bounds[idx + 1] - bounds[idx])
    return o, np.cumsum(obounds), indices


def getSampleBounds(args, matrix):
    """
    Given the sample labels, return an indexing array
    """
    bounds = matrix.header.sample_boundaries
    indices = resolveLabels(args.samples, matrix.header.sample_labels, 'sample')
    o = list()
    for idx in indices:
        o.extend(range(bounds[idx], bounds[idx + 1]))
    return o, indices


def subsetRegions(matrix, bounds):
    out = []
    for x in bounds:
        reg = matrix.regions[x]
        # we need to add a list of [chrom, [(start, end), (start, end)], name, 0, strand, score)]
        if isinstance(reg, dict):
            # This happens on occasion
            starts = reg["start"].split(",")
            starts = [int(x) for x in starts]
            ends = reg["end"].split(",")
            ends = [int(x) for x in ends]
            regs = [(x, y) for x, y in zip(starts, ends)]
            out.append([reg["chrom"], regs, reg["name"], 0, reg["strand"], reg["score"]])
        else:
            out.append(reg)
    return out


def insertMatrix(matrix, other, groupName):
    """
    Insert one owned matrix's named group into another owned matrix.
    """
    # get the bounds for hm
    idx = matrix.header.group_labels.index(groupName)
    hmEnd = matrix.header.group_boundaries[idx + 1]
    # get the bounds for hm2
    idx2 = other.header.group_labels.index(groupName)
    hm2Start = other.header.group_boundaries[idx2]
    hm2End = other.header.group_boundaries[idx2 + 1]

    # Insert the subset hm2 into hm along axis 0
    matrix.values = np.insert(matrix.values, hmEnd,
                              other.values[hm2Start:hm2End, :], axis=0)

    # Insert the regions
    matrix.regions[hmEnd:hmEnd] = other.regions[hm2Start:hm2End]

    # Increase the group boundaries
    bounds = []
    for idx3, bound in enumerate(matrix.header.group_boundaries):
        if idx3 > idx:
            bound += hm2End - hm2Start
        bounds.append(bound)
    _update_header(matrix, group_boundaries=bounds)


def appendMatrix(matrix, other, groupName):
    """
    Append one named group from an owned matrix.
    """
    # get the bounds for hm2
    idx2 = other.header.group_labels.index(groupName)
    hm2Start = other.header.group_boundaries[idx2]
    hm2End = other.header.group_boundaries[idx2 + 1]

    # Append the matrix
    matrix.values = np.concatenate(
        [matrix.values, other.values[hm2Start:hm2End, :]], axis=0)
    # Update the bounds
    bounds = list(matrix.header.group_boundaries)
    bounds.append(bounds[-1] + hm2End - hm2Start)
    _update_header(matrix, group_boundaries=bounds)
    # Append the regions
    matrix.regions.extend(other.regions[hm2Start:hm2End])


def stream_rbind(output_path, input_paths, same_group_labels, blind=False):
    with temporary_path_for(output_path, suffix='.matrix.gz') as temporary:
        completed = _stream_rbind_unchecked(
            temporary, input_paths, same_group_labels, blind=blind,
            compressed=matrix_output_is_compressed(output_path))
        if completed:
            os.replace(temporary, output_path)
        return completed


def _stream_rbind_unchecked(output_path, input_paths, same_group_labels,
                            blind=False, compressed=True):
    """Row-bind matrices by streaming, without loading any matrix body.

    Reads only the headers to validate compatibility and build the merged
    group boundaries/labels, then writes the merged header and copies each
    file's data lines through -- O(1) memory. Returns False (so the caller
    falls back to the in-memory path) when merge-matching would interleave
    groups, which needs random access.

    By default, sample labels, boundaries, and every per-sample geometry field
    must match. ``blind`` skips those checks, keeping only the same-number-of-
    columns check that keeps rows well-formed; the merged header keeps the first
    file's sample metadata.
    """
    base = _stream_header(input_paths[0])
    group_labels = list(base['group_labels'])
    group_boundaries = list(base['group_boundaries'])
    merge = same_group_labels == 'merge'
    geometry_fields = tuple(SPECIAL_PARAMS)
    base_geometry = {name: _header_sample_values(base, name)
                     for name in geometry_fields}
    others = [(path, _stream_header(path)) for path in input_paths[1:]]
    input_headers = [dict(base)] + [other for _path, other in others]

    # Validate every input before deciding whether group merging requires the
    # in-memory path. Otherwise an early overlapping group could leave later
    # files unchecked.
    for path, other in others:
        if blind:
            if other['sample_boundaries'][-1] != base['sample_boundaries'][-1]:
                raise SystemExit(
                    'Error: rbind --blind requires the same number of columns; '
                    '{} differs.'.format(path))
        else:
            differences = []
            if other['sample_labels'] != base['sample_labels']:
                differences.append('sample labels/order')
            if other['sample_boundaries'] != base['sample_boundaries']:
                differences.append('sample boundaries')
            for name in geometry_fields:
                if _header_sample_values(other, name) != base_geometry[name]:
                    differences.append(name)
            if differences:
                raise SystemExit(
                    'Error: rbind requires identical samples and sample geometry; {} differs '
                    'in {}. Use --blind to skip geometry validation.'.format(
                        path, ', '.join(differences)))

    # A duplicate label within one input also requires regrouping.  Merely
    # checking for overlap with earlier files would let a unique/disjoint file
    # such as ["new", "new"] take the streaming path and preserve both groups.
    if merge:
        headers = [base] + [other for _path, other in others]
        if any(len(header['group_labels']) !=
               len(set(header['group_labels'])) for header in headers):
            return False

    for path, other in others:
        if merge and set(other['group_labels']) & set(group_labels):
            return False  # interleaving needs the in-memory path
        for gi, label in enumerate(other['group_labels']):
            size = other['group_boundaries'][gi + 1] - other['group_boundaries'][gi]
            group_labels.append(label)
            group_boundaries.append(group_boundaries[-1] + size)

    base['group_labels'] = group_labels
    base['group_boundaries'] = group_boundaries
    update_antisense_sources(base, input_headers)
    header = serialize_header(base)

    with open_matrix_text(
            output_path, 'wt', compressed=compressed, newline='') as out:
        out.write('@' + header + '\n')
        for path, input_header in zip(input_paths, input_headers):
            with open_matrix_text(path, 'rt') as handle:
                handle.readline()  # skip the header line
                for line in validated_matrix_rows(handle, input_header, path):
                    # A file whose last line lacks a newline would otherwise fuse
                    # with the next file's first row; normalise the line ending.
                    out.write(line if line.endswith('\n') else line + '\n')
    return True


def stream_relabel(input_path, output_path, group_labels, sample_labels,
                   set_group, set_sample):
    with atomic_output_path(output_path, suffix='.matrix.gz') as temporary:
        return _stream_relabel_unchecked(
            input_path, temporary, group_labels, sample_labels,
            set_group, set_sample,
            compressed=matrix_output_is_compressed(output_path))


def _stream_relabel_unchecked(input_path, output_path, group_labels,
                              sample_labels, set_group, set_sample,
                              compressed=True):
    """Relabel groups/samples by rewriting only the header, copying the body.

    Supports full replacement (`group_labels`/`sample_labels`) and targeted
    per-entry relabel (`set_group`/`set_sample` = list of (selector, label),
    selector = 1-based index or exact current name). O(1) memory.
    """
    params = _stream_header(input_path)

    def relabel(current, full, targeted, kind):
        original = list(current)
        result = list(current)
        if full is not None:
            if len(full) != len(original):
                raise SystemExit(
                    "Error: {0} {1} labels given but {2} are required.".format(
                        len(full), kind, len(original)))
            result = list(full)
        for selector, new_label in (targeted or []):
            for idx in resolveLabels([selector], original, kind):
                result[idx] = new_label
        return result

    new_group_labels = relabel(params['group_labels'], group_labels,
                               set_group, 'group')
    update_antisense_sources(params, [params], new_group_labels)
    params['group_labels'] = new_group_labels
    params['sample_labels'] = relabel(params['sample_labels'], sample_labels,
                                      set_sample, 'sample')
    header = serialize_header(params)

    with open_matrix_text(
            output_path, 'wt', compressed=compressed, newline='') as out:
        out.write('@' + header + '\n')
        with open_matrix_text(input_path, 'rt') as handle:
            handle.readline()  # skip header
            for line in validated_matrix_rows(handle, params, input_path):
                out.write(line if line.endswith('\n') else line + '\n')


def stream_cbind_blind(output_path, input_paths):
    with atomic_output_path(output_path, suffix='.matrix.gz') as temporary:
        return _stream_cbind_blind_unchecked(
            temporary, input_paths,
            compressed=matrix_output_is_compressed(output_path))


def _stream_cbind_blind_unchecked(output_path, input_paths, compressed=True):
    """Column-bind matrices by streaming, matching rows by position (not name).

    Concatenates every file's value columns onto the first file's rows in
    lockstep, one row at a time (O(number-of-files) memory), after merging the
    per-sample header fields. Requires the same number of rows in every file;
    region metadata comes from the first file.
    """
    special_params = SPECIAL_PARAMS
    headers = [_stream_header(path) for path in input_paths]
    base = dict(headers[0])
    sample_labels = list(base['sample_labels'])
    sample_boundaries = list(base['sample_boundaries'])

    def sample_values(header, name):
        return sample_parameter_values(header.get(name), name, len(header['sample_labels']))
    merged = {name: sample_values(base, name) for name in special_params
              if name in base}

    for header in headers[1:]:
        offset = sample_boundaries[-1]
        sample_labels += list(header['sample_labels'])
        sample_boundaries += [offset + x for x in header['sample_boundaries'][1:]]
        for name in merged:
            merged[name] += sample_values(header, name)

    base['sample_labels'] = sample_labels
    base['sample_boundaries'] = sample_boundaries
    for name, values in merged.items():
        base[name] = values
    out_header = serialize_header(base)

    def mismatch():
        raise SystemExit('Error: cbind --blind requires the same number of '
                         'rows in every matrix.')

    with ExitStack() as stack:
        handles = [stack.enter_context(open_matrix_text(path, 'rt')) for path in input_paths]
        for handle in handles:
            handle.readline()  # skip each header line
        readers = [validated_matrix_rows(handle, header, path)
                   for handle, header, path in zip(handles, headers, input_paths)]
        with open_matrix_text(
                output_path, 'wt', compressed=compressed, newline='') as out:
            out.write('@' + out_header + '\n')
            while True:
                rows = [next(reader, None) for reader in readers]
                if not rows[0]:
                    if any(rows[1:]):
                        mismatch()
                    break
                if not all(rows[1:]):
                    mismatch()
                parts = [rows[0].rstrip('\r\n')]  # first file: metadata + values
                for row in rows[1:]:
                    fields = row.rstrip('\r\n').split('\t')
                    if len(fields) < 6:
                        raise SystemExit('Error: cbind --blind found a row with '
                                         'no value columns.')
                    parts.append('\t'.join(fields[6:]))
                out.write('\t'.join(parts) + '\n')


def _native_stream():
    return _compute_matrix_stream


def _assemble_filtered_output(input_path, output_path, new_group_boundaries,
                              temp_body_path, *, compressed):
    """Write the final file: rewritten header + the streamed temp body.

    The row-dropping filters stream survivor rows to a headerless temp. The
    new group boundaries aren't known until that pass finishes. The header
    and body use the same selected format; body bytes are copied verbatim.
    """
    params = _stream_header(input_path)
    params['group_boundaries'] = [int(x) for x in new_group_boundaries]
    header = serialize_header(params)
    with open(output_path, 'wb') as out:
        header_bytes = ('@' + header + '\n').encode('utf-8')
        if compressed:
            # GzipFile, not gzip.compress: before Python 3.13 the latter
            # writes the zlib OS byte, so header bytes varied by version.
            with gzip.GzipFile(fileobj=out, mode='wb', filename='',
                               mtime=0) as member:
                member.write(header_bytes)
        else:
            out.write(header_bytes)
        with open(temp_body_path, 'rb') as body:
            shutil.copyfileobj(body, out)


def stream_filter_strand(input_path, output_path, strand, *, threads):
    """Row-drop by strand, streaming in O(chunk) memory."""
    compressed = matrix_output_is_compressed(output_path)
    native = _native_stream()
    params = _stream_header(input_path)
    with temporary_path_for(output_path, suffix='.body.gz' if compressed
                            else '.body.txt') as temp_body:
        new_bounds = native.filter_strand(
            input_path, temp_body, [int(x) for x in params['group_boundaries']],
            strand, threads, compressed=compressed)
        with atomic_output_path(output_path, suffix='.matrix.gz' if compressed
                                else '.matrix.txt') as temp_output:
            _assemble_filtered_output(
                input_path, temp_output, new_bounds, temp_body,
                compressed=compressed)


_FILTER_STATISTICS = dict(_statistics.filter_statistics)
_FILTER_NAN_MODES = dict(kernels.FILTER_NAN_MODES)


def stream_filter_values(input_path, output_path, min_value, max_value,
                         filter_samples=None, statistic='perBin',
                         on_fail='removeRegion', nan_mode='keep', *, threads):
    """filterValues, streaming.

    Supports a per-sample gating statistic (`statistic`), a subset of gating
    samples (`filter_samples`), a NaN filter combined with the value filter
    (`nan_mode`), and two failure modes: drop the whole region (`removeRegion`)
    or blank only the failing sample with NaN (`maskSample`).
    """
    compressed = matrix_output_is_compressed(output_path)
    native = _native_stream()
    params = _stream_header(input_path)
    sample_labels = list(params['sample_labels'])
    sample_bounds = [int(x) for x in params['sample_boundaries']]
    stat = _FILTER_STATISTICS[statistic]
    nans = _FILTER_NAN_MODES[nan_mode]
    indices = resolveLabels(filter_samples, sample_labels, 'sample')
    has_min = min_value is not None
    has_max = max_value is not None
    min_v = float(min_value) if has_min else 0.0
    max_v = float(max_value) if has_max else 0.0
    procs = threads

    if on_fail == 'maskSample':
        # No rows removed: header is unchanged, write it up front.
        header = serialize_header(params)
        with atomic_output_path(output_path, suffix='.matrix.gz' if compressed
                                else '.matrix.txt') as temporary:
            native.filter_values_mask(
                input_path, temporary, header, sample_bounds, indices, stat,
                nans, has_min, min_v, has_max, max_v, procs,
                compressed=compressed)
        return

    group_bounds = [int(x) for x in params['group_boundaries']]
    with temporary_path_for(output_path, suffix='.body.gz' if compressed
                            else '.body.txt') as temp_body:
        new_bounds = native.filter_values(
            input_path, temp_body, group_bounds, sample_bounds, indices, stat, nans,
            has_min, min_v, has_max, max_v, procs, compressed=compressed)
        with atomic_output_path(output_path, suffix='.matrix.gz' if compressed
                                else '.matrix.txt') as temp_output:
            _assemble_filtered_output(
                input_path, temp_output, new_bounds, temp_body,
                compressed=compressed)


def stream_subset_columns(input_path, output_path, sample_selectors,
                          *, threads):
    """Column subset/reorder without moving rows, streaming.

    Only valid when every region group is kept in its original order (rows are
    untouched); the caller checks that before dispatching here.
    """
    compressed = matrix_output_is_compressed(output_path)
    native = _native_stream()
    special_params = SPECIAL_PARAMS
    params = _stream_header(input_path)
    sample_labels = list(params['sample_labels'])
    sample_bounds = [int(x) for x in params['sample_boundaries']]
    indices = resolveLabels(sample_selectors, sample_labels, 'sample')

    column_order = []
    widths = []
    for idx in indices:
        column_order.extend(range(sample_bounds[idx], sample_bounds[idx + 1]))
        widths.append(sample_bounds[idx + 1] - sample_bounds[idx])

    params['sample_labels'] = [sample_labels[idx] for idx in indices]
    params['sample_boundaries'] = np.cumsum([0] + widths).tolist()
    for name in special_params:
        value = params.get(name)
        if isinstance(value, list):
            params[name] = [value[idx] for idx in indices]
    header = serialize_header(params)
    with atomic_output_path(output_path, suffix='.matrix.gz' if compressed
                            else '.matrix.txt') as temporary:
        native.subset_columns(input_path, temporary, header, column_order,
                              threads, compressed=compressed)


def rbindMatrices(matrix, args):
    """
    Bind matrices, top to bottom while accounting for the groups.

    It's assumed that the same samples are present in both and in the exact same order
    """
    merge_matching = args.sameGroupLabels == 'merge'
    if merge_matching:
        # Matrix 1 owns the output order, but the merge policy applies to every
        # input, including duplicate labels already present in matrix 1.
        mergeSameNamedGroups(matrix)
    for idx in range(1, len(args.matrixFile)):
        other = OwnedMatrix.load(args.matrixFile[idx], args.threads)
        if merge_matching:
            # Consolidate internal duplicates before appendMatrix/insertMatrix,
            # whose label lookup intentionally selects a single group.
            mergeSameNamedGroups(other)
        parameters = dict(matrix.header.parameters)
        update_antisense_sources(parameters,
                                 [parameters, other.header.parameters])
        matrix.header = MatrixHeader.from_parameters(parameters)
        for group in other.header.group_labels:
            if merge_matching and group in matrix.header.group_labels:
                insertMatrix(matrix, other, group)
            else:
                appendMatrix(matrix, other, group)
                _update_header(matrix,
                               group_labels=[*matrix.header.group_labels, group])


def cbindMatrices(matrix, args):
    """
    Bind columns from different matrices according to the group and region names

    Missing regions are left as NA
    """
    # The first matrix owns the output's groups, regions, and row order.
    try:
        d = _group_row_index(matrix, args.matrixFile[0])
    except ValueError as error:
        raise SystemExit('Error: {}'.format(error))

    # Iterate through the other matrices
    for idx in range(1, len(args.matrixFile)):
        other = OwnedMatrix.load(args.matrixFile[idx], args.threads)
        try:
            _group_row_index(
                other, args.matrixFile[idx], matched_groups=set(d))
        except ValueError as error:
            raise SystemExit('Error: {}'.format(error))

        # Add the sample labels
        parameters = dict(matrix.header.parameters)
        parameters['sample_labels'] = [*matrix.header.sample_labels,
                                       *other.header.sample_labels]
        # Add the sample boundaries
        lens = [x + matrix.header.sample_boundaries[-1]
                for x in other.header.sample_boundaries][1:]
        parameters['sample_boundaries'] = [*matrix.header.sample_boundaries,
                                           *lens]

        # Add on additional NA initialized columns
        ncol = matrix.values.shape[1]
        added_columns = other.values.shape[1]
        added = np.full((matrix.values.shape[0], added_columns),
                        np.nan, dtype=np.float32)
        matrix.values = np.hstack((matrix.values, added))

        # Update the values
        for idx2, group in enumerate(other.header.group_labels):
            if group not in d:
                continue
            s = other.header.group_boundaries[idx2]
            e = other.header.group_boundaries[idx2 + 1]
            for idx3, reg in enumerate(other.regions[s:e]):
                identity = _region_bed_identity(reg)
                if identity not in d[group]:
                    continue
                matrix.values[d[group][identity], ncol:] = other.values[s + idx3, :]

        # Append the special params
        for name in SPECIAL_PARAMS:
            parameters[name] = [*matrix.header.parameters[name],
                                *other.header.parameters[name]]
        matrix.header = MatrixHeader.from_parameters(parameters)


def _region_text_lines(lines):
    """Stream decoded region lines for plain, gzip and bzip2 handles alike."""
    for line in lines:
        yield line if isinstance(line, str) else line.decode('utf-8')


def loadBED(line, fp, fname, labelColumn, labels, regions, defaultGroup):
    """Collect input order with one parsing path for every BED record."""
    labelIdx = None
    localRegions = {}
    for line in _region_text_lines(chain((line,), fp)):
        if line.startswith('#'):
            if labelColumn is None and localRegions:
                label = line[1:].strip() or os.path.basename(fname)
                labels[dti.findRandomLabel(labels, label)] = len(labels)
                regions.append(localRegions)
                localRegions = {}
            continue
        if line.startswith('track '):
            continue
        cols = line.strip().split('\t')
        if len(cols) < 3:
            continue
        if labelColumn is not None:
            label = cols.pop(labelColumn)
            if label not in labels:
                labels[label] = len(labels)
                regions.append({})
            labelIdx = labels[label]
            localRegions = regions[labelIdx]
        name = cols[3] if len(cols) >= 6 else '{}:{}-{}'.format(*cols[:3])
        name = dti.findRandomLabel(localRegions, name)
        localRegions[name] = len(localRegions)

    if labelIdx is None and localRegions:
        label = defaultGroup if defaultGroup is not None else os.path.basename(fname)
        labels[dti.findRandomLabel(labels, label)] = len(labels)
        regions.append(localRegions)


def loadGTFtranscript(cols, label, defaultGroup, transcript_id_designator):
    s = next(csv.reader([cols[8]], delimiter=' '))
    if "deepTools_group" in s and s[-1] != "deepTools_group":
        label = s[s.index("deepTools_group") + 1].rstrip(";")
    elif defaultGroup is not None:
        label = defaultGroup

    if transcript_id_designator not in s or s[-1] == transcript_id_designator:
        sys.stderr.write("Warning: {0} is malformed!\n".format("\t".join(cols)))
        return None, None

    name = s[s.index(transcript_id_designator) + 1].rstrip(";")
    return label, name


def loadGTF(line, fp, fname, labels, regions, transcriptID, transcript_id_designator, defaultGroup):
    """Collect transcript order using the same feature match on every line."""
    file_label = dti.findRandomLabel(labels, os.path.basename(fname))
    feature = transcriptID.lower()
    for line in _region_text_lines(chain((line,), fp)):
        if line.startswith('#') or not line.strip():
            continue
        cols = line.strip().split('\t')
        if len(cols) < 9 or cols[2].lower() != feature:
            continue
        label, name = loadGTFtranscript(cols, file_label, defaultGroup, transcript_id_designator)
        if label is None:
            continue
        if label not in labels:
            labels[label] = len(labels)
            regions.append({})
        localRegions = regions[labels[label]]
        if name not in localRegions:
            localRegions[name] = len(localRegions)


def _load_region_order(filenames, transcriptID, transcript_id_designator):
    """Read ordering metadata once, without retaining decompressed file text."""
    labels, regions = {}, []
    defaultGroup = 'genes' if len(filenames) == 1 else None
    for fname in filenames:
        with dti.openPossiblyCompressed(fname) as fp:
            lines = _region_text_lines(fp)
            labelColumn = None
            for line in lines:
                if line.startswith('#'):
                    if labelColumn is None:
                        labelColumn = dti.getLabel(line)
                elif line.strip() and not line.startswith('track '):
                    break
            else:
                raise RuntimeError('{} contains no region records!'.format(fname))
            cols = line.strip().split('\t')
            ncols = len(cols) - (labelColumn is not None)
            if ncols < 3:
                raise RuntimeError('{} does not seem to be a recognized file type!'.format(fname))
            if ncols > 6 and dti.seemsLikeGTF(cols):
                loadGTF(line, lines, fname, labels, regions, transcriptID,
                        transcript_id_designator, defaultGroup)
            else:
                loadBED(line, lines, fname, labelColumn, labels, regions, defaultGroup)
    return labels, regions


def sortMatrix(matrix, regionsFileName, transcriptID, transcript_id_designator,
               verbose=True, allow_missing_groups=False):
    """Restore within-group input order and return the source row indices.

    ``allow_missing_groups`` is for computeMatrix's already-filtered results;
    values and regions stay in place for write-time selection.
    """
    labels, regions = _load_region_order(
        regionsFileName, transcriptID, transcript_id_designator)
    generated = antisense_sources(matrix.header.parameters)
    labelsList = list(matrix.header.group_labels)
    sources = []
    for label in labelsList:
        # A BED describing the actual output always takes precedence. Only
        # explicit generation metadata permits mapping back to the source BED.
        if label in labels:
            sources.append((label, False))
        elif label in generated and generated[label] in labels:
            sources.append((generated[label], True))
        else:
            raise ValueError("The region group '{}' is absent from the specified regions.".format(label))

    if not allow_missing_groups:
        present = {label for label, _ in sources}
        for label in labels:
            if label not in present:
                raise ValueError("The computeMatrix output is missing the '{}' region group.".format(label))

    order, boundaries = [], [0]
    for group_idx, (source, is_antisense) in enumerate(sources):
        start, end = matrix.header.group_boundaries[group_idx:group_idx + 2]
        # Key by group index, not label: bound matrices can contain repeated
        # labels whose rows must remain independent.
        current = {}
        for row_idx in range(start, end):
            name = matrix.regions[row_idx][2]
            if is_antisense:
                if not name.endswith('_antisense'):
                    raise ValueError('Generated antisense region lacks its name suffix: {}'.format(name))
                name = name[:-len('_antisense')]
            if name not in current:
                current[name] = row_idx
            elif isinstance(current[name], list):
                current[name].append(row_idx)
            else:
                # Row binding can retain several records with one identity.
                # Selecting that name must not silently discard earlier rows.
                current[name] = [current[name], row_idx]
        for name in regions[labels[source]]:
            if name in current:
                matches = current[name]
                if isinstance(matches, list):
                    order.extend(matches)
                else:
                    order.append(matches)
            elif verbose:
                sys.stderr.write('Skipping {}, due to being absent in the computeMatrix output.\n'.format(name))
        if len(order) == boundaries[-1] and not allow_missing_groups:
            raise ValueError('The region group {} had no matching entries!'.format(labelsList[group_idx]))
        boundaries.append(len(order))

    _update_header(matrix, group_labels=labelsList,
                   group_boundaries=boundaries)
    return order


@concise_cli_errors("computeMatrixOperationsR")
def main(args=None):
    # if args none is need since otherwise pytest passes 'pytest' as sys.argv
    if args is None:
        if len(sys.argv) == 1:
            args = ["-h"]
        if len(sys.argv) == 2:
            args = [sys.argv[1], "-h"]

    raw_args = list(args) if args is not None else sys.argv[1:]
    parser = parse_arguments()
    args = parser.parse_args(raw_args)
    args.threads = run_options.resolve_run_options(args).threads
    supplied = specified_options(raw_args)

    if args.command == 'filterValues':
        range_error = min_max_error(args.min, args.max, '--min', '--max')
        if range_error:
            parser.error(range_error)

    if args.command == 'transform':
        row_statistics = {'row-sum', 'row-max', 'row-min', 'row-mean'}
        operations = [operation for operation, _values in args.pipeline or []]
        has_ratio = any(operation in ('ratio', 'log2fc')
                        for operation in operations)
        has_binary = any(operation in ('difference', 'ratio', 'log2fc')
                         for operation in operations)
        has_row_scale = any(
            operation == 'scale' and any(value in row_statistics
                                         for value in values)
            for operation, values in args.pipeline or [])
        validate_compatibility_rules(parser, args, supplied, [
            CompatibilityRule(
                lambda _ns, opts: not has_ratio and
                option_was_supplied(opts, '--pseudocount'),
                '--pseudocount is unused without --ratio or --log2FC',
                'warning'),
            CompatibilityRule(
                lambda _ns, opts: not has_binary and
                option_was_supplied(opts, '--zeroHandling'),
                '--zeroHandling is unused without --difference, --ratio, or '
                '--log2FC', 'warning'),
            CompatibilityRule(
                lambda _ns, opts: not has_row_scale and
                option_was_supplied(opts, '--scaleUsing'),
                '--scaleUsing is unused without a row-sum/row-max/row-min/'
                'row-mean --scale', 'warning'),
        ])
        try:
            for operation, values in args.pipeline or []:
                if operation in ('add', 'subtract_value'):
                    for value in values:
                        float32_finite(value)
                elif operation == 'scale':
                    for value in values:
                        if value not in row_statistics:
                            float32_finite(value)
            validate_pseudocount_specification(args.pseudocount)
        except (argparse.ArgumentTypeError, ValueError) as error:
            parser.error(str(error))

    inputs = args.matrixFile if isinstance(args.matrixFile, list) else [args.matrixFile]
    inputs += list(getattr(args, 'regionsFileName', None) or [])
    outputs = getattr(args, 'outFileName', None)
    outputs = outputs if isinstance(outputs, list) else [outputs]
    try:
        validate_input_output_paths(inputs, outputs)
    except ValueError as error:
        parser.error(str(error))

    if args.command == 'transform':
        try:
            transformMatrices(args)
        except ValueError as error:
            raise SystemExit('Error: {}'.format(error))
        return

    if args.command == 'relabel':
        # Header-only: never load the matrix body.
        stream_relabel(args.matrixFile, args.outFileName, args.groupLabels,
                       args.sampleLabels, args.setGroupLabel, args.setSampleLabel)
        return

    # Order-preserving ops stream the file chunk-by-chunk without loading it.
    if args.command == 'filterStrand':
        stream_filter_strand(args.matrixFile, args.outFileName, args.strand,
                             threads=args.threads)
        return
    elif args.command == 'filterValues':
        stream_filter_values(args.matrixFile, args.outFileName,
                             args.min, args.max, args.filterUsingSamples,
                             args.filterUsingStatistic, args.onFilterFail,
                             args.filterNans, threads=args.threads)
        return
    elif args.command in ('subset', 'reorder'):
        # Streamable only when rows are untouched: all groups kept in order.
        if not getattr(args, 'mergeSameNamedGroups', False) and args.groups is None:
            stream_subset_columns(args.matrixFile, args.outFileName,
                                  args.samples, threads=args.threads)
            return

    if args.command in ('info', 'dataRange'):
        _run_info_or_range(args)
        return

    if args.command in ('subset', 'reorder'):
        matrix = OwnedMatrix.load(args.matrixFile, args.threads)
        if args.mergeSameNamedGroups:
            mergeSameNamedGroups(matrix)
        original_sample_labels = list(matrix.header.sample_labels)
        original_sample_bounds = list(matrix.header.sample_boundaries)
        original_group_labels = list(matrix.header.group_labels)
        sIdx, sample_indices = getSampleBounds(args, matrix)
        gIdx, gBounds, group_indices = getGroupBounds(args, matrix)
        sample_widths = [original_sample_bounds[idx + 1] - original_sample_bounds[idx]
                         for idx in sample_indices]
        parameters = dict(matrix.header.parameters)
        parameters['sample_boundaries'] = np.cumsum([0] + sample_widths).tolist()
        parameters['group_boundaries'] = gBounds.tolist()
        for name in SPECIAL_PARAMS:
            parameters[name] = [parameters[name][idx] for idx in sample_indices]
        parameters['sample_labels'] = [original_sample_labels[idx]
                                       for idx in sample_indices]
        parameters['group_labels'] = [original_group_labels[idx]
                                      for idx in group_indices]
        matrix.header = MatrixHeader.from_parameters(parameters)
        matrix.save(args.outFileName,
                    compressed=matrix_output_is_compressed(args.outFileName),
                    row_order=gIdx, column_order=sIdx, threads=args.threads)
    elif args.command == 'rbind':
        if not stream_rbind(args.outFileName, args.matrixFile,
                            args.sameGroupLabels, args.blind):
            matrix = OwnedMatrix.load(args.matrixFile[0], args.threads)
            rbindMatrices(matrix, args)
            matrix.save(args.outFileName,
                        compressed=matrix_output_is_compressed(args.outFileName),
                        threads=args.threads)
    elif args.command == 'cbind':
        if args.blind:
            stream_cbind_blind(args.outFileName, args.matrixFile)
        else:
            matrix = OwnedMatrix.load(args.matrixFile[0], args.threads)
            cbindMatrices(matrix, args)
            matrix.save(args.outFileName,
                        compressed=matrix_output_is_compressed(args.outFileName),
                        threads=args.threads)
    elif args.command == 'sort':
        matrix = OwnedMatrix.load(args.matrixFile, args.threads)
        order = sortMatrix(matrix, args.regionsFileName, args.transcriptID,
                           args.transcript_id_designator)
        matrix.save(args.outFileName,
                    compressed=matrix_output_is_compressed(args.outFileName),
                    row_order=order, threads=args.threads)
    else:
        sys.exit("Unknown command {0}!\n".format(args.command))
