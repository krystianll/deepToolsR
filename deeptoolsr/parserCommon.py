import argparse
import os
from functools import lru_cache
from importlib.metadata import version
from deeptoolsr.stats import STATISTIC_CHOICES, FILTER_NAN_MODES

from deeptoolsr.numeric_validation import (
    assigned_finite_float,
    bounded_finite_float,
    decode_text_escapes,
    finite_float,
    fraction,
    nonnegative_int,
    positive_finite_float,
    positive_int,
)


def add_version(parser, prog):
    """Give a command its stable, public version line."""
    parser.add_argument('--version', action='version',
                        version='{} {}'.format(prog, version('deepToolsR')))


def check_float_0_1(value):
    return fraction(value)


def minor_tick_setting(value):
    value = str(value).lower()
    if value in ('none', 'auto'):
        return value
    try:
        count = nonnegative_int(value)
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError(
            '--minorTickMarks must be none, auto, or a non-negative integer')
    return count


def check_list_of_comma_values(value):
    if value is None:
        return None
    for foo in value:
        foo = value.split(",")
        if len(foo) < 2:
            raise argparse.ArgumentTypeError("%s is an invalid element of a list of comma separated values. "
                                             "Only argument elements of the following form are accepted: 'foo,bar'" % foo)
    return value


# --- Capability tags ------------------------------------------------------
# Help strings for plotHeatmapR / plotProfileR optionals are prefixed with
# a small tag saying how the option behaves: `[global]` for a single-valued
# setting, or `[multiple|explicit|recycled]` for a list-valued one, listing only
# the capabilities that apply. `explicit` means the option accepts per-target
# N,M= assignment; `recycled` means short lists are recycled across targets.

# Options that accept explicit ``N,M=`` / ``a-b=`` per-target assignment.
_EXPLICIT_TAG_DESTS = frozenset({
    'colorMap', 'colorList', 'zMin', 'zMax', 'zMid', 'colors',
    'sampleSetLabels', 'samplesLabel', 'xAxisLabel', 'yAxisLabel',
    'heatmapYAxisLabel', 'refPointLabel', 'startLabel', 'endLabel',
    'yMin', 'yMax', 'colorbarLabels'})
# Options whose short lists recycle across heatmaps / samples / sample sets.
# Series colours do not: a plain list names exactly one colour per series.
_RECYCLED_TAG_DESTS = _EXPLICIT_TAG_DESTS - {'colors'}


def capability_tag(action):
    """Return the ``[...]`` capability tag for one argparse action."""
    multiple = action.nargs in ('+', '*')
    if not multiple:
        return '[global]'
    parts = ['multiple']
    if (action.dest in _EXPLICIT_TAG_DESTS and
            getattr(action, '_capability_explicit', True)):
        parts.append('explicit')
    if action.dest in _RECYCLED_TAG_DESTS:
        parts.append('recycled')
    return '[' + '|'.join(parts) + ']'


def apply_capability_tags(parser):
    """Prefix every visible optional's help with its capability tag."""
    for action in parser._actions:
        if action.help in (None, argparse.SUPPRESS):
            continue
        if action.dest in ('help', argparse.SUPPRESS):
            continue
        if action.help.startswith(('[global]', '[multiple')):
            continue  # already tagged (idempotent)
        action.help = '{} {}'.format(capability_tag(action), action.help)
    return parser


def option_strings_by_dest(parser):
    """Capture each destination's spellings once for post-parse errors."""
    return {action.dest: tuple(action.option_strings)
            for action in parser._actions if action.option_strings}


@lru_cache(maxsize=3)
def plot_option_strings_by_dest(tool):
    """Read each plot tool's destination spellings from its parser once."""
    return option_strings_by_dest(plot_parser(tool, full_color_help=False))


def output(args=None):
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('Output')
    group.add_argument('--outFileName', '-o',
                       help='Output file name.',
                       metavar='FILENAME',
                       type=writableFile,
                       required=True)

    group.add_argument('--outFileFormat', '-of',
                       help='Output file type. Either "bigwig" or "bedgraph".',
                       choices=['bigwig', 'bedgraph'],
                       default='bigwig')

    return parser


# def read_options():
#     """Common arguments related to BAM files and the interpretation
#     of the read coverage
#     """
#     parser = argparse.ArgumentParser(add_help=False)
#     group = parser.add_argument_group('Read processing options')

#     group.add_argument('--extendReads', '-e',
#                        help='This parameter allows the extension of reads to '
#                        'fragment size. If set, each read is '
#                        'extended, without exception.\n'
#                        '*NOTE*: This feature is generally NOT recommended for '
#                        'spliced-read data, such as RNA-seq, as it would '
#                        'extend reads over skipped regions.\n'
#                        '*Single-end*: Requires a user specified value for the '
#                        'final fragment length. Reads that already exceed this '
#                        'fragment length will not be extended.\n'
#                        '*Paired-end*: Reads with mates are always extended to '
#                        'match the fragment size defined by the two read mates. '
#                        'Unmated reads, mate reads that map too far apart '
#                        '(>4x fragment length) or even map to different '
#                        'chromosomes are treated like single-end reads. The input '
#                        'of a fragment length value is optional. If '
#                        'no value is specified, it is estimated from the '
#                        'data (mean of the fragment size of all mate reads).\n',
#                        type=int,
#                        nargs='?',
#                        const=True,
#                        default=False,
#                        metavar="INT bp")

#     group.add_argument('--ignoreDuplicates',
#                        help='If set, reads that have the same orientation '
#                        'and start position will be considered only '
#                        'once. If reads are paired, the mate\'s position '
#                        'also has to coincide to ignore a read.',
#                        action='store_true'
#                        )

#     group.add_argument('--minMappingQuality',
#                        metavar='INT',
#                        help='If set, only reads that have a mapping '
#                        'quality score of at least this are '
#                        'considered.',
#                        type=int,
#                        )

#     group.add_argument('--centerReads',
#                        help='By adding this option, reads are centered with '
#                        'respect to the fragment length. For paired-end data, '
#                        'the read is centered at the fragment length defined '
#                        'by the two ends of the fragment. For single-end data, the '
#                        'given fragment length is used. This option is '
#                        'useful to get a sharper signal around enriched '
#                        'regions.',
#                        action='store_true')

#     group.add_argument('--samFlagInclude',
#                        help='Include reads based on the SAM flag. For example, '
#                        'to get only reads that are the first mate, use a flag of 64. '
#                        'This is useful to count properly paired reads only once, '
#                        'as otherwise the second mate will be also considered for the '
#                        'coverage. (Default: %(default)s)',
#                        metavar='INT',
#                        default=None,
#                        type=int,
#                        required=False)

#     group.add_argument('--samFlagExclude',
#                        help='Exclude reads based on the SAM flag. For example, '
#                        'to get only reads that map to the forward strand, use '
#                        '--samFlagExclude 16, where 16 is the SAM flag for reads '
#                        'that map to the reverse strand. (Default: %(default)s)',
#                        metavar='INT',
#                        default=None,
#                        type=int,
#                        required=False)

#     group.add_argument('--minFragmentLength',
#                        help='The minimum fragment length needed for read/pair '
#                        'inclusion. This option is primarily useful '
#                        'in ATACseq experiments, for filtering mono- or '
#                        'di-nucleosome fragments. (Default: %(default)s)',
#                        metavar='INT',
#                        default=0,
#                        type=int,
#                        required=False)

#     group.add_argument('--maxFragmentLength',
#                        help='The maximum fragment length needed for read/pair '
#                        'inclusion. (Default: %(default)s)',
#                        metavar='INT',
#                        default=0,
#                        type=int,
#                        required=False)

#     return parser


def gtf_options(suppress=False):
    """
    Arguments present whenever a BED/GTF file can be used
    """
    if suppress:
        parser = argparse.ArgumentParser(add_help=False)
        group = parser
    else:
        parser = argparse.ArgumentParser(add_help=False)
        group = parser.add_argument_group('GTF/BED12 options')

    if suppress:
        help = argparse.SUPPRESS
    else:
        help = 'When either a BED12 or GTF file are used to provide \
        regions, perform the computation on the merged exons, \
        rather than using the genomic interval defined by the \
        5-prime and 3-prime most transcript bound (i.e., columns \
        2 and 3 of a BED file). If a BED3 or BED6 file is used \
        as input, then columns 2 and 3 are used as an exon. (Default: %(default)s)'

    group.add_argument('--metagene',
                       help=help,
                       action='store_true',
                       dest='keepExons')

    if suppress is False:
        help = 'When a GTF file is used to provide regions, only \
        entries with this value as their feature (column 3) \
        will be processed as transcripts. (Default: %(default)s)'

    group.add_argument('--transcriptID',
                       help=help,
                       default='transcript')

    if suppress is False:
        help = 'When a GTF file is used to provide regions, only \
        entries with this value as their feature (column 3) \
        will be processed as exons. CDS would be another common \
        value for this. (Default: %(default)s)'

    group.add_argument('--exonID',
                       help=help,
                       default='exon')

    if suppress is False:
        help = 'Each region has an ID (e.g., ACTB) assigned to it, \
        which for BED files is either column 4 (if it exists) \
        or the interval bounds. For GTF files this is instead \
        stored in the last column as a key:value pair (e.g., as \
        \'transcript_id "ACTB"\', for a key of transcript_id \
        and a value of ACTB). In some cases it can be \
        convenient to use a different identifier. To do so, set \
        this to the desired key. (Default: %(default)s)'

    group.add_argument('--transcript_id_designator',
                       help=help,
                       default='transcript_id')

    return parser


def genomicRegion(string):
    # remove whitespaces using split,join trick
    region = ''.join(string.split())
    if region == '':
        return None
    # remove undesired characters that may be present and
    # replace - by :
    # N.B., the syntax for translate() differs between python 2 and 3
    try:
        region = region.translate(None, ",;|!{}()").replace("-", ":")
    except BaseException:
        region = region.translate({ord(i): None for i in ",;|!{}()"})
    if len(region) == 0:
        raise argparse.ArgumentTypeError(
            "{} is not a valid region".format(string))
    return region


def writableFile(string):
    """
    Validate an output path without creating, truncating, or removing it.

    Opening outputs while argparse is still running defeats the global
    input/output alias check and can destroy an input before validation.
    """
    path = os.path.abspath(os.path.expanduser(string))
    parent = os.path.dirname(path) or os.curdir
    if (not os.path.isdir(parent) or not os.access(parent, os.W_OK) or
            (os.path.exists(path) and
             (os.path.isdir(path) or not os.access(path, os.W_OK)))):
        msg = "{} file can't be opened for writing".format(string)
        raise argparse.ArgumentTypeError(msg)
    return string


def legend_location_alias(value):
    return {'top': 'above', 'bottom': 'below'}.get(value, value)


"""
Arguments used by matrix preparation and profile plotting
"""


def add_required_plot_args(required):
    required.add_argument('--matrixFile', '-m',
                          help='Matrix file from the computeMatrixR tool.',
                          type=str,
                          )

    required.add_argument('--outFileName', '-out', '-o',
                          help='File name to save the image to. The file '
                          'ending will be used to determine the image '
                          'format. The available options are: "png", '
                          '"eps", "pdf" and "svg", e.g., MyHeatmap.png.',
                          type=str,
                          required=True)


def add_shared_output_options(output):
    output.add_argument(
        '--outFileSortedRegions',
        help='File name into which the regions are saved '
        'after skipping zeros or min/max threshold values. The '
        'order of the regions in the file follows the sorting '
        'order selected. This is useful, for example, to '
        'generate other heatmaps while keeping the sorting of the '
        'first heatmap. Example: Heatmap1sortedRegions.bed',
        metavar='FILE',
        type=str)
    output.add_argument('--outFileNameMatrix',
                        help='If this option is given, then the matrix '
                        'of values underlying the plot will be saved '
                        'using this name. A .gz suffix selects gzip '
                        'compression; other suffixes write plain text.',
                        metavar='FILE',
                        type=str)


def add_out_file_name_data(output):
    output.add_argument('--outFileNameData',
                        help='File name for a tab-separated table containing '
                        'the exact plotted centre and optional lower/upper '
                        'profile bounds for every displayed panel and series, '
                        'e.g. myProfile.tab.',
                        type=str)


def add_output_dpi(output):
    output.add_argument(
        '--dpi',
        help='Set the DPI to save the figure.',
        type=positive_int,
        default=200)


def add_clustering(parser):
    cluster = parser.add_argument_group('Clustering arguments')
    cluster_method = cluster.add_mutually_exclusive_group()
    cluster_method.add_argument(
        '--kmeans',
        help='Number of clusters to compute. When this '
        'option is set, the matrix is split into clusters '
        'using the k-means algorithm. Clustering deliberately replaces '
        'all existing region groups with the requested clusters. If more '
        'specific clustering methods '
        'are required, then save the underlying matrix '
        'and run the clustering using other software. The plotting  '
        'of the clustering may fail with an error if a '
        'cluster has very few members compared to the total number '
        'or regions.',
        type=positive_int)
    cluster_method.add_argument(
        '--hclust',
        help='Number of clusters to compute. When this '
        'option is set, then the matrix is split into clusters '
        'using the hierarchical clustering algorithm, using "ward linkage". '
        'Clustering deliberately replaces all existing region groups with '
        'the requested clusters. --hclust could be very slow if you have '
        '>1000 regions. In those cases, you might prefer --kmeans or if more '
        'clustering methods are required you can save the underlying matrix and run '
        'the clustering using  other software. The plotting of the clustering may '
        'fail with an error if a cluster has very few members compared to the '
        'total number of regions.',
        type=positive_int)
    return cluster


def add_shared_plot_options(optional, *, prog):
    optional.add_argument("--help", "-h", action="help",
                          help="show this help message and exit")
    add_version(optional, prog)

    optional.add_argument('--clusterUsingSamples',
                          help='List of sample numbers (order as in '
                          'matrix), that are used for clustering by '
                          '--kmeans or --hclust if not given, all samples '
                          'are taken into account for clustering. '
                          'Example: --ClusterUsingSamples 1 3',
                          type=positive_int, nargs='+')

    optional.add_argument('--sortRegions',
                          help='Whether the heatmap should present '
                          'the regions sorted. The default is '
                          'to sort in descending order based on '
                          'the mean value per region. Note that "keep" and "no" are the same thing.',
                          choices=["descend", "ascend", "no", "keep"],
                          default='ascend')

    optional.add_argument('--sortUsing',
                          help='Indicate which method should be used for '
                          'sorting. For each row the method is computed. '
                          'For region_length, a dashed line is drawn at '
                          'the end of the region (reference point TSS and '
                          'center) or the beginning of the region '
                          '(reference point TES) as appropriate.',
                          choices=["mean", "median", "max", "min", "sum",
                                   "region_length", "score"],
                          default='mean')

    optional.add_argument('--sortUsingSamples',
                          help='List of sample numbers (order as in matrix), '
                          'which are used by --sortUsing for sorting. '
                          'If no value is set, it uses all samples. '
                          'Example: --sortUsingSamples 1 3',
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
                          'override any existing region groups. Note that this '
                          'cannot be used with --kmeans or --hclust.',
                          type=positive_int, default=1)


def add_distance_and_landmark_options(optional):
    optional.add_argument('--distanceUnit',
                          choices=['auto', 'bp', 'kb', 'mb'],
                          default='auto',
                          help='Unit for numeric distance labels. "auto" uses '
                          'bp below 1 kb, kb below 1 Mb, and mb otherwise.')
    optional.add_argument('--distanceUnitLocation',
                          choices=['ticks', 'axis'],
                          default='ticks',
                          help='Show the distance unit on every numeric tick '
                          'or once in square brackets on the X-axis label.')

    optional.add_argument('--startLabel', nargs='+',
                          default=['TSS'],
                          help='[only for scale-regions mode] Label shown '
                          'in the plot for the start of '
                          'the region. Default is TSS (transcription '
                          'start site), but could be changed to anything, '
                          'e.g. "peak start". '
                          'Same for the --endLabel option. See below. Use N=label for a 1-based sample set.')
    optional.add_argument('--endLabel', nargs='+',
                          default=['TES'],
                          help='[only for scale-regions mode] Label '
                          'shown in the plot for the region '
                          'end. Default is TES (transcription end site). Use N=label for a 1-based sample set.')
    optional.add_argument('--refPointLabel', nargs='+',
                          help='[only for reference-point mode] Label '
                          'shown in the plot for the '
                          'reference-point. Default '
                          'is the same as the reference point selected '
                          '(e.g. TSS), but could be anything, e.g. '
                          '"peak start". Use N=label for a 1-based sample set.',
                          default=None)

    optional.add_argument('--labelRotation',
                          dest='label_rotation',
                          help='Rotation of the X-axis labels in degrees. The default is 0, positive values denote a counter-clockwise rotation.',
                          type=finite_float,
                          default=0.0)


def add_regions_label(optional):
    optional.add_argument('--regionsLabel', '--regionLabels', '-z',
                          help='Labels for the regions plotted in the '
                          'heatmap. If more than one region is being '
                          'plotted, a list of labels separated by spaces is required. '
                          'If a label itself contains a space, then quotes are '
                          'needed. For example, --regionsLabel label_1, "label 2". ',
                          nargs='+', type=decode_text_escapes)


def add_show_region_counts(optional):
    optional.add_argument(
        '--showRegionCounts',
        action='store_true',
        help='Append the number of plotted regions to every region label, '
        'for example "genes [n = 1,234]". This applies to heatmap '
        'labels, profile labels, and summary-plot legends.')


def add_samples_label(optional):
    optional.add_argument('--samplesLabel', '--sampleLabels',
                          help='Labels for the samples plotted. The '
                          'default is to use the file name of the '
                          'sample. The sample labels should be separated '
                          'by spaces and quoted if a label itself'
                          'contains a space E.g. --samplesLabel label-1 "label 2". '
                          'Use N=label to name a 1-based sample; for example '
                          '--samplesLabel 2=Case. ',
                          nargs='+')


def add_plot_type(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'plotType')
    group.add_argument(*spellings, **kwargs)


def add_average_type(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'averageType')
    group.add_argument(*spellings, **kwargs)


def add_pseudocount(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'pseudocount')
    group.add_argument(*spellings, **kwargs)


def add_trim_perc(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'trim_perc')
    group.add_argument(*spellings, **kwargs)


def add_profile_height(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'profileHeight')
    kwargs.setdefault('type', positive_finite_float)
    kwargs.setdefault('default', None)
    kwargs.setdefault('help', 'Profile height in cm; by default the cell width '
                      'divided by the aspect ratio (5 cm when no size is set). '
                      'Any two of cell width, height and aspect ratio '
                      'determine the third.')
    group.add_argument(*spellings, **kwargs)


def add_profile_aspect_ratio(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'profileAspectRatio')
    group.add_argument(*spellings, **kwargs)


def add_y_min(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'yMin')
    group.add_argument(*spellings, **kwargs)


def add_y_max(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'yMax')
    group.add_argument(*spellings, **kwargs)


def add_x_axis_label(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'xAxisLabel')
    group.add_argument(*spellings, **kwargs)


def add_x_axis_visibility(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'xAxisVisibility')
    group.add_argument(*spellings, **kwargs)


def add_lines_at_tick_marks(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'linesAtTickMarks')
    group.add_argument(*spellings, **kwargs)


def add_y_axis_label(group, *spellings, explicit=True, **kwargs):
    kwargs.setdefault('dest', 'yAxisLabel')
    action = group.add_argument(*spellings, **kwargs)
    action._capability_explicit = explicit


def add_y_axis_visibility(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'yAxisVisibility')
    group.add_argument(*spellings, **kwargs)


def add_y_axis_limits(group, *spellings, **kwargs):
    kwargs.setdefault('dest', 'yAxisLimits')
    group.add_argument(*spellings, **kwargs)


def add_colors(group, *spellings, explicit=True, **kwargs):
    kwargs.setdefault('dest', 'colors')
    action = group.add_argument(*spellings, **kwargs)
    action._capability_explicit = explicit


def add_legend_location(optional, option_string):
    optional.add_argument(option_string,
                          dest='legendLocation',
                          type=legend_location_alias,
                          default='best',
                          choices=['best',
                                   'upper-right',
                                   'upper-left',
                                   'upper-center',
                                   'lower-left',
                                   'lower-right',
                                   'lower-center',
                                   'center',
                                   'center-left',
                                   'center-right',
                                   'above', "top",
                                   'below', "bottom",
                                   'right',
                                   'none'
                                   ],
                          help='Location for the legend in the summary plot. '
                          '"above" or "top" places the legend between the '
                          'main title and subplot titles; , "below" or '
                          '"bottom" places it below the Y-axis, "right" '
                          'places it outside the plots; "none" hides it.')


def _add_profile_statistics(optional):
    add_plot_type(optional,
                  '--plotType',
                  help='"lines" will plot the profile line based '
                  'on the average type selected. "fill" '
                  'fills the region between zero and the profile '
                  'curve. The fill in color is semi transparent to '
                  'distinguish different profiles. "se" and "std" '
                  'color the region between the profile and the '
                  'standard error or standard deviation of the data. '
                  'Standard error uses the finite observation count independently '
                  'in each bin (unlike stock deepTools) and is hidden when fewer '
                  'than two finite values are available. '
                  'As in the case of '
                  'fill, a semi-transparent color is used. '
                  '"ci" shades the analytic confidence interval of a '
                  'mean or geom_mean and "bootstrap" a percentile-bootstrap '
                  'interval of any average type (see --ci_level). '
                  '"heatmap" plots a '
                  'summary heatmap.',
                  choices=['lines', 'fill', 'se', 'std', 'ci', 'bootstrap',
                           'heatmap'],
                  default='lines')

    add_average_type(optional,
                     '--averageType',
                     default='mean',
                     choices=STATISTIC_CHOICES['profile'],
                     help='The type of statistic that should be used for the '
                     'profile. "geom_mean" requires nonzero observations within each bin '
                     'to have the same sign; '
                     '"trim_mean" trims values independently in each bin.')

    add_pseudocount(optional, '--pseudocount', type=finite_float, default=-1,
                    help='Pseudocount used by the geometric mean. '
                    'A negative value (the default) automatically uses '
                    'half the smallest positive magnitude in the profile '
                    'data, excluding zeros. All-zero data resolves to '
                    'zero. Zero is accepted explicitly and preserves '
                    'zero observations; analytic log-space bands need '
                    'a positive pseudocount for bins containing both '
                    'zero and nonzero observations.')
    add_trim_perc(optional, '--trim_perc', type=bounded_finite_float(
        0, 0.5, maximum_inclusive=False,
        requirement='a finite fraction from 0 up to, but not including, 0.5'),
        default=0.05,
        help='Fraction to trim from each tail for trimmed means.')
    add_interval_options(optional)
    add_plot_processor(optional)


def add_interval_options(optional, plot_type='--plotType'):
    optional.add_argument('--ci_level', type=bounded_finite_float(
        0, 1, minimum_inclusive=False,
        maximum_inclusive=False,
        requirement='a finite fraction strictly between 0 and 1'),
        default=0.95,
        help='Confidence level for confidence intervals. '
        f'Applicable if {plot_type} is "ci" or "bootstrap".')
    optional.add_argument('--bootstrapReplicates', type=positive_int, default=300,
                          help='Number of bootstrap resamples used by '
                               f'{plot_type} bootstrap.')


def add_plot_processor(group):
    from deeptoolsr.options import add_processor_option
    add_processor_option(group)


def _add_profile_panels(optional, *, matrix=False, common=None):
    add_profile_height(optional, '--plotHeight', '--profileHeight',
                       metavar='PLOTHEIGHT')
    if not matrix:
        optional.add_argument('--plotWidth', '--profileWidth',
                              dest='cellWidth', metavar='PLOTWIDTH',
                              help='Cell width in cm; default 5, and any two '
                                   'of width, profile height and aspect ratio '
                                   'determine the third.',
                              type=positive_finite_float, default=None)
    add_profile_aspect_ratio(optional,
                             '--profileAspectRatio' if matrix else '--aspectRatio',
                             metavar='ASPECTRATIO',
                             type=positive_finite_float,
                             default=None,
                             help='Cell width divided by profile height; default 1, '
                                  'and any two sizes determine the third.')

    add_sample_set_options(common or optional)


def add_sample_set_options(optional):
    optional.add_argument(
        '--arrangeSamples', nargs='+', metavar='SAMPLES',
        help='Create custom profile panels from comma-separated sample '
             'indices or names, for example "--arrangeSamples 1,2 3,4". '
             'Indices are 1-based. Samples not listed are omitted. '
             'By default, all samples are plotted in separate panels '
             'i.e. --arrangeSamples 1 2 3 ... n while --perGroup creates '
             'one set i.e. --arrangeSamples 1,2,3,...,n.')
    optional.add_argument(
        '--commonLegend', action='store_true',
        help='By default, every panel or sample set has its own legend. '
        'With this setting, you can replace them with one figure-level '
        'legend containing all unique keys.')
    optional.add_argument(
        '--sampleSetLabels', nargs='+', metavar='LABEL',
        help='Titles for the sample panels created by --arrangeSamples. '
             'By default, joined sample labels are used. Use N=label to '
             r'name a 1-based sample set. Text accepts \=, \\ and \n.')
    optional.add_argument(
        '--sampleSetGroupArrangement',
        choices=['overlay', 'adjacent', 'end', 'by_row', 'by_column'],
        default='overlay',
        help='Placement of additional logical region groups: overlay '
             'them as extra series, place group panels immediately after '
             'each sample set panel, or place them at the end after the first '
             'group has been plotted for every sample panel. "by_row" '
             'places region groups in one row while the next sample set '
             'occupies the next row; "by_column" places sample sets in columns '
             'while region groups are added as rows. These two modes determine '
             'the grid shape and override --gridRows/--gridColumns.')
    subplot_shape = optional.add_mutually_exclusive_group()
    subplot_shape.add_argument(
        '--gridColumns', type=positive_int, metavar='INT',
        help='Number of columns in the panel grid.')
    subplot_shape.add_argument(
        '--gridRows', type=positive_int, metavar='INT',
        help='Number of rows in the panel grid.')


def _add_profile_axes(optional, *, common=None):
    shared = common or optional
    add_x_axis_label(shared,
                     '--xAxisLabel', '-x',
                     nargs='+',
                     default=None,
                     help='Description for the X-axis label. By default, a label such '
                     'as "distance from TSS" is generated from the input. The '
                     'distance unit is placed according to --distanceUnitLocation. '
                     'You can have multiple X-axis labels and they are assigned '
                     'per sample set specified by --arrangeSamples. Use N=label for a 1-based sample set.')
    add_y_axis_label(optional,
                     '--yAxisLabel', '-y',
                     nargs='+', default=[''],
                     help='Description for the Y-axis label. You can have multiple '
                     'Y-axis labels and they are assigned per sample set specified '
                     'by --arrangeSamples, e.g. --yAxisLabel RNA-seq RNA-seq ChIP-seq ChIP-seq. Use N=label for a 1-based sample set.')
    add_x_axis_visibility(shared,
                          '--xAxisVisibility', choices=['show_all', 'outer', 'outer_merged'],
                          default='outer',
                          help='If you have multiple rows of panels, this setting controls '
                          'whether identical X-axis labels should be shown for all rows. "outer" keeps one '
                          'label per column per compatible set of rows; "outer_merged" '
                          'is like "outer" but only one label is shown for all compatible '
                          'columns. Tick marks remain visible.')
    add_y_axis_visibility(optional,
                          '--yAxisVisibility',
                          choices=['show_all', 'outer', 'outer_merged'],
                          default='outer',
                          help='If you have multiple columns of panels, this setting controls '
                          'whether identical Y-axis labels should be shown for all columns. "outer" keeps one '
                          'label per row per compatible set of columns; "outer_merged" '
                          'is like "outer" but only one label is shown for all compatible '
                          'rows. This will also hide Y-axis values if panels in the affected '
                          'row have the same limits. Tick marks remain visible.')
    add_axis_visibility(shared)
    add_y_axis_limits(optional,
                      '--yAxisLimits',
                      choices=['common', 'per_panel', 'per_sample_set', 'per_y_label'],
                      default='per_y_label',
                      help='Determine Y limits globally, independently per panel, per '
                      'sample set, or per distinct Y-axis label.')
    add_y_min(optional, '--yMin',
              default=None,
              nargs='+',
              type=assigned_finite_float(allow_empty=True),
              help='Minimum value for the Y-axis. Multiple values, separated by '
              'spaces can be set for each profile. If the number of yMin values is smaller than'
              'the number of plots, the values are recycled. Use N=value for a 1-based sample set.')
    add_y_max(optional, '--yMax',
              default=None,
              nargs='+',
              type=assigned_finite_float(allow_empty=True),
              help='Maximum value for the Y-axis. Multiple values, separated by '
              'spaces can be set for each profile. If the number of yMin values is smaller than'
              'the number of plots, the values are recycled. Use N=value for a 1-based sample set.')


def add_axis_visibility(group):
    group.add_argument(
        '--axisVisibility',
        choices=['show_all', 'outer', 'outer_merged'],
        default=None,
        help='Set the X-axis and Y-axis visibility at once, superseding '
             'both separate visibility options.')


def _add_profile_labels(optional, *, common=None):
    shared = common or optional
    add_distance_and_landmark_options(shared)

    add_lines_at_tick_marks(shared, '--linesAtTickMarks',
                            help='Draw dashed vertical lines from every '
                            'X-axis tick mark across each profile panel, '
                            'matching the equivalent plotHeatmap option.',
                            action='store_true')

    add_regions_label(shared)
    add_show_region_counts(shared)
    add_samples_label(shared)

    add_grouping_options(shared)

    add_colors(optional, '--colorsPerSample', '--colors', '--plotColors',
               help='One colour per plotted sample line, in the '
               'order the lines appear (which follows '
               '--arrangeSamples). Colour names and html hex '
               'strings (e.g., #eeff22) are accepted, space '
               'separated, for example --colorsPerSample red '
               'blue green. A plain list must name exactly one '
               'colour per distinctly coloured line (with --plotType '
               'heatmap, one colour map per panel); a shorter list is an '
               'error, extra values are ignored with a warning. '
               'N=colour sets coloured series N in legend order; any '
               'subset may be set, e.g. '
               '--colorsPerSample 1,2=red 3-4=blue (unnamed lines take '
               'the default); plain entries mixed with N= entries fill '
               'the remaining series in order. '
               'If you also use --sameSampleLabels, note that it '
               'is resolved before colours are assigned, so you '
               'should not provide colours twice for samples that '
               'are to be collapsed.',
               nargs='+')

    add_legend_location(optional, '--legendLocation')


def add_grouping_options(optional):
    optional.add_argument(
        '--sameGroupLabels',
        choices=['independent', 'together', 'merge'],
        default='independent',
        help='How repeated region-group labels are handled. '
             '"independent" preserves deepTools behavior; "together" '
             'keeps separate series but gives them the same colour and, '
             'with --perGroup, puts them in one panel; "merge" '
             'concatenates their regions before calculating profiles.')
    optional.add_argument(
        '--sameSampleLabels',
        choices=['independent', 'together'], default='independent',
        help='How repeated sample labels are handled if they\'re in the same panel. '
             '"independent" keeps separate colours and legend entries; '
             '"together" gives equally named samples the same colour '
             'and combines visually identical legend entries. Samples belonging '
             'to separate sample sets are treated as independent regardless.')


def add_filter_nans(optional):
    optional.add_argument(
        '--filterNans',
        choices=list(FILTER_NAN_MODES),
        default='keep',
        help="Drop regions with missing data before clustering, "
             "sorting, or plotting. 'keep' (default) never filters on "
             "NaNs; 'any_bin' drops a region if any sample has any NaN "
             "bin; 'any_sample' drops a region if any sample is "
             "entirely NaN; 'all_bins' drops a region only if it is "
             "entirely NaN across every sample. This is the same "
             "missing-data filter as computeMatrixOperationsR "
             "filterValues --filterNans, applied here in-place so a "
             "separate filtering step isn't needed just to plot.")


def add_heatmap_dimensions(optional, aspect_spelling, *, include_width=True):
    optional.add_argument('--heatmapHeight',
                          help='Whole heatmap-stack height in cm; by default the '
                               'cell width divided by the aspect ratio (10 cm when '
                               'no size is set). Any two of cell width, height and '
                               'aspect ratio determine the third.',
                          type=positive_finite_float, default=None)
    if include_width:
        optional.add_argument('--heatmapWidth', dest='cellWidth',
                              help='Cell width in cm; default 5, and any two '
                                   'of width, heatmap height and aspect ratio '
                                   'determine the third.',
                              type=positive_finite_float, default=None)
    optional.add_argument(aspect_spelling, dest='heatmapAspectRatio',
                          metavar='ASPECTRATIO',
                          type=positive_finite_float,
                          default=None,
                          help='Cell width divided by whole heatmap-stack height; '
                               'default 0.5, and any two sizes determine the third. '
                               'The stack includes blocks and gaps, excluding '
                               'titles, axes, summary plots and colorbars.')


def add_region_label_location(optional):
    optional.add_argument(
        '--regionLabelLocation', choices=['left', 'right'],
        default='right',
        help='Place region-group labels to the left of the heatmap or '
             'to its right, inside any right-hand colorbar.')


def add_heatmap_y_axis_label(optional, *spellings):
    optional.add_argument(
        *spellings,
        dest='heatmapYAxisLabel',
        metavar='YAXISLABEL',
        nargs='+', default=[''],
        help='Description for the heatmap Y axis, evaluated once per '
             'sample set/column and recycled as necessary.')


def _add_heatmap_structure(heatmap, common):
    add_filter_nans(common)
    common.add_argument(
        '--whatToShow',
        help='The default is to include a summary or profile plot on top '
        'of the heatmap and a heatmap colorbar. Other options are: '
        '"heatmap and colorbar" and the default '
        '"plot, heatmap and colorbar".',
        choices=["plot, heatmap and colorbar",
                 "heatmap and colorbar"],
        default='plot, heatmap and colorbar')

    add_heatmap_dimensions(heatmap, '--aspectRatio')
    add_regions_label(common)
    add_region_label_location(heatmap)
    add_show_region_counts(common)
    add_samples_label(common)

    add_sample_set_options(common)
    add_grouping_options(common)

    add_x_axis_label(common, '--xAxisLabel', '-x', nargs='+',
                     default=None,
                     help='Description for the X-axis label under '
                          'the heatmap. By default it is derived '
                          'independently for each sample set.')
    add_heatmap_y_axis_label(heatmap, '--yAxisLabel', '-y')
    add_x_axis_visibility(common,
                          '--xAxisVisibility', choices=['show_all', 'outer', 'outer_merged'],
                          default='outer')
    add_axis_visibility(common)
    add_distance_and_landmark_options(common)


def add_minor_tick_marks(group):
    group.add_argument('--minorTickMarks',
                       help='Minor X-axis ticks: none disables them, '
                            'auto chooses natural spacing based on the '
                            'available width while preserving landmark '
                            'ticks, and an integer sets the total number '
                            'of minor ticks across the axis, distributed '
                            'proportionally between landmark ticks.',
                       type=minor_tick_setting,
                       default='none', metavar='none|auto|INT')


def add_heatmap_axis_extras(optional):
    optional.add_argument(
        '--colorbarLocation',
        choices=['best', 'right', 'right_common', 'bottom', 'below',
                 'below_common'],
        default='best',
        help='Place heatmap colorbars automatically ("best": below the '
             'heatmaps when there are several colour scales, otherwise to '
             'the right of each row), per row to the right, or below the '
             'heatmaps. "below" is an alias for "bottom" (one bar '
             'per column). "below_common" dedups identical colour scales and '
             'packs the distinct bars into centred rows below the heatmaps. '
             '"right_common" packs one grid of distinct bars to the right '
             'of the full figure.')
    optional.add_argument(
        '--colorbarLabels', nargs='+', default=None, metavar='LABEL',
        help='Titles drawn above each colorbar. "none" (default) draws no '
             'title; "auto" uses the sample label(s) of the heatmap(s) that '
             'colorbar serves (joined with newlines when it serves '
             'several); otherwise the given labels are used, one per '
             'colorbar and recycled as necessary. N=LABEL assigns one '
             'colorbar by its 1-based index. A right colorbar title is '
             'left-aligned to the colorbar; a below colorbar title is '
             'centred on its heatmap.')
    optional.add_argument(
        '--sortIndicator', choices=['auto', 'none'], default='auto',
        help='Draw a sorting-direction triangle when the effective region '
        'sort is ascending or descending. "none" disables it.')


def _add_heatmap_axes(heatmap, common):
    add_lines_at_tick_marks(common, '--linesAtTickMarks',
                            help='Draw dashed lines from all tick marks through the heatmap. '
                            'This is then similar to the dashed line draw at region bounds '
                            'when using a reference point and --sortUsing region_length',
                            action='store_true')
    add_minor_tick_marks(common)
    add_heatmap_axis_extras(heatmap)


def _add_heatmap_colors(optional, *, full_color_help):
    if full_color_help:
        from deeptoolsr.plotting import colormap_names
        color_options = "', '".join(
            x for x in colormap_names() if not x.endswith('_r'))
    else:
        color_options = ''

    optional.add_argument(
        '--colorMap',
        help='Color map to use for the heatmap. If more than one heatmap is being plotted the color '
             'of each heatmap can be enter individually (e.g. `--colorMap Reds Blues`). Color maps '
             'are recycled if the number of color maps is smaller than the number of heatmaps being '
             'plotted. Use N=name for a 1-based sample set. Available values can be seen here: http://matplotlib.org/users/colormaps.html '
             'The available options are: \'' + color_options + '\'',
        default=['RdYlBu'],
        nargs='+')

    optional.add_argument(
        '--alpha',
        default=1.0,
        type=check_float_0_1,
        help='The alpha channel (transparency) to use for the heatmaps. The default is 1.0 and values '
             'must be between 0 and 1.')

    optional.add_argument(
        '--colorList',
        help='List of colors to use to create a colormap. For example, if `--colorList black,yellow,blue` '
             'is set (colors separated by comas) then a color map that starts with black, continues to '
             'yellow and finishes in blue is created. If this option is selected, it overrides the --colorMap '
             'chosen. The list of valid color names can be seen here: '
             'http://matplotlib.org/examples/color/named_colors.html  '
             'Hex colors are valid (e.g #34a2b1). If individual colors for different heatmaps '
             'need to be specified they need to be separated by space as for example: '
             '`--colorList "white,#cccccc" "white,darkred"` '
             'As for --colorMap, the color lists are recycled if their number is smaller thatn the number of'
             'plotted heatmaps. Use N=colors for a 1-based sample set. '
             'The number of transitions is defined by the --colorNumber option.',
        type=check_list_of_comma_values,
        nargs='+')

    optional.add_argument(
        '--colorNumber',
        help='N.B., --colorList is required for an effect. This controls the '
        'number of transitions from one color to the other. If --colorNumber is '
        'the number of colors in --colorList then there will be no transitions '
        'between the colors.',
        type=positive_int,
        default=256)

    optional.add_argument(
        '--missingDataColor',
        default='black',
        help='If --missingDataAsZero was not set, such cases '
        'will be colored in black by default. Using this '
        'parameter, a different color can be set. A value '
        'between 0 and 1 will be used for a gray scale '
        '(black is 0). For a list of possible color '
        'names see: http://packages.python.org/ete2/'
        'reference/reference_svgcolors.html. '
        'Other colors can be specified using the #rrggbb '
        'notation.')

    optional.add_argument(
        '--boxAroundHeatmaps',
        help='By default black boxes are plot around heatmaps. This can be turned off '
             'by setting --boxAroundHeatmaps no.',
        default='yes')


def _add_heatmap_limits(optional):
    optional.add_argument('--zMin', '-min',
                          default=None,
                          help='Minimum value for the heatmap intensities. Multiple values, separated by '
                               'spaces can be set for each heatmap. If the number of zMin values is smaller than'
                               'the number of heatmaps the values are recycled. If a value is set to "auto", it will be set '
                               ' to the first percentile of the matrix values. Use N=value for a 1-based sample set.',
                          type=assigned_finite_float(keywords=('auto',)),
                          nargs='+')
    optional.add_argument('--zMax', '-max',
                          default=None,
                          help='Maximum value for the heatmap intensities. Multiple values, separated by '
                               'spaces can be set for each heatmap. If the number of zMax values is smaller than'
                               'the number of heatmaps the values are recycled. If a value is set to "auto", it will be set '
                               ' to the 98th percentile of the matrix values. Use N=value for a 1-based sample set.',
                          type=assigned_finite_float(keywords=('auto',)),
                          nargs='+')
    optional.add_argument('--zMid',
                          default=None,
                          help='Value corresponding to the middle of the heatmap color scale. Multiple '
                               'values can be supplied and are recycled across heatmaps like --zMin and '
                               '--zMax, and explicit per-heatmap assignment is supported (e.g. '
                               '--zMid 1,2=0.5 for 1-based sample sets); heatmaps not named get no midpoint shift. '
                               'Each midpoint must be strictly between its effective zMin and zMax.',
                          type=assigned_finite_float(allow_empty=True),
                          nargs='+')


def _add_heatmap_summary(optional):
    add_out_file_name_data(optional)
    add_profile_height(optional, '--heightSummaryPlot',
                       metavar='HEIGHTSUMMARYPLOT',
                       help='Summary-profile height in cm; by default the '
                            'cell width divided by the aspect ratio. Any two '
                            'of cell width, height and aspect ratio determine '
                            'the third.')
    add_plot_type(optional,
                  '--plotTypeSummaryPlot',
                  help='"lines" will plot the profile line based '
                  'on the average type selected. "fill" '
                  'fills the region between zero and the profile '
                  'curve. The fill in color is semi transparent to '
                  'distinguish different profiles. "se" and "std" '
                  'color the region between the profile and the '
                  'standard error or standard deviation of the data. Standard error '
                  'uses finite observations independently in each bin and is hidden '
                  'when fewer than two are available; this intentionally differs '
                  'from stock deepTools. "ci" shades the analytic confidence '
                  'interval of a mean or geom_mean and "bootstrap" a '
                  'percentile-bootstrap interval of any average type '
                  '(see --ci_level).',
                  choices=['lines', 'fill', 'se', 'std', 'ci', 'bootstrap'],
                  default='lines')

    add_average_type(optional,
                     '--averageTypeSummaryPlot',
                     default='mean',
                     choices=STATISTIC_CHOICES['profile'],
                     help='Define the type of statistic that should be plotted in the '
                     'summary image above the heatmap.')

    add_pseudocount(optional,
                    '--pseudocountSummaryPlot',
                    metavar='PSEUDOCOUNTSUMMARYPLOT',
                    type=finite_float, default=-1,
                    help='Pseudocount used by geom_mean in the summary plot. A '
                    'negative value selects half the smallest nonzero magnitude. '
                    'All-zero data resolves to zero; mixed signs within one bin '
                    'are invalid.')
    add_trim_perc(optional,
                  '--trimPercSummaryPlot',
                  metavar='TRIMPERCSUMMARYPLOT', type=bounded_finite_float(
                      0, 0.5, maximum_inclusive=False,
                      requirement='a finite fraction from 0 up to, but not including, 0.5'),
                  default=0.05,
                  help='Fraction removed from each tail independently in every '
                       'bin when using trim_mean in the summary plot.')

    add_profile_aspect_ratio(optional,
                             '--aspectRatioSummaryPlot',
                             metavar='ASPECTRATIOSUMMARYPLOT',
                             type=positive_finite_float, default=None,
                             help='Cell width divided by summary-profile height; '
                                  'default 1, and any two sizes determine the third.')

    add_y_axis_visibility(optional,
                          '--yAxisVisibilitySummaryPlot',
                          choices=['show_all', 'outer', 'outer_merged'],
                          default='outer')
    add_y_axis_limits(optional,
                      '--yAxisLimitsSummaryPlot',
                      choices=['common', 'per_panel', 'per_sample_set', 'per_y_label'],
                      default='per_y_label')
    add_y_min(optional, '--yMinSummaryPlot',
              metavar='YMINSUMMARYPLOT',
              default=None,
              nargs='+',
              type=assigned_finite_float(allow_empty=True),
              help='Minimum value for the Y-axis. Multiple values, separated by '
              'spaces can be set for each profile. If the number of yMin values is smaller than'
              'the number of plots, the values are recycled. Use N=value for a 1-based sample set.')
    add_y_max(optional, '--yMaxSummaryPlot',
              metavar='YMAXSUMMARYPLOT',
              default=None,
              nargs='+',
              type=assigned_finite_float(allow_empty=True),
              help='Maximum value for the Y-axis. Multiple values, separated by '
              'spaces can be set for each profile. If the number of yMin values is smaller than'
              'the number of plots, the values are recycled. Use N=value for a 1-based sample set.')
    add_y_axis_label(optional,
                     '--yAxisLabelSummaryPlot',
                     explicit=False,
                     metavar='YAXISLABELSUMMARYPLOT',
                     nargs='+', default=[''],
                     help='Y-axis label for the summary plot.')

    add_colors(optional,
               '--colorsSummaryPlot', '--plotColorsSummaryPlot', nargs='+',
               explicit=False,
               metavar='COLORSSUMMARYPLOT',
               help='Colours for summary-plot series, evaluated in first '
               'plotting order, exactly one per distinctly coloured '
               'series (a shorter list is an error, extra values are '
               'ignored with a warning; N=colour sets some, and plain '
               'entries mixed with N= fill the remaining series in '
               'order). The same sample/group series keeps its '
               'colour between summary panels.')

    add_legend_location(optional, '--legendLocationSummaryPlot')
    add_interval_options(optional, '--plotTypeSummaryPlot')


def add_shared_plot_tail(optional):
    optional.add_argument('--nanAfterEnd',
                          help=argparse.SUPPRESS,
                          default=False)

    optional.add_argument('--plotTitle', '-T',
                          help='Title of the plot, to be printed on top of '
                          'the generated image. Leave blank for no title.',
                          default='', type=decode_text_escapes)

    optional.add_argument('--perGroup',
                          help='The default is to plot all groups of regions by '
                          'sample. Using this option instead plots all samples by '
                          'group of regions. Note that this is only useful if you '
                          'have multiple groups of regions. by sample rather than '
                          'group.',
                          action='store_true')

    optional.add_argument('--plotFileFormat',
                          metavar='',
                          help='Image format type. If given, this '
                          'option overrides the '
                          'image format based on the plotFile ending. '
                          'The available options are: "png", '
                          '"eps", "pdf" and "svg"',
                          choices=['png', 'pdf', 'svg', 'eps'])

    optional.add_argument('--verbose',
                          help='If set, warning messages and '
                          'additional information are given.',
                          action='store_true')


def add_interpolation_method(group):
    group.add_argument('--interpolationMethod',
                       help='If the heatmap image contains a large number of columns '
                       'is usually better to use an interpolation method to produce '
                       'better results (see '
                       'https://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html). '
                       'By default, plotHeatmapR uses the method `nearest` if the number of columns is 1000 or '
                       'less. Otherwise it uses the bilinear method. This default behaviour can be changed by '
                       'using any of the following options: "nearest", "bilinear", "bicubic", '
                       '"gaussian"',
                       choices=['auto', 'nearest', 'bilinear', 'bicubic', 'gaussian'],
                       metavar='STR',
                       default='auto')


def add_silhouette(group):
    group.add_argument(
        '--silhouette',
        help='Compute the silhouette score for regions. This requires '
        '--kmeans or --hclust. The score measures how similar a region is '
        'to its cluster versus other clusters and is reported in the final '
        'column of the BED output. Evaluation can be very slow above '
        '100,000 regions.',
        action='store_true')


def plot_heatmap_options(parser, *, full_color_help):
    parser.set_defaults(show_profile=True, show_heatmap=True)
    required = parser.add_argument_group('Required arguments')
    add_required_plot_args(required)
    output = parser.add_argument_group('Output options')
    add_shared_output_options(output)
    add_interpolation_method(output)
    add_output_dpi(output)
    cluster = add_clustering(parser)
    add_silhouette(cluster)
    common = parser.add_argument_group('Common options')
    heatmap = parser.add_argument_group('Heatmap options')
    summary = parser.add_argument_group('Summary plot options')
    add_shared_plot_options(common, prog='plotHeatmapR')
    _add_heatmap_structure(heatmap, common)
    _add_heatmap_axes(heatmap, common)
    _add_heatmap_colors(heatmap, full_color_help=full_color_help)
    _add_heatmap_limits(heatmap)
    _add_heatmap_summary(summary)
    add_shared_plot_tail(common)


def plot_profile_options(parser):
    parser.set_defaults(show_profile=True, show_heatmap=False)
    required = parser.add_argument_group('Required arguments')
    add_required_plot_args(required)
    output = parser.add_argument_group('Output options')
    add_shared_output_options(output)
    add_out_file_name_data(output)
    add_output_dpi(output)
    add_silhouette(add_clustering(parser))
    optional = parser.add_argument_group('Optional arguments')
    add_shared_plot_options(optional, prog='plotProfileR')
    add_filter_nans(optional)
    _add_profile_statistics(optional)
    _add_profile_panels(optional)
    _add_profile_axes(optional)
    _add_profile_labels(optional)
    add_minor_tick_marks(optional)
    add_shared_plot_tail(optional)


def plot_matrix_options(parser, *, full_color_help):
    """Build the common and per-kind option groups directly."""
    parser.set_defaults(show_profile=False, show_heatmap=False)
    required = parser.add_argument_group('Required arguments')
    add_required_plot_args(required)
    output = parser.add_argument_group('Output options')
    add_shared_output_options(output)
    add_out_file_name_data(output)
    add_output_dpi(output)
    add_clustering(parser)
    common = parser.add_argument_group('Common options')
    profile = parser.add_argument_group('Profile options')
    heatmap = parser.add_argument_group('Heatmap options')
    add_shared_plot_options(common, prog='plotMatrixR')
    common.add_argument('--profile', dest='show_profile',
                        action='store_true', help='Draw profile panels.')
    common.add_argument('--heatmap', dest='show_heatmap',
                        action='store_true', help='Draw heatmap blocks.')
    common.add_argument('--cellWidth', dest='cellWidth',
                        type=positive_finite_float, default=None,
                        help='Cell width in cm; default 5, and any two of '
                             'width, a drawn kind\'s height and aspect ratio '
                             'determine the third.')
    add_filter_nans(common)
    _add_profile_panels(profile, matrix=True, common=common)
    _add_profile_axes(profile, common=common)
    _add_profile_labels(profile, common=common)
    add_shared_plot_tail(common)
    _add_profile_statistics(profile)
    add_silhouette(heatmap)
    add_heatmap_dimensions(heatmap, '--heatmapAspectRatio',
                           include_width=False)
    add_heatmap_y_axis_label(heatmap, '--heatmapYAxisLabel')
    add_region_label_location(heatmap)
    add_interpolation_method(heatmap)
    add_minor_tick_marks(common)
    add_heatmap_axis_extras(heatmap)
    _add_heatmap_colors(heatmap, full_color_help=full_color_help)
    _add_heatmap_limits(heatmap)


def plot_parser(tool, *, full_color_help=True):
    """Build a plot parser without importing either rendering module."""
    from deeptoolsr import options as run_options

    if tool == 'plotHeatmapR':
        description = ('This tool creates a heatmap for '
                       'scores associated with genomic regions. '
                       'The program requires a matrix file '
                       'generated by the tool ``computeMatrixR``.')
        epilog = 'An example usage is: plotHeatmapR -m matrix.gz'
    elif tool == 'plotProfileR':
        description = ('This tool creates a profile plot for '
                       'scores over sets of genomic regions. '
                       'Typically, these regions are genes, but '
                       'any other regions defined in BED '
                       ' will work. A matrix generated '
                       'by computeMatrixR is required.')
        epilog = 'An example usage is: plotProfileR -m matrix.gz'
    elif tool == 'plotMatrixR':
        description = ('Plot profiles, heatmaps or both from a matrix. '
                       'A cell holds a profile, a heatmap stack, or both; '
                       'sample sets and group arrangements place cells in '
                       'rows and columns. Heatmap colours and limits apply '
                       'per cell. Matching labels form runs across adjacent '
                       'cells; outer_merged shares labels only across cells '
                       'with the same panel kind, and behaves as outer when '
                       'a cell has both kinds.')
        epilog = 'An example usage is: plotMatrixR -m matrix.gz --profile -o plot.png'
    else:
        raise ValueError(f'unsupported plot tool: {tool}')
    parser = argparse.ArgumentParser(
        prog=tool,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description, epilog=epilog,
        usage=f'{tool} -m matrix.gz\nhelp: {tool} -h / {tool} --help',
        add_help=False)
    if tool == 'plotHeatmapR':
        plot_heatmap_options(parser, full_color_help=full_color_help)
    elif tool == 'plotProfileR':
        plot_profile_options(parser)
    else:
        plot_matrix_options(parser, full_color_help=full_color_help)
    run_options.add_config_option(parser)
    if tool == 'plotHeatmapR':
        add_plot_processor(parser)
    return apply_capability_tags(parser)


# def requiredLength(minL, maxL):
#     """
#     This is an optional action that can be given to argparse.add_argument(..., nargs='+')
#     to allow a specified numeric range of arguments (e.g., "only 1 or 2 arguments").

#     minL and maxL are the minimum and maximum length
#     """
#     # https://stackoverflow.com/questions/4194948/python-argparse-is-there-a-way-to-specify-a-range-in-nargs
#     class RequiredLength(argparse.Action):
#         def __call__(self, parser, args, values, option_string=None):
#             if not minL <= len(values) <= maxL:
#                 msg = 'argument "{}" requires between {} and {} arguments'.format(self.dest, minL, maxL)
#                 raise argparse.ArgumentTypeError(msg)
#             setattr(args, self.dest, values)
#     return RequiredLength
