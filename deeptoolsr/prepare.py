"""Pure matrix preparation and per-request display-label replay."""

import argparse
from dataclasses import dataclass
import sys

import numpy as np

from deeptoolsr import _statistics
from deeptoolsr.matrix import (
    Labels, Matrix, PreparationWarning, RowLayout, apply_merge_relabel, cluster,
    filter_values, merge_groups_by_key, remove_empty_groups, silhouette, sort,
)
from deeptoolsr.numeric_validation import (
    CompatibilityError, CompatibilityRule, compatibility_warnings,
    decode_text_escapes, assignment_parts, option_was_supplied,
    recycled_min_max_error,
)


class DataError(ValueError):
    """An input error for the CLI or a future long-lived worker."""

    def __init__(self, message, *, parser_error=False, warnings=(), layout=None):
        super().__init__(message)
        self.parser_error = parser_error
        self.warnings = tuple(warnings)
        self.layout = layout


class OptionError(ValueError):
    """A plot argument error that never writes help or exits the process."""

    def __init__(self, message, *, parser_error=False, prefix='', exit_code=2):
        super().__init__(message)
        self.parser_error = parser_error
        self.prefix = prefix
        self.exit_code = exit_code


def option_error(error):
    """Every plot tool reports a request its plan or figure rejects alike."""
    return OptionError(str(error), prefix='Error: ', exit_code=1)


def command_labels(args, layout, spec, header):
    """Resolve the displayed group and sample labels of a parsed command."""
    return resolve_prepared_labels(
        layout, spec, regions_label=args.regionsLabel,
        samples_label=args.samplesLabel, show_counts=args.showRegionCounts,
        header=header,
        samples_label_option=dict(args._invoked_spellings).get(
            'samplesLabel', '--samplesLabel'))


def _configure_parse_mode(parser, raw, mode):
    if mode == 'cli':
        return
    for action in parser._actions:
        if action.dest in ('outFileName', 'outFileNameMatrix',
                           'outFileSortedRegions', 'outFileNameData'):
            action.required = False
        if isinstance(action, (argparse._HelpAction,
                               argparse._VersionAction)) and any(
                flag in raw for flag in action.option_strings):
            raise OptionError(f'{action.option_strings[0]} is unavailable '
                              f'in {mode} mode')

    def error(message):
        raise OptionError(message, parser_error=True)

    def parser_exit(_status=0, message=None):
        raise OptionError(message or 'parser exit')

    parser.error = error
    parser.exit = parser_exit


def _surface_parse_data_error(error, mode, tool, args, parser, emit):
    if mode != 'cli':
        if emit is not None:
            for warning in error.warnings:
                emit('stderr',
                     warning_text(warning, error.layout,
                                  regions_label=args.regionsLabel)
                     if tool == 'plotProfileR' and error.layout is not None
                     else warning)
        if error.parser_error:
            raise OptionError(str(error), parser_error=True) from error
        raise error
    for warning in error.warnings:
        message = (warning_text(warning, error.layout,
                                regions_label=args.regionsLabel)
                   if tool == 'plotProfileR' and error.layout is not None
                   else warning)
        (emit('stderr', message) if emit is not None else
         sys.stderr.write(message))
    if error.parser_error:
        parser.error(str(error))
    raise SystemExit(str(error)) from error


def parse_command(tool, argv=None, mode='cli', emit=None):
    """Parse and validate either plot command for CLI, metadata, or worker use."""
    if mode not in ('cli', 'describe', 'worker'):
        raise ValueError(f'unknown parse mode: {mode}')
    if tool not in ('plotHeatmapR', 'plotProfileR', 'plotMatrixR'):
        raise ValueError(f'unsupported plot tool: {tool}')
    from deeptoolsr import parserCommon
    from deeptoolsr.numeric_validation import specified_options

    parser = parserCommon.plot_parser(tool, full_color_help=mode == 'cli')
    raw = list(sys.argv[1:] if argv is None else argv)
    _configure_parse_mode(parser, raw, mode)
    args = parser.parse_args(raw)
    invoked = {}
    for token in raw:
        spelling = token.split('=', 1)[0]
        action = parser._option_string_actions.get(spelling)
        if action is not None:
            invoked[action.dest] = spelling
    args._invoked_spellings = tuple(sorted(invoked.items()))
    try:
        warnings = validate_plot(
            args, specified_options(raw),
            parserCommon.option_strings_by_dest(parser),
            heatmap=tool == 'plotHeatmapR', tool=tool)
    except DataError as error:
        _surface_parse_data_error(error, mode, tool, args, parser, emit)
    if mode == 'cli':
        for warning in warnings:
            (emit('stderr', warning) if emit is not None else
             sys.stderr.write(warning))
    elif emit is not None:
        for warning in warnings:
            emit('stderr', warning)
    return args


@dataclass(frozen=True)
class DataSpec:
    nan_mode: str
    cluster_k: int | None
    cluster_method: str | None
    cluster_columns: tuple[int, ...] | None
    cluster_seed: int
    ward_distance_budget_bytes: int
    silhouette: bool
    sort_method: str
    sort_using: str
    sort_columns: tuple[int, ...] | None
    sort_samples: tuple[int, ...] | None
    quantiles: int
    row_order_consumed: bool
    merge_keys: tuple[str, ...] | None

    def needs_values_for_membership(self):
        """Whether preparing this request can change groups or row counts."""
        return (self.nan_mode != 'keep' or self.cluster_k is not None or
                self.quantiles > 1 or self.merge_keys is not None)


@dataclass(frozen=True)
class PlotData:
    """The immutable matrix, row projection and labels consumed by a plot."""

    matrix: Matrix
    layout: RowLayout
    labels: Labels
    threads: int = 1
    distance_unit: str = 'auto'
    distance_unit_location: str = 'ticks'


def _sample_columns(samples, header, option):
    if samples is None:
        return None
    count = len(header.sample_labels)
    if any(index < 1 or index > count for index in samples):
        if option == '--clusterUsingSamples':
            raise DataError('--clusterUsingSamples indices must be between 1 '
                            f'and {count}')
        raise DataError(
            f'The value {list(samples)} for --sortUsingSamples is not valid. '
            f'Only values from 1 to {count} are allowed.')
    boundaries = header.sample_boundaries
    return tuple(column for index in samples for column in
                 range(boundaries[index - 1], boundaries[index]))


def _cluster_selection(args, header):
    # The old mains checked a supplied index even if clustering was disabled.
    columns = _sample_columns(args.clusterUsingSamples, header,
                              '--clusterUsingSamples')
    if args.kmeans is not None:
        return args.kmeans, 'kmeans', columns
    if args.hclust is not None:
        return args.hclust, 'hierarchical', columns
    return None, None, columns


def _common_spec(args, header):
    k, method, cluster_columns = _cluster_selection(args, header)
    # No/keep preserves input order, so both panel kinds share a DataSpec.
    # Keep the heatmap sample-index check that the old consumed path ran.
    consumed = (args.sortRegions not in ('no', 'keep') and (
        args.show_heatmap or
        (k is None and (args.quantileSortedRegions > 1 or
                        args.outFileSortedRegions is not None or
                        args.outFileNameMatrix is not None))))
    if (args.show_heatmap and args.sortRegions == 'keep' and
            args.sortUsingSamples is not None):
        _sample_columns(args.sortUsingSamples, header, '--sortUsingSamples')
    merge_keys = None
    if args.sameGroupLabels == 'merge':
        merge_keys = (tuple(args.regionsLabel) if args.regionsLabel else ())
    # The header's 'min/max threshold' record what computeMatrixR already
    # removed before --scale; deepTools 3.5.6 re-applied them to the stored
    # (scaled) values, which drops valid rows, so plotting deliberately
    # does not filter on them.
    return DataSpec(
        nan_mode=getattr(args, 'filterNans', 'keep'),
        cluster_k=k, cluster_method=method,
        cluster_columns=cluster_columns,
        cluster_seed=0,
        ward_distance_budget_bytes=getattr(
            getattr(args, 'run_options', None), 'ward_distance_budget_bytes',
            2 << 30),
        silhouette=getattr(args, 'silhouette', False),
        sort_method=args.sortRegions, sort_using=args.sortUsing,
        sort_columns=None,
        sort_samples=(tuple(args.sortUsingSamples)
                      if consumed and args.sortRegions != 'no' and
                      args.sortUsingSamples is not None else None),
        quantiles=args.quantileSortedRegions,
        row_order_consumed=consumed, merge_keys=merge_keys)


def heatmap_spec(args, header):
    """Project only heatmap options that affect row membership or order."""
    return _common_spec(args, header)


def profile_spec(args, header):
    """Project profile membership options and its output-dependent sort."""
    return _common_spec(args, header)


def _cluster_prepared(matrix, layout, spec, threads, warnings):
    if spec.cluster_k is None:
        return layout
    if spec.cluster_method == 'hierarchical':
        distance_bytes = layout.nrows * (layout.nrows - 1) // 2 * 8
        if distance_bytes > spec.ward_distance_budget_bytes:
            warnings.append(PreparationWarning(
                'ward_memory_saving',
                (distance_bytes, spec.ward_distance_budget_bytes)))
    layout, cluster_warning = cluster(
        matrix, layout, spec.cluster_k, method=spec.cluster_method,
        cols=spec.cluster_columns, seed=spec.cluster_seed,
        threads=threads,
        ward_distance_budget_bytes=spec.ward_distance_budget_bytes)
    if cluster_warning is not None:
        warnings.append(cluster_warning)
    return layout


def _silhouette_prepared(matrix, layout, spec, threads, warnings):
    if not spec.silhouette or spec.cluster_k is None:
        return layout
    layout = silhouette(matrix, layout, threads)
    if len(layout.origins) < 2:
        warnings.append(PreparationWarning('silhouette_undefined'))
    else:
        average = _statistics.reduce_axis(
            layout.silhouette.reshape(1, -1), 1, 'mean', threads)[0]
        warnings.append(PreparationWarning(
            'silhouette_average', (float(average),)))
    return layout


def prepare(matrix, spec, threads):
    """Apply the legacy operation order without mutating matrix values."""
    layout = RowLayout.identity(matrix)
    warnings = []
    if spec.nan_mode != 'keep':
        layout = filter_values(matrix, layout, None, None,
                               nan_mode=spec.nan_mode, threads=threads)
    if layout.nrows == 0:
        raise DataError('No regions remain after filtering; cannot plot an '
                        'empty matrix.')
    layout = _cluster_prepared(matrix, layout, spec, threads, warnings)
    if spec.merge_keys and len(spec.merge_keys) != len(layout.base_names):
        raise ValueError('length new labels != length original labels')
    layout, removed = remove_empty_groups(layout)
    if removed:
        warnings.append(PreparationWarning('empty_groups', (removed,)))
    sizes = np.diff(layout.group_bounds)
    small = np.flatnonzero(sizes / layout.nrows < 5.0 / 1000)
    if len(small):
        position = int(small[0])
        warnings.append(PreparationWarning(
            'small_group', (position, layout.origins[position])))
    if spec.merge_keys is not None:
        if spec.merge_keys:
            layout = apply_merge_relabel(layout, spec.merge_keys)
        else:
            keys = [layout.base_names[origin.source]
                    for origin in layout.origins]
            layout = merge_groups_by_key(layout, keys)
    if spec.row_order_consumed and spec.sort_method != 'no':
        try:
            columns = (_sample_columns(spec.sort_samples, matrix.header,
                                       '--sortUsingSamples')
                       if spec.sort_samples is not None else spec.sort_columns)
            layout = sort(matrix, layout, using=spec.sort_using,
                          method=spec.sort_method, cols=columns,
                          quantiles=spec.quantiles, threads=threads)
        except ValueError as error:
            raise DataError(str(error), warnings=warnings,
                            layout=layout) from error
    layout = _silhouette_prepared(matrix, layout, spec, threads, warnings)
    return layout, tuple(warnings)


def resolve_labels(layout, *, regions_label=None, samples_label=None,
                   show_counts=False, header,
                   samples_label_option='--samplesLabel'):
    """Replay labels at their original group positions without numeric work."""
    if regions_label and len(regions_label) != len(layout.base_names):
        raise ValueError('length new labels != length original labels')
    from deeptoolsr.plotting.series import (
        has_explicit_assignment, resolve_recyclable)
    explicit_samples = has_explicit_assignment(samples_label)
    if (samples_label and not explicit_samples and
            len(samples_label) != len(header.sample_labels)):
        raise ValueError('length new labels != length original labels')
    names = tuple(regions_label) if regions_label else layout.base_names
    keys = tuple(
        names[origin.source] + (f'_Q{origin.quantile}'
                                if origin.quantile is not None else '')
        for origin in layout.origins)
    groups = tuple(
        f'{key} [n = {layout.group_bounds[index + 1] - layout.group_bounds[index]:,}]'
        if show_counts and key else key
        for index, key in enumerate(keys))
    samples = (tuple(resolve_recyclable(
        samples_label, len(header.sample_labels),
        default=lambda index: header.sample_labels[index],
        option=samples_label_option, domain='sample', text=True))
        if explicit_samples else
        tuple(decode_text_escapes(value) for value in samples_label)
        if samples_label else header.sample_labels)
    return Labels(keys, groups, samples)


def resolve_prepared_labels(layout, spec, *, regions_label=None,
                            samples_label=None, show_counts=False, header,
                            samples_label_option='--samplesLabel'):
    """The one merge-aware route for CLI callers to resolve display labels."""
    return resolve_labels(
        layout, regions_label=None if spec.merge_keys is not None
        else regions_label, samples_label=samples_label,
        show_counts=show_counts, header=header,
        samples_label_option=samples_label_option)


def warning_text(warning, layout, *, regions_label=None):
    """Format a preparation warning at the CLI boundary."""
    kind, data = warning.kind, warning.data
    if kind == 'empty_groups':
        return 'WARNING: omitting {} empty region groups.\n'.format(*data)
    if kind == 'unpopulated_clusters':
        return ('WARNING: requested {} clusters, but only {} populated '
                'clusters could be formed.\n').format(*data)
    if kind == 'ward_memory_saving':
        distance_bytes, budget = data
        gib = float(1 << 30)
        return ('Hierarchical clustering uses the memory-saving method '
                '(distance table would need {:.3f} GiB > budget {:.3f} '
                'GiB); this is slower.\n').format(
                    distance_bytes / gib, budget / gib)
    if kind == 'small_group':
        _, origin = data
        names = tuple(regions_label) if regions_label else layout.base_names
        name = names[origin.source]
        return ("WARNING: Group '{}' is too small for plotting, you might "
                'want to remove it. \n').format(name)
    if kind == 'silhouette_undefined':
        return ('Silhouette scores are undefined with fewer than two '
                'populated clusters; writing NaN.\n')
    if kind == 'silhouette_average':
        return 'The average silhouette score is: {}\n'.format(*data)
    raise ValueError(f'unknown preparation warning: {kind}')


def _compatibility(args, supplied, rules):
    records = (CompatibilityRule(*rule) for rule in rules)
    try:
        return compatibility_warnings(args, supplied, records)
    except CompatibilityError as error:
        raise DataError(str(error), parser_error=True,
                        warnings=error.warnings) from error


def _parser_check(condition, message, warnings):
    if condition:
        raise DataError(message, parser_error=True, warnings=warnings)


def _spelling(spellings, dest):
    return next((option for option in spellings[dest]
                 if option == '--' + dest), spellings[dest][0])


def _supplied(supplied, spellings, dest):
    return option_was_supplied(supplied, *spellings[dest])


def _plot_rules(args, supplied, spellings, *, heatmap):
    def name(dest):
        return _spelling(spellings, dest)

    def present(dest):
        return _supplied(supplied, spellings, dest)
    rules = [
        (lambda ns, _: ns.quantileSortedRegions > 1 and
         (ns.kmeans is not None or ns.hclust is not None),
         '--quantiles cannot be combined with --kmeans or --hclust', 'error'),
    ]
    rules.append((lambda ns, _: getattr(ns, 'silhouette', False) and
                  ns.kmeans is None and ns.hclust is None,
                  '--silhouette requires --kmeans or --hclust', 'error'))
    # plotHeatmapR without a summary plot reports these below instead.
    intervals = not heatmap or args.show_profile
    rules.extend((
        (lambda ns, _: intervals and ns.plotType != 'bootstrap' and
         present('bootstrapReplicates'),
         f'{name("bootstrapReplicates")} is unused unless '
         f'{name("plotType")} bootstrap', 'warning'),
        (lambda ns, _: intervals and ns.plotType not in ('ci', 'bootstrap') and
         present('ci_level'),
         f'{name("ci_level")} is unused unless '
         f'{name("plotType")} ci or bootstrap', 'warning'),
    ))
    show_profile = args.show_profile
    rules.extend((
        (lambda ns, _: show_profile and ns.averageType != 'geom_mean' and
         present('pseudocount'),
         f'{name("pseudocount")} is unused unless '
         f'{name("averageType")} geom_mean', 'warning'),
        (lambda ns, _: show_profile and ns.averageType != 'trim_mean' and
         present('trim_perc'),
         f'{name("trim_perc")} is unused unless '
         f'{name("averageType")} trim_mean', 'warning'),
    ))
    if heatmap:
        rules.extend((
            (lambda ns, _: ns.colorList is None and present('colorNumber'),
             '--colorNumber is unused without --colorList', 'warning'),
            (lambda ns, _: not show_profile and any(present(dest) for dest in
             ('plotType', 'averageType', 'pseudocount', 'trim_perc',
              'ci_level', 'bootstrapReplicates')),
             'summary-plot statistic options are unused because --whatToShow '
             'does not include a summary plot', 'warning'),
        ))
    return rules


def _normalise_y_limits(args, warnings, spellings):
    def normalise(values):
        return [value if assignment_parts(value) else
                None if value in ('', None) else float(value)
                for value in (values or [None])]

    minimum, maximum = normalise(args.yMin), normalise(args.yMax)
    # Reject the pairs the lists spell out before any matrix I/O; every pair
    # the real sample-set count uses is checked in resolve_assigned_limits.
    if not any(assignment_parts(value) for value in minimum + maximum):
        error = recycled_min_max_error(
            minimum, maximum, max(len(minimum), len(maximum)),
            _spelling(spellings, 'yMin'), _spelling(spellings, 'yMax'),
            strict=True)
        _parser_check(error is not None, error, warnings)
    args.yMin, args.yMax = minimum, maximum


def _validate_heatmap_values(args, supplied, warnings, spellings):
    def name(dest):
        return _spelling(spellings, dest)
    _parser_check(not 0 <= args.trim_perc < 0.5,
                  f'{name("trim_perc")} must be at least 0 and less than 0.5',
                  warnings)
    if _supplied(supplied, spellings, 'missingDataColor'):
        from matplotlib.colors import is_color_like
        if not is_color_like(args.missingDataColor):
            raise DataError(
                'The value {}  for --missingDataColor is not valid'
                .format(args.missingDataColor), warnings=warnings)
    args.boxAroundHeatmaps = args.boxAroundHeatmaps == 'yes'


def _validate_profile_values(args, warnings, spellings):
    def name(dest):
        return _spelling(spellings, dest)
    _parser_check(not 0 < args.ci_level < 1,
                  f'{name("ci_level")} must be more than 0 and less than 1',
                  warnings)
    _parser_check(args.bootstrapReplicates < 1,
                  f'{name("bootstrapReplicates")} must be at least 1', warnings)
    _parser_check(args.plotType in ('se', 'std', 'ci') and
                  args.averageType not in ('mean', 'geom_mean'),
                  f'{name("plotType")} {args.plotType} pairs a spread with '
                  f'the centre, which is only meaningful for '
                  f'{name("averageType")} mean or geom_mean; use '
                  f'{name("plotType")} bootstrap to show a confidence '
                  f'interval for {args.averageType}', warnings)


def validate_plot(args, supplied, spellings, *, heatmap, tool=None):
    """Validate shared destinations and format errors with tool spellings."""
    if tool == 'plotMatrixR':
        if not args.show_profile and not args.show_heatmap:
            raise DataError('choose --profile, --heatmap or both',
                            parser_error=True)
    else:
        args.show_heatmap = heatmap
        args.show_profile = (args.whatToShow ==
                             'plot, heatmap and colorbar' if heatmap else True)
    warnings = _compatibility(
        args, supplied, _plot_rules(args, supplied, spellings,
                                    heatmap=heatmap))
    from deeptoolsr.plotting.sizes import SizeError, solve_sizes
    size_names = {
        dest: next((option for option in spellings[dest]
                    if option in supplied), _spelling(spellings, dest))
        for dest in ('cellWidth', 'profileHeight', 'profileAspectRatio',
                     'heatmapHeight', 'heatmapAspectRatio')
        if dest in spellings}
    try:
        solve_sizes(
            cell_width=args.cellWidth,
            profile_height=getattr(args, 'profileHeight', None),
            profile_aspect_ratio=args.profileAspectRatio,
            heatmap_height=getattr(args, 'heatmapHeight', None),
            heatmap_aspect_ratio=getattr(args, 'heatmapAspectRatio', None),
            show_profile=args.show_profile, show_heatmap=args.show_heatmap,
            names=size_names)
    except SizeError as error:
        raise DataError(str(error), parser_error=True,
                        warnings=warnings) from error
    _parser_check(not 0 <= args.trim_perc < 0.5,
                  f'{_spelling(spellings, "trim_perc")} must be at least 0 '
                  'and less than 0.5', warnings)
    _parser_check(args.quantileSortedRegions > 1 and
                  args.sortRegions not in ('ascend', 'descend'),
                  '--quantiles requires --sortRegions ascend or descend',
                  warnings)
    _parser_check(args.profileAspectRatio is not None and
                  args.profileAspectRatio <= 0,
                  f'{_spelling(spellings, "profileAspectRatio")} must be '
                  'greater than zero', warnings)
    _normalise_y_limits(args, warnings, spellings)
    if args.show_heatmap:
        _validate_heatmap_values(args, supplied, warnings, spellings)
    if not heatmap or args.show_profile:
        _validate_profile_values(args, warnings, spellings)
    if args.axisVisibility is not None:
        args.xAxisVisibility = args.axisVisibility
        args.yAxisVisibility = args.axisVisibility
    return warnings
