"""Pure matrix preparation and per-request display-label replay."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import re
import sys

import numpy as np

from deeptoolsr import _statistics
from deeptoolsr.matrix import (
    GroupOrigin, Labels, Matrix, PreparationWarning, RowLayout,
    apply_merge_relabel, check_quantiles, cluster, cluster_names,
    filter_values, merge_groups_by_key, quantile_groups, quantile_names,
    remove_empty_groups, silhouette, sort, take_groups,
)
from deeptoolsr.numeric_validation import (
    CompatibilityError, CompatibilityRule, compatibility_warnings,
    decode_text_escapes, assignment_parts, option_was_supplied,
    recycled_min_max_error,
)


class DataError(ValueError):
    """An input error for the CLI or a future long-lived worker."""

    def __init__(self, message, *, parser_error=False, warnings=(), layout=None,
                 option=None):
        super().__init__(message)
        self.parser_error = parser_error
        self.warnings = tuple(warnings)
        self.layout = layout
        self.option = option


class OptionError(ValueError):
    """A plot argument error that never writes help or exits the process.

    ``code``, ``option``, ``target``, ``slot`` (1-based) and ``samples``
    (1-based) locate the problem; ``describe`` reports them as data.
    """

    def __init__(self, message, *, parser_error=False, prefix='', exit_code=2,
                 code=None, option=None, target=None, slot=None, samples=None):
        super().__init__(message)
        self.parser_error = parser_error
        self.prefix = prefix
        self.exit_code = exit_code
        self.code = code
        self.option = option
        self.target = target
        self.slot = slot
        self.samples = None if samples is None else list(samples)

    def problem(self, severity='error'):
        """This error as one JSON-ready ``describe`` problem."""
        return {'code': self.code or 'option_error', 'severity': severity,
                'option': self.option, 'target': self.target,
                'slot': self.slot, 'samples': self.samples,
                'message': str(self)}


_COLLECTED = ContextVar('collected_problems', default=None)


@contextmanager
def collecting_problems():
    """Collect recoverable plan errors instead of raising them (describe)."""
    found = []
    token = _COLLECTED.set(found)
    try:
        yield found
    finally:
        _COLLECTED.reset(token)


def collected(error):
    """Keep ``error`` when describe collects problems; else the caller raises.

    A recoverable call site does ``if not collected(error): raise error``
    and then falls back to the automatic value.
    """
    found = _COLLECTED.get()
    if found is None:
        return False
    found.append(error)
    return True


def option_error(error):
    """Every plot tool reports a request its plan or figure rejects alike."""
    return OptionError(str(error), prefix='Error: ', exit_code=1)


def command_labels(args, layout, spec, header, *, counts=None):
    """Resolve the displayed group and sample labels of a parsed command.

    ``counts`` replaces the layout's group sizes in --showRegionCounts
    (describe passes ``UNKNOWN_COUNT`` ones when sizes are unknown).
    """
    return resolve_prepared_labels(
        layout, spec, regions_label=args.regionsLabel,
        samples_label=args.samplesLabel, show_counts=args.showRegionCounts,
        counts=counts,
        header=header,
        samples_label_option=dict(args._invoked_spellings).get(
            'samplesLabel', '--samplesLabel'),
        regions_label_option=dict(args._invoked_spellings).get(
            'regionsLabel', '--regionsLabel'))


def _configure_parse_mode(parser, mode):
    if mode == 'cli':
        return
    for action in parser._actions:
        if action.dest in ('outFileName', 'outFileNameMatrix',
                           'outFileSortedRegions', 'outFileNameData'):
            action.required = False

    def unavailable(option):
        raise OptionError(f'{option} is unavailable in {mode} mode')

    def print_message(message, file=None):
        # --version prints to stdout; argparse's own warnings go to stderr.
        if file is None or file is sys.stdout:
            unavailable('--version')
        original_print_message(message, file)

    original_print_message = parser._print_message
    parser.print_help = lambda file=None: unavailable('--help')
    parser._print_message = print_message

    def error(message):
        # argparse names the option as "argument --name/-n: ...".
        named = re.match(r'argument ([^:\s]+):', message)
        raise OptionError(message, parser_error=True,
                          option=named and named.group(1).split('/')[0])

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
            raise OptionError(str(error), parser_error=True,
                              option=error.option) from error
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
    _configure_parse_mode(parser, mode)
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
    # The 0-based matrix groups clustering or quantiles pool; others drop.
    group_selection: tuple[int, ...] | None
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
    # Group numbers; with --quantiles, a quantile name stays a string until
    # arrange_groups knows which group survived to name the quantiles.
    group_order: tuple[int | str, ...] | None
    # --regionsLabel, only to resolve those quantile names (so relabelling
    # alone keeps the layout).
    region_labels: tuple[str, ...] | None

    def needs_values_for_membership(self):
        """Whether preparing this request can change groups or row counts."""
        return (self.nan_mode != 'keep' or self.cluster_k is not None or
                self.quantiles > 1 or self.merge_keys is not None)

    def counts_from_header(self):
        """Whether the header's row counts give every group's size.

        Merging and quantile pooling only regroup rows; filtering and
        clustering decide them from values.
        """
        return self.nan_mode == 'keep' and self.cluster_k is None


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


_GROUP_TOKEN = re.compile(r'\d+\s*-\s*\d+|\d+')


def _listed_groups(tokens, names, option, noun, deferred=lambda token: False):
    """The 1-based groups that numbers, a-b ranges or ``names`` select.

    A token ``deferred`` accepts stays a string, resolved after preparation.
    """
    from deeptoolsr.plotting.series import (
        AssignmentIndexError, parse_index_spec)
    listed = []
    for token in tokens:
        if deferred(token):
            numbers = [token]
        elif _GROUP_TOKEN.fullmatch(token):
            try:
                numbers = [i + 1 for i in parse_index_spec(token, len(names))]
            except AssignmentIndexError as error:
                raise option_error(ValueError(
                    f'{option} {token}: there is no group {error.index}, '
                    f'only {error.count} {noun}')) from error
        elif token in names:
            numbers = [i + 1 for i, name in enumerate(names) if name == token]
        else:
            raise option_error(ValueError(
                f"{option}: unknown group '{token}'; the {noun} are "
                + ', '.join(f"'{name}'" for name in dict.fromkeys(names))))
        for number in numbers:
            if number in listed:
                raise option_error(ValueError(
                    f'{option} lists group {number} more than once'))
            listed.append(number)
    return listed


def _group_selection(args, header, k):
    """--clusterUsingGroups as 0-based matrix groups, or None when unused."""
    if args.clusterUsingGroups is None or (
            k is None and args.quantileSortedRegions == 1):
        return None
    option = dict(args._invoked_spellings).get('clusterUsingGroups',
                                               '--clusterUsingGroups')
    return tuple(sorted(number - 1 for number in _listed_groups(
        args.clusterUsingGroups, header.group_labels, option, 'groups')))


def _quantile_base(header, selection):
    """The group quantiles are named after: the first pooled one with rows."""
    sizes = np.diff(header.group_boundaries)
    sources = range(len(sizes)) if selection is None else selection
    return header.group_labels[next(
        (source for source in sources if sizes[source]), sources[0])]


def _defined_groups(args, header, k, selection):
    """Labels of the groups --arrangeGroups numbers, before any can vanish.

    The step that defines the final groups numbers them: clustering (rank),
    quantiles (position), the merge (first appearance) or the matrix.
    --regionsLabel names the groups that step defines.
    """
    labels = tuple(args.regionsLabel or ())
    quantiles = args.quantileSortedRegions
    if k is not None or quantiles > 1:
        defaults = (cluster_names(k) if k is not None else quantile_names(
            _quantile_base(header, selection), quantiles))
        names = (_quiet_region_names(labels, defaults) or defaults
                 if any(map(assignment_parts, labels)) else
                 tuple(decode_text_escapes(labels[i]) if i < len(labels)
                       else defaults[i] for i in range(len(defaults))))
    else:
        names = (_quiet_region_names(labels, header.group_labels) or
                 header.group_labels)
    if args.sameGroupLabels == 'merge':
        names = tuple(dict.fromkeys(names))
    return names


def _group_order(args, header, k, selection):
    """Resolve --arrangeGroups to defined group numbers in drawing order."""
    if args.arrangeGroups is None:
        return None
    quantiles = args.quantileSortedRegions
    names = _defined_groups(args, header, k, selection)
    noun = ('clusters' if k is not None else 'quantiles'
            if quantiles > 1 else 'groups')
    for token in args.arrangeGroups:
        if ',' in token:
            raise option_error(ValueError(
                f"--arrangeGroups takes one group per value, not '{token}'; "
                'to draw groups together give them equal labels and use '
                '--sameGroupLabels together'))
    listed = _listed_groups(
        args.arrangeGroups, names, '--arrangeGroups', noun,
        lambda token: quantiles > 1 and token not in names and
        _quantile_number(token, quantiles) is not None)
    if args.sameGroupLabels != 'together':
        return tuple(listed)
    order = []
    for number in listed:
        if isinstance(number, str):
            order.append(number)
            continue
        label = names[number - 1]
        members = [i + 1 for i, name in enumerate(names) if name == label]
        missing = [member for member in members if member not in listed]
        if missing:
            raise option_error(ValueError(
                f"--arrangeGroups with --sameGroupLabels together must list "
                f"every group labelled '{label}' or none; missing "
                + ', '.join(map(str, missing))))
        # An equal-label set is drawn at its first listed member.
        order += [member for member in listed if member in members and
                  member not in order]
    return tuple(order)


def _common_spec(args, header):
    k, method, cluster_columns = _cluster_selection(args, header)
    selection = _group_selection(args, header, k)
    # No/keep preserves input order, so both panel kinds share a DataSpec.
    # Keep the heatmap sample-index check that the old consumed path ran.
    consumed = (args.sortRegions not in ('no', 'keep') and (
        args.show_heatmap or
        (k is None and (args.quantileSortedRegions > 1 or
                        args.outFileSortedRegions is not None or
                        args.outFileSortedRegionsPerGroup is not None or
                        args.outFileNameMatrix is not None))))
    if (args.show_heatmap and args.sortRegions == 'keep' and
            args.sortUsingSamples is not None):
        _sample_columns(args.sortUsingSamples, header, '--sortUsingSamples')
    order = _group_order(args, header, k, selection)
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
        cluster_columns=cluster_columns, group_selection=selection,
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
        row_order_consumed=consumed, merge_keys=merge_keys,
        group_order=order,
        region_labels=(tuple(args.regionsLabel or ())
                       if any(isinstance(token, str) for token in order or ())
                       else None))


def _quantile_number(token, quantiles):
    """q of a '<group>_Q<q>' token within 1..quantiles, else None."""
    match = re.fullmatch(r'.+_Q(\d+)', token)
    q = int(match.group(1)) if match else 0
    return q if 1 <= q <= quantiles else None


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


def _sorted(matrix, layout, spec, threads, warnings):
    if not spec.row_order_consumed or spec.sort_method == 'no':
        return layout
    try:
        columns = (_sample_columns(spec.sort_samples, matrix.header,
                                   '--sortUsingSamples')
                   if spec.sort_samples is not None else spec.sort_columns)
        return sort(matrix, layout, using=spec.sort_using,
                    method=spec.sort_method, cols=columns,
                    quantiles=spec.quantiles, threads=threads)
    except ValueError as error:
        raise DataError(str(error), warnings=warnings,
                        layout=layout) from error


def _selected(layout, spec):
    """Drop the matrix groups --clusterUsingGroups leaves out."""
    if spec.group_selection is None:
        return layout
    return take_groups(layout, [
        position for position, origin in enumerate(layout.origins)
        if origin.source in spec.group_selection])


def prepare(matrix, spec, threads):
    """Apply the legacy operation order without mutating matrix values.

    Filter, select groups, define the groups (cluster, or cut quantiles
    from the sorted rows), merge equal labels, sort within groups, then
    --arrangeGroups.
    """
    layout = RowLayout.identity(matrix)
    warnings = []
    if spec.nan_mode != 'keep':
        layout = filter_values(matrix, layout, None, None,
                               nan_mode=spec.nan_mode, threads=threads)
    layout = _selected(layout, spec)
    _check_rows(layout)
    layout = _cluster_prepared(matrix, layout, spec, threads, warnings)
    layout = _drop_empty(layout, warnings)
    # The sort cuts quantiles, so it runs before their merge; other groups
    # are merged first and sorted as a whole.
    if spec.quantiles > 1:
        layout = _sorted(matrix, layout, spec, threads, warnings)
    layout = _regroup(layout, spec, warnings, sized=True)
    if spec.quantiles == 1:
        layout = _sorted(matrix, layout, spec, threads, warnings)
    layout = _silhouette_prepared(matrix, layout, spec, threads, warnings)
    layout, arranged = arrange_groups(layout, spec)
    return layout, tuple(warnings) + arranged


def predict_layout(header, spec):
    """The groups ``prepare`` forms, from the header alone (no values read).

    Follows ``prepare`` step by step: clusters are named and numbered as
    clustering names them, the merge keys and quantile pooling are the
    run's own, and --arrangeGroups selects as it does. Only sizes are
    unknown after filtering or clustering (``counts_from_header``): each
    cluster gets an equal share of the header's rows and filtering removes
    none, so the bounds are placeholders there. A group that the run finds empty (and drops) is
    still predicted.
    """
    layout = _selected(RowLayout.from_header(header), spec)
    warnings = []
    _check_rows(layout)
    if spec.cluster_k is not None:
        rows, k = layout.nrows, spec.cluster_k
        layout = RowLayout(
            None, tuple(int(i * rows / k) for i in range(k + 1)),
            cluster_names(k), tuple(GroupOrigin(i) for i in range(k)),
            layout.sort and layout.sort.reordered())
    layout = _drop_empty(layout, warnings)
    if spec.sort_samples is not None:
        _sample_columns(spec.sort_samples, header, '--sortUsingSamples')
    if spec.quantiles > 1:
        try:
            check_quantiles(spec.quantiles, spec.sort_method)
            layout = quantile_groups(layout, spec.quantiles)
        except ValueError as error:
            raise DataError(str(error), warnings=warnings,
                            layout=layout) from error
    layout = _regroup(layout, spec, warnings,
                      sized=spec.counts_from_header())
    layout, arranged = arrange_groups(layout, spec)
    return layout, tuple(warnings) + arranged


def _check_rows(layout):
    if layout.nrows == 0:
        raise DataError('No regions remain after filtering; cannot plot an '
                        'empty matrix.')


def _drop_empty(layout, warnings):
    layout, removed = remove_empty_groups(layout)
    if removed:
        warnings.append(PreparationWarning('empty_groups', (removed,)))
    return layout


def _regroup(layout, spec, warnings, *, sized):
    """Check small groups and merge equal labels, as ``prepare`` does.

    ``sized`` says the bounds are real, so the small-group check applies.
    """
    merge_keys = spec.merge_keys
    if merge_keys:
        merge_keys = region_names(merge_keys, layout.base_names) or ()
    sizes = np.diff(layout.group_bounds)
    small = np.flatnonzero(sizes / layout.nrows < 5.0 / 1000) if sized else ()
    if len(small):
        position = int(small[0])
        warnings.append(PreparationWarning(
            'small_group', (position, layout.origins[position])))
    if merge_keys is not None:
        merged = (apply_merge_relabel(layout, merge_keys) if merge_keys else
                  merge_groups_by_key(layout, [layout.base_names[origin.source]
                                               for origin in layout.origins]))
        # Merged quantiles are runs of whole quantiles, so still sorted.
        layout = (replace(merged, sort=layout.sort) if spec.quantiles > 1
                  else merged)
    return layout


def group_numbers(layout, spec):
    """Each drawn group's number as its defining step produced it."""
    merged = (tuple(dict.fromkeys(layout.base_names))
              if spec.merge_keys is not None else None)
    return tuple(
        merged.index(layout.base_names[origin.source]) + 1 if merged else
        origin.source + 1
        for origin in layout.origins)


def _resolve_quantile_name(token, layout, spec):
    """A quantile name's number; the quantiles are named after the group
    the sort took them from (the first one with regions)."""
    names = list(dict.fromkeys(layout.base_names)
                 if spec.merge_keys is not None else
                 _quiet_region_names(spec.region_labels, layout.base_names) or
                 layout.base_names)
    if token not in names:
        raise DataError(f"--arrangeGroups: unknown group '{token}'; the "
                        'quantiles are ' + ', '.join(f"'{name}'"
                                                     for name in names),
                        layout=layout)
    return names.index(token) + 1


def arrange_groups(layout, spec):
    """Keep and order the groups --arrangeGroups lists (one place for all)."""
    if spec.group_order is None:
        return layout, ()
    numbers = group_numbers(layout, spec)
    order = tuple(_resolve_quantile_name(token, layout, spec)
                  if isinstance(token, str) else token
                  for token in spec.group_order)
    if len(set(order)) < len(order):
        raise DataError('--arrangeGroups lists a quantile more than once',
                        layout=layout)
    drawn = tuple(n for n in order if n in numbers)
    missing = tuple(n for n in order if n not in numbers)
    warnings = ((PreparationWarning('arranged_groups_missing', (
        missing, drawn, len(numbers), spec.cluster_k)),) if missing else ())
    if not drawn:
        raise DataError('--arrangeGroups: none of the listed groups has '
                        'regions; nothing to draw.', warnings=warnings,
                        layout=layout)
    positions = [numbers.index(n) for n in drawn]
    if positions == list(range(len(numbers))):
        return layout, warnings
    return take_groups(layout, positions), warnings


def _label_count_collected(labels, count, option, target):
    """Raise, or let describe collect, a label list of the wrong length.

    Returns True when collected (the labels then stay automatic) and False
    when the count matches.
    """
    if len(labels) == count:
        return False
    problem = OptionError(
        f'{option}: {len(labels)} label{"s" * (len(labels) != 1)} given for '
        f'{count} {target}{"s" * (count != 1)}', code='label_count',
        option=option, target=target)
    if not collected(problem):
        raise problem
    return True


def region_names(labels, names, option='--regionsLabel'):
    """The region group names ``labels`` (--regionsLabel) give, or None.

    ``names`` are the groups' own names: the matrix groups (or clusters)
    before a merge, quantiles or --arrangeGroups regroup or reorder them;
    a plain list and ``N=`` both count these. A plain list must name every group (no recycling); ``N=``
    entries rename single groups and the others keep their own names;
    plain entries mixed with them must name every unassigned group, in
    order. A wrong count or an out-of-range ``N=`` raises, or describe
    collects it and those groups keep their own names (None when nothing
    applies).
    """
    if not labels:
        return None
    from deeptoolsr.plotting.series import (
        AssignmentIndexError, _assignment_error, parse_index_spec)
    result = list(names)
    assigned = set()
    plain = []
    for label in labels:
        parts = assignment_parts(label)
        if parts is None:
            plain.append(label)
            continue
        try:
            indices = parse_index_spec(parts[0], len(names))
        except AssignmentIndexError as error:
            _assignment_error(error, option, 'region group', None)
            continue
        for index in indices:
            result[index] = decode_text_escapes(parts[1])
        assigned.update(indices)
    free = [index for index in range(len(names)) if index not in assigned]
    if plain and _label_count_collected(plain, len(free), option,
                                        'region group'):
        plain = ()
    for index, label in zip(free, plain):
        result[index] = decode_text_escapes(label)
    return tuple(result) if plain or assigned else None


def _quiet_region_names(labels, names):
    """``region_names`` for a step before the labels are checked."""
    with collecting_problems():
        return region_names(labels, names)


def resolve_labels(layout, *, regions_label=None, samples_label=None,
                   show_counts=False, counts=None, header,
                   samples_label_option='--samplesLabel',
                   regions_label_option='--regionsLabel'):
    """Replay labels at their original group positions without numeric work.

    A --regionsLabel count that differs from the region groups raises, or
    describe collects it and the groups keep their own names.
    """
    from deeptoolsr.plotting.series import (
        has_explicit_assignment, resolve_recyclable)
    names = (region_names(regions_label, layout.base_names,
                          regions_label_option) or layout.base_names)
    explicit_samples = has_explicit_assignment(samples_label)
    if (samples_label and not explicit_samples and _label_count_collected(
            samples_label, len(header.sample_labels), samples_label_option,
            'sample')):
        samples_label = None
    keys = tuple(names[origin.source] for origin in layout.origins)
    if counts is None:
        counts = np.diff(layout.group_bounds).tolist()
    groups = tuple(
        f'{key} [n = {counts[index]:,}]' if show_counts and key else key
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
                            samples_label=None, show_counts=False,
                            counts=None, header,
                            samples_label_option='--samplesLabel',
                            regions_label_option='--regionsLabel'):
    """The one merge-aware route for CLI callers to resolve display labels."""
    return resolve_labels(
        layout, regions_label=None if spec.merge_keys is not None
        else regions_label, samples_label=samples_label,
        show_counts=show_counts, counts=counts, header=header,
        samples_label_option=samples_label_option,
        regions_label_option=regions_label_option)


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
        # Labels the plot rejects leave the group its own name, as
        # resolve_labels shows it.
        names = (_quiet_region_names(regions_label, layout.base_names) or
                 layout.base_names)
        name = names[origin.source]
        return ("WARNING: Group '{}' is too small for plotting, you might "
                'want to remove it. \n').format(name)
    if kind == 'arranged_groups_missing':
        missing, drawn, formed, k = data
        listed = ', '.join(map(str, missing))
        reason = (f'{"was" if len(missing) == 1 else "were"} not formed '
                  f'({formed} of {k} clusters)' if k is not None else
                  f'{"has" if len(missing) == 1 else "have"} no regions')
        return (f'WARNING: --arrangeGroups: group{"s" * (len(missing) > 1)} '
                f'{listed} {reason}'
                + (f'; drawing {", ".join(map(str, drawn))}' if drawn else '')
                + '.\n')
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
    rules.append((
        lambda ns, _: present('clusterUsingGroups') and ns.kmeans is None and
        ns.hclust is None and ns.quantileSortedRegions == 1,
        f'{name("clusterUsingGroups")} is unused without --kmeans, --hclust '
        'or --quantileSortedRegions; to draw only some groups use '
        '--arrangeGroups', 'warning'))
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
    rules.append((
        lambda ns, _: ns.groupsAsSampleSets and not ns.perGroup and
        ns.sampleSetGroupArrangement == 'overlay',
        f'{name("groupsAsSampleSets")} has no effect because region groups '
        f'are drawn in the same cell ({name("sampleSetGroupArrangement")} '
        f'overlay); use adjacent, end, by_row, by_column or '
        f'{name("perGroup")} to split them', 'warning'))
    rules.append((
        lambda ns, _: ns.perGroup and not ns.groupsAsSampleSets and
        present('sampleSetLabels') and
        (ns.arrangeSamples is None or len(ns.arrangeSamples) == 1),
        f'{name("sampleSetLabels")} is unused with {name("perGroup")}: '
        'panels are titled by region group', 'warning'))
    pseudolog = any((assignment_parts(value) or (None, value))[1] ==
                    'pseudolog' for value in args.yScale)
    rules.extend((
        (lambda ns, _: not pseudolog and present('yPseudologSigma'),
         f'{name("yPseudologSigma")} is unused unless '
         f'{name("yScale")} pseudolog', 'warning'),
        (lambda ns, _: ns.plotType == 'heatmap' and present('yScale'),
         f'{name("yScale")} is unused with {name("plotType")} heatmap',
         'warning'),
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
    if 'colorbarLocation' in spellings:
        rules.append((
            lambda ns, _: ns.colorbarLocation == 'none' and
            present('colorbarLabels'),
            f'{name("colorbarLabels")} is unused with '
            f'{name("colorbarLocation")} none', 'warning'))
    if heatmap:
        rules.extend((
            (lambda ns, _: ns.colorList is None and present('colorNumber'),
             '--colorNumber is unused without --colorList', 'warning'),
            (lambda ns, _: not show_profile and any(present(dest) for dest in
             ('plotType', 'averageType', 'pseudocount', 'trim_perc',
              'ci_level', 'bootstrapReplicates')),
             'summary-plot statistic options are unused because '
             '--showSummaryPlot off does not include a summary plot', 'warning'),
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
        args.show_profile = (args.showSummaryPlot == 'on' if heatmap
                             else True)
    warnings = _compatibility(
        args, supplied, _plot_rules(args, supplied, spellings,
                                    heatmap=heatmap))
    from deeptoolsr.plotting.sizes import (
        SIZE_OPTIONS, SizeError, solve_sizes)
    size_names = {
        dest: next((option for option in spellings[dest]
                    if option in supplied), _spelling(spellings, dest))
        for dest in SIZE_OPTIONS if dest in spellings}
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
        raise DataError(str(error), parser_error=True, warnings=warnings,
                        option=error.option) from error
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
