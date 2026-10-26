"""Pure series, colour, and sample-set planning for both plot tools.

The active plotters remain responsible for Matplotlib drawing and physical
measurement.  This module only resolves sample-set metadata, compatible
matrix geometry, axis ownership, and grouped Y limits before rendering.
"""

import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable

from deeptoolsr.numeric_validation import (
    assignment_parts, decode_text_escapes, recycled_min_max_error)


@dataclass(frozen=True)
class DataSeries:
    key: str
    group: int
    sample: int
    label: str


@dataclass(frozen=True)
class ColorSlot:
    position: int
    series: tuple[str, ...]
    panels: tuple[int, ...]
    cells: tuple[int, ...] = ()  # cells whose heatmap stacks use the slot


@dataclass(frozen=True)
class ColorDomain:
    option: str
    kind: str
    slots: tuple[ColorSlot, ...]


@dataclass(frozen=True)
class Panel:
    index: int
    kind: str
    title: str
    export_label: str | None
    row: int
    column: int
    series: tuple[str, ...]


@dataclass(frozen=True)
class StackBlock:
    group: int
    sample: int


@dataclass(frozen=True)
class Cell:
    index: int
    row: int
    column: int
    samples: tuple[int, ...]
    groups: tuple[int, ...]
    title: str
    export_label: str | None
    profile: Panel | None
    blocks: tuple[StackBlock, ...]
    stack_labels: tuple[str, ...]
    sample_set: int = 0  # also numbers the cell's heatmap colour scale
    # Per block: the line after which the label's newline is only preferred
    # (between sample name and group), or None when every newline is hard.
    stack_label_breaks: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class SeriesPlan:
    panels: tuple[Panel, ...]
    row_labels: tuple[str, ...]
    series: tuple[DataSeries, ...]
    color_domains: tuple[ColorDomain, ...]
    cells: tuple[Cell, ...]


@dataclass(frozen=True)
class CellOptions:
    placement: str = 'overlay'
    per_group: bool = False
    groups_as_sample_sets: bool = False
    same_group_labels: str = 'independent'
    same_sample_labels: str = 'independent'
    explicit_sample_sets: bool = False
    per_group_titles: bool = False
    grid_columns: int | None = None
    grid_rows: int | None = None
    show_profile: bool = True
    show_heatmap: bool = False
    profile_style: str = 'lines'
    color_option: str = '--colorMap'
    colors_option: str = '--colorsPerSample'


@dataclass(frozen=True)
class CellRun:
    """Adjacent cells sharing one label value (before measurement)."""

    axis: int
    labels: object
    cells: tuple[int, ...]


@dataclass(frozen=True)
class LabelKind:
    edge: str
    vector: Callable[[Cell], object]


def effective_per_group(spec):
    """--perGroup lays cells out by region group unless sample sets are given."""
    return spec.per_group and spec.arrange_samples is None


def per_group_titles(spec):
    """--perGroup titles panels by region group alone unless several sample
    sets must be told apart."""
    return spec.per_group and (spec.arrange_samples is None or
                               len(spec.arrange_samples) == 1)


def effective_placement(spec):
    """--perGroup splits overlaid region groups into adjacent panels."""
    if spec.per_group and spec.placement == 'overlay':
        return 'adjacent'
    return spec.placement


def cell_options(spec, *, colors_option=None):
    """Project cell ownership from the shared figure destinations."""
    if colors_option is None:
        colors_option = spec.series_options.colors_option
    placement = effective_placement(spec)
    return CellOptions(
        placement, effective_per_group(spec),
        spec.groups_as_sample_sets and placement != 'overlay',
        spec.same_group_labels,
        spec.same_sample_labels, spec.arrange_samples is not None,
        per_group_titles(spec), spec.grid_columns, spec.grid_rows, spec.show_profile,
        spec.show_heatmap, spec.plot_type,
        '--colorList' if spec.color_list else '--colorMap', colors_option)


def _key(group, sample):
    return f'g{group + 1}:s{sample + 1}'


def _slots(assignments):
    """Make slots from (position, series key, panel index) in drawing order."""
    grouped = {}
    for position, key, panel in assignments:
        keys, panels = grouped.setdefault(position, ([], []))
        if key is not None and key not in keys:
            keys.append(key)
        if panel not in panels:
            panels.append(panel)
    return tuple(ColorSlot(position, tuple(keys), tuple(panels))
                 for position, (keys, panels) in sorted(grouped.items()))


def series_slot(plan, domain, panel_index, key=None):
    """Look up a one-based colour position for a panel/series occurrence."""
    for slot in plan.color_domains[domain].slots:
        if panel_index in slot.panels and (key is None or key in slot.series):
            return slot.position
    raise KeyError((domain, panel_index, key))


def _profile_entries(selected_groups, selected_samples, index, sid, labels,
                     opts, default_group_series, series, colors, assignments):
    group_keys = labels.group_keys
    panel_groups = {g for group in selected_groups for g in group}
    panel_samples = set(selected_samples)
    keys = []
    for groups in selected_groups:
        for group in groups:
            for sample in selected_samples:
                key = _key(group, sample)
                if default_group_series:
                    label = (labels.groups[group]
                             if opts.same_group_labels == 'independent'
                             else group_keys[group])
                elif len(selected_groups) == 1:
                    label = labels.samples[sample]
                elif len(panel_samples) == 1:
                    label = (labels.groups[group]
                             if opts.same_group_labels == 'independent'
                             else group_keys[group])
                else:
                    label = '{}: {}'.format(group_keys[group],
                                            labels.samples[sample])
                series.setdefault(key, DataSeries(key, group, sample, label))
                keys.append(key)
                groups_merged = opts.same_group_labels in ('together', 'merge')
                samples_merged = opts.same_sample_labels == 'together'
                group_identity = group_keys[group] if groups_merged else group
                sample_identity = (labels.samples[sample] if samples_merged
                                   else sample)
                # A series' colour names what its label names.
                if default_group_series or (len(panel_groups) > 1 and
                                            len(panel_samples) == 1):
                    identity, merged = ('group', group_identity), groups_merged
                elif len(panel_groups) > 1:
                    identity = ('group-sample', group_identity, sample_identity)
                    merged = groups_merged and samples_merged
                else:
                    identity, merged = ('sample', sample_identity), samples_merged
                # Promoted sample sets colour their series apart, except
                # series merged by label, which keep one colour figure-wide.
                if opts.groups_as_sample_sets and not merged:
                    identity += (sid,)
                position = colors.setdefault(identity, len(colors) + 1)
                assignments.append((position, key, index + 1))
    return tuple(keys)


def region_group_sets(labels, opts):
    """Region groups sharing a panel: equal labels with --sameGroupLabels together."""
    keys = labels.group_keys
    if opts.same_group_labels == 'together':
        return tuple(tuple(i for i, value in enumerate(keys) if value == key)
                     for key in dict.fromkeys(keys))
    return tuple((i,) for i in range(len(labels.groups)))


class _UnknownCount:
    """A region count describe cannot know; it sums and formats as ???."""

    def __add__(self, other):
        return self

    __radd__ = __add__

    def __format__(self, spec):
        return '???'


UNKNOWN_COUNT = _UnknownCount()


def group_title(groups, labels, sizes):
    key = labels.group_keys[groups[0]]
    if key and any('[n = ' in labels.groups[g] for g in groups):
        return '{} [n = {:,}]'.format(key, sum(sizes[g] for g in groups))
    return key


def panel_order(set_count, group_sets, placement):
    """(sample set, its group sets) of every panel, in drawing order."""
    if placement == 'overlay':
        return tuple((sid, group_sets) for sid in range(set_count))
    if placement in ('adjacent', 'by_row'):
        return tuple((sid, (groups,)) for sid in range(set_count)
                     for groups in group_sets)
    return tuple((sid, (groups,)) for groups in group_sets
                 for sid in range(set_count))


def _split_title(sample_label, group_label, opts):
    """Title of a split group panel outside by_row/by_column."""
    if opts.per_group_titles:
        return group_label
    return '{} — {}'.format(sample_label, group_label)


def promote_group_panels(sample_sets, labels, sizes, opts):
    """Make every split group panel its own sample set (--groupsAsSampleSets).

    Returns the panels' samples, region groups and default labels (the
    titles the panels have without promotion), in drawing order.
    """
    group_sets = region_group_sets(labels, opts)
    order = panel_order(len(sample_sets), group_sets, opts.placement)
    sets = tuple(sample_sets[sid] for sid, _ in order)
    groups = tuple(selected[0] for _, selected in order)
    if len(group_sets) == 1:
        return sets, groups, None
    titles = tuple(_split_title(', '.join(labels.samples[s] for s in samples),
                                group_title(members, labels, sizes), opts)
                   for samples, members in zip(sets, groups))
    return sets, groups, titles


def _profile_grid(samples, group_sets, opts, base_labels):
    if opts.groups_as_sample_sets:
        set_count = len(samples) // len(group_sets)
        descriptors = [(sid, plan.samples, (plan.groups,))
                       for sid, plan in enumerate(samples)]
        base_labels = base_labels[::len(group_sets)]
    else:
        set_count = len(samples)
        descriptors = [(sid, samples[sid].samples, groups) for sid, groups
                       in panel_order(set_count, group_sets, opts.placement)]
    count = len(descriptors)
    if opts.placement == 'by_row':
        columns = len(group_sets)
        row_labels = base_labels
    elif opts.placement == 'by_column':
        columns = set_count
        row_labels = ()
    elif opts.grid_columns is not None:
        if opts.grid_columns < 1:
            raise ValueError('--gridColumns must be greater than zero')
        columns = min(opts.grid_columns, count)
        row_labels = ()
    elif opts.grid_rows is not None:
        if opts.grid_rows < 1:
            raise ValueError('--gridRows must be greater than zero')
        rows = min(opts.grid_rows, count)
        columns = (count + rows - 1) // rows
        row_labels = ()
    else:
        columns, row_labels = count, ()
    return descriptors, columns, row_labels


def _profile_projection(sizes, samples, labels, opts):
    """Resolve profile panels and figure-wide colour identities from labels.

    Promoted sample sets (--groupsAsSampleSets) title their panels with
    their own labels; their export and row labels name their samples.
    """
    promoted = opts.groups_as_sample_sets
    base_labels = tuple(', '.join(labels.samples[s] for s in plan.samples)
                        if promoted else plan.subplot_label
                        for plan in samples)
    group_sets = region_group_sets(labels, opts)
    # Series are named by region group alone unless a panel mixes samples;
    # explicit one-sample sets (--arrangeSamples 1 2 3) match the default.
    default_group_series = (not opts.per_group and
                            opts.placement == 'overlay' and
                            all(len(plan.samples) == 1 for plan in samples))

    descriptors, columns, row_labels = _profile_grid(
        samples, group_sets, opts, base_labels)
    if opts.placement == 'by_column':
        row_labels = tuple(group_title(groups, labels, sizes)
                           for groups in group_sets)

    panels = []
    series = {}
    colors = {}
    assignments = []
    for index, (sid, selected_samples, selected_groups) in enumerate(descriptors):
        sample_label = base_labels[sid]
        group_label = group_title(selected_groups[0], labels, sizes)
        multiple = len(group_sets) > 1 and opts.placement != 'overlay'
        title = sample_label
        if promoted:
            title = samples[sid].subplot_label
        elif opts.placement == 'by_row':
            title = group_label
        elif multiple:
            title = (sample_label if opts.placement == 'by_column' and
                     not opts.per_group_titles else
                     _split_title(sample_label, group_label, opts))
        parts = [sample_label]
        if len(selected_groups) == 1 and not default_group_series:
            parts.append(group_label)
        parts = [part for i, part in enumerate(parts)
                 if part and part not in parts[:i]]
        export_label = ' — '.join(parts) or title
        if (not promoted and opts.placement in ('by_row', 'by_column') and
                index // columns > 0):
            title = ''
        keys = _profile_entries(
            selected_groups, selected_samples, index, sid, labels, opts,
            default_group_series, series, colors, assignments)
        panels.append(Panel(index + 1,
                            'heatmap' if opts.profile_style == 'heatmap' else 'profile',
                            title, export_label, index // columns,
                            index % columns, keys))
    if opts.profile_style == 'heatmap':
        domain = ColorDomain(opts.colors_option, 'colormap',
                             _slots((index + 1, key, index + 1)
                                    for index, panel in enumerate(panels)
                                    for key in panel.series))
    else:
        domain = ColorDomain(opts.colors_option, 'series', _slots(assignments))
    return (SeriesPlan(tuple(panels), tuple(row_labels),
                       tuple(series.values()), (domain,), ()), descriptors)


def _block_labels(blocks, labels):
    """Name each block by what distinguishes it from the cell's other blocks.

    Returns the texts and, per block, the line index of the preferred break
    between sample name and group (None when only one of them is shown).
    """
    samples = len({block.sample for block in blocks}) > 1
    groups = len({block.group for block in blocks}) > 1 or not samples
    texts = tuple('\n'.join(
        ((labels.samples[block.sample],) if samples else ()) +
        ((labels.groups[block.group],) if groups else ()))
        for block in blocks)
    breaks = tuple(labels.samples[block.sample].count('\n')
                   if samples and groups else None for block in blocks)
    return texts, breaks


def _cells_from_profile(plan, descriptors, labels, opts):
    """Build cells; a cell's colour map and limits follow its sample set."""
    cells = []
    for panel, (sid, samples, group_sets) in zip(plan.panels, descriptors):
        groups = tuple(g for members in group_sets for g in members)
        blocks = _stack_blocks(groups, samples, opts.per_group) if opts.show_heatmap else ()
        stack_labels, stack_label_breaks = _block_labels(blocks, labels)
        profile = panel if opts.show_profile else None
        if profile is not None and profile.kind == 'heatmap':
            profile = Panel(profile.index, 'series_heatmap', profile.title,
                            profile.export_label, profile.row, profile.column,
                            profile.series)
        cells.append(Cell(panel.index, panel.row, panel.column,
                          tuple(samples), groups, panel.title,
                          panel.export_label, profile, blocks, stack_labels,
                          sid, stack_label_breaks))
    return tuple(cells)


def _stack_blocks(groups, samples, per_group):
    pairs = ((g, s) for g in groups for s in samples) if per_group else (
        (g, s) for s in samples for g in groups)
    return tuple(StackBlock(g, s) for g, s in pairs)


def _stack_color_domain(cells, opts):
    """One colormap slot per sample set, naming the cells that use it."""
    sets = sorted({cell.sample_set for cell in cells})
    return ColorDomain(opts.color_option, 'colormap', tuple(
        ColorSlot(sid + 1, (), (), tuple(cell.index for cell in cells
                                         if cell.sample_set == sid))
        for sid in sets))


def resolve_cells(sizes, samples, labels, opts: CellOptions) -> SeriesPlan:
    """Resolve physical cells, series identities and colour domains."""
    profile_plan, descriptors = _profile_projection(
        sizes, samples, labels, opts)
    cells = _cells_from_profile(profile_plan, descriptors, labels, opts)
    if opts.show_heatmap:
        # Profiles are the profile plan's, unchanged, so a profile looks the
        # same with or without a heatmap stack under it.
        profiles = profile_plan if opts.show_profile else SeriesPlan(
            (), profile_plan.row_labels, (), (), ())
        return SeriesPlan(profiles.panels, profile_plan.row_labels,
                          profiles.series, profiles.color_domains +
                          (_stack_color_domain(cells, opts),), cells)
    panels = tuple(replace(cell.profile, kind='heatmap') if
                   cell.profile.kind == 'series_heatmap' else cell.profile
                   for cell in cells)
    return SeriesPlan(panels, profile_plan.row_labels,
                      profile_plan.series, profile_plan.color_domains, cells)


class AssignmentIndexError(ValueError):
    """One explicit assignment selected an unavailable 1-based index."""

    def __init__(self, spec, index, count):
        self.spec, self.index, self.count = spec, index, count
        super().__init__(f'Assignment index {index} is outside 1..{count}')


def parse_index_spec(spec, count):
    """Expand a 1-based INDEXSPEC (commas + ``a-b`` ranges) to 0-based indices.

    Ranges may run in either direction (``3-1`` == ``1-3``).  Every index must
    fall within ``1..count``.
    """
    indices = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            start_text, end_text = token.split('-', 1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            values = range(start, end + step, step)
        else:
            values = (int(token),)
        for value in values:
            if value < 1 or value > count:
                raise AssignmentIndexError(spec, value, count)
            indices.append(value - 1)
    return indices


def has_explicit_assignment(values):
    """True if any entry carries an ``INDEXSPEC=`` prefix."""
    if isinstance(values, str):
        values = (values,)
    return any(assignment_parts(value) is not None
               for value in (values or ()))


def _assignment_error(error, option, domain, context):
    """Raise, or let describe collect, one out-of-range ``N=`` assignment."""
    from deeptoolsr.prepare import OptionError, collected

    plural = {'series': 'coloured series'}.get(domain, domain + 's')
    message = (f'{option}: {error.spec} names {domain} {error.index}, '
               f'but there are only {error.count} {plural}')
    if context:
        message += f' ({context})'
    problem = OptionError(message, code='assignment_out_of_range',
                          option=option, target=domain, slot=error.index)
    if not collected(problem):
        raise problem from error


def resolve_recyclable(values, count, default=None, *, option='--value',
                       domain='item', context=None, text=False):
    """Resolve a recyclable option that may carry explicit ``N,M=`` assignments.

    Behaviour:

    * No ``INDEXSPEC=`` entry anywhere -> plain positional recycling over
      ``count`` (the historical behaviour).
    * Otherwise, each explicit entry assigns its value to the named 1-based
      positions; the remaining prefix-less entries form a positional pool that
      recycles across the *unassigned* positions only.  Positions left with no
      pool fall back to ``default``.

    ``default`` may be a callable ``index -> value`` or a plain value used for
    every unfilled position (``''`` when ``None``).
    """
    if isinstance(values, str):
        values = [values]
    values = list(values or [])

    def fallback(index):
        if callable(default):
            return default(index)
        return '' if default is None else default

    if not values:
        return [fallback(index) for index in range(count)]
    if not has_explicit_assignment(values):
        result = [values[index % len(values)] for index in range(count)]
        return ([decode_text_escapes(value) for value in result]
                if text else result)

    assigned = {}
    pool = []
    for entry in values:
        assignment = assignment_parts(entry)
        if assignment:
            try:
                indices = parse_index_spec(assignment[0], count)
            except AssignmentIndexError as error:
                _assignment_error(error, option, domain, context)
                continue  # collected: these slots keep their fallback
            for index in indices:
                assigned[index] = assignment[1]
        else:
            pool.append(entry)
    result = [None] * count
    remaining = [index for index in range(count) if index not in assigned]
    for position, index in enumerate(remaining):
        result[index] = (pool[position % len(pool)] if pool
                         else fallback(index))
    for index, value in assigned.items():
        result[index] = value
    return ([decode_text_escapes(value) for value in result]
            if text else result)


@dataclass(frozen=True)
class SampleGeometry:
    mode: str
    upstream: int
    downstream: int
    body: int
    unscaled_5_prime: int
    unscaled_3_prime: int
    bin_size: int
    bin_count: int
    reference_point: object = None

    def compatibility_key(self):
        if self.mode == 'reference-point':
            return (self.mode, self.reference_point, self.upstream,
                    self.downstream, self.bin_size, self.bin_count)
        return (self.mode, self.upstream, self.body, self.downstream,
                self.unscaled_5_prime, self.unscaled_3_prime,
                self.bin_size, self.bin_count)

    def summary(self):
        """This geometry in words, e.g. for a clash message."""
        if self.mode == 'reference-point':
            parts = [f'reference-point at {self.reference_point}',
                     f'{self.upstream} bp upstream',
                     f'{self.downstream} bp downstream']
        else:
            parts = ['scale-regions', f'{self.upstream} bp upstream',
                     f'{self.body} bp body',
                     f'{self.downstream} bp downstream']
            if self.unscaled_5_prime or self.unscaled_3_prime:
                parts += [f"{self.unscaled_5_prime} bp unscaled 5'",
                          f"{self.unscaled_3_prime} bp unscaled 3'"]
        return ', '.join(parts + [f'bin size {self.bin_size}'])


@dataclass(frozen=True)
class SampleSetPlan:
    index: int
    samples: tuple
    subplot_label: str
    x_axis_label: str
    y_axis_label: str
    geometry: SampleGeometry
    reference_label: object = None
    start_label: object = None
    end_label: object = None
    groups: tuple | None = None  # its one group panel (--groupsAsSampleSets)
    heatmap_y_axis_label: str = ''
    # Each text's value without its option (SAMPLE_SET_TEXTS keys).
    automatic: dict = field(default_factory=dict, compare=False)
    # True when its samples' X geometries differ (describe only; plotting
    # raises instead).  ``geometry`` is then its first sample's.
    geometry_clash: bool = False


# The per-sample-set texts, keyed as describe reports them.
SAMPLE_SET_TEXTS = ('subplot_title', 'x_axis_label', 'y_axis_label',
                    'heatmap_y_axis_label', 'reference_label', 'start_label',
                    'end_label')


SampleSetPlans = tuple[SampleSetPlan, ...]


# @dataclass(frozen=True)
# class AxisRun:
#     axis: str
#     panel_indices: tuple
#     label: str
#     merged: bool = False


def _parameter(parameters, name, sample, default=0):
    value = parameters.get(name, default)
    if isinstance(value, (list, tuple)):
        return value[sample]
    return value


def geometry_for_sample(sample_boundaries, parameters, sample):
    """Return the complete stored X geometry for one matrix sample."""
    body = int(_parameter(parameters, 'body', sample, 0) or 0)
    reference = _parameter(parameters, 'ref point', sample, None)
    mode = ('reference-point'
            if body == 0 and reference is not None else 'scale-regions')
    return SampleGeometry(
        mode=mode,
        upstream=int(_parameter(parameters, 'upstream', sample, 0) or 0),
        downstream=int(_parameter(parameters, 'downstream', sample, 0) or 0),
        body=body,
        unscaled_5_prime=int(_parameter(
            parameters, 'unscaled 5 prime', sample, 0) or 0),
        unscaled_3_prime=int(_parameter(
            parameters, 'unscaled 3 prime', sample, 0) or 0),
        bin_size=int(_parameter(parameters, 'bin size', sample, 1) or 1),
        bin_count=int(sample_boundaries[sample + 1] -
                      sample_boundaries[sample]),
        reference_point=reference)


def parse_sample_sets(sample_labels, specifications=None, per_group=False):
    """Resolve 1-based indices/names while preserving displayed order."""
    labels = list(sample_labels)
    if not specifications:
        specifications = ([','.join(str(index + 1)
                                    for index in range(len(labels)))]
                          if per_group else
                          [str(index + 1) for index in range(len(labels))])
    result = []
    used = set()
    for specification in specifications:
        samples = []
        for token in specification.split(','):
            token = token.strip()
            # ``a-b`` (both numeric) expands to an inclusive 1-based range;
            # names may themselves contain hyphens, so only all-digit endpoints
            # are treated as a range.
            range_match = re.match(r'^(\d+)\s*-\s*(\d+)$', token)
            if range_match:
                expanded = parse_index_spec(token, len(labels))
            elif token.isdigit():
                sample = int(token) - 1
                if sample < 0 or sample >= len(labels):
                    raise ValueError('Sample {} is outside 1..{}'.format(
                        token, len(labels)))
                expanded = [sample]
            else:
                if token not in labels:
                    raise ValueError("Unknown sample '{}'".format(token))
                expanded = [labels.index(token)]
            for sample in expanded:
                if sample in used:
                    raise ValueError("Sample '{}' occurs in more than one "
                                     'sample set'.format(labels[sample]))
                used.add(sample)
                samples.append(sample)
        if not samples:
            raise ValueError('A sample set cannot be empty')
        result.append(tuple(samples))
    return tuple(result)


def recycle(values, count, defaults=None, *, option='--value',
            domain='sample set', context=None, text=False):
    if isinstance(values, str):
        values = (values,)
    values = tuple(values or ())
    if values and has_explicit_assignment(values):
        # Explicit N,M= assignment; unfilled positions fall back to ``defaults``
        # (the value they would have received with no user input).
        default_fn = ((lambda index: defaults[index])
                      if defaults is not None else None)
        return tuple(resolve_recyclable(
            list(values), count, default=default_fn, option=option,
            domain=domain, context=context, text=text))
    if values:
        result = tuple(values[index % len(values)] for index in range(count))
        return (tuple(decode_text_escapes(value) for value in result)
                if text else result)
    if defaults is None:
        return ('',) * count
    defaults = tuple(defaults)
    if len(defaults) != count:
        raise ValueError('defaults must contain one item per sample set')
    return defaults


def resolve_assigned_limits(minimum, maximum, count, *, minimum_option,
                            maximum_option, context=None):
    """Resolve optional Y limits over 1-based sample-set positions."""
    lower = [None] if minimum is None else list(minimum)
    upper = [None] if maximum is None else list(maximum)
    if has_explicit_assignment(lower + upper):
        def resolve(values, option):
            result = resolve_recyclable(
                values, count, default=None, option=option,
                domain='sample set', context=context)
            return [None if value in ('', None) else float(value)
                    for value in result]

        lower = resolve(lower, minimum_option)
        upper = resolve(upper, maximum_option)
    error = recycled_min_max_error(
        lower, upper, count, minimum_option, maximum_option, strict=True)
    if error:
        from deeptoolsr.prepare import OptionError
        raise OptionError(error)
    return lower, upper


def _recycle_applicable(values, applicable, defaults, count, *, option):
    if isinstance(values, str):
        values = (values,)
    result = dict(defaults)
    values = tuple(values or ())
    if values:
        if has_explicit_assignment(values):
            resolved = resolve_recyclable(
                values, count, default=lambda index: defaults.get(index),
                option=option, domain='sample set', text=True)
            for sample_set in applicable:
                result[sample_set] = resolved[sample_set]
        else:
            for position, sample_set in enumerate(applicable):
                result[sample_set] = decode_text_escapes(
                    values[position % len(values)])
    return result


def build_sample_set_plans(sample_labels, sample_boundaries, parameters, sample_sets,
                           subplot_labels=None, x_axis_labels=None,
                           y_axis_labels=None, reference_labels=None,
                           start_labels=None, end_labels=None,
                           heatmap_y_axis_labels=None, *,
                           groups=None, default_labels=None,
                           option_spellings=None, context=None):
    """Resolve recycled labels and validate geometry within each sample set.

    ``groups`` gives promoted sample sets their region groups, and
    ``default_labels`` their default --sampleSetLabels.  Each plan also
    keeps the automatic value of every text (``SampleSetPlan.automatic``).
    """
    from deeptoolsr.prepare import OptionError, collected

    geometries = tuple(
        geometry_for_sample(sample_boundaries, parameters, sample)
        for sample in range(len(sample_labels)))
    set_geometries = []
    clashes = set()
    for set_index, samples in enumerate(sample_sets):
        first = geometries[samples[0]]
        incompatible = [sample for sample in samples[1:]
                        if geometries[sample].compatibility_key() !=
                        first.compatibility_key()]
        if incompatible:
            details = '; '.join(
                '{} is {}'.format(sample_labels[sample],
                                  geometries[sample].summary())
                for sample in (samples[0], *incompatible))
            problem = OptionError(
                f'Samples in sample set {set_index + 1} cannot share a panel '
                f'because their X-axis geometries differ: {details}.',
                prefix='Error: ', exit_code=1, code='geometry_clash',
                target='sample set', slot=set_index + 1,
                samples=[sample + 1 for sample in samples])
            if not collected(problem):
                raise problem
            clashes.add(set_index)
        set_geometries.append(first)

    count = len(sample_sets)
    spellings = option_spellings or {}

    def option(dest):
        return spellings.get(dest, '--' + dest)

    default_subplots = default_labels or tuple(
        ', '.join(sample_labels[sample] for sample in samples)
        for samples in sample_sets)
    resolved_subplots = recycle(
        subplot_labels, count, default_subplots,
        option=option('sampleSetLabels'), context=context, text=True)
    resolved_y = recycle(
        y_axis_labels, count, option=option('yAxisLabel'),
        context=context, text=True)
    resolved_heatmap_y = recycle(
        heatmap_y_axis_labels, count, option=option('heatmapYAxisLabel'),
        text=True)
    reference_sets = [index for index, geometry in enumerate(set_geometries)
                      if geometry.mode == 'reference-point']
    scaled_sets = [index for index, geometry in enumerate(set_geometries)
                   if geometry.mode == 'scale-regions']
    automatic_references = {
        index: set_geometries[index].reference_point or 'feature'
        for index in range(count)}
    references = _recycle_applicable(
        reference_labels, reference_sets, automatic_references, count,
        option=option('refPointLabel'))
    starts = _recycle_applicable(
        start_labels, scaled_sets, {index: 'TSS' for index in range(count)},
        count, option=option('startLabel'))
    ends = _recycle_applicable(
        end_labels, scaled_sets, {index: 'TES' for index in range(count)},
        count, option=option('endLabel'))
    # The automatic X label names the reference label in effect; an empty
    # one leaves plain "distance".  A given label, even an empty one, wins.
    automatic_xs = tuple(
        ('distance from {}'.format(references[index]) if references[index]
         else 'distance') if index in reference_sets else 'gene distance'
        for index in range(count))
    resolved_x = recycle(
        x_axis_labels, count, automatic_xs, option=option('xAxisLabel'),
        context=context, text=True)

    plans = []
    for index, samples in enumerate(sample_sets):
        geometry = set_geometries[index]
        reference = index in reference_sets
        automatic_x = automatic_xs[index]
        x_label = resolved_x[index]
        plans.append(SampleSetPlan(
            index=index, samples=tuple(samples),
            subplot_label=resolved_subplots[index],
            x_axis_label=x_label, y_axis_label=resolved_y[index],
            geometry=geometry,
            reference_label=(references[index]
                             if index in reference_sets else None),
            start_label=(starts[index] if index in scaled_sets else None),
            end_label=(ends[index] if index in scaled_sets else None),
            groups=groups[index] if groups else None,
            heatmap_y_axis_label=resolved_heatmap_y[index],
            automatic={
                'subplot_title': default_subplots[index],
                'x_axis_label': automatic_x, 'y_axis_label': '',
                'heatmap_y_axis_label': '',
                'reference_label': (automatic_references[index]
                                    if reference else None),
                'start_label': None if reference else 'TSS',
                'end_label': None if reference else 'TES'},
            geometry_clash=index in clashes))
    return tuple(plans)


def _padded_on_own_scale(axis, low, high):
    """``[low, high]`` with the axis' Y margin added on its own scale, as
    Matplotlib's autoscaling adds it."""
    transform = axis.yaxis.get_transform()
    lower, upper = transform.transform([low, high])
    pad = (upper - lower) * axis.margins()[1]
    return [float(value) for value in
            transform.inverted().transform([lower - pad, upper + pad])]


def y_limit_keys(panel_set_ids, panel_y_labels, mode='per_subplot'):
    """One key per profile panel; panels with equal keys share Y limits."""
    if mode == 'common':
        return [('common',)] * len(panel_set_ids)
    if mode == 'per_panel':
        return [('panel', index) for index in range(len(panel_set_ids))]
    if mode == 'per_y_label':
        return [('label', label) for label in panel_y_labels]
    return [('subplot', sample_set) for sample_set in panel_set_ids]


def grouped_y_limits(axes, keys, scales, panel_set_ids, y_min, y_max):
    """Automatic and drawn Y limits of autoscaled profile axes.

    A group whose panels share one scale (``scales``, one per axis) unites
    their autoscaled limits. A group mixing scales unites the raw data
    range instead, and each panel then adds its margin on its own scale:
    a margin padded on one scale is far too wide on another. Typed limits
    (``y_min``/``y_max``, recycled over sample sets) replace either bound.

    Returns ``(shared, automatic, drawn)``: ``shared`` maps each key to the
    limits its panels unite, or ``None`` for a group mixing scales;
    ``automatic`` and ``drawn`` hold one ``(low, high)`` per axis, without
    and with the typed limits.
    """
    y_min = [None] if y_min is None else list(y_min)
    y_max = [None] if y_max is None else list(y_max)
    mixed = {key for key in keys
             if len({scale for other, scale in zip(keys, scales)
                     if other == key}) > 1}
    ranges = {}
    for axis, key in zip(axes, keys):
        low, high = (axis.dataLim.intervaly if key in mixed
                     else axis.get_ylim())
        if key not in ranges:
            ranges[key] = [low, high]
        else:
            ranges[key][0] = min(ranges[key][0], low)
            ranges[key][1] = max(ranges[key][1], high)
    automatic, drawn = [], []
    for axis, key, sample_set in zip(axes, keys, panel_set_ids):
        limits = list(ranges[key])
        if key in mixed and all(map(math.isfinite, limits)):
            limits = _padded_on_own_scale(axis, *limits)
        elif key in mixed:
            limits = list(axis.get_ylim())
        automatic.append(_bounded(limits, None, None))
        drawn.append(_bounded(limits, y_min[sample_set % len(y_min)],
                              y_max[sample_set % len(y_max)]))
    shared = {key: None if key in mixed else tuple(limits)
              for key, limits in ranges.items()}
    return shared, tuple(automatic), tuple(drawn)


def _bounded(limits, low, high):
    """``limits`` with typed bounds; widened when the bounds meet.

    Explicit pairs were validated in resolve_assigned_limits; this only
    widens a data-derived bound that meets an explicit one (or a flat
    profile), as deepTools does.
    """
    low = limits[0] if low is None else float(low)
    high = limits[1] if high is None else float(high)
    return (low, low + 1) if low >= high else (low, high)


def apply_grouped_y_limits(axes, panel_set_ids, panel_y_labels,
                           y_min, y_max, mode='per_subplot', scales=None):
    """Apply Y limits by the requested semantic grouping (grouped_y_limits)."""
    axes = tuple(axes)
    if not axes:
        return ()
    scales = (None,) * len(axes) if scales is None else tuple(scales)
    _, _, drawn = grouped_y_limits(
        axes, y_limit_keys(panel_set_ids, panel_y_labels, mode), scales,
        panel_set_ids, y_min, y_max)
    for axis, limits in zip(axes, drawn):
        axis.set_ylim(limits)
        # draw a horizontal line through 0 when limits have opposite signs
        if (limits[0] <= 0 and limits[1] >= 0):
            axis.axhline(y=0, color="black", linewidth=0.8)
    return drawn


def contiguous_runs(items, key):
    """Return maximal contiguous runs of items sharing ``key(item)``."""
    runs = []
    current = []
    current_key = object()
    for item in items:
        item_key = key(item)
        if current and item_key != current_key:
            runs.append((current_key, tuple(current)))
            current = []
        current_key = item_key
        current.append(item)
    if current:
        runs.append((current_key, tuple(current)))
    return tuple(runs)


def label_runs(cells, kind):
    """Group adjacent equal vectors across rows or top/bottom columns."""
    if kind == 'stack':
        kind = LabelKind('right', lambda cell: cell.stack_labels)
    if kind.edge not in ('left', 'right', 'top', 'bottom'):
        raise ValueError('unknown label edge: {}'.format(kind.edge))
    columnwise = kind.edge in ('top', 'bottom')
    labels = kind.vector
    axes = sorted({cell.column if columnwise else cell.row for cell in cells})
    result = []
    for axis in axes:
        ordered = sorted((cell for cell in cells if
                          (cell.column if columnwise else cell.row) == axis),
                         key=lambda cell: cell.row if columnwise
                         else cell.column)
        contiguous = []
        for cell in ordered:
            position = cell.row if columnwise else cell.column
            if contiguous and position != (
                    contiguous[-1].row if columnwise else
                    contiguous[-1].column) + 1:
                result.extend(CellRun(axis, value,
                                      tuple(item.index for item in run))
                              for value, run in contiguous_runs(contiguous, labels))
                contiguous = []
            contiguous.append(cell)
        result.extend(CellRun(axis, value, tuple(item.index for item in run))
                      for value, run in contiguous_runs(contiguous, labels))
    return tuple(result)
