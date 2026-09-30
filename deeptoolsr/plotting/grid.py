"""Matplotlib-free physical layout for the unified cell grid."""

import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Tuple

from .geometry import (DEFAULT_GEOMETRY, DEFAULT_LABEL_LAYOUT,
                       GeometrySpec, Insets, LabelLayoutSpec, Rect, Size)
from .series import StackBlock
from .text_layout import (LabelCandidate, SharedGapBandSpec, SharedGapLayout,
                          optimize_shared_gap_layout)


_EMPTY_LABEL = LabelCandidate(('',), 0.0, 0.0, 1, 0.0, 0.0)


@dataclass(frozen=True)
class MeasuredLabelSpec:
    """A label's literal choice and optional measured wrap alternatives."""

    candidates: Tuple[LabelCandidate, ...]

    def __post_init__(self):
        if not self.candidates:
            raise ValueError('a measured label needs a candidate')


def _expand(rect, insets):
    return Rect(rect.x0 - insets.left, rect.y0 - insets.bottom,
                rect.width + insets.left + insets.right,
                rect.height + insets.bottom + insets.top)


def _move(rect, dx, dy):
    return (None if rect is None else
            Rect(rect.x + dx, rect.y + dy, rect.width, rect.height))


def _fit_canvas(rects, padding):
    """Return a tight padded canvas and the translation for its rectangles."""
    visible = tuple(rect for rect in rects
                    if rect is not None and rect.width > 0 and rect.height > 0)
    x0 = min(rect.x0 for rect in visible)
    y0 = min(rect.y0 for rect in visible)
    x1 = max(rect.x1 for rect in visible)
    y1 = max(rect.y1 for rect in visible)
    return (Size(x1 - x0 + 2 * padding, y1 - y0 + 2 * padding),
            padding - x0, padding - y0)


@dataclass(frozen=True)
class BelowColorbarEntry:
    """One below_common colorbar's measured decoration extents (points)."""
    left_overhang: float = 0.0
    right_overhang: float = 0.0
    tick_height: float = 0.0
    label_width: float = 0.0
    label_height: float = 0.0


def pack_colorbar_grid(count, block_height, base_x, top_y,
                       thickness=None,
                       min_height=None,
                       row_gap=None,
                       column_gap=None,
                       tick_label_widths=(),
                       label_band_heights=(),
                       label_widths=(),
                       geometry=DEFAULT_GEOMETRY):
    """Pack ``count`` colorbars into a top-aligned grid beside a heatmap block.

    Colorbars fill the grid **column-major** -- down the first column, then the
    next column to the right -- so a short last column stays aligned to the top
    of the grid.  Every box shares one height, chosen so that as many rows as
    possible fit within ``block_height`` while respecting ``min_height``:

        rows_max   = floor((block_height + row_gap) / (min_height + band + row_gap))
        columns    = ceil(count / rows_max)
        rows       = ceil(count / columns)          # balance the columns
        box_height = (block_height - rows*band - (rows-1)*row_gap) / rows

    ``band`` is the vertical space a colorbar title occupies above its box
    (title height + label gap); it is 0 when titles are off (the PR-A default).
    ``tick_label_widths`` gives each colorbar's rendered tick-label width so the
    per-column advance clears the widest label in that column; ``column_gap`` is
    then measured from that widest label to the next colorbar's border.
    ``label_widths`` gives each colorbar title's width (titles are left-aligned
    to the colorbar's left edge, so a title wider than the bar+tick labels
    widens its column too).

    Returns ``(rects, total_width)`` where ``rects`` is one ``Rect`` per colorbar
    (the rectangle only, tick labels excluded) in input order, and
    ``total_width`` spans from the first colorbar's left edge to the right edge
    of the last column's widest decoration.
    """
    # Unspecified gaps come from the resolved geometry spec (its defaults equal
    # the historical constants, so a default spec reproduces the prior layout).
    if thickness is None:
        thickness = geometry.heatmap_colorbar_thickness
    if min_height is None:
        min_height = geometry.colorbar_min_height
    if row_gap is None:
        row_gap = geometry.colorbar_short_edge_gap
    if column_gap is None:
        column_gap = geometry.colorbar_long_edge_gap
    count = max(0, int(count))
    if count == 0:
        return (), 0.0
    widths = list(tick_label_widths) + [0.0] * max(
        0, count - len(tick_label_widths))
    bands = list(label_band_heights) + [0.0] * max(
        0, count - len(label_band_heights))
    titles = list(label_widths) + [0.0] * max(0, count - len(label_widths))
    # Reserve the tallest requested title band for every row so each box clears.
    band = max(bands[:count]) if bands else 0.0
    box_cost = min_height + band + row_gap
    rows_max = max(1, int((block_height + row_gap) // box_cost))
    columns = int(math.ceil(count / rows_max))
    rows = int(math.ceil(count / columns))
    box_height = (block_height - rows * band - row_gap * (rows - 1)) / rows
    rects = [None] * count
    x = base_x
    total_width = 0.0
    for column in range(columns):
        members = [index for index in range(count) if index // rows == column]
        column_tick = max((widths[index] for index in members), default=0.0)
        column_title = max((titles[index] for index in members), default=0.0)
        for position, index in enumerate(members):
            box_top = top_y - band - position * (box_height + band + row_gap)
            rects[index] = Rect(x, box_top - box_height, thickness, box_height)
        # The column must clear whichever is wider: bar + tick labels, or the
        # left-aligned title spilling to the right of the bar.
        advance = max(thickness + column_tick, column_title)
        total_width += advance
        x += advance + column_gap
        if column < columns - 1:
            total_width += column_gap
    return tuple(rects), total_width


def _below_common_footprint(entry, bar_width, label_width=0.0):
    title_overhang = max(0.0, (label_width - bar_width) / 2.0)
    left = max(entry.left_overhang, title_overhang)
    right = max(entry.right_overhang, title_overhang)
    return left, right, bar_width + left + right


def _below_common_row_advance(width, footprint, gap, limit, has_items):
    """Return the next greedy row width and whether this bar starts a row."""
    trial = width + gap + footprint if has_items else footprint
    return (footprint, True) if has_items and trial > limit else (trial, False)


def pack_below_common(entries, bar_width, thickness, block_width,
                      origin_x, top_y,
                      horizontal_gap=None,
                      vertical_gap=None,
                      label_gap=None,
                      geometry=DEFAULT_GEOMETRY,
                      include_label_widths=False):
    """Pack below colorbars into independently-centred rows (common-legend style).

    Unlike the side grid this is **not** a rigid grid.  Distinct bars (each one
    heatmap-width wide and un-stretched) are laid left-to-right into rows; when
    the next bar's footprint would overflow ``block_width`` a new row starts, and
    each row is then centred within ``block_width`` on its own -- like an
    above/below common legend.  Rows therefore may hold different numbers of bars
    because tick-label widths differ.

    The gaps are swapped relative to the side grid: the horizontal gap between
    bars in a row is the side grid's *row* gap, and the vertical gap between rows
    is the side grid's *column* gap (both 8 pt today).

    ``entries`` is a sequence with ``.left_overhang`` / ``.right_overhang`` (how
    far the tick *labels* actually protrude past the bar's left/right border --
    like a heatmap's end x labels, a tucked end label protrudes ~0 and so costs
    nothing, while a centred or mitigation-pushed one protrudes and is counted),
    ``.tick_height`` (label height below the bar), and ``.label_width`` /
    ``.label_height`` (0 = no title). When ``include_label_widths`` is enabled,
    a title wider than its bar contributes its actual centered left/right
    protrusion to that bar's footprint. Consecutive footprints are separated by
    ``horizontal_gap``; rows wrap when their total width exceeds ``block_width``
    and each row is then centred within that envelope.

    Returns ``(bar_rects, label_rects, total_height)``; ``label_rects`` holds
    ``None`` for bars without a title.
    """
    if horizontal_gap is None:
        horizontal_gap = geometry.colorbar_short_edge_gap
    if vertical_gap is None:
        vertical_gap = geometry.colorbar_long_edge_gap
    if label_gap is None:
        label_gap = geometry.colorbar_title_to_bar_gap
    count = len(entries)
    if count == 0:
        return (), (), 0.0
    extents = tuple(_below_common_footprint(
        entry, bar_width,
        entry.label_width if include_label_widths else 0.0)
        for entry in entries)
    left_overhangs = tuple(value[0] for value in extents)
    footprints = tuple(value[2] for value in extents)
    bands = [(entry.label_height + label_gap) if entry.label_height else 0.0
             for entry in entries]

    # Greedy row assignment: keep adding bars until the row would overflow.
    rows = []
    row_widths = []
    current = []
    current_width = 0.0
    for index, footprint in enumerate(footprints):
        next_width, new_row = _below_common_row_advance(
            current_width, footprint, horizontal_gap, block_width,
            bool(current))
        if new_row:
            rows.append(current)
            row_widths.append(current_width)
            current = [index]
        else:
            current.append(index)
        current_width = next_width
    if current:
        rows.append(current)
        row_widths.append(current_width)
    row_label_bands = [max(bands[index] for index in row) for row in rows]
    row_tick_heights = [max(entries[index].tick_height for index in row)
                        for row in rows]
    row_heights = [row_label_bands[row_index] + thickness +
                   row_tick_heights[row_index]
                   for row_index in range(len(rows))]
    total_height = sum(row_heights) + vertical_gap * max(0, len(rows) - 1)
    bar_rects = [None] * count
    label_rects = [None] * count
    row_top = top_y
    for row_index, row in enumerate(rows):
        cursor = origin_x + (block_width - row_widths[row_index]) / 2.0
        for index in row:
            bar_x = cursor + left_overhangs[index]
            bar_top = row_top - row_label_bands[row_index]
            bar_rects[index] = Rect(bar_x, bar_top - thickness,
                                    bar_width, thickness)
            if entries[index].label_height:
                label_rects[index] = Rect(
                    bar_x + (bar_width - entries[index].label_width) / 2.0,
                    bar_top + label_gap,
                    entries[index].label_width, entries[index].label_height)
            cursor += footprints[index] + horizontal_gap
        row_top -= row_heights[row_index] + vertical_gap
    return tuple(bar_rects), tuple(label_rects), total_height


def _select_below_common_label_candidates(  # noqa: C901
        candidate_sets, entries, bar_width, bar_thickness, block_width,
        label_layout,
        horizontal_gap, geometry):
    """Choose wraps and greedy packing rows against one fixed envelope."""
    if len(candidate_sets) != len(entries):
        raise ValueError('below_common candidates must match entries')
    if not candidate_sets:
        return ()
    candidate_minima = tuple(
        min(candidate.line_count for candidate in candidates)
        if candidates else 0 for candidates in candidate_sets)
    if any(not candidates for candidates in candidate_sets):
        raise ValueError('each colorbar title needs a candidate')
    line_heights = [candidate.height / max(candidate.line_count, 1)
                    for candidates in candidate_sets for candidate in candidates
                    if candidate.height > 0.0]
    penalty_scale = max(line_heights, default=1.0)

    # Track row label-band and tick heights separately: a shared bar baseline
    # makes a row's vertical extent their maxima plus the bar thickness.
    # The state also carries width, closed-row height, overflow, penalty, path.
    states = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ())]
    for index, (candidates, entry) in enumerate(zip(candidate_sets, entries)):
        next_states = []
        for state in states:
            (row_width, row_label_band, row_tick_height, closed_height,
             overflow, penalty, path) = state
            for candidate in candidates:
                _, _, footprint = _below_common_footprint(
                    entry, bar_width, candidate.width)
                title_band = (candidate.height +
                              geometry.colorbar_title_to_bar_gap
                              if candidate.height else 0.0)
                new_width, new_row = _below_common_row_advance(
                    row_width, footprint, horizontal_gap, block_width,
                    bool(path))
                if new_row:
                    row_height = (row_label_band + bar_thickness +
                                  row_tick_height)
                    new_closed_height = (closed_height + row_height +
                                         geometry.colorbar_long_edge_gap)
                    new_label_band = title_band
                    new_tick_height = entry.tick_height
                else:
                    new_closed_height = closed_height
                    new_label_band = max(row_label_band, title_band)
                    new_tick_height = max(row_tick_height, entry.tick_height)
                title_overflow = max(0.0, footprint - block_width)
                excess_lines = max(
                    0, candidate.line_count - candidate_minima[index])
                new_penalty = penalty + (
                    label_layout.horizontal_colorbar_label_extra_row_penalty *
                    excess_lines +
                    label_layout.horizontal_colorbar_label_balance_weight *
                    candidate.balance_penalty +
                    label_layout.horizontal_colorbar_label_orphan_weight *
                    candidate.orphan_penalty +
                    candidate.break_penalty)
                next_states.append((
                    new_width, new_label_band, new_tick_height,
                    new_closed_height, overflow + title_overflow, new_penalty,
                    path + (candidate,)))

        def dominates(left, right):
            # Equal active-row widths are future-equivalent for the greedy
            # break rule. Compare the remaining monotone costs only within
            # that class; a narrower current row is not a safe substitute,
            # since the next item may reset a wider row and change later
            # row-height maxima.
            left_metrics, right_metrics = left[1:6], right[1:6]
            if not all(a <= b for a, b in zip(left_metrics, right_metrics)):
                return False
            if any(a < b for a, b in zip(left_metrics, right_metrics)):
                return True
            return tuple(c.text for c in left[6]) <= tuple(
                c.text for c in right[6])

        # First coalesce exact states, then keep a Pareto frontier within each
        # exact active-row width. Width equality preserves every possible
        # future greedy row-break decision while the other fields are
        # componentwise-monotone under appending candidates.
        exact_states = {}
        for state in next_states:
            key = state[:6] + (tuple(candidate.text for candidate in state[6]),)
            exact_states.setdefault(key, state)
        states_by_width = {}
        for state in exact_states.values():
            states_by_width.setdefault(state[0], []).append(state)
        states = []
        for same_width_states in states_by_width.values():
            ordered = sorted(same_width_states, key=lambda state: (
                *state[1:6], tuple(candidate.text for candidate in state[6])))
            frontier = []
            for state in ordered:
                if any(dominates(other, state) for other in frontier):
                    continue
                frontier = [other for other in frontier
                            if not dominates(state, other)]
                frontier.append(state)
            states.extend(frontier)

    def final_cost(state):
        (_, row_label_band, row_tick_height, closed_height, overflow,
         penalty, path) = state
        packed_height = (closed_height + row_label_band + bar_thickness +
                         row_tick_height)
        typography_height = penalty_scale * penalty / len(path)
        return (overflow, packed_height + typography_height,
                packed_height, penalty,
                tuple(candidate.text for candidate in path))

    return min(states, key=final_cost)[6]


@dataclass(frozen=True)
class PanelInsets:
    """Measured decorations around one profile or stack panel."""

    axis_insets: Insets = Insets()
    total_extents: Insets = Insets()
    y_label_width: float = 0.0
    y_tick_label_inset: float = 0.0
    x_label_height: float = 0.0
    legend_size: Size | None = None


@dataclass(frozen=True)
class CellInsets:
    """Independent measured extents for the profile and heatmap stack."""

    profile: PanelInsets = PanelInsets()
    stack: PanelInsets = PanelInsets()
    blocks: tuple[Insets, ...] = ()


@dataclass(frozen=True)
class CellLabels:
    """Measured candidates owned by a cell, independent of its renderer."""

    title_text: str = ''
    title_candidates: tuple[LabelCandidate, ...] = ()
    title_order: int | None = None
    y_candidates: tuple[LabelCandidate, ...] = ()
    x_candidates: tuple[LabelCandidate, ...] = ()


@dataclass(frozen=True)
class Decoration:
    """One measured or requested decoration with an explicit owner."""

    kind: str
    scope: str
    owner: int
    location: str
    size: Size | None = None
    ncols: int = 1
    text: str = ''
    candidates: tuple[LabelCandidate, ...] = ()
    extra: tuple = ()
    covered_cells: tuple[int, int] | None = None
    insets: Insets = Insets()
    tick_label_width: float = 0.0
    below_entry: BelowColorbarEntry | None = None
    ordinal: int = 0


@dataclass(frozen=True)
class CellLayout:
    row: int
    column: int
    profile: Size | None
    stack: tuple[float, ...]
    width: float
    insets: CellInsets
    labels: CellLabels
    legend: Decoration | None = None
    show_x_tick_labels: bool = True
    show_x_label: bool = True
    show_y_tick_labels: bool = True
    show_y_label: bool = True
    blocks: tuple[StackBlock, ...] = ()


@dataclass(frozen=True)
class LabelRun:
    """One measured structural axis label over a set of cell indices."""

    axis: str
    panel_indices: tuple[int, ...]
    candidates: tuple[LabelCandidate, ...]
    label_to_tick_gap: float | None = None
    block_index: int | None = None
    location: str = ''


@dataclass(frozen=True)
class GridLayoutSpec:
    rows: int
    columns: int
    cells: tuple[CellLayout, ...]
    decorations: tuple[Decoration, ...] = ()
    runs: tuple[LabelRun, ...] = ()
    title: MeasuredLabelSpec | None = None
    x_decoration_height: float = 0.0
    geometry: GeometrySpec = DEFAULT_GEOMETRY
    label_layout: LabelLayoutSpec = DEFAULT_LABEL_LAYOUT


@dataclass(frozen=True)
class SolvedGrid:
    figure_size: Size
    rects: Mapping[str, Rect]
    selections: Mapping[str, int] = field(default_factory=dict)


def cell_profile(row, column):
    return f'cell/{row + 1}/{column + 1}/profile'


def cell_block(row, column, block):
    return f'cell/{row + 1}/{column + 1}/block/{block + 1}'


def cell_title(row, column):
    return f'cell/{row + 1}/{column + 1}/title'


def label_role(kind, anchor, run=0, block=None):
    result = f'label/{kind}/{anchor + 1}/{run + 1}'
    return result if block is None else f'{result}/{block + 1}'


def row_label(row):
    return f'label/row/{row + 1}'


def figure_title():
    return 'title'


def legend_role(scope, owner):
    return f'legend/{scope}/{owner + 1}'


def colorbar_role(scope, index, ordinal=0):
    suffix = f'/{ordinal + 1}' if scope == 'row' else ''
    return f'colorbar/{scope}/{index + 1}{suffix}'


def colorbar_title_role(scope, index, ordinal=0):
    suffix = f'/{ordinal + 1}' if scope == 'row' else ''
    return f'colorbar_title/{scope}/{index + 1}{suffix}'


def indicator_role(row=0):
    return f'indicator/{row + 1}'


_DECORATION_LOCATIONS = {
    ('legend', 'cell'): frozenset((
        'above', 'below', 'right', 'best', 'upper-left', 'upper-right',
        'lower-left', 'lower-right', 'center', 'center-left',
        'center-right', 'upper-center', 'lower-center')),
    ('legend', 'row'): frozenset(('above', 'below', 'right')),
    ('legend', 'column'): frozenset(('above', 'below', 'right')),
    ('legend', 'figure'): frozenset(('above', 'below', 'right')),
    ('colorbar', 'cell'): frozenset(('below', 'right')),
    ('colorbar', 'row'): frozenset(('right',)),
    ('colorbar', 'figure'): frozenset(('below', 'right')),
    ('indicator', 'row'): frozenset(('left',)),
    ('row_label', 'row'): frozenset(('right',)),
}


def _validate_decorations(spec):
    for item in spec.decorations:
        allowed = _DECORATION_LOCATIONS.get((item.kind, item.scope), ())
        if item.location not in allowed:
            raise ValueError('unsupported decoration: {} / {} / {}'.format(
                item.kind, item.scope, item.location))
        if (item.kind == 'colorbar' and item.scope == 'cell' and
                item.location == 'right' and spec.cells[item.owner].stack):
            raise ValueError('cell-right colorbar requires a series heatmap')


def stack_heights(cell, group_sizes, height, gap):
    """Share one fixed stack height across all blocks and intervening gaps."""
    weights = tuple(group_sizes[block.group] for block in cell.blocks)
    if not weights:
        return ()
    total = sum(weights)
    if total <= 0:
        raise ValueError('heatmap stack has no rows')
    drawable = height - (len(weights) - 1) * gap
    if drawable <= 0:
        raise ValueError(
            'the requested heatmap height of {:.2f} points leaves no room '
            'for {} rows separated by {:.2f}-point gaps; increase the '
            'height or reduce the number of groups'.format(
                height, len(weights), gap))
    return tuple(drawable * weight / total for weight in weights)


def _padding(spec):
    return spec.geometry.figure_edge_padding


def _title_gap(spec):
    return spec.geometry.figure_title_to_content_gap


def _legend_gap(spec):
    return spec.geometry.external_legend_to_content_gap


def _label_clearance(spec):
    return spec.geometry.label_min_clearance


def _right_legend_gap(spec):
    return spec.geometry.right_legend_to_content_gap


def _axis_gap(spec):
    return spec.geometry.axis_label_to_tick_labels_gap


def _common_axis_gap(spec):
    return spec.geometry.common_axis_label_to_tick_labels_gap


def _column_title_gap(spec):
    return spec.geometry.column_header_to_panel_gap


def _profile_base_gap(spec):
    return spec.geometry.profile_min_column_gap


def _heatmap_base_gap(spec):
    return spec.geometry.heatmap_min_column_gap


def _heatmap_decoration_gap(spec):
    return spec.geometry.heatmap_decoration_gap


def _bar_gap(spec):
    return spec.geometry.heatmap_colorbar_gap


def _bar_title_gap(spec):
    return spec.geometry.colorbar_title_to_bar_gap


def _bar_thickness(spec):
    return spec.geometry.heatmap_colorbar_thickness


def axis_run_roles(runs, cells):
    """Name structural axis-label runs in their source order."""
    counts = {}
    names = []
    for run in runs:
        if run.axis not in ('x', 'y'):
            continue
        positions = tuple(cells[index] for index in run.panel_indices)
        anchor = min((cell.column if run.axis == 'x' else cell.row)
                     for cell in positions)
        key = run.axis, anchor
        ordinal = counts.get(key, 0)
        counts[key] = ordinal + 1
        names.append(label_role(run.axis, anchor, ordinal))
    return tuple(names)


def _stack_label_runs(spec, row=None):
    return tuple(run for run in spec.runs if run.axis == 'stack' and
                 (row is None or any(spec.cells[index].row == row
                                     for index in run.panel_indices)))


def stack_run_groups(spec, row):
    groups = []
    for run in _stack_label_runs(spec, row):
        if groups and groups[-1][0] == run.panel_indices:
            groups[-1][1].append(run)
        else:
            groups.append((run.panel_indices, [run]))
    return tuple((members, tuple(runs)) for members, runs in groups)


def _stack_region_gap(spec):
    return spec.geometry.heatmap_to_row_header_gap


def _stack_y_label_gap(spec):
    return spec.geometry.row_header_to_y_label_gap


@dataclass(frozen=True)
class _VerticalWidths:
    size: Size
    run_candidates: tuple[LabelCandidate, ...]
    row_candidates: tuple[LabelCandidate, ...]
    vertical_layout: SharedGapLayout | None
    row_gap: float
    label_gap: float
    title_gap: float
    legend_gap: float
    padding: float
    stacks: tuple['_StackWidths', ...] = ()


@dataclass(frozen=True)
class _Columns:
    gaps: tuple[float, ...]
    common_gap: float
    private_gaps: tuple[float, ...]
    right_legend_gap: float
    axis_gap: float
    cell_bar_gap: float = 0.0
    cell_bar_thickness: float = 0.0
    starts: tuple[float, ...] = ()
    width: float = 0.0
    # Smallest boundary gap each right legend needs; labels never spill
    # into this room, so it is kept apart from the label-shared gaps.
    legend_gaps: tuple[float, ...] = ()


@dataclass(frozen=True)
class _Bands:
    columns: _Columns
    run_candidates: tuple[LabelCandidate, ...]
    title_candidates: tuple[LabelCandidate, ...]
    bar_titles: tuple[LabelCandidate, ...] = ()


@dataclass(frozen=True)
class _Rows:
    panels: tuple[Rect | None, ...]
    cell_legends: tuple[Rect | None, ...]
    blocks: tuple[tuple[Rect, ...], ...] = ()
    bars: tuple[tuple[Decoration, Rect, Rect | None], ...] = ()
    indicators: tuple[Rect | None, ...] = ()
    stack_bottoms: tuple[float, ...] = ()


@dataclass(frozen=True)
class _RowDecorations:
    rows: _Rows
    row_labels: tuple[tuple[int, Rect, LabelCandidate], ...]
    panel_titles: tuple[tuple[int, Rect], ...]
    grid_legends: tuple[tuple[Decoration, Rect], ...]
    axis_labels: tuple[tuple[int, Rect], ...]
    region_labels: tuple[tuple[int, int, Rect], ...] = ()
    heatmap_y_labels: tuple[tuple[int, int, Rect], ...] = ()


@dataclass(frozen=True)
class _FigureDecorations:
    rows: _RowDecorations
    legend: Rect | None
    body: tuple[Rect, ...]
    body_top: float
    bars: tuple[tuple[Decoration, Rect, Rect | None], ...] = ()


@dataclass(frozen=True)
class _FigureBands:
    figure: _FigureDecorations
    title: Rect | None


@dataclass(frozen=True)
class _StackWidths:
    row: int
    block_gap: float
    stack_height: float
    region_candidates: tuple[LabelCandidate, ...]
    y_candidates: tuple[LabelCandidate | None, ...]
    summary_gap: float
    summary_private: float
    summary_height: float
    region_groups: tuple = ()


def _profile_size(spec):
    if not spec.cells:
        raise ValueError('at least one panel is required')
    sizes = tuple(cell.profile for cell in spec.cells
                  if cell.profile is not None)
    widths = {cell.width for cell in spec.cells}
    heights = {size.height for size in sizes}
    if len(widths) > 1 or len(heights) > 1:
        raise ValueError(
            'every panel in a profile grid must request the same data size; '
            'got widths {} and heights {}'.format(
                sorted(widths), sorted(heights)))
    stack_height = max((sum(cell.stack) for cell in spec.cells), default=0.0)
    return Size(next(iter(widths)), next(iter(heights), stack_height))


def _panels(cell):
    return ((cell.insets.profile,) if cell.profile is not None else ()) + \
        ((cell.insets.stack,) if cell.stack else ())


def _titles(spec):
    return tuple(sorted(
        ((index, cell) for index, cell in enumerate(spec.cells)
         if cell.labels.title_order is not None and
         cell.labels.title_candidates),
        key=lambda pair: pair[1].labels.title_order))


def _facets(spec):
    return tuple(item for item in spec.decorations
                 if item.kind == 'row_label' and item.text)


def _legends(spec, scope):
    return tuple(item for item in spec.decorations
                 if item.kind == 'legend' and item.scope == scope)


def _profiles_on_stacks(spec, row):
    """Whether the row's profiles stand on heatmap stacks."""
    cells = tuple(cell for cell in spec.cells if cell.row == row)
    return (any(cell.stack for cell in cells) and
            any(cell.profile is not None for cell in cells))


def _below_row_legend(spec, row):
    """The band a row's 'below' legend takes under the row's profiles."""
    return sum(_legend_gap(spec) + legend.size.height
               for legend in _legends(spec, 'row')
               if legend.location == 'below' and legend.owner == row)


def _top_insets(cell):
    """Measured extents of a cell's top element: its profile, else its stack."""
    return cell.insets.profile if cell.profile is not None else \
        cell.insets.stack


def _span_extent(spec, run, axis):
    values = tuple(getattr(spec.cells[index], axis)
                   for index in run.panel_indices)
    return min(values), max(values)


def _run_gap(run, default):
    return (default if run.label_to_tick_gap is None else
            max(0.0, float(run.label_to_tick_gap)))


def _row_boundary_baseline(spec, widths, reserves, title_heights,
                           include_auto_x=True):
    """Reserve what lies between two rows' data extents.

    Every row keeps only its data (profile, or stack and profile) inside its
    height.  The upper row's bottom decorations (profile insets and X labels,
    or a stack's X decorations and bars), the lower row's top insets, titles
    and row legends all go into the gap.
    """
    cells = {(cell.row, cell.column): cell for cell in spec.cells}
    title_extras = tuple(widths.title_gap + value if value else 0.0
                         for value in title_heights)
    automatic = bool(spec.label_layout.auto_axis_label_layout and spec.runs)
    stacked = {stack.row for stack in widths.stacks}
    gaps = []
    for row in range(spec.rows - 1):
        reserve = reserves[row]
        gap = widths.row_gap
        for column in range(spec.columns):
            upper = cells.get((row, column))
            lower = cells.get((row + 1, column))
            if upper and lower:
                upper_bottom = (reserve if row in stacked else
                                upper.insets.profile.total_extents.bottom)
                gap = max(gap, upper_bottom + widths.row_gap +
                          _top_insets(lower).total_extents.top +
                          title_extras[row + 1])
        gap = max(gap, reserve + widths.row_gap + title_extras[row + 1])
        # A below row legend hangs under the profiles, over any stack.
        if not _profiles_on_stacks(spec, row):
            gap += _below_row_legend(spec, row)
        gap += sum(widths.legend_gap + legend.size.height
                   for legend in _legends(spec, 'row')
                   if legend.location == 'above' and legend.owner == row + 1)
        for index, run in enumerate(spec.runs):
            if run.axis != 'x' or not run.panel_indices or \
                    _span_extent(spec, run, 'row')[1] != row or \
                    row in stacked:
                continue
            if automatic and include_auto_x:
                gap = max(gap, widths.row_gap + _run_gap(run, widths.label_gap)
                          + min(candidate.height for candidate in run.candidates))
            elif not automatic:
                gap += widths.label_gap + run.candidates[0].height
        gaps.append(gap)
    return tuple(gaps)


def _vertical_band_specs(spec, widths, scale):
    """Build the Y-run and row-label bands with their source indices."""
    bands = []
    sources = []
    if spec.label_layout.auto_axis_label_layout:
        groups = {}
        for index, run in enumerate(spec.runs):
            if run.axis != 'y':
                continue
            first, _ = _span_extent(spec, run, 'column')
            groups.setdefault(None if first == 0 else first - 1, []).append(index)
        for owner in sorted(groups, key=lambda value: -1 if value is None else value):
            ordered = sorted(groups[owner], key=lambda index: (
                _span_extent(spec, spec.runs[index], 'row'), index))
            current = []
            previous_end = -1
            for index in ordered:
                start, end = _span_extent(spec, spec.runs[index], 'row')
                if current and start <= previous_end:
                    bands.append(_y_band(spec, widths, tuple(current), scale))
                    sources.append(('y', tuple(current)))
                    current = []
                current.append(index)
                previous_end = max(previous_end, end)
            if current:
                bands.append(_y_band(spec, widths, tuple(current), scale))
                sources.append(('y', tuple(current)))
    facets = _facets(spec)
    if spec.label_layout.auto_facet_label_layout and facets:
        ordered = tuple(sorted(range(len(facets)),
                               key=lambda index: (facets[index].owner, index)))
        bands.append(SharedGapBandSpec(
            candidate_sets=tuple(facets[index].candidates for index in ordered),
            spans=tuple((facets[index].owner, facets[index].owner)
                        for index in ordered),
            primary_dimension='height',
            extra_row_penalty=spec.label_layout.panel_title_extra_row_penalty,
            balance_weight=spec.label_layout.panel_title_balance_weight,
            orphan_weight=spec.label_layout.panel_title_orphan_weight,
            secondary_costs=tuple(tuple(candidate.width / scale
                                        for candidate in facets[index].candidates)
                                  for index in ordered)))
        sources.append(('facet', ordered))
    return tuple(bands), tuple(sources)


def _y_band(spec, widths, indices, scale):
    return SharedGapBandSpec(
        candidate_sets=tuple(spec.runs[index].candidates for index in indices),
        spans=tuple(_span_extent(spec, spec.runs[index], 'row')
                    for index in indices),
        primary_dimension='height',
        extra_row_penalty=spec.label_layout.axis_label_extra_row_penalty,
        balance_weight=spec.label_layout.axis_label_balance_weight,
        orphan_weight=spec.label_layout.axis_label_orphan_weight,
        secondary_costs=tuple(tuple(
            (_run_gap(spec.runs[index], widths.label_gap) + candidate.width)
            / scale for candidate in spec.runs[index].candidates)
            for index in indices))


def _width_affecting_vertical(spec):
    """Choose vertical Y and row-label candidates before column gaps."""
    size = _profile_size(spec)
    stacks = tuple(_stack_widths_for_row(spec, row)
                   for row in range(spec.rows)
                   if any(cell.stack for cell in spec.cells if cell.row == row))
    automatic = bool(spec.label_layout.auto_axis_label_layout and spec.runs)
    facets = _facets(spec)
    automatic_facets = bool(spec.label_layout.auto_facet_label_layout and facets)
    base_row_gap = spec.geometry.profile_row_gap
    row_gap = (min(base_row_gap, spec.label_layout.max_row_gap_points)
               if automatic or automatic_facets else base_row_gap)
    widths = _VerticalWidths(
        size, tuple(run.candidates[0] for run in spec.runs),
        tuple(facet.candidates[0] for facet in facets), None, row_gap,
        _common_axis_gap(spec), _column_title_gap(spec),
        _legend_gap(spec), _padding(spec), stacks)
    if not (automatic or automatic_facets):
        return widths
    title_heights = [0.0] * spec.rows
    for _, cell in _titles(spec):
        title_heights[cell.row] = max(
            title_heights[cell.row],
            min(c.height for c in cell.labels.title_candidates))
    base_gaps = _row_boundary_baseline(
        spec, widths, _reservations(spec, widths,
                                    _provisional_bands(spec, widths)),
        title_heights)
    private = tuple(max(0.0, gap - row_gap) for gap in base_gaps)
    scale = max(spec.columns * size.width, 1.0)
    bands, sources = _vertical_band_specs(spec, widths, scale)
    if not bands:
        return widths
    top = max(cell.insets.profile.total_extents.top for cell in spec.cells)
    bottom = max(cell.insets.profile.total_extents.bottom for cell in spec.cells)
    heights = tuple(_row_height(spec, widths, row) for row in range(spec.rows))
    selected = optimize_shared_gap_layout(
        bands, cell_extent=heights, base_common_gap=row_gap,
        base_private_gaps=private,
        original_block_extent=sum(heights) + sum(base_gaps),
        max_common_gap_points=spec.label_layout.max_row_gap_points,
        outer_left_allowance=top + widths.padding,
        outer_right_allowance=bottom + widths.padding,
        hard_outer_containment=True,
        label_min_clearance=_label_clearance(spec))
    runs = list(widths.run_candidates)
    rows = list(widths.row_candidates)
    for (kind, indices), band in zip(sources, selected.bands):
        target = runs if kind == 'y' else rows
        for index, candidate in zip(indices, band.candidates):
            target[index] = candidate
    return _VerticalWidths(size, tuple(runs), tuple(rows), selected,
                           row_gap, widths.label_gap, widths.title_gap,
                           widths.legend_gap, widths.padding, stacks)


def _column_boundary(spec, widths, left, right, row, column, bar_owners,
                     axis_gap, bar_gap, bar_thickness):
    """One inset rule across every panel at a column boundary."""
    has_stack = bool(left.stack or right.stack)
    base = _heatmap_base_gap(spec) if has_stack else _profile_base_gap(spec)
    decoration = (_heatmap_decoration_gap(spec) if has_stack else base)
    bar = bar_owners.get(left)
    extra = (bar_gap + bar_thickness + bar.insets.right
             if bar is not None else 0.0)
    labels, right_labelled = _stack_boundary_labels(
        spec, widths, left, right, row)
    extra += labels
    # Left region labels of the right cell sit between its heatmap Y label
    # and its stack, as in the first column.
    y_gap = _stack_y_label_gap(spec) if right_labelled else axis_gap
    selected_y = None
    for item in widths.stacks:
        if item.row != row:
            continue
        members = tuple(cell for cell in spec.cells if cell.row == row)
        selected_y = next((item.y_candidates[index]
                           for index, cell in enumerate(members)
                           if cell.column == column), None)
        break
    y_width = selected_y.width if selected_y is not None else 0.0
    common = float(base)
    private = 0.0
    for lhs in _panels(left):
        for rhs in _panels(right):
            without_y = max(
                0.0, max(rhs.total_extents.left, rhs.axis_insets.left) -
                rhs.y_label_width -
                rhs.y_tick_label_inset -
                (axis_gap if rhs.y_label_width else 0.0) -
                y_width - (y_gap if y_width else 0.0))
            common = max(common, max(lhs.total_extents.right,
                                     lhs.axis_insets.right) + decoration +
                         without_y + extra)
            private = max(
                private, rhs.y_tick_label_inset +
                (axis_gap + rhs.y_label_width if rhs.y_label_width else 0.0),
                y_gap + y_width if y_width else 0.0)
    return common, private


def _stack_boundary_labels(spec, widths, left, right, row):
    """Reserve the stack-label runs between two adjacent cells.

    A run ending at the left cell with labels on its right, and a run
    starting at the right cell with labels on its left, both sit in the gap.
    Returns the reserved width and whether the right cell has left labels.
    """
    selected = next((item for item in widths.stacks if item.row == row), None)
    if selected is None:
        return 0.0, False
    left_index = spec.cells.index(left)
    right_index = spec.cells.index(right)
    reserved, right_labelled = 0.0, False
    for (members, candidates), (_, runs) in zip(
            selected.region_groups, stack_run_groups(spec, row)):
        location = runs[0].location
        if location == 'right':
            boundary = members[-1] == left_index and right_index not in members
        else:
            boundary = members[0] == right_index and left_index not in members
        width = max((candidate.width for candidate in candidates),
                    default=0.0)
        if boundary and width:
            reserved += width + _stack_region_gap(spec)
            right_labelled = right_labelled or location == 'left'
    return reserved, right_labelled


def _column_start_positions(spec, gaps, width):
    starts = [0.0]
    for column in range(spec.columns - 1):
        starts.append(starts[-1] + width + gaps[column])
    return tuple(starts)


def _columns(spec, widths):
    """Reserve all panel insets, selected Y widths and structural legends."""
    by_position = {(cell.row, cell.column): cell for cell in spec.cells}
    axis_gap = _axis_gap(spec)
    right_legend_gap = _right_legend_gap(spec)
    bars = tuple(item for item in spec.decorations
                 if item.kind == 'colorbar' and item.scope == 'cell'
                 and item.location == 'right')
    bar_owners = {spec.cells[item.owner]: item for item in bars}
    bar_gap = _bar_gap(spec) if bars else 0.0
    bar_thickness = _bar_thickness(spec) if bars else 0.0
    common = 0.0
    boundary_private = [0.0] * max(0, spec.columns - 1)
    for column in range(spec.columns - 1):
        for row in range(spec.rows):
            left = by_position.get((row, column))
            right = by_position.get((row, column + 1))
            if left is None or right is None:
                continue
            gap, private = _column_boundary(
                spec, widths, left, right, row, column + 1, bar_owners,
                axis_gap, bar_gap, bar_thickness)
            common = max(common, gap)
            boundary_private[column] = max(boundary_private[column], private)
    if not boundary_private:
        common = _heatmap_base_gap(spec) if any(c.stack for c in spec.cells) \
            else _profile_base_gap(spec)
    run_widths = [0.0] * len(boundary_private)
    for index, run in enumerate(spec.runs):
        if run.axis != 'y':
            continue
        first, _ = _span_extent(spec, run, 'column')
        if first:
            candidate = widths.run_candidates[index]
            run_widths[first - 1] = max(
                run_widths[first - 1],
                _run_gap(run, widths.label_gap) + candidate.width)
    boundary_private = [value + extra for value, extra in
                        zip(boundary_private, run_widths)]
    capped = bool(spec.label_layout.auto_axis_label_layout or _titles(spec))
    selected_common = min(common, spec.label_layout.max_column_gap_points) \
        if capped else common
    # A right legend (a column's, or a cell's own) owns room in the gap it
    # sits in; labels share only the common gap and the label privates.
    label_private = tuple(boundary_private)
    legend_gaps = [0.0] * len(boundary_private)
    beside = tuple((legend, legend.owner) for legend in _legends(spec, 'column')
                   ) + tuple((cell.legend, cell.column) for cell in spec.cells
                             if cell.legend is not None)
    for legend, column in beside:
        if legend.location != 'right' or column >= spec.columns - 1:
            continue
        left = max((panel.axis_insets.right
                    for cell in spec.cells if cell.column == column
                    for panel in _panels(cell)), default=0.0)
        right = max((panel.axis_insets.left
                     for cell in spec.cells if cell.column == column + 1
                     for panel in _panels(cell)), default=0.0)
        legend_gaps[column] = max(
            legend_gaps[column],
            left + right_legend_gap + legend.size.width +
            _profile_base_gap(spec) + right)
    gaps = tuple(max(selected_common + value, legend)
                 for value, legend in zip(label_private, legend_gaps))
    starts = _column_start_positions(spec, gaps, widths.size.width)
    width = starts[-1] + widths.size.width
    return _Columns(gaps, selected_common, label_private,
                    right_legend_gap, axis_gap, bar_gap, bar_thickness,
                    starts, width, tuple(legend_gaps))


def _horizontal_band_specs(spec):
    bands = []
    sources = []
    titles = _titles(spec)
    for row in range(spec.rows):
        indices = tuple(index for index, (_, cell) in enumerate(titles)
                        if cell.row == row)
        indices = tuple(sorted(indices, key=lambda i: (titles[i][1].column, i)))
        if indices and spec.label_layout.auto_panel_title_column_gap:
            bands.append(SharedGapBandSpec(
                candidate_sets=tuple(titles[i][1].labels.title_candidates
                                     for i in indices),
                spans=tuple((titles[i][1].column, titles[i][1].column)
                            for i in indices),
                primary_dimension='width',
                extra_row_penalty=spec.label_layout.panel_title_extra_row_penalty,
                balance_weight=spec.label_layout.panel_title_balance_weight,
                orphan_weight=spec.label_layout.panel_title_orphan_weight))
            sources.append(('title', indices))
    if spec.label_layout.auto_axis_label_layout:
        for row in range(spec.rows):
            indices = tuple(index for index, run in enumerate(spec.runs)
                            if run.axis == 'x' and run.panel_indices and
                            _span_extent(spec, run, 'row')[1] == row)
            indices = tuple(sorted(indices, key=lambda i: (
                _span_extent(spec, spec.runs[i], 'column'), i)))
            if indices:
                bands.append(SharedGapBandSpec(
                    candidate_sets=tuple(spec.runs[i].candidates
                                         for i in indices),
                    spans=tuple(_span_extent(spec, spec.runs[i], 'column')
                                for i in indices),
                    primary_dimension='width',
                    extra_row_penalty=spec.label_layout.axis_label_extra_row_penalty,
                    balance_weight=spec.label_layout.axis_label_balance_weight,
                    orphan_weight=spec.label_layout.axis_label_orphan_weight))
                sources.append(('x', indices))
    bars = tuple(item for item in spec.decorations
                 if item.kind == 'colorbar')
    below = tuple((index, item) for index, item in enumerate(bars)
                  if item.scope == 'cell' and item.location == 'below')
    if not (below and spec.label_layout.auto_horizontal_colorbar_label_layout
            and all(item.candidates for _, item in below)):
        return tuple(bands), tuple(sources)
    # One band per row; a bar spans the columns of the cells it covers.
    for row in sorted({spec.cells[item.owner].row for _, item in below}):
        members = tuple((index, item) for index, item in below
                        if spec.cells[item.owner].row == row)
        bands.append(SharedGapBandSpec(
            candidate_sets=tuple(item.candidates for _, item in members),
            spans=tuple(tuple(spec.cells[cell].column for cell in
                              (item.covered_cells or (item.owner,) * 2))
                        for _, item in members),
            primary_dimension='width',
            extra_row_penalty=(spec.label_layout.
                               horizontal_colorbar_label_extra_row_penalty),
            balance_weight=(spec.label_layout.
                            horizontal_colorbar_label_balance_weight),
            orphan_weight=(spec.label_layout.
                           horizontal_colorbar_label_orphan_weight)))
        sources.append(('bar', tuple(index for index, _ in members)))
    return tuple(bands), tuple(sources)


def _horizontal_bands(spec, columns, widths):
    """Choose title, X-run and below-bar labels in one shared-gap solve."""
    titles = _titles(spec)
    selected_titles = [cell.labels.title_candidates[0]
                       for _, cell in titles]
    selected_runs = list(widths.run_candidates)
    bars = tuple(item for item in spec.decorations
                 if item.kind == 'colorbar')
    selected_bars = [item.candidates[0] if item.candidates else _EMPTY_LABEL
                     for item in bars]
    bands, sources = _horizontal_band_specs(spec)
    if bands:
        columns, selected_titles, selected_runs, selected_bars = \
            _solve_horizontal_bands(spec, columns, widths, bands, sources,
                                    selected_titles, selected_runs,
                                    selected_bars)
    common = tuple(item for item in bars if item.scope == 'figure'
                   and item.location == 'below')
    if (common and all(item.below_entry is not None for item in common) and
            spec.label_layout.auto_horizontal_colorbar_label_layout and
            all(item.candidates for item in common)):
        selected_bars = _select_below_common_label_candidates(
            tuple(item.candidates for item in common),
            tuple(item.below_entry for item in common),
            widths.size.width, _bar_thickness(spec), columns.width,
            spec.label_layout,
            min(columns.common_gap, _below_common_cap(spec)), spec.geometry)
    return _Bands(columns, tuple(selected_runs), tuple(selected_titles),
                  tuple(selected_bars))


def _solve_horizontal_bands(spec, columns, widths, bands, sources,
                            selected_titles, selected_runs, selected_bars):
    outer_y = max((
        selected_runs[index].width + _run_gap(run, widths.label_gap)
        for index, run in enumerate(spec.runs)
        if run.axis == 'y' and run.panel_indices and
        _span_extent(spec, run, 'column')[0] == 0), default=0.0)
    left = max(panel.total_extents.left for cell in spec.cells
               for panel in _panels(cell))
    right = max(panel.total_extents.right for cell in spec.cells
                for panel in _panels(cell))
    stacked = any(cell.stack for cell in spec.cells)
    layout = optimize_shared_gap_layout(
        bands, cell_extent=widths.size.width,
        base_common_gap=columns.common_gap,
        base_private_gaps=columns.private_gaps,
        original_block_extent=columns.width,
        max_common_gap_points=spec.label_layout.max_column_gap_points,
        outer_left_allowance=0.0 if stacked else left + outer_y + widths.padding,
        outer_right_allowance=0.0 if stacked else right + widths.padding,
        hard_outer_containment=not stacked,
        label_min_clearance=_label_clearance(spec))
    for (kind, indices), band in zip(sources, layout.bands):
        target = {'title': selected_titles, 'x': selected_runs,
                  'bar': selected_bars}[kind]
        for index, candidate in zip(indices, band.candidates):
            target[index] = candidate
    gaps = tuple(max(gap, legend) for gap, legend in
                 zip(layout.boundary_gaps, columns.legend_gaps))
    updated = _Columns(gaps, layout.common_gap, columns.private_gaps,
                       columns.right_legend_gap, columns.axis_gap,
                       columns.cell_bar_gap, columns.cell_bar_thickness,
                       _column_start_positions(spec, gaps, widths.size.width),
                       spec.columns * widths.size.width + sum(gaps),
                       columns.legend_gaps)
    return updated, selected_titles, selected_runs, selected_bars


def _x_label_stack_height(spec, widths, bands, row, column=None):
    heights = []
    for index, run in enumerate(spec.runs):
        if run.axis != 'x' or _span_extent(spec, run, 'row')[1] != row:
            continue
        if column is not None and not any(
                spec.cells[item].column == column for item in run.panel_indices):
            continue
        heights.append(_run_gap(run, widths.label_gap) +
                       bands.run_candidates[index].height)
    return max(heights, default=0.0)


def _row_gaps(spec, widths, bands, reserves):
    """Apply the final three-pass row solve and keep private reservations."""
    titles = _titles(spec)
    heights = [0.0] * spec.rows
    offsets = [0.0] * len(spec.cells)
    for index, (cell_index, cell) in enumerate(titles):
        candidate = bands.title_candidates[index]
        if candidate is None:
            raise ValueError('profile panel title selection is incomplete')
        heights[cell.row] = max(heights[cell.row], candidate.height)
        offsets[cell_index] = max(offsets[cell_index],
                                  widths.title_gap + candidate.height)
    automatic = bool(spec.label_layout.auto_axis_label_layout and spec.runs)
    x_indices = tuple(index for index, run in enumerate(spec.runs)
                      if run.axis == 'x') if automatic else ()
    stacked = {stack.row for stack in widths.stacks}
    selected_x_stack = max((
        _run_gap(spec.runs[index], widths.label_gap) +
        bands.run_candidates[index].height
        for index in x_indices
        if _span_extent(spec, spec.runs[index], 'row')[1] < spec.rows - 1 and
        _span_extent(spec, spec.runs[index], 'row')[1] not in stacked),
        default=0.0)
    common = (min(widths.row_gap + selected_x_stack,
                  spec.label_layout.max_row_gap_points)
              if x_indices else widths.row_gap)
    if widths.vertical_layout is not None:
        common = max(common, widths.vertical_layout.common_gap)
    baseline = _row_boundary_baseline(spec, widths, reserves, heights,
                                      include_auto_x=False)
    gaps = [common + max(0.0, value - widths.row_gap) for value in baseline]
    if x_indices:
        expansion = max(0.0, common - widths.row_gap)
        below = {}
        for cell in spec.cells:
            if cell.legend is not None and cell.legend.location == 'below' \
                    and cell.row not in stacked:
                below.setdefault(cell.row, set()).add(cell.column)
        for row, columns in below.items():
            if row < spec.rows - 1:
                stack = max((_x_label_stack_height(spec, widths, bands,
                                                   row, column)
                             for column in columns), default=0.0)
                gaps[row] += max(0.0, stack - expansion)
    return tuple(gaps), tuple(offsets)


def _cell_legend_rect(cell, data, gap, right_gap, above, below):
    """Place a profile's own legend ``above`` or ``below`` points away."""
    legend = cell.legend
    size = cell.insets.profile.legend_size
    if legend is None or size is None:
        return None
    x, y, width, height = data.x, data.y, data.width, data.height
    if legend.location == 'above':
        return Rect(x + (width - size.width) / 2, y + height + above + gap,
                    size.width, size.height)
    if legend.location == 'below':
        return Rect(x + (width - size.width) / 2,
                    y - below - gap - size.height, size.width, size.height)
    if legend.location == 'right':
        return Rect(x + width + cell.insets.profile.axis_insets.right +
                    right_gap, y + (height - size.height) / 2,
                    size.width, size.height)
    return None


def _stack_for_row(widths, row):
    return next((item for item in widths.stacks if item.row == row), None)


def _row_height(spec, widths, row):
    """A row's data extent: its profiles, or its stack and the profiles on it."""
    stack = _stack_for_row(widths, row)
    if stack is None:
        return widths.size.height
    if not _profiles_on_stacks(spec, row):
        return stack.stack_height
    return (stack.stack_height + stack.summary_gap + stack.summary_private +
            stack.summary_height)


def _row_origins(heights, gaps):
    origins = [0.0] * len(heights)
    current = 0.0
    for row in range(len(heights) - 1, -1, -1):
        origins[row] = current
        current += heights[row]
        if row:
            current += gaps[row - 1]
    return tuple(origins)


def _row_indicator(spec, columns, members, stack, bottom):
    if not any(item.kind == 'indicator' and item.owner == members[0][1].row
               for item in spec.decorations):
        return None
    left = min(columns.starts[cell.column] for _, cell in members
               if cell.stack)
    thickness = _bar_thickness(spec)
    return Rect(left - spec.geometry.sort_indicator_gap - thickness, bottom,
                thickness, stack.stack_height)


def _rows(spec, columns, bands, widths):
    """Place every row from its origin: its stack, then the profiles on it."""
    reserves = _reservations(spec, widths, bands)
    row_gaps, offsets = _row_gaps(spec, widths, bands, reserves)
    origins = _row_origins(tuple(_row_height(spec, widths, row)
                                 for row in range(spec.rows)), row_gaps)
    panels = [None] * len(spec.cells)
    cell_legends = [None] * len(spec.cells)
    blocks = [()] * len(spec.cells)
    bars = []
    stack_bottoms = [0.0] * spec.rows
    indicators = [None] * spec.rows
    for row in range(spec.rows):
        stack = _stack_for_row(widths, row)
        members = tuple((index, cell) for index, cell in
                        enumerate(spec.cells) if cell.row == row)
        profile_y = origins[row]
        if stack is not None:
            bottom = stack_bottoms[row] = origins[row]
            placed = _place_row_blocks(tuple(cell for _, cell in members),
                                       stack, columns, bottom)
            for (index, _), rects in zip(members, placed):
                blocks[index] = rects
            profile_y = (bottom + stack.stack_height + stack.summary_gap +
                         stack.summary_private)
            indicators[row] = _row_indicator(spec, columns, members, stack,
                                             bottom)
            bars.extend(_place_row_bars(spec, stack, bands, bottom,
                                        reserves[row], row))
        # Above legends share one baseline over the row's tallest title.
        above = max((max(cell.insets.profile.axis_insets.top, offsets[index])
                     for index, cell in members if cell.profile is not None),
                    default=0.0)
        for index, cell in members:
            if cell.profile is None:
                continue
            panels[index] = Rect(columns.starts[cell.column], profile_y,
                                 cell.profile.width, cell.profile.height)
            # X labels hang under the profile only when no stack is below it.
            below = cell.insets.profile.axis_insets.bottom + (
                0.0 if stack is not None else
                _x_label_stack_height(spec, widths, bands, row, cell.column))
            cell_legends[index] = _cell_legend_rect(
                cell, panels[index], widths.legend_gap,
                columns.right_legend_gap, above, below)
            bars.extend((item, Rect(panels[index].x1 + columns.cell_bar_gap,
                                    panels[index].y,
                                    columns.cell_bar_thickness,
                                    panels[index].height), None)
                        for item in spec.decorations
                        if item.kind == 'colorbar' and item.scope == 'cell'
                        and item.owner == index and not cell.stack)
    return _Rows(tuple(panels), tuple(cell_legends), tuple(blocks),
                 tuple(bars), tuple(indicators), tuple(stack_bottoms))


def _lowest(spec, rows, index):
    """A cell's lowest element and the height of the X decorations under it."""
    if rows.blocks[index]:
        return rows.blocks[index][-1], spec.x_decoration_height
    return rows.panels[index], spec.cells[index].insets.profile.axis_insets.bottom


def _edge_under(spec, rows, axis_labels, elements, last_row=None):
    """Top of the free space under ``elements`` and what hangs from them.

    ``elements`` are ``(cell index, rect, bottom inset)``. An X label or a
    below colorbar (with its title) counts when it hangs from one of them:
    the element is its cell's lowest and the label or bar sits under it.
    """
    edge = min(rect.y0 - inset for _, rect, inset in elements)
    hanging = {index: rect for index, rect, _ in elements
               if _lowest(spec, rows, index)[0] == rect}
    labels = (rect.y0 for index, rect in axis_labels
              if spec.runs[index].axis == 'x' and
              hanging.keys() & set(spec.runs[index].panel_indices) and
              (last_row is None or
               _span_extent(spec, spec.runs[index], 'row')[1] == last_row))
    bars = (min(bar.y0 - item.insets.bottom,
                title.y0 if title is not None else bar.y0)
            for item, bar, title in rows.bars
            if item.location == 'below' and
            any(bar.x0 < rect.x1 and rect.x0 < bar.x1 and bar.y1 <= rect.y0
                for rect in hanging.values()))
    return min((edge, *labels, *bars))


def _place_grid_legend(spec, widths, columns, rows, titles, axis_labels,
                       legend):
    scope, owner = legend.scope, legend.owner
    members = tuple(index for index, cell in enumerate(spec.cells)
                    if getattr(cell, scope) == owner)
    panels = tuple((index, rows.panels[index]) for index in members
                   if rows.panels[index] is not None)
    # Legends frame the profiles they describe; stacks only when there are none.
    owned = tuple(rect for _, rect in panels) or tuple(
        rect for index in members for rect in rows.blocks[index])
    x0, x1 = min(r.x0 for r in owned), max(r.x1 for r in owned)
    y0, y1 = min(r.y0 for r in owned), max(r.y1 for r in owned)
    if legend.location == 'above':
        top = max(spec.cells[index].insets.profile.axis_insets.top
                  for index in members)
        edge = max((y1 + top, *(rect.y1 for index, rect in titles
                                if getattr(spec.cells[index], scope) == owner)))
        return Rect(x0 + (x1 - x0 - legend.size.width) / 2,
                    edge + widths.legend_gap,
                    legend.size.width, legend.size.height)
    if legend.location == 'below':
        # A row legend hangs under the row's profiles; a column legend under
        # the whole column.
        elements = (tuple((index, rect, spec.cells[index].insets.profile.
                           axis_insets.bottom) for index, rect in panels)
                    if scope == 'row' and panels else
                    tuple((index, *_lowest(spec, rows, index))
                          for index in members))
        edge = _edge_under(spec, rows, axis_labels, elements,
                           owner if scope == 'row' else None)
        return Rect(x0 + (x1 - x0 - legend.size.width) / 2,
                    edge - widths.legend_gap - legend.size.height,
                    legend.size.width, legend.size.height)
    # A right legend starts beside its profiles; row labels clear it.
    right = max(spec.cells[index].insets.profile.axis_insets.right
                for index, _ in panels)
    return Rect(x1 + right + columns.right_legend_gap,
                y0 + (y1 - y0 - legend.size.height) / 2,
                legend.size.width, legend.size.height)


def _axis_label_rect(spec, widths, bands, rows, index, run):
    candidate = bands.run_candidates[index]
    if run.axis == 'y':
        owned = tuple(rows.panels[item] for item in run.panel_indices)
        x0 = min(r.x0 for r in owned)
        y0, y1 = min(r.y0 for r in owned), max(r.y1 for r in owned)
        first, _ = _span_extent(spec, run, 'column')
        inset = max(spec.cells[item].insets.profile.axis_insets.left
                    for item in run.panel_indices
                    if spec.cells[item].column == first)
        x = x0 - inset - _run_gap(run, widths.label_gap) - candidate.width
        y = y0 + (y1 - y0 - candidate.height) / 2
    else:
        lowest = tuple(_lowest(spec, rows, item) for item in run.panel_indices)
        x0 = min(rect.x0 for rect, _ in lowest)
        x1 = max(rect.x1 for rect, _ in lowest)
        x = x0 + (x1 - x0 - candidate.width) / 2
        y = (min(rect.y0 - inset for rect, inset in lowest) -
             _run_gap(run, widths.label_gap) - candidate.height)
    return Rect(x, y, candidate.width, candidate.height)


def _stack_labels_for_row(spec, columns, widths, rows, row):
    stack = _stack_for_row(widths, row)
    if stack is None:
        return (), ()
    labels = _stack_label_runs(spec, row)
    indicator = rows.indicators[row] if rows.indicators else None
    on_left = bool(labels and labels[0].location == 'left')
    region_rects = []
    # Left edge of the left-side labels of the run starting at each cell.
    left_edges = {}
    for ordinal, (members, candidates) in enumerate(stack.region_groups):
        label_width = max((item.width for item in candidates), default=0.0)
        gap = _stack_region_gap(spec) if label_width else 0.0
        if on_left:
            nearest = (indicator.x0 if ordinal == 0 and indicator else
                       min(columns.starts[spec.cells[index].column]
                           for index in members))
            region_left = nearest - gap - label_width
            if label_width:
                left_edges[members[0]] = region_left
        else:
            right = max(columns.starts[spec.cells[index].column] +
                        spec.cells[index].width for index in members)
            region_left = right + gap
        first = members[0]
        for index, candidate in enumerate(candidates):
            block = rows.blocks[first][index]
            x = region_left + label_width - candidate.width if on_left \
                else region_left
            region_rects.append((row, ordinal, index, Rect(
                x, block.y + (block.height - candidate.height) / 2,
                candidate.width, candidate.height)))
    y_rects = []
    cells = tuple((index, cell) for index, cell in enumerate(spec.cells)
                  if cell.row == row)
    for (index, cell), candidate in zip(cells, stack.y_candidates):
        if candidate is None or not candidate.width or not candidate.height:
            continue
        # Keyed by the cell's column, as its label selection is.
        column = cell.column
        if index in left_edges:
            x = left_edges[index] - _stack_y_label_gap(spec) - candidate.width
        elif column == 0:
            marker = indicator
            gap = (_stack_y_label_gap(spec) if marker
                   else _stack_region_gap(spec))
            nearest = marker.x0 if marker else columns.starts[0]
            x = nearest - gap - candidate.width
        else:
            x = columns.starts[column] - _axis_gap(spec) - candidate.width
        y_rects.append((row, column, Rect(
            x, rows.stack_bottoms[row] +
            (stack.stack_height - candidate.height) / 2,
            candidate.width, candidate.height)))
    return tuple(region_rects), tuple(y_rects)


def _row_extent(spec, rows, region, row, right_legends=()):
    """The rectangle a row label must clear: profiles, stacks and their
    region labels, colorbars and colorbar titles, the cells' own legends and
    right grid legends beside the row."""
    boxes = []
    for index, cell in enumerate(spec.cells):
        if cell.row != row:
            continue
        if rows.panels[index] is not None:
            boxes.append(_expand(rows.panels[index],
                                 cell.insets.profile.axis_insets))
        if rows.cell_legends[index] is not None:
            boxes.append(rows.cell_legends[index])
        boxes.extend(_expand(rect, inset) for rect, inset in zip(
            rows.blocks[index], cell.insets.blocks))
    boxes.extend(rect for owner, _, _, rect in region if owner == row)
    # Bars beside the row count; below bars sit in its bottom reservation.
    for item, rect, title in rows.bars:
        if item.location == 'right' and (
                item.owner == row if item.scope == 'row' else
                spec.cells[item.owner].row == row):
            boxes.append(_expand(rect, item.insets))
            if title is not None:
                boxes.append(title)
    y0 = min(box.y0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    boxes.extend(rect for rect in right_legends
                 if rect.y0 < y1 and rect.y1 > y0)
    return Rect(min(box.x0 for box in boxes), min(box.y0 for box in boxes),
                max(box.x1 for box in boxes) - min(box.x0 for box in boxes),
                max(box.y1 for box in boxes) - min(box.y0 for box in boxes))


def _row_decorations(spec, columns, bands, widths, rows):
    """Place row labels, titles, legends and all structural label runs."""
    row_label_gap = spec.geometry.profile_panel_to_row_header_gap
    region = []
    heatmap_y = []
    for row in range(spec.rows):
        row_region, row_y = _stack_labels_for_row(
            spec, columns, widths, rows, row)
        region.extend(row_region)
        heatmap_y.extend(row_y)
    titles = []
    for index, (cell_index, cell) in enumerate(_titles(spec)):
        candidate = bands.title_candidates[index]
        if candidate is None:
            raise ValueError('panel title selection is incomplete')
        # A title sits over the cell's top element: its profile, else stack.
        top = (rows.panels[cell_index] if rows.panels[cell_index] is not None
               else rows.blocks[cell_index][0])
        width = candidate.width if candidate.width > 0 else widths.size.width
        # Titles also clear the right legends rising above the row's
        # profiles, as the row gap reserves them over the top extents.
        edge = max((rect.y1 for other, rect in zip(spec.cells,
                                                   rows.cell_legends)
                    if rect is not None and other.row == cell.row and
                    other.legend.location == 'right'), default=top.y1)
        titles.append((cell_index, Rect(
            top.x + (widths.size.width - width) / 2,
            max(top.y1, edge) + widths.title_gap, width, candidate.height)))
    axes = tuple((index, _axis_label_rect(spec, widths, bands, rows,
                                          index, run))
                 for index, run in enumerate(spec.runs)
                 if run.axis in ('x', 'y') and run.panel_indices)
    grid_legends = tuple((legend, _place_grid_legend(
        spec, widths, columns, rows, titles, axes, legend))
        for legend in spec.decorations
        if legend.kind == 'legend' and legend.scope in ('row', 'column'))
    # Row labels clear the row, including right legends beside its profiles.
    right_legends = tuple(rect for legend, rect in grid_legends
                          if legend.location == 'right')
    facets = tuple(
        (facet.owner, candidate,
         _row_extent(spec, rows, region, facet.owner, right_legends))
        for facet, candidate in zip(_facets(spec), widths.row_candidates)
        if any(cell.row == facet.owner for cell in spec.cells) and
        candidate.width > 0 and candidate.height > 0)
    # Row labels share one line right of the widest row.
    x = max((extent.x1 for _, _, extent in facets), default=0.0) + \
        row_label_gap
    labels = [(owner, Rect(x, extent.y0 +
                           (extent.height - candidate.height) / 2,
                           candidate.width, candidate.height), candidate)
              for owner, candidate, extent in facets]
    return _RowDecorations(
        rows, tuple(labels), tuple(titles), grid_legends, axes,
        tuple(region), tuple(heatmap_y))


def _figure_body(spec, rows):
    body = []
    for index, cell in enumerate(spec.cells):
        panel = rows.rows.panels[index]
        if panel is not None:
            body.append(_expand(panel, cell.insets.profile.total_extents))
        for block_index, rect in enumerate(rows.rows.blocks[index]):
            inset = (cell.insets.blocks[block_index]
                     if block_index < len(cell.insets.blocks) else
                     cell.insets.stack.axis_insets)
            body.append(_expand(rect, inset))
    body.extend(rect for rect in rows.rows.cell_legends if rect is not None)
    body.extend(rect for _, rect in rows.grid_legends)
    body.extend(rect for _, rect, _ in rows.row_labels)
    body.extend(rect for _, rect in rows.axis_labels)
    body.extend(rect for _, rect in rows.panel_titles)
    body.extend(rect for _, _, _, rect in rows.region_labels)
    body.extend(rect for _, _, rect in rows.heatmap_y_labels)
    body.extend(rect for rect in rows.rows.indicators if rect is not None)
    for item, rect, title in rows.rows.bars:
        inset = item.insets
        if item.below_entry is not None and item.location == 'below':
            entry = item.below_entry
            inset = Insets(entry.left_overhang, entry.right_overhang,
                           0.0, entry.tick_height)
        body.append(_expand(rect, inset))
        if title is not None:
            body.append(title)
    return tuple(body)


def _figure_legend(spec, widths, columns, rows, body, left, y0, y1):
    common = next(iter(_legends(spec, 'figure')), None)
    if common is None or common.size is None:
        return None
    size = common.size
    right = max(rect.x1 for rect in body)
    bottom = min(rect.y0 for rect in body)
    top = max(rect.y1 for rect in body)
    if common.location == 'above':
        return Rect((columns.width - size.width) / 2,
                    top + widths.legend_gap, size.width, size.height)
    if common.location == 'below':
        return Rect((columns.width - size.width) / 2,
                    bottom - widths.legend_gap - size.height,
                    size.width, size.height)
    if common.location == 'right':
        return Rect(right + columns.right_legend_gap,
                    y0 + (y1 - y0 - size.height) / 2,
                    size.width, size.height)
    if common.location == 'left':
        return Rect(left - widths.legend_gap - size.width,
                    y0 + (y1 - y0 - size.height) / 2,
                    size.width, size.height)
    return None


def _figure_decorations(spec, columns, bands, widths, rows):
    """Collect the measured body and position shared figure decorations."""
    body = list(_figure_body(spec, rows))
    bars = list(rows.rows.bars)
    common = tuple(item for item in spec.decorations
                   if item.kind == 'colorbar' and item.scope == 'figure'
                   and item.location == 'right')
    if common:
        data = tuple(rect for index, panel in enumerate(rows.rows.panels)
                     for rect in (((panel,) if panel is not None else ()) +
                                  rows.rows.blocks[index]))
        bottom = min(rect.y0 for rect in data)
        top = max(rect.y1 for rect in data)
        right = max(rect.x1 for rect in body)
        thickness = _bar_thickness(spec)
        gap = _bar_gap(spec)
        colorbars = tuple(item for item in spec.decorations
                          if item.kind == 'colorbar')
        start_index = colorbars.index(common[0])
        chosen = bands.bar_titles[start_index:start_index + len(common)]
        label_heights = tuple((item.height + _bar_title_gap(spec))
                              if item.height else 0.0 for item in chosen)
        placed, _ = pack_colorbar_grid(
            len(common), top - bottom, right + gap + common[0].insets.left,
            top, thickness=thickness,
            tick_label_widths=tuple(item.tick_label_width for item in common),
            label_band_heights=label_heights,
            label_widths=tuple(item.width for item in chosen),
            geometry=spec.geometry)
        for item, rect, candidate in zip(common, placed, chosen):
            title = (Rect(rect.x, rect.y1 + _bar_title_gap(spec),
                          candidate.width, candidate.height)
                     if candidate.height else None)
            bars.append((item, rect, title))
            body.append(_expand(rect, item.insets))
            if title is not None:
                body.append(title)
    body = tuple(body)
    # Figure decorations centre on every data rectangle: profiles and stacks.
    data = tuple(rect for index, panel in enumerate(rows.rows.panels)
                 for rect in ((panel,) if panel is not None else ()) +
                 rows.rows.blocks[index])
    left = min((rect.x0 for rect in body), default=0.0)
    left = min((left, *(rect.x0 for index, rect in rows.axis_labels
                        if spec.runs[index].axis == 'y')))
    y0 = min(rect.y0 for rect in data)
    y1 = max(rect.y1 for rect in data)
    legend = _figure_legend(spec, widths, columns, rows, body,
                            left, y0, y1)
    top = max((rect.y1 for rect in body), default=0.0)
    return _FigureDecorations(rows, legend, body, top, tuple(bars))


def _figure_bands(spec, columns, figure):
    """Place the figure title above the solved grid."""
    title = None
    if spec.title is not None:
        candidate = spec.title.candidates[0]
        width = candidate.width or columns.width
        top = (figure.legend.y1 if figure.legend is not None and
               next(iter(_legends(spec, 'figure')), None).location == 'above'
               else figure.body_top)
        title = Rect((columns.width - width) / 2,
                     top + _title_gap(spec),
                     width, candidate.height)
    return _FigureBands(figure, title)


def _selected_indices(spec, widths, bands, rows):
    selected = {}
    for index, (_, cell) in enumerate(_titles(spec)):
        candidate = bands.title_candidates[index]
        selected[cell_title(cell.row, cell.column)] = _candidate_index(
            cell.labels.title_candidates, candidate)
    for index, facet in enumerate(_facets(spec)):
        if any(owner == facet.owner for owner, _, _ in rows.row_labels):
            selected[row_label(facet.owner)] = _candidate_index(
                facet.candidates, widths.row_candidates[index])
    axis_runs = tuple((run, bands.run_candidates[index])
                      for index, run in enumerate(spec.runs)
                      if run.axis in ('x', 'y'))
    for role, (run, candidate) in zip(
            axis_run_roles(spec.runs, spec.cells), axis_runs):
        selected[role] = _candidate_index(run.candidates, candidate)
    for stack in widths.stacks:
        for ordinal, ((_, runs), (_, candidates)) in enumerate(zip(
                stack_run_groups(spec, stack.row), stack.region_groups)):
            for index, (run, chosen) in enumerate(zip(runs, candidates)):
                selected[label_role('stack', stack.row, ordinal, index)] = \
                    _candidate_index(run.candidates, chosen)
        for column, chosen in enumerate(stack.y_candidates):
            if chosen is None or not chosen.width:
                continue
            cell = tuple(cell for cell in spec.cells
                         if cell.row == stack.row)[column]
            selected[label_role('heatmap_y', stack.row, cell.column)] = \
                _candidate_index(cell.labels.y_candidates, chosen)
    bars = tuple(item for item in spec.decorations
                 if item.kind == 'colorbar')
    for item, chosen in zip(bars, bands.bar_titles):
        if item.candidates:
            selected[colorbar_title_role(item.scope, item.owner,
                                         item.ordinal)] = \
                _candidate_index(item.candidates, chosen)
    return selected


def _candidate_index(candidates, chosen):
    for index, candidate in enumerate(candidates):
        if candidate == chosen:
            return index
    raise ValueError('selected label candidate is absent from input')


def _stack_region_selection(spec, heights, gap, row, max_gap, labels=None):
    if labels is None:
        labels = _stack_label_runs(spec, row)
    selected = tuple(item.candidates[0] for item in labels)
    if not labels or not spec.label_layout.auto_heatmap_region_label_layout:
        return selected, gap
    reversed_sets = tuple(item.candidates for item in reversed(labels))
    extent = sum(heights) + gap * (len(heights) - 1)
    scale = max(spec.columns * spec.cells[0].width, 1.0)
    band = SharedGapBandSpec(
        candidate_sets=reversed_sets,
        spans=tuple((index, index) for index in range(len(labels))),
        primary_dimension='height',
        extra_row_penalty=spec.label_layout.heatmap_region_label_extra_row_penalty,
        balance_weight=spec.label_layout.heatmap_region_label_balance_weight,
        orphan_weight=spec.label_layout.heatmap_region_label_orphan_weight,
        secondary_costs=tuple(tuple(c.width / scale for c in candidates)
                              for candidates in reversed_sets))
    chosen = optimize_shared_gap_layout(
        (band,), cell_extent=tuple(reversed(heights)),
        base_common_gap=gap,
        base_private_gaps=(0.0,) * max(0, len(heights) - 1),
        original_block_extent=extent,
        max_common_gap_points=max_gap,
        hard_outer_containment=False,
        label_min_clearance=_label_clearance(spec))
    return (tuple(reversed(chosen.bands[0].candidates)),
            chosen.common_gap if len(heights) > 1 else gap)


def _below_legend_reaches_labels(spec, cells, row):
    """Whether a below legend under a row's profiles can reach the region
    labels beside its stacks.

    A right cell legend hanging into the band sits over the labels. A below
    legend is centred on the profiles it describes and the labels start a
    region gap beyond them, so it reaches them only when it is wider than
    those profiles plus a region gap on each side. Column gaps are not known
    yet, so the profiles' own widths bound the span from below.
    """
    reach = 2 * _stack_region_gap(spec)
    if any(cell.legend is not None and (
            cell.legend.location == 'right' or
            (cell.legend.location == 'below' and
             cell.legend.size.width > cell.width + reach))
           for cell in cells):
        return True
    span = sum(cell.width for cell in cells if cell.profile is not None)
    return any(legend.size.width > span + reach
               for legend in _legends(spec, 'row')
               if legend.location == 'below' and legend.owner == row)


def _stack_y_selection(spec, stack_height, block_gap, row, overhang=0.0):
    """``overhang`` is how far region labels rise above the stack's top."""
    cells = tuple(cell for cell in spec.cells if cell.row == row)
    y = tuple((cell.labels.y_candidates[0]
               if cell.labels.y_candidates else None) for cell in cells)
    summary_gap = spec.geometry.heatmap_to_summary_plot_gap
    has_profile = any(cell.profile is not None for cell in cells)
    summary_height = max((cell.profile.height for cell in cells
                          if cell.profile is not None), default=0.0)
    bottom = max((cell.insets.profile.total_extents.bottom for cell in cells
                  if cell.profile is not None), default=0.0)
    axis = max((cell.insets.profile.axis_insets.bottom for cell in cells
                if cell.profile is not None), default=0.0)
    private = bottom + (_below_row_legend(spec, row)
                        if has_profile else 0.0)
    # A legend hanging below the profiles stays above the region labels
    # rising over the stack when it is wide enough to reach them.
    if private > axis and _below_legend_reaches_labels(spec, cells, row):
        private += overhang
    if not spec.label_layout.auto_axis_label_layout:
        return y, summary_gap, private, summary_height
    scale = max(spec.columns * cells[0].width, 1.0)
    indexed = tuple((index, cell.labels.y_candidates)
                    for index, cell in enumerate(cells)
                    if cell.labels.y_candidates and
                    cell.labels.y_candidates[0].width)
    if not indexed:
        return y, summary_gap, private, summary_height
    bands = tuple(SharedGapBandSpec(
        candidate_sets=(candidates,), spans=((0, 0),),
        primary_dimension='height',
        extra_row_penalty=spec.label_layout.axis_label_extra_row_penalty,
        balance_weight=spec.label_layout.axis_label_balance_weight,
        orphan_weight=spec.label_layout.axis_label_orphan_weight,
        secondary_costs=(tuple(candidate.width / scale for candidate in candidates),))
        for _, candidates in indexed)
    extents = (stack_height, summary_height) if has_profile else (stack_height,)
    original = (sum(extents) + summary_gap + private if has_profile
                else stack_height)
    chosen = optimize_shared_gap_layout(
        bands, cell_extent=extents,
        base_common_gap=summary_gap if has_profile else 0.0,
        base_private_gaps=(private,) if has_profile else (),
        original_block_extent=max(original, 1.0),
        max_common_gap_points=(spec.label_layout.max_row_gap_points
                               if has_profile else 0.0),
        outer_left_allowance=block_gap / 2,
        outer_right_allowance=0.0,
        hard_outer_containment=has_profile,
        label_min_clearance=_label_clearance(spec))
    result = list(y)
    for (index, _), band in zip(indexed, chosen.bands):
        result[index] = band.candidates[0]
    return (tuple(result), chosen.common_gap if has_profile else 0.0,
            private, summary_height)


def _stack_widths_for_row(spec, row):
    """Select stack labels and heatmap Y widths for one occupied row."""
    cells = tuple(cell for cell in spec.cells if cell.row == row and cell.stack)
    gap = spec.geometry.heatmap_group_gap
    heights = cells[0].stack
    growth = spec.label_layout.heatmap_region_label_gap_growth
    cap = (spec.label_layout.max_heatmap_region_gap_points if growth else gap)
    groups = stack_run_groups(spec, row)
    selected_groups = []
    selected_gap = gap
    for members, labels in groups:
        selected, group_gap = _stack_region_selection(
            spec, spec.cells[members[0]].stack, gap, row, cap, labels)
        selected_groups.append((members, selected))
        selected_gap = max(selected_gap, group_gap)
    region = tuple(candidate for _, selected in selected_groups
                   for candidate in selected)
    if growth:
        gap = selected_gap
    height = sum(heights) + gap * max(0, len(heights) - 1)
    overhang = max((max(0.0, (selected[0].height -
                              spec.cells[members[0]].stack[0]) / 2)
                    for members, selected in selected_groups if selected),
                   default=0.0)
    y, summary_gap, private, summary_height = \
        _stack_y_selection(spec, height, gap, row, overhang)
    return _StackWidths(row, gap, height, region, y,
                        summary_gap, private, summary_height,
                        tuple(selected_groups))


def _provisional_bands(spec, widths):
    """First-choice labels for row reservations before the column solve."""
    bars = tuple(item for item in spec.decorations if item.kind == 'colorbar')
    return _Bands(None, widths.run_candidates, (),
                  tuple(item.candidates[0] if item.candidates else _EMPTY_LABEL
                        for item in bars))


def _below_common_cap(spec):
    return spec.geometry.colorbar_below_common_max_gap


def _below_entries(bars, titles):
    return tuple(replace(item.below_entry, label_width=title.width,
                         label_height=title.height)
                 for item, title in zip(bars, titles))


def _span(columns, cells, start, end):
    left = columns.starts[start]
    return left, columns.starts[end] + cells[end].width - left


def _row_colorbars(spec, row):
    occupied = tuple(index for index in range(spec.rows)
                     if any(cell.stack for cell in spec.cells
                            if cell.row == index))
    return tuple(item for item in spec.decorations
                 if item.kind == 'colorbar' and
                 ((item.scope == 'figure' and
                   item.location == 'below' and row == occupied[-1]) or
                  (item.scope == 'row' and item.owner == row) or
                  (item.scope == 'cell' and
                   spec.cells[item.owner].row == row)))


def _bar_titles(spec, bands, items):
    """Each colorbar's own chosen title candidate."""
    colorbars = tuple(item for item in spec.decorations
                      if item.kind == 'colorbar')
    return tuple(bands.bar_titles[colorbars.index(item)] for item in items)


def _pack_common_below(spec, bands, common, top):
    return pack_below_common(
        _below_entries(common, _bar_titles(spec, bands, common)),
        spec.cells[0].width, _bar_thickness(spec), bands.columns.width,
        0.0, top,
        horizontal_gap=min(bands.columns.common_gap, _below_common_cap(spec)),
        geometry=spec.geometry,
        include_label_widths=(spec.label_layout.
                              auto_horizontal_colorbar_label_layout))


def _reservations(spec, widths, bands):
    """Each row's bottom reservation, computed once (0 without a stack)."""
    return tuple(_row_bottom_reservation(spec, bands, row)
                 if _stack_for_row(widths, row) is not None else 0.0
                 for row in range(spec.rows))


def _row_bottom_reservation(spec, bands, row):
    """Height of the X decorations and below bars beneath one stack row."""
    bars = _row_colorbars(spec, row)
    below = tuple(item for item in bars if item.scope == 'cell'
                  and item.location == 'below')
    common = tuple(item for item in bars if item.scope == 'figure'
                   and item.location == 'below')
    gap = _bar_gap(spec)
    indices = tuple(index for index, run in enumerate(spec.runs)
                    if run.axis == 'x' and run.panel_indices and
                    _span_extent(spec, run, 'row')[1] == row)
    x_height = max((bands.run_candidates[index].height for index in indices),
                   default=0.0)
    clearance = max((_run_gap(spec.runs[index], _common_axis_gap(spec))
                     for index in indices), default=0.0)
    bottom = spec.x_decoration_height if indices else 0.0
    bottom += clearance + x_height if x_height else 0.0
    if below:
        bottom += gap + _bar_thickness(spec) + max(item.insets.bottom
                                                   for item in below)
        label = max(title.height for title in
                    _bar_titles(spec, bands, below))
        if label:
            bottom += label + _bar_title_gap(spec)
    if common and bands.columns is not None and \
            any(item.below_entry is not None for item in common):
        bottom += gap + _pack_common_below(spec, bands, common, 0.0)[2]
    return bottom


def _place_row_blocks(cells, stack, columns, bottom):
    blocks = []
    for cell in cells:
        cursor = bottom + stack.stack_height
        placed = []
        for height in cell.stack:
            cursor -= height
            placed.append(Rect(columns.starts[cell.column], cursor,
                               cell.width, height))
            cursor -= stack.block_gap
        blocks.append(tuple(placed))
    return tuple(blocks)


def _place_row_bars(spec, stack, bands, bottom, reserve, row):
    """Place a stack row's cell, row or figure bars and their titles.

    ``reserve`` is the row's bottom reservation; below bars fill it from
    its lower edge.
    """
    bars = _row_colorbars(spec, row)
    thickness = _bar_thickness(spec)
    title_gap = _bar_title_gap(spec)
    base = bottom - reserve
    cell_bars = tuple(item for item in bars if item.scope == 'cell')
    figure_below = tuple(item for item in bars if item.scope == 'figure'
                         and item.location == 'below')
    row_right = tuple(item for item in bars if item.scope == 'row'
                      and item.location == 'right')
    if cell_bars:
        y = base + max(item.insets.bottom for item in cell_bars)
        placed = []
        for item in cell_bars:
            start, end = item.covered_cells or (item.owner, item.owner)
            left, width = _span(bands.columns, spec.cells,
                                spec.cells[start].column,
                                spec.cells[end].column)
            data_width = spec.cells[start].width
            placed.append(Rect(left + (width - data_width) / 2,
                               y, data_width, thickness))
        titles = tuple(
            Rect(rect.x + (rect.width - title.width) / 2, rect.y1 + title_gap,
                 title.width, title.height) if title.height else None
            for rect, title in zip(placed,
                                   _bar_titles(spec, bands, cell_bars)))
        return tuple(zip(cell_bars, placed, titles))
    if figure_below:
        placed, titles, height = _pack_common_below(
            spec, bands, figure_below, 0.0)
        return tuple((item, _move(rect, 0.0, base + height),
                      _move(title, 0.0, base + height))
                     for item, rect, title in zip(figure_below, placed, titles))
    if not row_right:
        return ()
    labels = _stack_label_runs(spec, row)
    width = max((candidate.width for candidate in stack.region_candidates),
                default=0.0)
    region_right = (width + _stack_region_gap(spec)
                    if labels and width and labels[0].location == 'right'
                    else 0.0)
    chosen = _bar_titles(spec, bands, row_right)
    placed, _ = pack_colorbar_grid(
        len(row_right), stack.stack_height,
        bands.columns.width + region_right + _bar_gap(spec) +
        row_right[0].insets.left,
        bottom + stack.stack_height, thickness=thickness,
        tick_label_widths=tuple(item.tick_label_width for item in row_right),
        label_band_heights=tuple(title.height + title_gap if title.height
                                 else 0.0 for title in chosen),
        label_widths=tuple(title.width for title in chosen),
        geometry=spec.geometry)
    return tuple((item, rect, Rect(rect.x, rect.y1 + title_gap, title.width,
                                   title.height) if title.height else None)
                 for item, rect, title in zip(row_right, placed, chosen))


def _emit_cell_rects(spec, placed, dx, dy):
    rects = {}
    for index, cell in enumerate(spec.cells):
        panel = placed.panels[index]
        if panel is not None:
            rects[cell_profile(cell.row, cell.column)] = _move(panel, dx, dy)
        for block, rectangle in enumerate(placed.blocks[index]):
            rects[cell_block(cell.row, cell.column, block)] = \
                _move(rectangle, dx, dy)
        legend = placed.cell_legends[index]
        if legend is not None:
            rects[legend_role('cell', index)] = _move(legend, dx, dy)
    return rects


def _emit_label_rects(spec, rows, dx, dy):
    rects = {}
    for owner, rect, _ in rows.row_labels:
        rects[row_label(owner)] = _move(rect, dx, dy)
    roles = axis_run_roles(spec.runs, spec.cells)
    role_by_index = dict(zip(
        (index for index, run in enumerate(spec.runs)
         if run.axis in ('x', 'y')), roles))
    for index, rect in rows.axis_labels:
        rects[role_by_index[index]] = _move(rect, dx, dy)
    for index, rect in rows.panel_titles:
        cell = spec.cells[index]
        rects[cell_title(cell.row, cell.column)] = _move(rect, dx, dy)
    for row, run, block, rect in rows.region_labels:
        if rect.width and rect.height:
            rects[label_role('stack', row, run, block)] = _move(rect, dx, dy)
    for row, column, rect in rows.heatmap_y_labels:
        rects[label_role('heatmap_y', row, column)] = _move(rect, dx, dy)
    return rects


def _emit_decoration_rects(spec, decorated, figure, dx, dy):
    rows = decorated.rows
    rects = {}
    for role, rect in ((figure_title(), figure.title),
                       (legend_role('figure', 0), decorated.legend)):
        if rect is not None:
            rects[role] = _move(rect, dx, dy)
    for row, indicator in enumerate(rows.rows.indicators):
        if indicator is not None:
            rects[indicator_role(row)] = _move(indicator, dx, dy)
    for item, rect in rows.grid_legends:
        rects[legend_role(item.scope, item.owner)] = _move(rect, dx, dy)
    for item, rect, title in decorated.bars:
        rects[colorbar_role(item.scope, item.owner, item.ordinal)] = \
            _move(rect, dx, dy)
        if title is not None:
            rects[colorbar_title_role(item.scope, item.owner,
                                      item.ordinal)] = \
                _move(title, dx, dy)
    return rects


def _envelope(spec, widths, bands, figure):
    """Fit the full decoration envelope and emit role-keyed rectangles."""
    decorated = figure.figure
    rows = decorated.rows
    size, dx, dy = _fit_canvas(
        (*decorated.body, figure.title, decorated.legend),
        widths.padding)
    rects = _emit_cell_rects(spec, rows.rows, dx, dy)
    rects.update(_emit_label_rects(spec, rows, dx, dy))
    rects.update(_emit_decoration_rects(spec, decorated, figure, dx, dy))
    return SolvedGrid(size, rects, _selected_indices(spec, widths, bands, rows))


def solve(spec):
    """Measure-independent eight-phase solve for a cell grid."""
    _validate_decorations(spec)
    widths = _width_affecting_vertical(spec)
    columns = _columns(spec, widths)
    bands = _horizontal_bands(spec, columns, widths)
    rows = _rows(spec, bands.columns, bands, widths)
    row_decorations = _row_decorations(spec, bands.columns, bands,
                                       widths, rows)
    figure_decorations = _figure_decorations(
        spec, bands.columns, bands, widths, row_decorations)
    figure_bands = _figure_bands(spec, bands.columns, figure_decorations)
    return _envelope(spec, widths, bands, figure_bands)
