"""Renderer-measured word wrapping candidates for structural plot labels.

Candidate generation is independent of Matplotlib: callers inject a function
that measures a string and returns point-based ``Size`` data (or a
``(width, height)`` pair). Small candidate sets are enumerated exhaustively.
Large sets are sampled deterministically within each line-count group and then
Pareto-pruned, so the global layout optimiser still receives width/height and
typography trade-offs instead of only the most balanced wrap.
"""

import itertools
import math
import numbers
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import cache
from typing import Tuple

from uniseg.graphemecluster import grapheme_cluster_boundaries
from uniseg.linebreak import line_break_boundaries

from .geometry import (
    DEFAULT_LABEL_BALANCE_WEIGHT, DEFAULT_LABEL_EXTRA_ROW_PENALTY,
    DEFAULT_LABEL_MAX_LINES, DEFAULT_LABEL_ORPHAN_WEIGHT, Size)


DEFAULT_CANDIDATE_CAP = 512
_SHORT_FINAL_LINE_THRESHOLD = 0.45
_MATPLOTLIB_REFERENCE_PIXELS_PER_POINT = 2.0
_FEASIBILITY_TOLERANCE = 1e-9
# Emergency breaks are available only for tokens longer than this measured
# em threshold, and carry a substantial additive cost so legal UAX/word breaks
# remain preferable whenever the layout envelope can accommodate them.
_EMERGENCY_WORD_MIN_EM = 12.0
_EMERGENCY_BREAK_PENALTY = 1.0
_EMERGENCY_BOUNDARY_CAP = 64
_MATH_PLACEHOLDER = "\ufffc"
_NUMERIC_EXPRESSION = re.compile(
    r"^[+\-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+\-−]?\d+)?(?:/[+\-−]?\d+(?:\.\d+)?)?$")
_SCIENTIFIC_OPERATORS = frozenset("+−-=×*÷/|<>")
_NONSPACE_RUN = re.compile(r"\S+")
# Adding the same common gutter should cost nearly the same at different grid
# sizes; the small logarithmic factor only reflects the number of boundaries.
_SHARED_GAP_BOUNDARY_LOG_WEIGHT = 0.1


@dataclass(frozen=True)
class LabelCandidate:
    """One measured layout for a label.

    ``width`` and ``height`` are in typographic points.
    ``balance_penalty`` and ``orphan_penalty`` are dimensionless and minimized
    independently by the title-layout policy. ``break_penalty`` is an additive
    cost for last-resort grapheme breaks. ``text`` joins ``lines`` with explicit
    newlines for direct use by a renderer.
    """

    lines: Tuple[str, ...]
    width: float
    height: float
    line_count: int
    balance_penalty: float
    orphan_penalty: float
    break_penalty: float = 0.0

    @property
    def text(self):
        return "\n".join(self.lines)


def literal_label_candidate(text, size):
    """Represent already-measured literal text without generating wraps."""
    lines = tuple(text.split('\n'))
    return LabelCandidate(lines, size.width, size.height,
                          len(lines), 0.0, 0.0)


@dataclass(frozen=True)
class SharedGapBandSpec:
    """One ordered label band participating in a shared-gap solve.

    ``primary_dimension`` selects the candidate extent that lies along the
    shared grid axis. ``width`` is the column-oriented default; ``height``
    gives the orientation-neutral row analogue. Bands are separate visual
    tracks: this coordinator checks collisions only among spans within the
    same band and never compares rectangles from different bands/classes.
    """

    candidate_sets: Tuple[Tuple[LabelCandidate, ...], ...]
    spans: Tuple[Tuple[int, int], ...]
    primary_dimension: str = 'width'
    extra_row_penalty: float = DEFAULT_LABEL_EXTRA_ROW_PENALTY
    balance_weight: float = DEFAULT_LABEL_BALANCE_WEIGHT
    orphan_weight: float = DEFAULT_LABEL_ORPHAN_WEIGHT
    # Optional normalized cost for the orthogonal reservation (for example a
    # Y label's private horizontal thickness while solving row gaps).  The
    # caller supplies these costs explicitly because the normalization scale
    # belongs to the opposite orientation.
    secondary_costs: Tuple[Tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class SharedGapBandLayout:
    """Selected candidates for one shared-gap band."""

    candidates: Tuple[LabelCandidate, ...]


@dataclass(frozen=True)
class SharedGapLayout:
    """Common-gap geometry and selected candidates for every band."""

    common_gap: float
    boundary_gaps: Tuple[float, ...]
    bands: Tuple[SharedGapBandLayout, ...]
    objective_cost: float


@dataclass(frozen=True)
class _BandSelection:
    candidates: Tuple[LabelCandidate, ...]
    residual: float
    objective: float


def _expand_cell_extents(cell_extent, cell_count):
    if isinstance(cell_extent, bool):
        raise ValueError("cell extent must be a finite non-negative number")
    if isinstance(cell_extent, numbers.Real):
        return (float(cell_extent),) * cell_count
    try:
        return tuple(cell_extent)
    except TypeError:
        raise ValueError(
            "cell extent must be a number or per-cell sequence")


def _column_starts(cell_extents, common_gap, private_gaps):
    cell_extents = _expand_cell_extents(cell_extents, len(private_gaps) + 1)
    starts = [0.0]
    for index in range(1, len(cell_extents)):
        starts.append(starts[-1] + cell_extents[index - 1] + common_gap +
                      private_gaps[index - 1])
    return tuple(starts)


def _title_span_interval(span, candidate, cell_extent, common_gap,
                         private_gaps, dimension='width'):
    cell_extents = _expand_cell_extents(cell_extent, len(private_gaps) + 1)
    starts = _column_starts(cell_extents, common_gap, private_gaps)
    start, end = span
    span_width = (sum(cell_extents[start:end + 1]) +
                  sum(common_gap + private_gaps[index]
                      for index in range(start, end)))
    extent = getattr(candidate, dimension)
    x0 = starts[start] + (span_width - extent) / 2.0
    return x0, x0 + extent


def _title_reserved_encroachment(span, candidate, cell_extent, common_gap,
                                 private_gaps, dimension='width'):
    """Measure title width that spills outside its span's personal interval.

    Adjacent spans split each common gutter at its midpoint. A merged span owns
    its internal separators in full. Existing private reservations remain
    available to the horizontal label track, while their common-gap component
    is still split between the neighboring spans.
    """
    cell_extents = _expand_cell_extents(cell_extent, len(private_gaps) + 1)
    x0, x1 = _title_span_interval(
        span, candidate, cell_extents, common_gap, private_gaps, dimension)
    starts = _column_starts(cell_extents, common_gap, private_gaps)
    start, end = span
    left_spill = 0.0
    right_spill = 0.0
    if start > 0:
        left_limit = (starts[start] - common_gap / 2.0 -
                      private_gaps[start - 1])
        left_spill = max(0.0, left_limit - x0)
    if end < len(private_gaps):
        right_limit = (starts[end] + cell_extents[end] + common_gap / 2.0 +
                       private_gaps[end])
        right_spill = max(0.0, x1 - right_limit)
    return left_spill + right_spill


def _affine_fit_gap(at_lower, at_upper, lower, upper):
    """First gap where a linearly decreasing signed overflow reaches zero."""
    if at_lower <= 0:
        return lower
    if at_upper > 0 or lower == upper:
        return math.inf
    return lower + (upper - lower) * at_lower / (at_lower - at_upper)


def _outer_overhang(spans, candidates, cell_extent, common_gap, private_gaps,
                    dimension, left_allowance=0.0, right_allowance=0.0):
    cell_extents = _expand_cell_extents(cell_extent, len(private_gaps) + 1)
    block_extent = (sum(cell_extents) +
                    sum(common_gap + value for value in private_gaps))
    left = right = 0.0
    for span, candidate in zip(spans, candidates):
        start, end = _title_span_interval(
            span, candidate, cell_extents, common_gap, private_gaps, dimension)
        left = max(left, -start)
        right = max(right, end - block_extent)
    return (max(0.0, left - left_allowance) +
            max(0.0, right - right_allowance))


def _band_records(band, candidate_sets, spans, cell_extent, gap,
                  private_gaps, original_extent, left_allowance,
                  right_allowance, hard_outer_containment):
    """Measure each candidate's interval and local objective terms."""
    count = len(candidate_sets)
    minima = tuple(min(candidate.line_count for candidate in values)
                   for values in candidate_sets)
    minimum_global_lines = max(minima)
    secondary_costs = band.secondary_costs or tuple(
        tuple(0.0 for _ in values) for values in candidate_sets)
    cell_extents = _expand_cell_extents(cell_extent, len(private_gaps) + 1)
    block_extent = (sum(cell_extents) +
                    sum(gap + value for value in private_gaps))

    def typography(index, candidate):
        return (
            band.balance_weight * candidate.balance_penalty +
            band.orphan_weight * candidate.orphan_penalty +
            candidate.break_penalty +
            band.extra_row_penalty * max(
                0, candidate.line_count - minima[index])) / count

    intervals = tuple(tuple(_title_span_interval(
        span, candidate, cell_extents, gap, private_gaps,
        band.primary_dimension) for candidate in values)
        for span, values in zip(spans, candidate_sets))
    # Candidate costs are local; only overlaps couple neighboring spans.
    # The line and secondary maxima are small threshold sets, so each
    # threshold combination needs only one best path per final candidate.
    records = []
    for index, (span, values) in enumerate(zip(spans, candidate_sets)):
        group = []
        for choice, candidate in enumerate(values):
            start, end = intervals[index][choice]
            outer = (max(0.0, -start - left_allowance) +
                     max(0.0, end - block_extent - right_allowance))
            own = _title_reserved_encroachment(
                span, candidate, cell_extents, gap, private_gaps,
                band.primary_dimension)
            group.append((secondary_costs[index][choice],
                          typography(index, candidate) + outer / original_extent,
                          own + (outer if hard_outer_containment else 0.0),
                          outer))
        records.append(group)
    return records, intervals, minimum_global_lines, cell_extents


def _shared_midpoints(spans, cell_extents, gap, private_gaps):
    starts = _column_starts(cell_extents, gap, private_gaps)
    result = []
    for left, right in zip(spans, spans[1:]):
        boundary = left[1]
        if boundary + 1 != right[0]:
            result.append(None)
        else:
            result.append(starts[boundary] + cell_extents[boundary] +
                          (gap + private_gaps[boundary]) / 2)
    return tuple(result)


def _clearance_pairs(spans, candidate_sets, cell_extents, gap, private_gaps,
                     dimension):
    midpoints = _shared_midpoints(spans, cell_extents, gap, private_gaps)
    intervals = tuple(tuple(_title_span_interval(
        span, candidate, cell_extents, gap, private_gaps, dimension)
        for candidate in values)
        for span, values in zip(spans, candidate_sets))
    return tuple(tuple(tuple(
        midpoint is not None and left[1] >= midpoint and
        right[0] <= midpoint for right in intervals[index])
        for left in intervals[index - 1])
        for index, midpoint in enumerate(midpoints, 1))


def _pair_residual(left_end, right_start, needs_clearance, clearance):
    overlap = max(0.0, left_end - right_start)
    if needs_clearance and clearance > 0:
        return max(overlap, clearance - (right_start - left_end))
    return overlap


def _band_paths_at_limits(candidate_sets, records, intervals,
                          line_limit, secondary_limit, pressures, clearance):
    """Keep one best chain for each final candidate at fixed thresholds."""
    allowed = tuple(tuple(choice for choice, record in enumerate(group)
                          if (record[0] <= secondary_limit and
                              candidate_sets[index][choice].line_count
                              <= line_limit))
                    for index, group in enumerate(records))
    if any(not group for group in allowed):
        return ()
    states = {choice: (records[0][choice][2],
                       records[0][choice][1], (choice,))
              for choice in allowed[0]}
    for index in range(1, len(candidate_sets)):
        next_states = {}
        for choice in allowed[index]:
            own = records[index][choice]
            start = intervals[index][choice][0]
            next_states[choice] = min(
                (path[0] + own[2] + _pair_residual(
                    intervals[index - 1][previous][1], start,
                    pressures[index - 1][previous][choice], clearance),
                 path[1] + own[1], path[2] + (choice,))
                for previous, path in states.items())
        states = next_states
    return states.values()


def _band_finalists(band, candidate_sets, records, intervals,
                    minimum_global_lines, pressures, clearance):
    """Enumerate line and secondary thresholds without changing DP order."""
    line_limits = sorted({candidate.line_count for values in candidate_sets
                          for candidate in values})
    secondary_limits = sorted({record[0] for group in records
                               for record in group})
    finalists = []
    for line_limit in line_limits:
        for secondary_limit in secondary_limits:
            paths = _band_paths_at_limits(
                candidate_sets, records, intervals, line_limit,
                secondary_limit, pressures, clearance)
            extra_rows = max(0, line_limit - minimum_global_lines)
            for residual, local_cost, choices in paths:
                cost = (local_cost + secondary_limit +
                        extra_rows * band.extra_row_penalty)
                finalists.append((residual, cost, choices))
    return finalists


def _solve_band_at_gap(band, candidate_sets, spans, cell_extent, gap,
                       private_gaps, original_extent, left_allowance,
                       right_allowance, hard_outer_containment, clearance,
                       pressure_gap):
    """Select one band's wraps at a fixed physical gap."""
    records, intervals, minimum_global_lines, cell_extents = _band_records(
        band, candidate_sets, spans, cell_extent, gap, private_gaps,
        original_extent, left_allowance, right_allowance,
        hard_outer_containment)
    pressures = _clearance_pairs(
        spans, candidate_sets, cell_extents, pressure_gap, private_gaps,
        band.primary_dimension)
    finalists = _band_finalists(
        band, candidate_sets, records, intervals, minimum_global_lines,
        pressures, clearance)
    feasible = [item for item in finalists
                if item[0] <= _FEASIBILITY_TOLERANCE]
    chosen = min(feasible, key=lambda item: item[1:3]) if feasible else min(
        finalists, key=lambda item: item[:3])
    _, cost, choices = chosen
    selected = tuple(values[choice] for values, choice in zip(
        candidate_sets, choices))
    cost += (_outer_overhang(
        spans, selected, cell_extents, gap, private_gaps,
        band.primary_dimension, left_allowance, right_allowance) -
        sum(records[index][choice][3]
            for index, choice in enumerate(choices))) / original_extent
    residual = sum(records[index][choice][2] -
                   (records[index][choice][3]
                    if hard_outer_containment else 0.0)
                   for index, choice in enumerate(choices))
    residual += sum(_pair_residual(
        intervals[index - 1][choices[index - 1]][1],
        intervals[index][choices[index]][0],
        pressures[index - 1][choices[index - 1]][choices[index]], clearance)
                    for index in range(1, len(candidate_sets)))
    return _BandSelection(selected, residual, cost)


def _pair_breakpoints(intervals, clearance, first_fit):
    points = set()
    for index in range(1, len(intervals)):
        for left_low, left_high in intervals[index - 1]:
            for right_low, right_high in intervals[index]:
                required = first_fit(
                    (left_low[1] - right_low[0],
                     left_high[1] - right_high[0]))
                if math.isfinite(required):
                    points.add(required)
                if not clearance:
                    continue
                required = first_fit(
                    (left_low[1] + clearance - right_low[0],
                     left_high[1] + clearance - right_high[0]))
                if math.isfinite(required):
                    points.add(required)
    return points


def _band_breakpoints(candidate_sets, spans, dimension, cell_extents,
                      raw_private, lower_common, max_common_gap_points,
                      outer_left_allowance, outer_right_allowance,
                      clearance):
    points = {lower_common, max_common_gap_points}

    def first_fit(*constraints):
        return max((_affine_fit_gap(a, b, lower_common,
                                    max_common_gap_points)
                    for a, b in constraints), default=lower_common)

    block_at_lower = (sum(cell_extents) + sum(raw_private) +
                      len(raw_private) * lower_common)
    block_at_upper = (sum(cell_extents) + sum(raw_private) +
                      len(raw_private) * max_common_gap_points)
    starts_at_lower = _column_starts(
        cell_extents, lower_common, raw_private)
    starts_at_upper = _column_starts(
        cell_extents, max_common_gap_points, raw_private)
    intervals = []
    for index, values in enumerate(candidate_sets):
        span = spans[index]
        start, end = span
        candidates = []
        for candidate in values:
            low = _title_span_interval(
                span, candidate, cell_extents, lower_common,
                raw_private, dimension)
            high = _title_span_interval(
                span, candidate, cell_extents, max_common_gap_points,
                raw_private, dimension)
            candidates.append((low, high))
            constraints = []
            if start > 0:
                low_limit = (starts_at_lower[start] - lower_common / 2 -
                             raw_private[start - 1])
                high_limit = (starts_at_upper[start] -
                              max_common_gap_points / 2 -
                              raw_private[start - 1])
                constraints.append((low_limit - low[0],
                                    high_limit - high[0]))
            if end < len(raw_private):
                low_limit = (starts_at_lower[end] + cell_extents[end] +
                             lower_common / 2 + raw_private[end])
                high_limit = (starts_at_upper[end] + cell_extents[end] +
                              max_common_gap_points / 2 + raw_private[end])
                constraints.append((low[1] - low_limit,
                                    high[1] - high_limit))
            required = first_fit(*constraints)
            if math.isfinite(required):
                points.add(required)
            # Include outer-fit crossings even when no internal boundary
            # exists, as in a one-column layout.
            required = first_fit(
                (-low[0] - outer_left_allowance,
                 -high[0] - outer_left_allowance),
                (low[1] - block_at_lower - outer_right_allowance,
                 high[1] - block_at_upper - outer_right_allowance))
            if math.isfinite(required):
                points.add(required)
        intervals.append(candidates)
    points.update(_pair_breakpoints(
        intervals, clearance, first_fit))
    return tuple(sorted(points))


def _evaluate_shared_gap(common_gap, prepared_bands, cell_extents,
                         raw_private, original_block_extent,
                         outer_left_allowance, outer_right_allowance,
                         hard_outer_containment, base_common_gap,
                         mean_cell_extent, gap_boundary_factor, clearance):
    band_layouts = []
    ranking_residual = 0.0
    gap_delta = max(0.0, common_gap - base_common_gap)
    objective = (gap_delta / (2.0 * mean_cell_extent) *
                 gap_boundary_factor)
    for band, candidate_sets, spans in prepared_bands:
        if not candidate_sets:
            band_layouts.append(SharedGapBandLayout(candidates=()))
            continue
        optimized = _solve_band_at_gap(
            band, candidate_sets, spans, cell_extents, common_gap,
            raw_private, original_block_extent, outer_left_allowance,
            outer_right_allowance, hard_outer_containment, clearance,
            min(base_common_gap, common_gap))
        selected = optimized.candidates
        outer = _outer_overhang(
            spans, selected, cell_extents, common_gap, raw_private,
            band.primary_dimension, outer_left_allowance,
            outer_right_allowance)
        ranking_residual += (optimized.residual +
                             (outer if hard_outer_containment else 0.0))
        band_layouts.append(SharedGapBandLayout(candidates=selected))
        objective += optimized.objective
    return (SharedGapLayout(
        common_gap=common_gap,
        boundary_gaps=tuple(common_gap + value
                            for value in raw_private),
        bands=tuple(band_layouts),
        objective_cost=objective), ranking_residual)


def optimize_shared_gap_layout(
        bands, cell_extent, base_common_gap, base_private_gaps,
        original_block_extent, max_common_gap_points,
        outer_left_allowance=0.0, outer_right_allowance=0.0,
        hard_outer_containment=True, label_min_clearance=0.0):
    """Coordinate multiple independent label bands over one common gap.

    ``cell_extent`` may be one shared positive cell size or a sequence of
    positive per-cell extents. ``bands`` are evaluated independently at each
    relevant common-gap breakpoint. Common-gap extension cost is normalized
    by the data-cell scale, with only a small logarithmic adjustment for the
    boundary count. Each band's global line penalty and typography costs
    remain local to that band. Pre-existing physical private gutter
    reservations remain available to separate label bands but do not affect
    the cost of added common-gap width.
    Adjacent spans each own half of their external common gutter, while merged
    spans own internal separators in full. A band may contain one-cell or
    merged spans. ``secondary_costs`` are explicit normalized costs supplied
    by the caller for the orthogonal reservation; taking their maximum prices
    one private track rather than a sum of labels.

    This helper deliberately performs no cross-band rectangle collision
    checks. For example, sample-title-vs-X-label or facet-vs-Y-label overlap
    must be handled by a later coordinator with the relevant visual tracks.
    With ``hard_outer_containment=True``, overhang beyond the supplied edge
    allowances participates in feasibility (profile). With ``False``, it
    affects cost but not feasibility; the heatmap caller may reserve a
    directional outer margin. Neither rule relaxes the common-gap cap.
    """
    raw_private = tuple(float(value) for value in base_private_gaps)
    base_common_gap = float(base_common_gap)
    original_block_extent = float(original_block_extent)
    max_common_gap_points = float(max_common_gap_points)
    outer_left_allowance = float(outer_left_allowance)
    outer_right_allowance = float(outer_right_allowance)
    cell_count = len(raw_private) + 1
    cell_extents = _expand_cell_extents(cell_extent, cell_count)
    cell_extents = tuple(float(value) for value in cell_extents)
    lower_common = min(base_common_gap, max_common_gap_points)
    normalized_bands = tuple(
        (band, tuple(tuple(values) for values in band.candidate_sets),
         tuple(tuple(span) for span in band.spans))
        for band in bands)
    prepared_bands = []
    all_breakpoints = {lower_common, max_common_gap_points}
    for band, candidate_sets, spans in normalized_bands:
        points = _band_breakpoints(
            candidate_sets, spans, band.primary_dimension, cell_extents,
            raw_private, lower_common, max_common_gap_points,
            outer_left_allowance, outer_right_allowance,
            label_min_clearance)
        prepared_bands.append((band, candidate_sets, spans))
        all_breakpoints.update(points)

    boundary_count = len(raw_private)
    mean_cell_extent = sum(cell_extents) / len(cell_extents)
    gap_boundary_factor = (
        1.0 + _SHARED_GAP_BOUNDARY_LOG_WEIGHT * math.log(boundary_count)
        if boundary_count else 0.0)

    evaluations = [
        _evaluate_shared_gap(
            value, prepared_bands, cell_extents, raw_private,
            original_block_extent, outer_left_allowance,
            outer_right_allowance, hard_outer_containment, base_common_gap,
            mean_cell_extent, gap_boundary_factor, label_min_clearance)
        for value in sorted(all_breakpoints)]
    feasible = [item for item in evaluations
                if item[1] <= _FEASIBILITY_TOLERANCE]
    if feasible:
        return min(
            (layout for layout, _ in feasible),
            key=lambda layout: (
                layout.objective_cost, layout.common_gap,
                tuple(tuple(candidate.text for candidate in band.candidates)
                      for band in layout.bands)))
    # No band combination is collision-free at any breakpoint.  The maximum
    # is an upper bound rather than an unconditional target: widening an
    # outer-only or otherwise plateaued label cannot improve its residual.
    # Prefer the smallest equally good gap after residual and normal costs;
    # when residual improves monotonically, this naturally selects the cap.
    return min(
        evaluations,
        key=lambda item: (
            item[1], item[0].objective_cost, item[0].common_gap,
            tuple(tuple(candidate.text for candidate in band.candidates)
                  for band in item[0].bands)))[0]


def _validate_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))


def _coerce_size(value):
    if isinstance(value, Size):
        width, height = value.width, value.height
    elif (isinstance(value, (tuple, list)) and len(value) == 2):
        width, height = value
    else:
        raise ValueError(
            "text measurer must return Size or a (width, height) pair")
    result = []
    for dimension in (width, height):
        if (isinstance(dimension, bool) or
                not isinstance(dimension, numbers.Real)):
            raise ValueError(
                "text measurement dimensions must be finite non-negative numbers")
        dimension = float(dimension)
        if not math.isfinite(dimension) or dimension < 0:
            raise ValueError(
                "text measurement dimensions must be finite non-negative numbers")
        result.append(dimension)
    return Size(*result)


def _is_escaped(text, index):
    """Whether the character at ``index`` is preceded by odd backslashes."""
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _matched_math_end(line, index):
    """Return the end of a matched, unescaped ``$...$`` span, if any."""
    if line[index] != "$" or _is_escaped(line, index):
        return None
    end = index + 1
    while end < len(line):
        if line[end] == "$" and not _is_escaped(line, end):
            return end + 1
        end += 1
    return None


def _normalise_hard_line(line):
    """Return normalized text, its math-masked form, and offset mapping."""
    parts = []
    masked_parts = []
    source_offsets = [0]
    text_length = 0
    pending_space = False

    def append_plain(value):
        nonlocal text_length
        parts.append(value)
        masked_parts.append(value)
        for _ in value:
            text_length += 1
            source_offsets.append(text_length)

    def append_math(value):
        nonlocal text_length
        parts.append(value)
        masked_parts.append(_MATH_PLACEHOLDER)
        text_length += len(value)
        source_offsets.append(text_length)

    index = 0
    while index < len(line):
        math_end = _matched_math_end(line, index)
        if math_end is not None:
            if pending_space and parts:
                append_plain(" ")
            append_math(line[index:math_end])
            pending_space = False
            index = math_end
            continue
        character = line[index]
        if character.isspace():
            pending_space = bool(parts)
        else:
            if pending_space:
                append_plain(" ")
            append_plain(character)
            pending_space = False
        index += 1
    return ("".join(parts), "".join(masked_parts),
            tuple(source_offsets))


def _normalise_text(text):
    """Normalize soft whitespace while retaining every explicit hard break."""
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return tuple(_normalise_hard_line(line)
                 for line in canonical.split("\n"))


def _is_numeric_expression(value):
    """Whether a token is a decimal/grouped number or a numeric ratio."""
    return bool(_NUMERIC_EXPRESSION.fullmatch(
        value.strip("()[]{}.,;:!?")))


def _scientific_operator_breaks(masked_line):
    """Add after-operator breaks for compact, non-numeric expressions."""
    result = set()
    for match in _NONSPACE_RUN.finditer(masked_line):
        token_start, token_end = match.span()
        token = masked_line[token_start:token_end]
        if _is_numeric_expression(token):
            continue
        index = token_start
        while index < token_end:
            operator = masked_line[index]
            if operator not in _SCIENTIFIC_OPERATORS:
                index += 1
                continue
            if operator in "-−+":
                previous_is_operator = (
                    index > token_start and
                    masked_line[index - 1] in _SCIENTIFIC_OPERATORS)
                exponent_sign = (
                    operator in "-−+" and index >= token_start + 2 and
                    masked_line[index - 1] in "eE" and
                    masked_line[index - 2].isdigit() and
                    index + 1 < token_end and
                    masked_line[index + 1].isdigit())
                if (index > token_start and not previous_is_operator and
                        index + 1 < token_end and not exponent_sign):
                    result.add(index + 1)
                index += 1
                continue

            end = index + 1
            if operator in "=|<>*":
                while end < token_end and masked_line[end] == operator:
                    end += 1
            numeric_ratio = (
                operator == "/" and index > token_start and
                end < token_end and masked_line[index - 1].isdigit() and
                masked_line[end].isdigit())
            if numeric_ratio:
                index = end
                continue
            if index > token_start and end < token_end:
                result.add(end)
            index = end
    return result


def _emergency_breaks(line, masked_line, normal_breaks, source_offsets,
                      line_measure):
    """Return bounded grapheme cuts for measured, very long plain tokens."""
    if line_measure is None:
        return ()
    positions = set()
    normal = tuple(sorted(normal_breaks))
    em_height = None
    for match in _NONSPACE_RUN.finditer(masked_line):
        token_start, token_end = match.span()
        first = bisect_right(normal, token_start)
        last = bisect_left(normal, token_end)
        token_breaks = normal[first:last]
        chunk_starts = (token_start,) + tuple(token_breaks)
        chunk_ends = tuple(token_breaks) + (token_end,)
        for start, end in zip(chunk_starts, chunk_ends):
            masked_chunk = masked_line[start:end]
            if not masked_chunk or _MATH_PLACEHOLDER in masked_chunk:
                continue
            source_start, source_end = source_offsets[start], source_offsets[end]
            chunk = line[source_start:source_end]
            if (len(chunk) <= _EMERGENCY_WORD_MIN_EM or
                    _is_numeric_expression(chunk)):
                continue
            cluster_boundaries = tuple(grapheme_cluster_boundaries(chunk))
            cluster_count = max(0, len(cluster_boundaries) - 1)
            if cluster_count <= int(_EMERGENCY_WORD_MIN_EM):
                continue
            measured = _coerce_size(line_measure(chunk))
            if em_height is None:
                em_height = _coerce_size(line_measure("M")).height
            if em_height <= 0 or measured.width <= \
                    _EMERGENCY_WORD_MIN_EM * em_height:
                continue
            internal = cluster_boundaries[1:-1]
            if len(internal) > _EMERGENCY_BOUNDARY_CAP:
                indexes = _sampled_ranks(
                    len(internal), _EMERGENCY_BOUNDARY_CAP)
                internal = tuple(internal[index] for index in indexes)
            positions.update(source_start + boundary for boundary in internal)
    return tuple(sorted(positions))


def _bracketed_offsets(masked):
    """Offsets that fall inside square brackets."""
    inside = set()
    depth = 0
    for offset, character in enumerate(masked):
        if depth:
            inside.add(offset)
        depth = (depth + 1 if character == '[' else
                 max(0, depth - 1) if character == ']' else depth)
    return inside


def _soft_breaks(line_segments, line_measure=None):
    """Map legal and emergency breaks to (hard-line, offset, cost)."""
    legal_breaks = []
    emergency_breaks = []
    for segment_index, (line, masked, source_offsets) in enumerate(
            line_segments):
        if not line:
            continue
        normal_boundaries = {
            boundary
            for boundary in line_break_boundaries(masked)
            if 0 < boundary < len(masked)
        }
        normal_boundaries.update(_scientific_operator_breaks(masked))
        normal_boundaries = {
            boundary for boundary in normal_boundaries
            if 0 < boundary < len(masked)
        }
        # A bracketed unit such as "[n = 1,234]" breaks only as a last resort.
        bracketed = _bracketed_offsets(masked) & normal_boundaries
        normal_boundaries -= bracketed
        emergency_positions = _emergency_breaks(
            line, masked, normal_boundaries, source_offsets, line_measure)
        normal_positions = {source_offsets[value]
                            for value in normal_boundaries}
        emergency_positions = tuple(sorted(
            {source_offsets[value] for value in bracketed} |
            {position for position in emergency_positions
             if position not in normal_positions}))
        legal_breaks.extend((segment_index, position, 0.0)
                            for position in sorted(normal_positions))
        emergency_breaks.extend((segment_index, position,
                                 _EMERGENCY_BREAK_PENALTY)
                                for position in emergency_positions)
    return tuple(legal_breaks + emergency_breaks)


def _unrank_combination_colex(n, k, rank):
    """Return one k-combination by colexicographic rank in O(k log n)."""
    if k == 0:
        return ()
    chosen = [0] * k
    upper = n - 1
    for order in range(k, 0, -1):
        low, high = order - 1, upper
        while low < high:
            middle = (low + high + 1) // 2
            if math.comb(middle, order) <= rank:
                low = middle
            else:
                high = middle - 1
        value = low
        chosen[order - 1] = value
        rank -= math.comb(value, order)
        upper = value - 1
    return tuple(chosen)


def _sampled_ranks(count, budget):
    """Choose deterministic ranks spanning a combination stratum."""
    if count <= budget:
        return tuple(range(count))
    if budget == 1:
        return (count // 2,)
    denominator = budget - 1
    return tuple(
        (index * (count - 1) + denominator // 2) // denominator
        for index in range(budget))


def _sample_two_group_combinations(first_count, second_count, total_count,
                                   budget):
    """Sample combinations across two break-priority groups deterministically."""
    strata = []
    for selected_first in range(min(total_count - 1, first_count) + 1):
        selected_second = total_count - selected_first
        if selected_second > second_count:
            continue
        first_combinations = math.comb(first_count, selected_first)
        second_combinations = math.comb(second_count, selected_second)
        strata.append((selected_first, selected_second,
                       first_combinations * second_combinations,
                       second_combinations))
    if not strata or budget <= 0:
        return ()
    stratum_budgets = _stratum_budgets(len(strata) - 1, budget)
    result = []
    for stratum_index, (selected_first, selected_second, count,
                        second_combinations) in enumerate(strata):
        stratum_budget = stratum_budgets.get(stratum_index, 0)
        ranks = _sampled_ranks(count, min(count, stratum_budget))
        for rank in ranks:
            first_rank, second_rank = divmod(rank, second_combinations)
            first = _unrank_combination_colex(
                first_count, selected_first, first_rank)
            second = tuple(
                first_count + index for index in
                _unrank_combination_colex(
                    second_count, selected_second, second_rank))
            result.append(tuple(first) + second)
    return tuple(result)


def _stratum_budgets(last, cap):
    """Allocate a candidate cap across line counts without dropping all of one."""
    count = last + 1
    if count <= cap:
        base, extra = divmod(cap, count)
        return {index: base + (index < extra) for index in range(count)}
    if cap == 1:
        return {0: 1}
    return {index: 1 for index in _sampled_ranks(count, cap)}


def _candidate_lines(word_segments, soft_breaks, selected_breaks):
    cuts_by_segment = [[] for _ in word_segments]
    break_penalty = 0.0
    for break_index in selected_breaks:
        segment_index, offset, penalty = soft_breaks[break_index]
        cuts_by_segment[segment_index].append(offset)
        break_penalty += penalty
    lines = []
    for (text, _, _), cuts in zip(word_segments, cuts_by_segment):
        if not text:
            lines.append("")
            continue
        positions = (0,) + tuple(sorted(cuts)) + (len(text),)
        lines.extend(text[start:end].strip()
                     for start, end in zip(positions, positions[1:]))
    return tuple(lines), break_penalty


def _line_penalties(line_widths):
    if len(line_widths) < 2:
        return 0.0, 0.0
    widest = max(line_widths)
    if widest <= 0:
        return 0.0, 0.0
    # fsum is exact on every Python version (3.12 changed float sum()), so
    # reorderings of the same line widths tie exactly and break the same way.
    average = math.fsum(line_widths) / float(len(line_widths))
    balance = math.fsum(((width - average) / widest) ** 2
                        for width in line_widths) / float(len(line_widths))
    final_ratio = line_widths[-1] / widest
    threshold = _SHORT_FINAL_LINE_THRESHOLD
    orphan = (((threshold - final_ratio) / threshold) ** 2
              if final_ratio < threshold else 0.0)
    return balance, orphan


def _dominates(left, right):
    left_values = (left.width, left.height,
                   left.line_count, left.balance_penalty,
                   left.orphan_penalty, left.break_penalty)
    right_values = (right.width, right.height,
                    right.line_count, right.balance_penalty,
                    right.orphan_penalty, right.break_penalty)
    return (all(a <= b for a, b in zip(left_values, right_values)) and
            any(a < b for a, b in zip(left_values, right_values)))


def _pareto_frontier(candidates):
    """Remove candidates no better in any geometry or typography objective."""
    ordered = sorted(candidates, key=lambda candidate: (
        candidate.line_count, candidate.lines, candidate.width,
        candidate.height, candidate.balance_penalty,
        candidate.orphan_penalty, candidate.break_penalty))
    frontier = []
    for candidate in ordered:
        if not any(_dominates(other, candidate) for other in ordered
                   if other is not candidate):
            frontier.append(candidate)
    return tuple(frontier)


def generate_label_candidates(
        text, measure, max_lines=DEFAULT_LABEL_MAX_LINES,
        candidate_cap=DEFAULT_CANDIDATE_CAP, line_measure=None):
    """Measure useful Unicode line-break candidates for ``text``.

    ``measure`` must be a callable accepting text and returning either a
    point-based :class:`~deeptoolsr.plotting.geometry.Size` or a
    ``(width, height)`` pair in points. Explicit newlines are mandatory and
    override ``max_lines`` when the text already contains more lines.

    All layouts are considered when the candidate count fits the cap. Larger
    sets are sampled deterministically across line counts and combination
    ranks, with at most ``candidate_cap`` layouts measured before Pareto
    pruning. The default cap leaves every candidate for ordinary 2–10-word
    labels intact while bounding long-label work. ``line_measure`` optionally
    measures individual unrotated line advances for balance/orphan penalties;
    the primary ``measure`` callback always determines final candidate
    geometry. It defaults to ``measure`` for compatibility.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _validate_positive_integer(max_lines, "max_lines")
    _validate_positive_integer(candidate_cap, "candidate_cap")
    if not callable(measure):
        raise TypeError("measure must be callable")
    if line_measure is not None and not callable(line_measure):
        raise TypeError("line_measure must be callable")

    line_segments = _normalise_text(text)
    mandatory_lines = len(line_segments)
    max_lines = max(max_lines, mandatory_lines)

    if not any(text for text, _, _ in line_segments):
        blank_lines = tuple("" for _ in line_segments)
        # A truly empty (or whitespace-only) single line has no drawable
        # geometry. Explicit newlines, however, represent mandatory blank
        # lines and Matplotlib assigns those lines vertical extent.
        if "\n" in text.replace("\r\n", "\n").replace("\r", "\n"):
            rendered = _coerce_size(measure("\n".join(blank_lines)))
            return (LabelCandidate(
                lines=blank_lines, width=rendered.width,
                height=rendered.height,
                line_count=len(blank_lines), balance_penalty=0.0,
                orphan_penalty=0.0),)
        return (LabelCandidate(
            lines=blank_lines, width=0.0, height=0.0,
            line_count=len(blank_lines), balance_penalty=0.0,
            orphan_penalty=0.0),)

    @cache
    def measured(value):
        return _coerce_size(measure(value))

    @cache
    def measured_line(value):
        callback = measure if line_measure is None else line_measure
        return _coerce_size(callback(value))

    boundaries = _soft_breaks(line_segments, measured_line)
    maximum_soft_breaks = min(max_lines - mandatory_lines, len(boundaries))
    budgets = _stratum_budgets(maximum_soft_breaks, candidate_cap)
    legal_count = sum(penalty == 0.0 for _, _, penalty in boundaries)
    emergency_count = len(boundaries) - legal_count

    candidates = []
    seen_lines = set()
    for soft_break_count, budget in budgets.items():
        legal_combinations = (math.comb(legal_count, soft_break_count)
                              if soft_break_count <= legal_count else 0)
        if legal_combinations <= budget:
            selected_sets = list(itertools.combinations(
                range(legal_count), soft_break_count))
        else:
            selected_sets = [
                _unrank_combination_colex(
                    legal_count, soft_break_count, rank)
                for rank in _sampled_ranks(legal_combinations, budget)]
        remaining_budget = budget - len(selected_sets)

        # Legal opportunities are sampled before emergency grapheme cuts, so
        # fallback choices cannot displace a normal word/operator wrap.
        if remaining_budget and emergency_count:
            mixed_sets = _sample_two_group_combinations(
                legal_count, emergency_count, soft_break_count,
                remaining_budget)
            selected_sets.extend(mixed_sets)

        for selected in selected_sets:
            lines, break_penalty = _candidate_lines(
                line_segments, boundaries, selected)
            if lines in seen_lines:
                continue
            seen_lines.add(lines)
            joined = "\n".join(lines)
            line_widths = tuple(
                measured_line(line).width if line else 0.0 for line in lines)
            rendered = measured(joined)
            balance, orphan = _line_penalties(line_widths)
            candidates.append(LabelCandidate(
                lines=lines,
                width=rendered.width,
                height=rendered.height,
                line_count=len(lines),
                balance_penalty=balance,
                orphan_penalty=orphan,
                break_penalty=break_penalty))

    return _pareto_frontier(candidates)


def measure_matplotlib_text(text, renderer, font_properties, figure=None,
                            rotation=0.0):
    """Return a Matplotlib text bounding box as a point-based ``Size``.

    The renderer and font properties are supplied by the caller, keeping this
    adapter independent of figure creation and allowing candidate generation
    to use a scratch Agg renderer or an existing plot renderer. Matplotlib's
    ``Text.get_window_extent`` also reads the associated figure's DPI, so pass
    the renderer's figure when available. If omitted, a temporary figure is
    created at the renderer's effective DPI. Agg can round glyph metrics
    differently at different DPIs, so font properties are scaled to a
    canonical 144-DPI reference before measuring and the result is converted
    back to points. This keeps point geometry stable while still using the
    supplied renderer and font resolution. ``rotation`` mirrors the final
    text artist when a structural label is rotated.
    """
    from matplotlib.text import Text

    pixels_per_point = float(renderer.points_to_pixels(1.0))
    if not math.isfinite(pixels_per_point) or pixels_per_point <= 0:
        raise ValueError("renderer returned an invalid point-to-pixel scale")
    if figure is None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(dpi=pixels_per_point * 72.0)
        FigureCanvasAgg(figure)

    reference_scale = _MATPLOTLIB_REFERENCE_PIXELS_PER_POINT
    measured_font = font_properties.copy()
    measured_font.set_size(
        font_properties.get_size_in_points() * reference_scale /
        pixels_per_point)
    artist = Text(0.0, 0.0, text=text, fontproperties=measured_font,
                  rotation=rotation)
    # Figure.add_artist establishes the DPI context required by Text's public
    # extent API. Remove the scratch artist immediately so a supplied plot
    # figure is left unchanged.
    figure.add_artist(artist)
    try:
        bbox = artist.get_window_extent(renderer=renderer)
    finally:
        artist.remove()
    return Size(bbox.width / reference_scale,
                bbox.height / reference_scale)


ExtentCache = dict


class TextMeasurer:
    """Cache renderer extents for one figure's measurement pass."""

    def __init__(self, renderer, figure, dpi, cache):
        self.renderer = renderer
        self.figure = figure
        self.dpi = dpi
        self.cache = cache

    def measure(self, text, font_properties, rotation=0.0):
        from matplotlib import rcParams

        key = (text, tuple(font_properties.get_family()),
               font_properties.get_style(), font_properties.get_variant(),
               font_properties.get_weight(), font_properties.get_stretch(),
               font_properties.get_size_in_points(), rotation, 1.2,
               rcParams['text.usetex'], self.dpi)
        if key not in self.cache:
            self.cache[key] = measure_matplotlib_text(
                text, self.renderer, font_properties, self.figure,
                rotation)
        return self.cache[key]


def measure_label_candidates(text, renderer, font_properties, figure,
                             max_lines, rotation=0.0, measurer=None):
    """Generate wraps using rendered geometry and unrotated line advances."""
    def measure(value, angle):
        if measurer is not None:
            return measurer.measure(value, font_properties, angle)
        return measure_matplotlib_text(
            value, renderer, font_properties, figure=figure, rotation=angle)

    return generate_label_candidates(
        text, lambda value: measure(value, rotation), max_lines=max_lines,
        line_measure=(lambda value: measure(value, 0.0))
        if rotation else None)
