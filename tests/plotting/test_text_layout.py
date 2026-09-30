"""Tests for plotting text layout."""

import math
import time
import pytest
from uniseg.graphemecluster import grapheme_cluster_boundaries
from tests.helpers.label_solver_cases import r1_case
from deeptoolsr.plotting.geometry import Size
from deeptoolsr.plotting.scene import MatplotlibMeasureContext
from deeptoolsr.plotting.text_layout import (
    LabelCandidate, SharedGapBandSpec, SharedGapLayout,
    generate_label_candidates, measure_matplotlib_text,
    optimize_shared_gap_layout)
from deeptoolsr.plotting.text_layout import (
    DEFAULT_CANDIDATE_CAP, _SHARED_GAP_BOUNDARY_LOG_WEIGHT,
    _line_penalties, _outer_overhang,
    _title_reserved_encroachment, _title_span_interval)


# From test_text_layout.py.

def _fake_measurer(word_widths=None, default_width=1.0, line_height=10.0):
    word_widths = word_widths or {}
    calls = []

    def measure(text):
        calls.append(text)
        lines = text.split('\n')
        widths = [
            sum(word_widths.get(word, default_width) for word in line.split()) +
            max(0, len(line.split()) - 1)
            for line in lines
        ]
        return Size(max(widths, default=0.0),
                    0.0 if text == '' else line_height * len(lines))

    return measure, calls


def test_generates_one_two_and_three_line_candidates():
    measure, _ = _fake_measurer()

    candidates = generate_label_candidates(
        'one two three', measure, max_lines=3)

    assert {candidate.line_count for candidate in candidates} == {1, 2, 3}
    assert ('one two three',) in [candidate.lines for candidate in candidates]
    assert all(candidate.line_count <= 3 for candidate in candidates)


def test_wrap_balance_uses_measured_glyph_proportions_not_character_counts():
    measure, _ = _fake_measurer({'iiii': 4.0, 'WWWW': 12.0})

    candidates = generate_label_candidates(
        'iiii WWWW', measure, max_lines=2)

    wrapped = next(candidate for candidate in candidates
                   if candidate.lines == ('iiii', 'WWWW'))
    assert wrapped.width == 12.0
    assert wrapped.balance_penalty == pytest.approx(
        _line_penalties((4.0, 12.0))[0])


def test_balanced_two_line_break_scores_better_than_uneven_break():
    measure, _ = _fake_measurer({'a': 1.0, 'b': 5.0, 'c': 1.0, 'd': 4.0})

    candidates = generate_label_candidates('a b c d', measure, max_lines=2)
    balanced = next(candidate for candidate in candidates
                    if candidate.lines == ('a b', 'c d'))
    ugly_balance, _ = _line_penalties((9.0, 4.0))

    assert balanced.balance_penalty == pytest.approx(
        _line_penalties((7.0, 6.0))[0])
    assert balanced.balance_penalty < ugly_balance


def test_balance_and_short_final_line_penalties_are_separate():
    balanced_penalty, balanced_orphan = _line_penalties((10.0, 10.0))
    uneven_penalty, long_final_orphan = _line_penalties((2.0, 10.0))
    short_final_balance, short_final_orphan = _line_penalties((10.0, 2.0))

    assert balanced_penalty == 0.0
    assert balanced_orphan == 0.0
    assert uneven_penalty > balanced_penalty
    assert long_final_orphan == 0.0
    assert short_final_balance == uneven_penalty
    assert short_final_orphan > 0.0
    assert math.isfinite(short_final_orphan)


def test_explicit_newlines_are_mandatory_but_segments_can_wrap():
    measure, _ = _fake_measurer()

    candidates = generate_label_candidates(
        'alpha  beta gamma\n delta', measure, max_lines=3)

    assert ('alpha beta gamma', 'delta') in [
        candidate.lines for candidate in candidates]
    assert any(candidate.line_count == 3 for candidate in candidates)
    assert all(not any('gamma delta' in line for line in candidate.lines)
               for candidate in candidates)
    assert all(candidate.text == candidate.text.replace('  ', ' ')
               for candidate in candidates)


def test_shared_midpoint_contact_reserves_four_point_clearance():
    wide = LabelCandidate(('wide',), 20.0, 10.0, 1, 0.0, 0.0)
    band = SharedGapBandSpec(((wide,), (wide,)), ((0, 0), (1, 1)))
    kwargs = dict(cell_extent=(10.0, 10.0), base_common_gap=10.0,
                  base_private_gaps=(0.0,), original_block_extent=30.0,
                  max_common_gap_points=30.0,
                  outer_left_allowance=5.0, outer_right_allowance=5.0)
    old = optimize_shared_gap_layout((band,), **kwargs)
    clear = optimize_shared_gap_layout(
        (band,), label_min_clearance=4.0, **kwargs)
    assert old.common_gap == 10.0
    assert clear.common_gap == pytest.approx(14.0)
    left = _title_span_interval((0, 0), wide, (10.0, 10.0),
                                clear.common_gap, (0.0,))
    right = _title_span_interval((1, 1), wide, (10.0, 10.0),
                                 clear.common_gap, (0.0,))
    assert right[0] - left[1] == pytest.approx(4.0)


def test_one_sided_midpoint_pressure_keeps_original_gap():
    left = LabelCandidate(('left',), 20.0, 10.0, 1, 0.0, 0.0)
    right = LabelCandidate(('right',), 17.0, 10.0, 1, 0.0, 0.0)
    band = SharedGapBandSpec(((left,), (right,)), ((0, 0), (1, 1)))
    chosen = optimize_shared_gap_layout(
        (band,), cell_extent=(10.0, 10.0), base_common_gap=10.0,
        base_private_gaps=(0.0,), original_block_extent=30.0,
        max_common_gap_points=30.0, outer_left_allowance=5.0,
        outer_right_allowance=5.0, label_min_clearance=4.0)
    assert chosen.common_gap == 10.0


def test_mathtext_spans_are_indivisible_and_keep_internal_whitespace():
    measure, _ = _fake_measurer()
    text = r'left $x^2 + y^2$ middle $\alpha + \beta$ right'

    candidates = generate_label_candidates(text, measure, max_lines=3)

    assert candidates
    for candidate in candidates:
        assert candidate.text.count('$') == 4
        assert r'$x^2 + y^2$' in candidate.text
        assert r'$\alpha + \beta$' in candidate.text


def test_long_mathtext_span_is_not_used_for_emergency_breaks():
    measure, _ = _fake_measurer()
    expression = '$' + ('x + ' * 40) + 'y$'

    candidates = generate_label_candidates(
        expression, measure, max_lines=3)

    assert len(candidates) == 1
    assert candidates[0].text == expression
    assert candidates[0].break_penalty == 0.0


def test_mathtext_escaped_and_unmatched_dollars_remain_literal():
    measure, _ = _fake_measurer()
    text = r'price \$5 and $unfinished value'

    candidates = generate_label_candidates(text, measure, max_lines=2)

    assert candidates
    assert all(r'\$5' in candidate.text for candidate in candidates)
    assert all('$unfinished value' in candidate.text
               for candidate in candidates)


@pytest.mark.parametrize(('text', 'expected_first_line'), [
    ('treated/control', 'treated/'),
    ('a+b', 'a+'),
    ('x=y', 'x='),
    ('a−b', 'a−'),
    ('a-b', 'a-'),
    ('a+-b', 'a+'),
    ('x=-1/12', 'x='),
    ('x=1e-12', 'x='),
    ('a×b', 'a×'),
    ('a*b', 'a*'),
    ('a÷b', 'a÷'),
    ('x|y', 'x|'),
])
def test_unicode_and_scientific_operator_breaks_preserve_source_text(
        text, expected_first_line):
    def measure(value):
        lines = value.split('\n')
        return Size(max(map(len, lines), default=0), 10.0 * len(lines))

    candidates = generate_label_candidates(text, measure, max_lines=2)

    assert any(candidate.lines == (expected_first_line,
                                   text[len(expected_first_line):])
               for candidate in candidates)
    assert all(candidate.text.replace('\n', '') == text
               for candidate in candidates)


@pytest.mark.parametrize('text', [
    '1/12', '-1/12', '−1/12', '7.619', '7,619', '1e-12',
])
def test_numeric_expressions_do_not_get_operator_or_emergency_breaks(text):
    def measure(value):
        lines = value.split('\n')
        return Size(max(map(len, lines), default=0), 10.0 * len(lines))

    candidates = generate_label_candidates(text, measure, max_lines=3)

    assert tuple(candidate.lines for candidate in candidates) == ((text,),)
    assert candidates[0].break_penalty == 0.0


@pytest.mark.parametrize('text', ['x=-1/12', 'x=1/12'])
def test_numeric_ratio_stays_together_after_a_scientific_operator(text):
    def measure(value):
        lines = value.split('\n')
        return Size(max(map(len, lines), default=0), 10.0 * len(lines))

    candidates = generate_label_candidates(text, measure, max_lines=3)

    assert any(candidate.lines == ('x=', text[2:])
               for candidate in candidates)
    assert not any(line.endswith('/') for candidate in candidates
                   for line in candidate.lines[:-1])


def test_long_slash_halves_retain_the_legal_split_under_default_candidate_cap():
    left = 'alpha' * 40
    right = 'beta' * 32
    text = left + '/' + right

    def measure(value):
        lines = value.split('\n')
        return Size(max(map(len, lines), default=0), 10.0 * len(lines))

    candidates = generate_label_candidates(text, measure, max_lines=3)

    assert any(candidate.lines == (left + '/', right)
               for candidate in candidates)


def test_emergency_word_breaks_are_bounded_costed_and_grapheme_safe():
    text = 'a\u0301' * 80

    def measure(value):
        lines = value.split('\n')
        return Size(max(map(len, lines), default=0), 10.0 * len(lines))

    candidates = generate_label_candidates(
        text, measure, max_lines=3, candidate_cap=37)
    legal_boundaries = set(grapheme_cluster_boundaries(text))

    assert candidates
    assert len(candidates) <= 37
    assert any(candidate.break_penalty > 0.0 for candidate in candidates)
    for candidate in candidates:
        offset = 0
        for line in candidate.lines[:-1]:
            offset += len(line)
            assert offset in legal_boundaries
        assert candidate.text.replace('\n', '') == text


def test_default_candidate_cap_bounds_long_unicode_fallback_work():
    calls = []

    def measure(text):
        calls.append(text)
        lines = text.split('\n')
        return Size(max((len(line) for line in lines), default=0) * 5.0,
                    max(1, len(lines)) * 10.0)

    candidates = generate_label_candidates(
        'a\u0301' * 1000, measure, candidate_cap=DEFAULT_CANDIDATE_CAP)

    assert len(candidates) <= DEFAULT_CANDIDATE_CAP
    # At most three line measurements and one joined-layout measurement are
    # needed per sampled layout; no wall-clock threshold makes this stable.
    assert len(calls) <= DEFAULT_CANDIDATE_CAP * 4


def test_unicode_line_breaks_do_not_split_emoji_graphemes():
    text = '👩\u200d🔬' * 40

    def measure(value):
        lines = value.split('\n')
        return Size(max(map(len, lines), default=0), 10.0 * len(lines))

    candidates = generate_label_candidates(text, measure, max_lines=3)
    legal_boundaries = set(grapheme_cluster_boundaries(text))

    for candidate in candidates:
        offset = 0
        for line in candidate.lines[:-1]:
            offset += len(line)
            assert offset in legal_boundaries


def test_matplotlib_mathtext_candidates_render_without_broken_delimiters():
    from matplotlib.font_manager import FontProperties

    context = MatplotlibMeasureContext(dpi=100)
    font = FontProperties(size=10.0)

    def measure(text):
        return measure_matplotlib_text(
            text, context.renderer, font, figure=context.figure)

    expression = r'$x^2 + y^2$'
    candidates = generate_label_candidates(
        'left {} right'.format(expression), measure, max_lines=2)

    assert candidates
    math_size = measure(expression)
    literal_size = measure(r'\$x^2 + y^2\$')
    assert abs(math_size.width - literal_size.width) > 0.1
    for candidate in candidates:
        assert candidate.text.count('$') == 2
        assert expression in candidate.text
        assert measure(candidate.text).width > 0
    context.close()


def test_rotated_geometry_uses_unrotated_line_advances_for_typography():
    from matplotlib.font_manager import FontProperties

    context = MatplotlibMeasureContext(dpi=100)
    font = FontProperties(size=10.0)

    def rotated(text):
        return measure_matplotlib_text(
            text, context.renderer, font, figure=context.figure, rotation=90)

    def unrotated_line(text):
        return measure_matplotlib_text(
            text, context.renderer, font, figure=context.figure, rotation=0)

    candidates = generate_label_candidates(
        'AAAAAAAAAAAA i', rotated, max_lines=2,
        line_measure=unrotated_line)
    wrapped = next(candidate for candidate in candidates
                   if candidate.lines == ('AAAAAAAAAAAA', 'i'))

    assert unrotated_line(wrapped.lines[0]).width > \
        unrotated_line(wrapped.lines[1]).width * 5
    assert wrapped.balance_penalty > 0.1
    assert wrapped.width == pytest.approx(rotated(wrapped.text).width, abs=.05)
    assert wrapped.height == pytest.approx(rotated(wrapped.text).height, abs=.05)
    context.close()


def test_mandatory_breaks_override_max_lines():
    measure, _ = _fake_measurer()
    candidates = generate_label_candidates(
        'one\ntwo\nthree', measure, max_lines=2)
    assert any(candidate.text == 'one\ntwo\nthree'
               for candidate in candidates)


def test_explicit_blank_lines_keep_measured_height():
    measure, calls = _fake_measurer()

    candidates = generate_label_candidates('\n', measure, max_lines=2)

    assert len(candidates) == 1
    assert candidates[0].lines == ('', '')
    assert candidates[0].line_count == 2
    assert candidates[0].width == 0.0
    assert candidates[0].height == 20.0
    assert calls == ['\n']


def test_empty_text_is_a_deterministic_zero_size_candidate():
    measure, calls = _fake_measurer()

    candidates = generate_label_candidates('', measure)

    assert len(candidates) == 1
    assert candidates[0].lines == ('',)
    assert candidates[0].width == 0.0
    assert candidates[0].height == 0.0
    assert calls == []


def test_single_unbreakable_token_is_not_split():
    measure, _ = _fake_measurer()

    candidates = generate_label_candidates(
        'UnbreakableToken' * 20, measure, max_lines=3)

    assert len(candidates) == 1
    assert candidates[0].lines == ('UnbreakableToken' * 20,)
    assert candidates[0].line_count == 1


def test_pareto_pruning_retains_width_balance_tradeoffs_deterministically():
    measure, _ = _fake_measurer({
        'a': 13.0, 'b': 9.0, 'c': 9.0, 'd': 7.0, 'e': 11.0, 'f': 2.0})
    text = 'a b c d e f'

    first = generate_label_candidates(text, measure, max_lines=3)
    second = generate_label_candidates(text, measure, max_lines=3)

    narrower_less_balanced = next(
        candidate for candidate in first
        if candidate.lines == ('a', 'b c', 'd e f'))
    wider_more_balanced = next(
        candidate for candidate in first
        if candidate.lines == ('a b', 'c d', 'e f'))
    assert first == second
    assert narrower_less_balanced.width < wider_more_balanced.width
    assert (narrower_less_balanced.balance_penalty >
            wider_more_balanced.balance_penalty)


def test_pareto_pruning_retains_line_count_tradeoffs_for_global_row_cost():
    def measure(text):
        return Size(50.0 if '\n' in text else 100.0, 20.0)

    candidates = generate_label_candidates('alpha beta', measure, max_lines=2)

    assert {candidate.line_count for candidate in candidates} == {1, 2}
    assert any(candidate.lines == ('alpha beta',) for candidate in candidates)
    assert any(candidate.lines == ('alpha', 'beta') for candidate in candidates)


def test_long_labels_have_a_deterministic_candidate_evaluation_bound():
    measure, calls = _fake_measurer()
    text = ' '.join('word{}'.format(index) for index in range(100))
    cap = 17
    max_lines = 3

    candidates = generate_label_candidates(
        text, measure, max_lines=max_lines, candidate_cap=cap)

    assert len(candidates) <= cap
    # Every sampled layout needs at most one measurement per line plus the
    # joined multiline extent. Shared lines are cached, so this is an upper
    # bound on real measurement work as well as the layout count.
    assert len(calls) <= cap * (max_lines + 1)
    assert candidates == generate_label_candidates(
        text, measure, max_lines=max_lines, candidate_cap=cap)


@pytest.mark.parametrize('max_lines', [0, -1, True, 1.5])
def test_invalid_max_lines_are_rejected(max_lines):
    measure, _ = _fake_measurer()

    with pytest.raises(ValueError, match='max_lines'):
        generate_label_candidates('label', measure, max_lines=max_lines)


@pytest.mark.parametrize('measurement', [
    None, (1.0,), (math.nan, 1.0), (1.0, math.inf), (-1.0, 1.0),
    (True, 1.0),
])
def test_malformed_measurements_are_rejected(measurement):
    with pytest.raises(ValueError):
        generate_label_candidates('label', lambda text: measurement)


def test_matplotlib_adapter_returns_point_measurements_stable_across_dpi():
    from matplotlib.font_manager import FontProperties

    font = FontProperties(size=10.0)
    measurements = []
    for dpi in (72, 144):
        context = MatplotlibMeasureContext(dpi=dpi)
        artist_count = len(context.figure.artists)
        measured = measure_matplotlib_text(
            'alpha beta\ngamma delta', context.renderer, font,
            figure=context.figure)
        measurements.append(measured)
        assert len(context.figure.artists) == artist_count
        context.close()

    assert measurements[0].width == pytest.approx(
        measurements[1].width, abs=0.05)
    assert measurements[0].height == pytest.approx(
        measurements[1].height, abs=0.05)
    assert measurements[0].width > 0
    assert measurements[0].height > 0


def test_matplotlib_adapter_supplies_figure_context_when_omitted():
    from matplotlib.font_manager import FontProperties

    context = MatplotlibMeasureContext(dpi=100)
    measured = measure_matplotlib_text(
        'label', context.renderer, FontProperties(size=10.0))
    context.close()

    assert measured.width > 0
    assert measured.height > 0


def _manual_candidate(text, width, height=10.0, balance=0.0, orphan=0.0):
    lines = tuple(text.split('\n'))
    return LabelCandidate(
        lines=lines, width=width, height=height,
        line_count=len(lines),
        balance_penalty=balance, orphan_penalty=orphan)


def _solve_title_band(candidate_sets, title_spans, base_column_gaps,
                      column_width, original_block_width,
                      max_column_gap_points, extra_row_penalty=0.08,
                      balance_weight=0.02, orphan_weight=0.05,
                      secondary_costs=()):
    common = min(base_column_gaps, default=0.0)
    band = SharedGapBandSpec(
        candidate_sets=candidate_sets, spans=title_spans,
        extra_row_penalty=extra_row_penalty,
        balance_weight=balance_weight, orphan_weight=orphan_weight,
        secondary_costs=secondary_costs)
    return optimize_shared_gap_layout(
        (band,), cell_extent=column_width, base_common_gap=common,
        base_private_gaps=tuple(gap - common for gap in base_column_gaps),
        original_block_extent=original_block_width,
        max_common_gap_points=max_column_gap_points,
        hard_outer_containment=False)


def _optimize_two_title_sets(candidate_sets, gap=8.0, cap=96.0,
                             row_penalty=0.08, balance_weight=0.02,
                             orphan_weight=0.05):
    return _solve_title_band(
        candidate_sets=candidate_sets,
        title_spans=((0, 0), (1, 1)),
        base_column_gaps=(gap,), column_width=50.0,
        original_block_width=100.0 + gap,
        max_column_gap_points=cap,
        extra_row_penalty=row_penalty,
        balance_weight=balance_weight,
        orphan_weight=orphan_weight)


def test_shared_gap_objective_includes_emergency_break_penalty():
    emergency = LabelCandidate(
        lines=('long', 'label'), width=40.0, height=20.0,
        line_count=2,
        balance_penalty=0.0, orphan_penalty=0.0, break_penalty=0.75)
    band = SharedGapBandSpec(
        candidate_sets=((emergency,),), spans=((0, 0),),
        extra_row_penalty=0.0)

    result = optimize_shared_gap_layout(
        (band,), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(), original_block_extent=50.0,
        max_common_gap_points=8.0)

    assert result.bands[0].candidates == (emergency,)
    assert result.objective_cost == pytest.approx(0.75)


def test_global_title_optimizer_wraps_when_it_saves_enough_gap():
    one_line = _manual_candidate('long structural title', 90.0)
    wrapped = _manual_candidate('long structural\ntitle', 50.0, 20.0)

    result = _optimize_two_title_sets(
        ((one_line, wrapped), (one_line, wrapped)))

    assert tuple(candidate.line_count for candidate in result.bands[0].candidates) == (2, 2)
    assert result.boundary_gaps == pytest.approx((8.0,))
    # The global band row and the normalized per-label excess-row preference
    # are both charged when every label takes an extra line.
    assert result.objective_cost == pytest.approx(0.16)


def test_global_title_optimizer_prices_small_edge_overhang_saving():
    one_line = _manual_candidate('moderately long', 60.0)
    wrapped = _manual_candidate('moderately\nlong', 50.0, 20.0)

    result = _optimize_two_title_sets(
        ((one_line, wrapped), (one_line, wrapped)))

    # The one-line choice is sufficient at a modest widened gap; the new
    # per-label line preference no longer wraps merely because another title
    # could choose a taller band.
    assert tuple(candidate.line_count for candidate in result.bands[0].candidates) == (1, 1)
    assert result.boundary_gaps == pytest.approx((10.0,))


def test_outer_overhang_survives_frontier_pruning_for_first_title():
    # Both paths converge on the same final-title state.  The first path has
    # the lower typography cost but its wider first title grows the left
    # envelope; the narrower first title must survive the DP frontier so that
    # the outer-width term can choose it.
    wide_edge = _manual_candidate('wide edge', 60.0, balance=0.0)
    narrow_edge = _manual_candidate('narrow edge', 50.0, balance=1.0)
    final = _manual_candidate('final', 50.0)
    result = _optimize_two_title_sets(
        ((wide_edge, narrow_edge), (final,)))

    assert result.bands[0].candidates == (narrow_edge, final)


def test_shared_gap_bands_pay_one_common_gap_not_one_per_band():
    band40 = SharedGapBandSpec(
        candidate_sets=((_manual_candidate('a', 140.0),),),
        spans=((0, 1),))
    band50 = SharedGapBandSpec(
        candidate_sets=((_manual_candidate('c', 150.0),),),
        spans=((0, 1),))

    result = optimize_shared_gap_layout(
        (band40, band50), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(0.0,), original_block_extent=108.0,
        max_common_gap_points=96.0)

    assert isinstance(result, SharedGapLayout)
    assert result.common_gap == pytest.approx(50.0)
    assert result.boundary_gaps == pytest.approx((50.0,))
    assert result.common_gap - 8.0 == pytest.approx(42.0)
    assert len(result.bands) == 2


def test_shared_gap_can_change_a_band_after_another_band_pays_gap():
    one_line = _manual_candidate('wide edge', 90.0, 10.0)
    wrapped = _manual_candidate('wrapped\nedge', 50.0, 20.0)
    band_a = SharedGapBandSpec(
        candidate_sets=((one_line, wrapped), (one_line, wrapped)),
        spans=((1, 1), (2, 2)))
    band_b = SharedGapBandSpec(
        candidate_sets=((_manual_candidate('b1', 100.0),),
                        (_manual_candidate('b2', 100.0),)),
        spans=((1, 1), (2, 2)))
    sequential = _solve_title_band(
        candidate_sets=band_a.candidate_sets, title_spans=band_a.spans,
        base_column_gaps=(8.0,) * 4, column_width=50.0,
        original_block_width=282.0, max_column_gap_points=96.0)
    result = optimize_shared_gap_layout(
        (band_a, band_b), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(0.0,) * 4, original_block_extent=282.0,
        max_common_gap_points=96.0)

    assert tuple(candidate.line_count for candidate in sequential.bands[0].candidates) == (2, 2)
    assert result.common_gap == pytest.approx(50.0)
    assert tuple(candidate.line_count for candidate in result.bands[0].candidates) == (1, 1)
    # The common expansion is paid once for the four shared boundaries; the
    # two bands do not each add the same gap cost.
    assert result.objective_cost == pytest.approx(
        (50.0 - 8.0) / (2.0 * 50.0) *
        (1.0 + _SHARED_GAP_BOUNDARY_LOG_WEIGHT * math.log(4)))


def test_shared_gap_cost_grows_logarithmically_with_grid_boundaries():
    def cost_for_cell_count(cell_count):
        cell_extent = 50.0
        base_gap = 8.0
        selected_gap = 20.0
        boundary_count = cell_count - 1
        full_span_width = (cell_count * cell_extent +
                           boundary_count * selected_gap)
        band = SharedGapBandSpec(
            candidate_sets=((_manual_candidate(
                'merged label', full_span_width),),),
            spans=((0, cell_count - 1),), extra_row_penalty=0.0,
            balance_weight=0.0, orphan_weight=0.0)
        result = optimize_shared_gap_layout(
            (band,), cell_extent=cell_extent, base_common_gap=base_gap,
            base_private_gaps=(0.0,) * boundary_count,
            original_block_extent=(cell_count * cell_extent +
                                   boundary_count * base_gap),
            max_common_gap_points=40.0)
        assert result.common_gap == pytest.approx(selected_gap)
        return result.objective_cost

    two_cell_cost = cost_for_cell_count(2)
    five_cell_cost = cost_for_cell_count(5)
    cost_ratio = five_cell_cost / two_cell_cost

    assert 1.10 <= cost_ratio <= 1.15
    assert cost_ratio == pytest.approx(
        1.0 + _SHARED_GAP_BOUNDARY_LOG_WEIGHT * math.log(4))


def test_shared_gap_cost_is_invariant_to_existing_private_reservations():
    cell_extent = 50.0
    base_gap = 8.0
    selected_gap = 10.0
    candidate = _manual_candidate(
        'merged label', 2.0 * cell_extent + 2.0 * selected_gap)
    band = SharedGapBandSpec(
        candidate_sets=((candidate,),), spans=((1, 2),),
        extra_row_penalty=0.0, balance_weight=0.0, orphan_weight=0.0)

    def solve(private_gaps):
        boundary_count = len(private_gaps)
        result = optimize_shared_gap_layout(
            (band,), cell_extent=cell_extent, base_common_gap=base_gap,
            base_private_gaps=private_gaps,
            original_block_extent=(5.0 * cell_extent +
                                   boundary_count * base_gap +
                                   sum(private_gaps)),
            max_common_gap_points=40.0)
        assert result.common_gap == pytest.approx(selected_gap)
        return result.objective_cost

    no_private = solve((0.0, 0.0, 0.0, 0.0))
    existing_private = solve((0.0, 0.0, 0.0, 40.0))

    assert existing_private == pytest.approx(no_private)


def test_max_secondary_frontier_keeps_lower_additive_path_for_later_max():
    first = LabelCandidate(('first',), 20.0, 10.0, 1, 0.0, 0.0)
    lower_additive = LabelCandidate(
        ('lower-additive',), 20.0, 10.0, 1, 0.0, 0.0)
    higher_additive = LabelCandidate(
        ('higher-additive',), 20.0, 10.0, 1, 2.1, 0.0)
    last = LabelCandidate(('last',), 20.0, 10.0, 1, 0.0, 0.0)

    result = _solve_title_band(
        candidate_sets=((first,), (lower_additive, higher_additive), (last,)),
        title_spans=((0, 0), (1, 1), (2, 2)),
        base_column_gaps=(8.0, 8.0), column_width=50.0,
        original_block_width=166.0, max_column_gap_points=96.0,
        balance_weight=1.0, orphan_weight=0.0,
        secondary_costs=((0.0,), (0.8, 0.0), (1.0,)))

    # The initially cheaper path has secondary cost .8, but the final label
    # raises both paths to the same maximum of 1.0.  The lower additive path
    # must therefore survive the intermediate Pareto pruning and win.
    assert result.bands[0].candidates[1].text == 'lower-additive'
    assert result.objective_cost == pytest.approx(1.0)


def test_max_secondary_hard_cap_frontier_keeps_same_future_max_path():
    first = LabelCandidate(('first',), 60.0, 10.0, 1, 0.0, 0.0)
    lower_additive = LabelCandidate(
        ('lower-additive',), 60.0, 10.0, 1, 0.0, 0.0)
    higher_additive = LabelCandidate(
        ('higher-additive',), 60.0, 10.0, 1, 2.1, 0.0)
    last = LabelCandidate(('last',), 60.0, 10.0, 1, 0.0, 0.0)

    result = _solve_title_band(
        candidate_sets=((first,), (lower_additive, higher_additive), (last,)),
        title_spans=((0, 0), (1, 1), (2, 2)),
        base_column_gaps=(8.0, 8.0), column_width=50.0,
        original_block_width=166.0, max_column_gap_points=8.0,
        balance_weight=1.0, orphan_weight=0.0,
        secondary_costs=((0.0,), (0.8, 0.0), (1.0,)))

    assert result.bands[0].candidates[1].text == 'lower-additive'


def test_shared_gap_treats_one_cell_merged_and_nonmerged_spans_identically():
    candidate = _manual_candidate('label', 70.0)
    nonmerged = SharedGapBandSpec(
        candidate_sets=((candidate,),), spans=((1, 1),))
    merged_one_cell = SharedGapBandSpec(
        candidate_sets=((candidate,),), spans=((1, 1),))
    arguments = dict(
        cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(0.0, 0.0), original_block_extent=166.0,
        max_common_gap_points=96.0)

    first = optimize_shared_gap_layout((nonmerged,), **arguments)
    second = optimize_shared_gap_layout((merged_one_cell,), **arguments)
    assert first == second


def test_shared_gap_allocates_adjacent_common_gap_without_double_ownership():
    candidate = _manual_candidate('wide', 70.0)
    band = SharedGapBandSpec(
        candidate_sets=((candidate,), (candidate,)),
        spans=((1, 1), (2, 2)))
    result = optimize_shared_gap_layout(
        (band,), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(0.0, 0.0, 0.0), original_block_extent=208.0,
        max_common_gap_points=96.0)
    private = (0.0, 0.0, 0.0)
    first = _title_span_interval((1, 1), candidate, 50.0,
                                 result.common_gap, private)
    second = _title_span_interval((2, 2), candidate, 50.0,
                                  result.common_gap, private)
    starts = (0.0, 50.0 + result.common_gap,
              100.0 + 2.0 * result.common_gap)
    midpoint = (starts[1] + 50.0 + starts[2]) / 2.0

    assert result.common_gap == pytest.approx(20.0)
    assert first[1] <= midpoint + 1e-8
    assert second[0] + 1e-8 >= midpoint


def test_shared_gap_title_uses_half_of_each_external_common_gutter():
    one_line = _manual_candidate('one line', 135.0, height=10.0)
    wrapped = _manual_candidate('two\nlines', 75.5, height=20.0)
    band = SharedGapBandSpec(
        candidate_sets=((one_line, wrapped),), spans=((1, 1),))
    result = optimize_shared_gap_layout(
        (band,), cell_extent=85.0, base_common_gap=25.0,
        base_private_gaps=(0.0, 0.0), original_block_extent=305.0,
        max_common_gap_points=25.0)

    assert result.common_gap == pytest.approx(25.0)
    assert result.bands[0].candidates == (wrapped,)


def test_shared_gap_multicell_span_uses_internal_common_space():
    candidate = _manual_candidate('long merged label', 130.0)
    multi = SharedGapBandSpec(
        candidate_sets=((candidate,),), spans=((0, 1),))
    one_cell = SharedGapBandSpec(
        candidate_sets=((candidate,),), spans=((1, 1),))
    arguments = dict(
        cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(0.0, 0.0), original_block_extent=166.0,
        max_common_gap_points=96.0)

    multi_result = optimize_shared_gap_layout((multi,), **arguments)
    one_cell_result = optimize_shared_gap_layout((one_cell,), **arguments)
    assert multi_result.common_gap == pytest.approx(30.0)
    interval = _title_span_interval(
        (0, 1), candidate, 50.0, multi_result.common_gap, (0.0, 0.0))
    block_extent = 3.0 * 50.0 + 2.0 * multi_result.common_gap
    assert interval[0] >= -1e-9
    assert interval[1] <= block_extent + 1e-9
    # A single-column label gets half of each external common gutter, while
    # the merged span above owns its internal separator in full.
    assert one_cell_result.common_gap == pytest.approx(80.0)


def test_shared_gap_private_reservation_is_not_common_space():
    candidate = _manual_candidate('label', 70.0)
    band = SharedGapBandSpec(
        candidate_sets=((candidate,),), spans=((1, 1),))
    without_private = optimize_shared_gap_layout(
        (band,), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(0.0, 0.0), original_block_extent=158.0,
        max_common_gap_points=96.0)
    with_private = optimize_shared_gap_layout(
        (band,), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(0.0, 30.0), original_block_extent=188.0,
        max_common_gap_points=96.0)

    # The left common half is the binding side. A private reservation on the
    # right remains available to the label but cannot replace that half.

    assert without_private.common_gap == pytest.approx(20.0)
    assert with_private.common_gap == pytest.approx(20.0)
    assert with_private.boundary_gaps == pytest.approx((20.0, 50.0))


def test_shared_gap_hard_cap_reports_residual_without_margin_expansion():
    wide = _manual_candidate('unbreakable', 200.0)
    band = SharedGapBandSpec(
        candidate_sets=((wide,), (wide,)), spans=((0, 0), (1, 1)))
    result = optimize_shared_gap_layout(
        (band,), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(0.0,), original_block_extent=108.0,
        max_common_gap_points=10.0)
    outer_band = SharedGapBandSpec(
        candidate_sets=((wide,),), spans=((0, 0),))
    outer = optimize_shared_gap_layout(
        (outer_band,), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(), original_block_extent=50.0,
        max_common_gap_points=96.0)

    assert result.common_gap == pytest.approx(10.0)
    # The adjacent unbreakable labels improve as the only common gap widens,
    # so the infeasible fallback still reaches its hard cap.
    # Conversely, there is no boundary that can repair this one-column
    # outermost overhang; the cap is only an upper bound and the base gap is
    # retained when the residual is unchanged.
    assert outer.common_gap == pytest.approx(8.0)
    assert outer.boundary_gaps == ()


def test_shared_gap_hard_containment_prefers_a_fitting_wrap_over_edge_overhang():
    one_line = _manual_candidate('literal label', 80.0, 10.0)
    wrapped = _manual_candidate('literal\nlabel', 50.0, 20.0)
    band = SharedGapBandSpec(
        candidate_sets=((one_line, wrapped),), spans=((0, 0),),
        extra_row_penalty=10.0)

    result = optimize_shared_gap_layout(
        (band,), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(), original_block_extent=50.0,
        max_common_gap_points=96.0)

    # The literal candidate has no internal boundary to widen, and its
    # outward overhang is therefore a hard residual.  A very large row cost
    # cannot make that invalid candidate feasible.
    assert result.common_gap == pytest.approx(8.0)
    assert result.bands[0].candidates == (wrapped,)


def test_shared_gap_is_orientation_neutral():
    column_candidate = _manual_candidate('wide', 70.0, 10.0)
    row_candidate = LabelCandidate(
        lines=column_candidate.lines, width=10.0, height=70.0,
        line_count=column_candidate.line_count,
        balance_penalty=0.0, orphan_penalty=0.0)
    column_band = SharedGapBandSpec(
        candidate_sets=((column_candidate,), (column_candidate,)),
        spans=((0, 0), (1, 1)), primary_dimension='width')
    row_band = SharedGapBandSpec(
        candidate_sets=((row_candidate,), (row_candidate,)),
        spans=((0, 0), (1, 1)), primary_dimension='height')
    arguments = dict(
        cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(0.0,), original_block_extent=108.0,
        max_common_gap_points=96.0)

    column_result = optimize_shared_gap_layout((column_band,), **arguments)
    row_result = optimize_shared_gap_layout((row_band,), **arguments)
    assert row_result.common_gap == pytest.approx(column_result.common_gap)
    assert row_result.boundary_gaps == pytest.approx(column_result.boundary_gaps)
    assert column_result.bands[0].candidates[0].height == pytest.approx(10.0)
    assert row_result.bands[0].candidates[0].width == pytest.approx(10.0)


def test_shared_gap_uses_explicit_orthogonal_thickness_costs():
    narrow = LabelCandidate(('narrow',), 10.0, 30.0, 1, 0.0, 0.0)
    wide = LabelCandidate(('wide',), 100.0, 30.0, 1, 0.0, 0.0)
    band = SharedGapBandSpec(
        candidate_sets=((narrow, wide),), spans=((0, 0),),
        primary_dimension='height', secondary_costs=((0.1, 1.0),))

    result = optimize_shared_gap_layout(
        (band,), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(), original_block_extent=200.0,
        max_common_gap_points=96.0)

    assert result.bands[0].candidates == (narrow,)
    assert result.objective_cost == pytest.approx(0.1)


def test_shared_gap_height_projection_preserves_selected_original_and_thickness():
    narrower = LabelCandidate(
        lines=('narrower',), width=40.0, height=70.0,
        line_count=1,
        balance_penalty=1.0, orphan_penalty=0.0)
    wider = LabelCandidate(
        lines=('wider',), width=100.0, height=70.0,
        line_count=1,
        balance_penalty=0.0, orphan_penalty=0.0)
    band = SharedGapBandSpec(
        candidate_sets=((narrower, wider),), spans=((0, 1),),
        primary_dimension='height', balance_weight=1.0,
        extra_row_penalty=0.0)

    result = optimize_shared_gap_layout(
        (band,), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(0.0,), original_block_extent=108.0,
        max_common_gap_points=96.0)

    # Both candidates have the same primary (height) extent, but the selected
    # original retains its distinct width for the orthogonal band thickness.
    assert result.bands[0].candidates == (wider,)
    assert result.bands[0].candidates[0].width == pytest.approx(100.0)


def test_shared_gap_does_not_compare_rectangles_between_bands():
    candidate = _manual_candidate('same cell', 50.0)
    bands = tuple(SharedGapBandSpec(
        candidate_sets=((candidate,),), spans=((0, 0),)) for _ in range(3))
    result = optimize_shared_gap_layout(
        bands, cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(), original_block_extent=50.0,
        max_common_gap_points=96.0)

    # The three visual tracks intentionally occupy the same data span. The
    # helper solves each track independently; cross-band rectangle handling
    # belongs to a later coordinator.
    assert len(result.bands) == 3


def test_shared_gap_many_bands_is_deterministic_and_bounded():
    candidates = (
        _manual_candidate('wide', 90.0),
        _manual_candidate('wrapped\nwide', 50.0, 20.0),
    )
    bands = tuple(SharedGapBandSpec(
        candidate_sets=(candidates, candidates),
        spans=((0, 0), (1, 1))) for _ in range(12))
    arguments = dict(
        cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(0.0,), original_block_extent=108.0,
        max_common_gap_points=96.0)
    first = optimize_shared_gap_layout(bands, **arguments)
    second = optimize_shared_gap_layout(bands, **arguments)

    assert first == second
    assert len(first.bands) == 12


def test_hard_cap_many_columns_keeps_solver_bounded():
    count = 24
    candidates = tuple(tuple(LabelCandidate(
        (f'{index}-{choice}',), width, 10.0, 1,
        penalty, 0.0) for choice, (width, penalty) in enumerate((
            (105.0 + index * 0.7, 0.0),
            (88.0 + index * 0.9, 5.0),
            (70.0 + index * 1.1, 10.0))))
        for index in range(count))
    band = SharedGapBandSpec(
        candidates, tuple((index, index) for index in range(count)),
        balance_weight=1.0)
    start = time.perf_counter()
    result = optimize_shared_gap_layout(
        (band, band), cell_extent=50.0, base_common_gap=4.0,
        base_private_gaps=(0.0,) * (count - 1),
        original_block_extent=50.0 * count + 4.0 * (count - 1),
        max_common_gap_points=20.0)

    assert time.perf_counter() - start < 2.0
    assert result.common_gap == pytest.approx(20.0)


def test_global_extra_row_cost_is_paid_once_for_multiple_wrapped_titles():
    one_line = _manual_candidate('title stays wide', 70.0)
    wrapped = _manual_candidate('title stays\nwide', 50.0, 20.0)
    result = _solve_title_band(
        candidate_sets=((one_line, wrapped),) * 3,
        title_spans=((0, 0), (1, 1), (2, 2)),
        base_column_gaps=(19.0, 19.0), column_width=50.0,
        original_block_width=188.0, max_column_gap_points=96.0,
        extra_row_penalty=0.04, balance_weight=0.0, orphan_weight=0.0)

    assert tuple(candidate.line_count for candidate in result.bands[0].candidates) == (2, 1, 2)
    assert result.objective_cost == pytest.approx(
        0.04 + 2 * 0.04 / 3 +
        1.0 / (2.0 * 50.0) *
        (1.0 + _SHARED_GAP_BOUNDARY_LOG_WEIGHT * math.log(2)))


@pytest.mark.parametrize(('field', 'weight'), [
    ('balance_penalty', 'balance_weight'),
    ('orphan_penalty', 'orphan_weight'),
])
def test_typography_weights_break_identical_geometry_ties(field, weight):
    metrics = {'balance': 0.0, 'orphan': 0.0}
    preferred = _manual_candidate('balanced\nlines', 50.0, 20.0, **metrics)
    metrics['balance' if field == 'balance_penalty' else 'orphan'] = 1.0
    disfavoured = _manual_candidate('uneven\nlines', 50.0, 20.0, **metrics)
    weights = {'balance_weight': 0.0, 'orphan_weight': 0.0}
    weights[weight] = 1.0
    result = _solve_title_band(
        candidate_sets=((disfavoured, preferred),),
        title_spans=((0, 0),), base_column_gaps=(8.0,),
        column_width=50.0, original_block_width=108.0,
        max_column_gap_points=96.0, extra_row_penalty=0.0,
        **weights)

    assert result.bands[0].candidates == (preferred,)


def test_title_optimizer_enforces_the_hard_gap_cap():
    huge = _manual_candidate('unbreakable long title', 150.0)
    result = _optimize_two_title_sets(
        ((huge,), (huge,)), cap=20.0)

    assert result.boundary_gaps == pytest.approx((20.0,))
    assert (result.common_gap - 8.0,) == pytest.approx((12.0,))
    assert result.boundary_gaps[0] <= 20.0


def test_collision_free_layout_beats_a_prettier_but_overlapping_layout():
    wide_balanced = _manual_candidate(
        'single line', 80.0, balance=0.0)
    narrow_ugly = _manual_candidate(
        'two\nlines', 50.0, 20.0, balance=100.0)
    result = _optimize_two_title_sets(
        ((wide_balanced, narrow_ugly),) * 2,
        gap=0.0, cap=0.0, row_penalty=0.08,
        balance_weight=100.0, orphan_weight=0.0)

    assert all(candidate.width == 50.0 for candidate in result.bands[0].candidates)


def test_impossible_title_chain_chooses_the_least_residual_overlap():
    width_100 = _manual_candidate('wide', 100.0)
    width_80 = _manual_candidate('narrower', 80.0)
    result = _optimize_two_title_sets(
        ((width_100, width_80), (width_100,)), gap=0.0, cap=0.0)

    assert result.bands[0].candidates[0] == width_80
    first, second = (_title_span_interval(
        (index, index), candidate, 50.0, result.common_gap, (0.0,))
        for index, candidate in enumerate(result.bands[0].candidates))
    assert first[1] > second[0]
    assert result.boundary_gaps == pytest.approx((0.0,))
    own = sum(_title_reserved_encroachment(
        (index, index), candidate, 50.0, 0.0, (0.0,))
        for index, candidate in enumerate(result.bands[0].candidates))
    assert own == 40.0
    assert first[1] - second[0] == 40.0
    assert own + first[1] - second[0] == 80.0


def test_r1_pairwise_selection_uses_the_wide_narrow_zero_residual_path():
    bands, options = r1_case()
    result = optimize_shared_gap_layout(bands, **options)
    assert result.common_gap == 10.0
    assert result.boundary_gaps == (20.0,)
    assert result.bands[0].candidates == (
        bands[0].candidate_sets[0][0], bands[0].candidate_sets[1][1])
    selected = result.bands[0].candidates
    intervals = [_title_span_interval(
        span, candidate, 10.0, 10.0, (10.0,))
        for span, candidate in zip(bands[0].spans, selected)]
    residual = sum(_title_reserved_encroachment(
        span, candidate, 10.0, 10.0, (10.0,))
        for span, candidate in zip(bands[0].spans, selected))
    residual += max(0.0, intervals[0][1] - intervals[1][0])
    residual += _outer_overhang(
        bands[0].spans, selected, 10.0, 10.0, (10.0,),
        'width', 10.0, 10.0)
    assert residual == 0.0


def test_many_columns_stop_widening_when_only_outer_overhang_remains():
    # The last title cannot fit the outer edge by widening an internal gap.
    # Exact affine crossings should find the first residual plateau, even
    # when many distinct candidate widths create different breakpoints.
    count = 16
    candidates = tuple(
        (_manual_candidate(str(index), 71.0 + index * 1.7),)
        for index in range(count))
    band = SharedGapBandSpec(
        candidate_sets=candidates,
        spans=tuple((index, index) for index in range(count)))
    result = optimize_shared_gap_layout(
        (band,), cell_extent=80.0, base_common_gap=4.0,
        base_private_gaps=(0.0,) * (count - 1),
        original_block_extent=80.0 * count + 4.0 * (count - 1),
        max_common_gap_points=60.0)

    assert result.common_gap == pytest.approx(16.5)
    assert _outer_overhang(
        band.spans, result.bands[0].candidates, 80.0,
        result.common_gap, (0.0,) * (count - 1), 'width', 0.0, 0.0
    ) == pytest.approx(8.25)


def test_common_gutter_and_existing_private_space_are_available_to_the_band():
    candidate = _manual_candidate('uses the gutter', 100.0)
    x0, x1 = _title_span_interval(
        (0, 0), candidate, cell_extent=50.0, common_gap=20.0,
        private_gaps=(15.0,))

    # The title crosses the first data edge through half of the common gutter
    # and five points into the available private reservation.
    assert x0 < 50.0 < 70.0 < x1 == pytest.approx(75.0)
    assert _title_reserved_encroachment(
        (0, 0), candidate, 50.0, 20.0, (15.0,)) == pytest.approx(0.0)
    foreign = _manual_candidate('enters foreign data', 160.0)
    assert _title_reserved_encroachment(
        (0, 0), foreign, 50.0, 20.0, (15.0,)) > 0.0


def test_shared_gap_solver_supports_unequal_per_cell_extents():
    candidates = (
        _manual_candidate('first cell', 40.0),
        _manual_candidate('wider middle cell', 100.0),
        _manual_candidate('last cell', 60.0))
    band = SharedGapBandSpec(
        candidate_sets=tuple((candidate,) for candidate in candidates),
        spans=((0, 0), (1, 1), (2, 2)))
    result = optimize_shared_gap_layout(
        (band,), cell_extent=(40.0, 80.0, 60.0),
        base_common_gap=8.0, base_private_gaps=(0.0, 0.0),
        original_block_extent=196.0, max_common_gap_points=96.0)

    assert result.common_gap == pytest.approx(20.0)
    assert result.boundary_gaps == pytest.approx((20.0, 20.0))
    assert result.bands[0].candidates == candidates
    intervals = tuple(_title_span_interval(
        span, candidate, (40.0, 80.0, 60.0), result.common_gap,
        (0.0, 0.0)) for span, candidate in zip(
            band.spans, result.bands[0].candidates))
    assert intervals[0] == pytest.approx((0.0, 40.0))
    assert intervals[1] == pytest.approx((50.0, 150.0))
    assert intervals[2] == pytest.approx((160.0, 220.0))


def test_outer_allowance_makes_existing_edge_room_free():
    one_line = _manual_candidate('edge label', 60.0)
    wrapped = _manual_candidate('edge\nlabel', 50.0, 20.0)
    band = SharedGapBandSpec(
        candidate_sets=((one_line, wrapped),), spans=((0, 0),))

    without_allowance = optimize_shared_gap_layout(
        (band,), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(), original_block_extent=50.0,
        max_common_gap_points=96.0)
    with_allowance = optimize_shared_gap_layout(
        (band,), cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=(), original_block_extent=50.0,
        max_common_gap_points=96.0,
        outer_left_allowance=5.0, outer_right_allowance=5.0)

    assert without_allowance.bands[0].candidates == (wrapped,)
    assert with_allowance.bands[0].candidates == (one_line,)


def test_per_label_line_preference_beats_balanced_extra_wrap():
    establishing = _manual_candidate('three\nline\nlabel', 40.0, 30.0)
    two_line = _manual_candidate(
        'sufficient\ntwo', 50.0, 20.0, balance=0.5)
    three_line = _manual_candidate(
        'balanced\nthree\nlines', 50.0, 30.0, balance=0.0)
    result = _solve_title_band(
        candidate_sets=((establishing,), (two_line, three_line)),
        title_spans=((0, 0), (1, 1)), base_column_gaps=(8.0,),
        column_width=50.0, original_block_width=108.0,
        max_column_gap_points=96.0, extra_row_penalty=0.08,
        balance_weight=0.02, orphan_weight=0.0)

    assert result.bands[0].candidates == (establishing, two_line)


def test_optimizer_preserves_explicit_newlines_and_empty_titles():
    measure, _ = _fake_measurer()
    fixed_break = generate_label_candidates(
        'first\nsecond', measure, max_lines=2)[0]
    empty = generate_label_candidates('', measure)[0]
    result = _optimize_two_title_sets(((fixed_break,), (empty,)))

    assert tuple(candidate.text for candidate in result.bands[0].candidates) == (
        'first\nsecond', '')
    assert max(candidate.height for candidate in result.bands[0].candidates) == fixed_break.height


def test_bracketed_units_break_only_as_a_last_resort():
    def measure(text):
        lines = text.split('\n')
        return max(len(line) for line in lines) * 5.0, 10.0 * len(lines)

    candidates = generate_label_candidates(
        'log2FC [treated/control] [n = 7,619]', measure, max_lines=3)
    for candidate in candidates:
        whole = all(line.count('[') == line.count(']')
                    for line in candidate.lines)
        assert whole == (candidate.break_penalty == 0)
    assert ('log2FC', '[treated/control]', '[n = 7,619]') in {
        candidate.lines for candidate in candidates}
