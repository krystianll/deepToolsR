"""Capture solver choices at their existing import sites without code hooks."""

import inspect
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from tests.helpers import label_solver_cases as label_solver_probe
from deeptoolsr.plotting import grid
from tests import test_plot_baselines
from tests.plotting import test_text_layout
from tests.plotting import test_grid_solver


def _choice(values, selected):
    for index, value in enumerate(values):
        if value is selected:
            return index
    return next(index for index, value in enumerate(values)
                if value == selected)


def _record_below(original_below, calls):
    def below(candidate_sets, entries, bar_width, bar_thickness, block_width,
              label_layout, horizontal_gap, geometry):
        selected = original_below(
            candidate_sets, entries, bar_width, bar_thickness, block_width,
            label_layout, horizontal_gap, geometry)
        calls.append({
            'solver': 'below_common',
            'choices': [_choice(values, item)
                        for values, item in zip(candidate_sets, selected)]})
        return selected
    return below


@contextmanager
def _record_calls(target, calls):
    original_gap = target.optimize_shared_gap_layout
    original_below = grid._select_below_common_label_candidates

    def gap(bands, *args, **kwargs):
        bands = tuple(bands)
        result = original_gap(bands, *args, **kwargs)
        calls.append({
            'solver': 'shared_gap',
            'choices': [[_choice(values, selected)
                         for values, selected in zip(band.candidate_sets,
                                                     solved.candidates)]
                        for band, solved in zip(bands, result.bands)],
            'objective_cost': repr(result.objective_cost)})
        return result

    with ExitStack() as patches:
        patches.enter_context(patch.object(
            target, 'optimize_shared_gap_layout', gap))
        if target is grid:
            patches.enter_context(patch.object(
                grid, '_select_below_common_label_candidates',
                _record_below(original_below, calls)))
        yield


def capture(tmp_path):
    """Key by case/test name, then by call ordinal within that render/test."""
    result = {'baselines': {}, 'probe': {}, 'text_layout': {},
              'below_common_direct': {}}
    for kind, name, case in test_plot_baselines.ALL_CASES:
        calls = []
        with _record_calls(grid, calls):
            render = {
                'heatmap': test_plot_baselines._render_heatmap,
                'profile': test_plot_baselines._render_profile,
                'matrix': test_plot_baselines._render_matrix,
            }[kind]
            render(case, tmp_path / f'{name}.png')
        result['baselines'][name] = calls
    for name, (bands, options) in label_solver_probe.cases().items():
        calls = []
        with _record_calls(label_solver_probe, calls):
            label_solver_probe.optimize_shared_gap_layout(bands, **options)
        result['probe'][name] = calls
    for name, function in inspect.getmembers(test_text_layout,
                                             inspect.isfunction):
        if (not name.startswith('test_') or name.startswith('test_r1_') or
                inspect.signature(function).parameters):
            continue
        calls = []
        with _record_calls(test_text_layout, calls):
            function()
        if calls:
            result['text_layout'][name] = calls
    for field, weight in (('balance_penalty', 'balance_weight'),
                          ('orphan_penalty', 'orphan_weight')):
        name = f'test_typography_weights_break_identical_geometry_ties[{field}]'
        calls = []
        with _record_calls(test_text_layout, calls):
            test_text_layout.test_typography_weights_break_identical_geometry_ties(
                field, weight)
        result['text_layout'][name] = calls
    for name in (
            'test_below_common_pareto_pruning_matches_exhaustive_greedy_oracle',
            'test_below_common_selector_prefers_legal_break_over_equal_geometry_fallback'):
        calls = []
        with patch.object(
                grid,
                '_select_below_common_label_candidates',
                _record_below(
                    grid._select_below_common_label_candidates, calls)):
            getattr(test_grid_solver, name)()
        result['below_common_direct'][name] = calls
    return result
