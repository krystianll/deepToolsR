"""Pure grid vocabulary and height-allocation checks."""

import ast
import json
from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest

from deeptoolsr.plotting import grid
from deeptoolsr.plotting.series import Cell, LabelKind, StackBlock, label_runs
from deeptoolsr.plotting.text_layout import LabelCandidate

BASELINES = Path(__file__).resolve().parents[1] / 'test_baselines' / \
    'plot_baselines.json'
BASELINE_GEOMETRY = json.loads(BASELINES.read_text())['geometry']
GRID_SOURCE = Path(__file__).resolve().parents[2] / 'deeptoolsr' / 'plotting' / 'grid.py'

MOVED_HELPERS = {
    '_expand', '_move', '_fit_canvas', 'pack_colorbar_grid',
    '_below_common_footprint', '_below_common_row_advance',
    'pack_below_common', '_select_below_common_label_candidates',
}
B1_FIELDS = {
    'figure_edge_padding', 'figure_title_to_content_gap',
    'external_legend_to_content_gap', 'right_legend_to_content_gap',
    'column_header_to_panel_gap', 'axis_label_to_tick_labels_gap',
    'common_axis_label_to_tick_labels_gap', 'profile_min_column_gap',
    'profile_row_gap', 'profile_panel_to_row_header_gap',
    'label_min_clearance',
}
B2_FIELDS = {
    'heatmap_min_column_gap', 'heatmap_decoration_gap', 'heatmap_group_gap',
    'heatmap_colorbar_gap', 'sort_indicator_gap',
    'heatmap_to_row_header_gap', 'row_header_to_y_label_gap',
    'heatmap_to_summary_plot_gap', 'colorbar_title_to_bar_gap',
    'heatmap_colorbar_thickness', 'colorbar_below_common_max_gap',
}
MOVED_HELPER_ONLY = {'tick_label_to_tick_gap', 'colorbar_short_edge_gap',
                     'colorbar_long_edge_gap', 'colorbar_min_height'}


def test_each_geometry_boundary_has_one_read_outside_moved_helpers():
    tree = ast.parse(GRID_SOURCE.read_text())
    reads = Counter()
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or \
                function.name in MOVED_HELPERS:
            continue
        for node in ast.walk(function):
            if (isinstance(node, ast.Attribute) and
                    isinstance(node.value, ast.Attribute) and
                    node.value.attr == 'geometry'):
                reads[node.attr] += 1
    assert {name: reads[name] for name in B1_FIELDS} == \
        dict.fromkeys(B1_FIELDS, 1)
    assert {name: reads[name] for name in B2_FIELDS} == \
        dict.fromkeys(B2_FIELDS, 1)
    assert all(reads[name] == 0 for name in MOVED_HELPER_ONLY)


def _cell(blocks):
    return grid.CellLayout(0, 0, None, (), 100, grid.CellInsets(),
                           grid.CellLabels(), blocks=tuple(blocks))


def test_stack_heights_normal_mode():
    cell = _cell((StackBlock(0, 0), StackBlock(1, 0)))
    assert grid.stack_heights(cell, (10, 30), 100, 4) == (24, 72)


def test_stack_heights_per_group():
    small = _cell((StackBlock(0, 0), StackBlock(0, 1)))
    large = _cell((StackBlock(1, 0), StackBlock(1, 1)))
    assert grid.stack_heights(small, (10, 30), 100, 4) == (48, 48)
    assert grid.stack_heights(large, (10, 30), 100, 4) == (48, 48)


def test_stack_heights_multiple_samples_share_one_stack():
    cell = _cell((StackBlock(0, 0), StackBlock(1, 0),
                  StackBlock(0, 1), StackBlock(1, 1)))
    heights = grid.stack_heights(cell, (10, 30), 100, 4)
    assert heights == (11, 33, 11, 33)
    assert sum(heights) + 3 * 4 == 100


def test_region_labels_can_wrap_without_growing_stack_height():
    heights = (224 / 6, 448 / 6, 672 / 6)
    cell = grid.CellLayout(
        0, 0, None, heights, 100, grid.CellInsets(), grid.CellLabels(),
        blocks=tuple(StackBlock(index, 0) for index in range(3)))
    candidates = (
        _candidate('', 0, 0),
        LabelCandidate(('middle', 'region'), 16, 130, 2, 0, 0, 0),
        _candidate('', 0, 0),
    )
    runs = tuple(grid.LabelRun('stack', (0,), (candidate,),
                               block_index=index, location='left')
                 for index, candidate in enumerate(candidates))
    spec = grid.GridLayoutSpec(1, 1, (cell,), runs=runs)
    growing = grid.solve(spec)
    strict = grid.solve(replace(spec, label_layout=replace(
        spec.label_layout, heatmap_region_label_gap_growth=False)))

    def stack_span(solution):
        top = solution.rects['cell/1/1/block/1']
        bottom = solution.rects['cell/1/1/block/3']
        return top.y1 - bottom.y0, (
            top.y0 - solution.rects['cell/1/1/block/2'].y1,
            solution.rects['cell/1/1/block/2'].y0 - bottom.y1)

    assert stack_span(growing)[0] > 240
    assert stack_span(strict)[0] == pytest.approx(240)
    assert stack_span(strict)[1] == pytest.approx((8, 8))
    assert strict.rects['label/stack/1/1/2'].height == 130
    assert strict.selections['label/stack/1/1/2'] == 0

    alternatives = replace(runs[1], candidates=(
        _candidate('middle region', 16, 130),
        LabelCandidate(('middle', 'region'), 40, 60, 2, 0, 0, 0)))
    spec = replace(spec, runs=(runs[0], alternatives, runs[2]))
    assert grid.solve(spec).selections['label/stack/1/1/2'] == 0
    fixed = grid.solve(replace(spec, label_layout=replace(
        spec.label_layout, heatmap_region_label_gap_growth=False)))
    assert fixed.selections['label/stack/1/1/2'] == 1
    assert stack_span(fixed)[0] == pytest.approx(240)


def _role_uses_grid_naming(role):
    parts = role.split('/')
    if role == grid.figure_title():
        return True
    if parts[0] == 'indicator' and len(parts) == 2:
        return role == grid.indicator_role(int(parts[1]) - 1)
    try:
        if parts[0] == 'cell':
            row, column = int(parts[1]) - 1, int(parts[2]) - 1
            if len(parts) == 5:
                return role == grid.cell_block(row, column,
                                               int(parts[4]) - 1)
            maker = {'profile': grid.cell_profile,
                     'title': grid.cell_title}[parts[3]]
            return role == maker(row, column)
        if parts[0] == 'label':
            if parts[1] == 'row':
                return role == grid.row_label(int(parts[2]) - 1)
            block = int(parts[4]) - 1 if len(parts) == 5 else None
            return role == grid.label_role(parts[1], int(parts[2]) - 1,
                                           int(parts[3]) - 1, block)
        maker = {'legend': grid.legend_role,
                 'colorbar': grid.colorbar_role,
                 'colorbar_title': grid.colorbar_title_role}[parts[0]]
        ordinal = int(parts[3]) - 1 if len(parts) == 4 else 0
        return role == maker(parts[1], int(parts[2]) - 1, ordinal) \
            if parts[0] != 'legend' else role == maker(
                parts[1], int(parts[2]) - 1)
    except (IndexError, KeyError, ValueError):
        return False


def test_mixed_cells_and_stacks_share_one_multirow_solve():
    cells = (
        grid.CellLayout(0, 0, grid.Size(50, 20), (30.0,), 50.0,
                        grid.CellInsets(), grid.CellLabels()),
        grid.CellLayout(0, 1, None, (30.0,), 50.0,
                        grid.CellInsets(), grid.CellLabels()),
        grid.CellLayout(1, 0, grid.Size(50, 20), (), 50.0,
                        grid.CellInsets(), grid.CellLabels()),
        grid.CellLayout(1, 1, grid.Size(50, 20), (30.0,), 50.0,
                        grid.CellInsets(), grid.CellLabels()),
    )
    solved = grid.solve(grid.GridLayoutSpec(2, 2, cells))
    assert {'cell/1/1/profile', 'cell/1/1/block/1',
            'cell/1/2/block/1', 'cell/2/1/profile',
            'cell/2/2/profile', 'cell/2/2/block/1'} <= solved.rects.keys()
    assert (solved.rects['cell/1/1/block/1'].y0 >
            solved.rects['cell/2/2/profile'].y1)


def test_indicator_stays_left_of_top_stack_row():
    cells = tuple(grid.CellLayout(
        row, column, None, (30.0,), 50.0, grid.CellInsets(),
        grid.CellLabels()) for row in range(2) for column in range(2))
    indicator = grid.Decoration('indicator', 'row', 0, 'left')
    geometry = replace(grid.DEFAULT_GEOMETRY, sort_indicator_gap=7)
    solved = grid.solve(grid.GridLayoutSpec(
        2, 2, cells, decorations=(indicator,), geometry=geometry))
    marker = solved.rects['indicator/1']
    top_left = solved.rects['cell/1/1/block/1']
    bottom_left = solved.rects['cell/2/1/block/1']
    assert top_left.x0 - marker.x1 == 7
    assert marker.y0 == top_left.y0
    assert marker.y0 > bottom_left.y1


@pytest.mark.parametrize('name,record', BASELINE_GEOMETRY.items(),
                         ids=BASELINE_GEOMETRY)
def test_baseline_role_uses_grid_naming_function(name, record):
    assert all(_role_uses_grid_naming(role) for role in record['roles']), name


def _candidate(text, width=10, height=10):
    return LabelCandidate((text,), width, height, 1, 0.0, 0.0, 0.0)


def _stack_spec(*, columns=1, profile=False, blocks=1,
                decorations=(), geometry=grid.DEFAULT_GEOMETRY,
                labels=None):
    cells = tuple(grid.CellLayout(
        0, column, grid.Size(50, 30) if profile else None,
        (40.0,) * blocks, 50.0, grid.CellInsets(),
        labels[column] if labels else grid.CellLabels(),
        blocks=tuple(StackBlock(index, column) for index in range(blocks)))
        for column in range(columns))
    return grid.GridLayoutSpec(1, columns, cells,
                               decorations=tuple(decorations),
                               geometry=geometry)


def _gap(left, right):
    return right.x0 - left.x1


def test_heatmap_column_and_stack_gaps_read_their_geometry_fields():
    geometry = replace(grid.DEFAULT_GEOMETRY,
                       heatmap_min_column_gap=13,
                       heatmap_group_gap=7)
    solved = grid.solve(_stack_spec(columns=2, blocks=2,
                                    geometry=geometry))
    left = solved.rects['cell/1/1/block/1']
    right = solved.rects['cell/1/2/block/1']
    lower = solved.rects['cell/1/1/block/2']
    assert _gap(left, right) == 13
    assert left.y0 - lower.y1 == 7


def test_summary_gap_and_column_insets_are_independent():
    geometry = replace(grid.DEFAULT_GEOMETRY,
                       heatmap_to_summary_plot_gap=17,
                       heatmap_decoration_gap=9,
                       heatmap_min_column_gap=1)
    spec = _stack_spec(columns=2, profile=True, geometry=geometry)
    first = replace(spec.cells[0], insets=grid.CellInsets(
        stack=grid.PanelInsets(axis_insets=grid.Insets(right=10))))
    spec = replace(spec, cells=(first, spec.cells[1]))
    solved = grid.solve(spec)
    block = solved.rects['cell/1/1/block/1']
    profile = solved.rects['cell/1/1/profile']
    assert profile.y0 - block.y1 == 17
    assert _gap(block, solved.rects['cell/1/2/block/1']) == 19


def test_heatmap_title_and_canvas_edge_gaps():
    geometry = replace(grid.DEFAULT_GEOMETRY,
                       figure_edge_padding=12,
                       column_header_to_panel_gap=5,
                       figure_title_to_content_gap=9)
    labels = grid.CellLabels(
        title_text='sample', title_order=0,
        title_candidates=(_candidate('sample', 20, 7),))
    spec = _stack_spec(geometry=geometry, labels=(labels,))
    spec = replace(spec, title=grid.MeasuredLabelSpec((
        _candidate('figure', 30, 8),)))
    solved = grid.solve(spec)
    block = solved.rects['cell/1/1/block/1']
    title = solved.rects['cell/1/1/title']
    figure_title = solved.rects['title']
    assert block.x0 == 12
    assert title.y0 - block.y1 == 5
    assert figure_title.y0 - title.y1 == 9


def test_phase_one_y_choice_changes_the_column_gap():
    wide = LabelCandidate(('one line',), 7, 126, 1, 0, 0, 0)
    wrapped = LabelCandidate(('two', 'lines'), 14, 70, 2, 0, 0, 0)
    spec = _stack_spec(columns=2, blocks=2, profile=True,
                       labels=(grid.CellLabels(),
                               grid.CellLabels(y_candidates=(wide, wrapped))))
    spec = replace(spec, cells=tuple(replace(cell, stack=(56, 56))
                                     for cell in spec.cells))
    narrow_cap = replace(spec, label_layout=replace(
        spec.label_layout, max_row_gap_points=0))
    generous_cap = replace(spec, label_layout=replace(
        spec.label_layout, max_row_gap_points=10))
    narrow = grid.solve(narrow_cap)
    generous = grid.solve(generous_cap)
    role = 'label/heatmap_y/1/2'
    assert narrow.selections[role] == 1
    assert generous.selections[role] == 0
    left, right = 'cell/1/1/block/1', 'cell/1/2/block/1'
    assert _gap(narrow.rects[left], narrow.rects[right]) == 24
    assert _gap(generous.rects[left], generous.rects[right]) == 17


@pytest.mark.parametrize('scope,location', [
    ('cell', 'below'), ('figure', 'right'), ('figure', 'below')])
def test_heatmap_colorbar_scopes_are_solved(scope, location):
    entry = grid.BelowColorbarEntry(tick_height=5)
    bar = grid.Decoration('colorbar', scope, 0, location,
                          covered_cells=(0, 0) if scope == 'cell' else None,
                          below_entry=entry if scope == 'figure' and
                          location == 'below' else None)
    solved = grid.solve(_stack_spec(decorations=(bar,)))
    assert grid.colorbar_role(scope, 0) in solved.rects


def test_below_colorbar_title_keeps_the_measured_gap():
    bar = grid.Decoration('colorbar', 'cell', 0, 'below',
                          candidates=(_candidate('scale', 20, 8),),
                          covered_cells=(0, 0))
    geometry = replace(grid.DEFAULT_GEOMETRY,
                       colorbar_title_to_bar_gap=6)
    solved = grid.solve(_stack_spec(decorations=(bar,), geometry=geometry))
    rect = solved.rects['colorbar/cell/1']
    title = solved.rects['colorbar_title/cell/1']
    assert title.y0 - rect.y1 == 6


@pytest.mark.parametrize('scope,location', [
    ('cell', 'below'), ('figure', 'right'), ('figure', 'below')])
def test_heatmap_to_colorbar_gap(scope, location):
    bar = grid.Decoration(
        'colorbar', scope, 0, location,
        covered_cells=(0, 0) if scope == 'cell' else None,
        below_entry=(grid.BelowColorbarEntry()
                     if scope == 'figure' and location == 'below' else None))
    geometry = replace(grid.DEFAULT_GEOMETRY,
                       heatmap_colorbar_gap=13)
    solved = grid.solve(_stack_spec(decorations=(bar,), geometry=geometry))
    block = solved.rects['cell/1/1/block/1']
    colorbar = solved.rects[grid.colorbar_role(scope, 0)]
    separation = (colorbar.x0 - block.x1 if location == 'right' else
                  block.y0 - colorbar.y1)
    assert separation == 13


@pytest.mark.parametrize('scope,location', [
    ('row', 'above'), ('row', 'below'), ('row', 'right'),
    ('column', 'above'), ('column', 'below'), ('column', 'right'),
    ('figure', 'above'), ('figure', 'below'), ('figure', 'right')])
def test_profile_grid_legend_scope_and_location(scope, location):
    cell = grid.CellLayout(0, 0, grid.Size(50, 30), (), 50,
                           grid.CellInsets(), grid.CellLabels())
    item = grid.Decoration('legend', scope, 0, location,
                           size=grid.Size(12, 8))
    solved = grid.solve(grid.GridLayoutSpec(1, 1, (cell,), (item,)))
    assert grid.legend_role(scope, 0) in solved.rects


@pytest.mark.parametrize('scope', ['row', 'column'])
def test_grid_legend_above_stacks_outside_selected_column_titles(scope):
    cells = tuple(grid.CellLayout(
        row, column, grid.Size(50, 30), (), 50,
        grid.CellInsets(), grid.CellLabels(
            'long title', (_candidate('long\ntitle', 45, 20),),
            row * 2 + column))
        for row in range(2) for column in range(2))
    owner = 1 if scope == 'row' else 0
    legend = grid.Decoration('legend', scope, owner, 'above',
                             size=grid.Size(30, 12))
    solved = grid.solve(grid.GridLayoutSpec(2, 2, cells, (legend,)))
    legend_rect = solved.rects[grid.legend_role(scope, owner)]
    title_roles = (grid.cell_title(1, column) for column in range(2)) \
        if scope == 'row' else (grid.cell_title(0, 0),)
    for role in title_roles:
        title = solved.rects[role]
        assert legend_rect.y0 - title.y1 >= \
            grid.DEFAULT_GEOMETRY.external_legend_to_content_gap - 1e-9


@pytest.mark.parametrize('scope', ['row', 'column'])
def test_grid_legend_below_stacks_outside_selected_x_label(scope):
    cells = tuple(grid.CellLayout(
        row, column, grid.Size(50, 30), (), 50,
        grid.CellInsets(), grid.CellLabels())
        for row in range(2) for column in range(2))
    run = grid.LabelRun('x', (2,), (_candidate('distance', 35, 14),))
    owner = 1 if scope == 'row' else 0
    legend = grid.Decoration('legend', scope, owner, 'below',
                             size=grid.Size(30, 12))
    solved = grid.solve(grid.GridLayoutSpec(
        2, 2, cells, (legend,), (run,)))
    legend_rect = solved.rects[grid.legend_role(scope, owner)]
    label = solved.rects[grid.label_role('x', 0, 0)]
    assert label.y0 - legend_rect.y1 >= \
        grid.DEFAULT_GEOMETRY.external_legend_to_content_gap - 1e-9


@pytest.mark.parametrize('location', ['above', 'below', 'right'])
def test_profile_cell_legend_location(location):
    item = grid.Decoration('legend', 'cell', 0, location,
                           size=grid.Size(12, 8))
    insets = grid.CellInsets(profile=grid.PanelInsets(
        legend_size=item.size))
    cell = grid.CellLayout(0, 0, grid.Size(50, 30), (), 50,
                           insets, grid.CellLabels(), legend=item)
    solved = grid.solve(grid.GridLayoutSpec(1, 1, (cell,)))
    assert 'legend/cell/1' in solved.rects


@pytest.mark.parametrize('location', ['best', 'upper-left', 'lower-right'])
def test_profile_internal_cell_legend_uses_the_data_axes(location):
    item = grid.Decoration('legend', 'cell', 0, location,
                           size=grid.Size(12, 8))
    insets = grid.CellInsets(profile=grid.PanelInsets(
        legend_size=item.size))
    cell = grid.CellLayout(0, 0, grid.Size(50, 30), (), 50,
                           insets, grid.CellLabels(), legend=item)
    solved = grid.solve(grid.GridLayoutSpec(1, 1, (cell,)))
    assert 'cell/1/1/profile' in solved.rects
    assert 'legend/cell/1' not in solved.rects


def test_row_label_decoration_is_placed():
    cell = grid.CellLayout(0, 0, grid.Size(50, 30), (), 50,
                           grid.CellInsets(), grid.CellLabels())
    item = grid.Decoration('row_label', 'row', 0, 'right',
                           text='facet', candidates=(_candidate('facet'),))
    solved = grid.solve(grid.GridLayoutSpec(1, 1, (cell,), (item,)))
    assert 'label/row/1' in solved.rects


def test_series_heatmap_cell_right_colorbar_uses_gap_and_thickness():
    geometry = replace(grid.DEFAULT_GEOMETRY,
                       heatmap_colorbar_gap=11,
                       heatmap_colorbar_thickness=7)
    cell = grid.CellLayout(0, 0, grid.Size(50, 40), (), 50,
                           grid.CellInsets(), grid.CellLabels())
    bar = grid.Decoration('colorbar', 'cell', 0, 'right')
    solved = grid.solve(grid.GridLayoutSpec(
        1, 1, (cell,), (bar,), geometry=geometry))
    panel = solved.rects['cell/1/1/profile']
    colorbar = solved.rects['colorbar/cell/1']
    assert _gap(panel, colorbar) == 11
    assert colorbar.width == 7


def test_cell_right_colorbar_reserves_measured_tick_inset_in_two_rows():
    """Every row keeps colorbar ticks clear of its next data cell."""
    cells = tuple(grid.CellLayout(
        row, column, grid.Size(50, 40), (), 50,
        grid.CellInsets(), grid.CellLabels())
        for row in range(2) for column in range(2))
    bars = tuple(grid.Decoration('colorbar', 'cell', owner, 'right',
                                 insets=grid.Insets(right=20))
                 for owner in (0, 2))
    solved = grid.solve(grid.GridLayoutSpec(2, 2, cells, bars))
    for row, owner in ((1, 1), (2, 3)):
        bar = solved.rects[f'colorbar/cell/{owner}']
        next_cell = solved.rects[f'cell/{row}/2/profile']
        assert next_cell.x0 - bar.x1 >= bars[row - 1].insets.right


@pytest.mark.parametrize('rows,columns', [(2, 2), (1, 2)],
                         ids=['by_row_or_by_column', 'overlay'])
def test_row_labels_clear_series_heatmap_colorbars(rows, columns):
    """Row labels sit right of every cell-right colorbar and its ticks."""
    cells = tuple(grid.CellLayout(
        row, column, grid.Size(50, 40), (), 50,
        grid.CellInsets(), grid.CellLabels())
        for row in range(rows) for column in range(columns))
    bars = tuple(grid.Decoration('colorbar', 'cell', owner, 'right',
                                 insets=grid.Insets(right=20))
                 for owner in range(len(cells)))
    labels = tuple(grid.Decoration('row_label', 'row', row, 'right',
                                   text='facet',
                                   candidates=(_candidate('facet', 30),))
                   for row in range(rows))
    solved = grid.solve(grid.GridLayoutSpec(rows, columns, cells,
                                            bars + labels))
    for row in range(rows):
        label = solved.rects[grid.row_label(row)]
        for owner, bar in enumerate(bars):
            rect = grid._expand(solved.rects[f'colorbar/cell/{owner + 1}'],
                                bar.insets)
            assert label.x0 >= rect.x1 or label.y0 >= rect.y1 or \
                label.y1 <= rect.y0


def test_indicator_and_stack_label_use_their_own_boundaries():
    label = grid.LabelRun('stack', (0,), (_candidate('group'),),
                          block_index=0, location='left')
    y_label = grid.CellLabels(y_candidates=(_candidate('heatmap y'),))
    indicator = grid.Decoration('indicator', 'row', 0, 'left')
    geometry = replace(grid.DEFAULT_GEOMETRY, sort_indicator_gap=9,
                       heatmap_to_row_header_gap=5,
                       row_header_to_y_label_gap=7)
    spec = replace(_stack_spec(labels=(y_label,),
                               decorations=(indicator,), geometry=geometry),
                   runs=(label,))
    solved = grid.solve(spec)
    block = solved.rects['cell/1/1/block/1']
    marker = solved.rects['indicator/1']
    region = solved.rects['label/stack/1/1/1']
    heatmap_y = solved.rects['label/heatmap_y/1/1']
    assert _gap(marker, block) == 9
    assert _gap(region, marker) == 5
    assert _gap(heatmap_y, region) == 7


def test_indicator_precedes_left_heatmap_y_when_stack_labels_are_right():
    label = grid.LabelRun('stack', (0,), (_candidate('group'),),
                          block_index=0, location='right')
    y_label = grid.CellLabels(y_candidates=(_candidate('heatmap y'),))
    indicator = grid.Decoration('indicator', 'row', 0, 'left')
    geometry = replace(grid.DEFAULT_GEOMETRY, sort_indicator_gap=9,
                       row_header_to_y_label_gap=7)
    spec = replace(_stack_spec(labels=(y_label,),
                               decorations=(indicator,), geometry=geometry),
                   runs=(label,))
    rects = grid.solve(spec).rects
    assert _gap(rects['indicator/1'], rects['cell/1/1/block/1']) == 9
    assert _gap(rects['label/heatmap_y/1/1'], rects['indicator/1']) == 7


def test_left_indicator_and_right_region_labels_and_side_colorbar():
    label = grid.LabelRun('stack', (0,), (_candidate('group'),),
                          block_index=0, location='right')
    indicator = grid.Decoration('indicator', 'row', 0, 'left')
    bar = grid.Decoration('colorbar', 'figure', 0, 'right')
    spec = replace(_stack_spec(decorations=(indicator, bar)), runs=(label,))
    rects = grid.solve(spec).rects
    block = rects['cell/1/1/block/1']
    marker = rects['indicator/1']
    region = rects['label/stack/1/1/1']
    colorbar = rects['colorbar/figure/1']
    assert marker.x1 < block.x0 < block.x1 < region.x0 < colorbar.x0


def test_both_panel_kinds_keep_separate_y_labels():
    spec = _stack_spec(profile=True,
                       labels=(grid.CellLabels(y_candidates=(
                           _candidate('heatmap y'),)),))
    spec = replace(spec, runs=(
        grid.LabelRun('y', (0,), (_candidate('profile y'),)),))
    solved = grid.solve(spec)
    assert 'label/heatmap_y/1/1' in solved.rects
    assert 'label/y/1/1' in solved.rects


def test_row_colorbar_is_supported():
    bar = grid.Decoration('colorbar', 'row', 0, 'right')
    solved = grid.solve(_stack_spec(decorations=(bar,)))
    assert 'colorbar/row/1/1' in solved.rects


def _series_cell(index, column, stack_labels):
    return Cell(index, 0, column, (index,), (0,), '', None, None,
                (StackBlock(0, index),), stack_labels)


@pytest.mark.parametrize('vectors,expected', [
    (('A', 'A', 'A', 'A'), ((0, 1, 2, 3),)),
    (('A', 'B', 'A', 'B'), ((0,), (1,), (2,), (3,))),
    (('A', 'A', 'B', 'B'), ((0, 1), (2, 3))),
])
def test_stack_label_runs_are_maximal(vectors, expected):
    cells = tuple(_series_cell(i, i, (label,))
                  for i, label in enumerate(vectors))
    assert tuple(run.cells for run in label_runs(cells, 'stack')) == expected


def test_x_label_runs_keep_distinct_signatures():
    cells = tuple(replace(_series_cell(i, 0, ('A',)), row=i)
                  for i in range(3))
    kind = LabelKind('top', lambda cell: ('TSS',) if cell.row < 2
                     else ('TES',))
    assert tuple(run.cells for run in label_runs(cells, kind)) == \
        ((0, 1), (2,))


def test_feasible_stack_rects_are_contained_and_nonoverlapping():
    spec = _stack_spec(columns=2, blocks=2, profile=True)
    solved = grid.solve(spec)
    for rect in solved.rects.values():
        assert rect.x0 >= -1e-9 and rect.y0 >= -1e-9
        assert rect.x1 <= solved.figure_size.width + 1e-9
        assert rect.y1 <= solved.figure_size.height + 1e-9
    for column in (1, 2):
        top = solved.rects[f'cell/1/{column}/block/1']
        bottom = solved.rects[f'cell/1/{column}/block/2']
        profile = solved.rects[f'cell/1/{column}/profile']
        assert bottom.y1 < top.y0 < top.y1 < profile.y0


def _profile_grid(geometry=grid.DEFAULT_GEOMETRY, *, rows=2, columns=2,
                  runs=()):
    cells = tuple(grid.CellLayout(
        row, column, grid.Size(100, 80), (), 100,
        grid.CellInsets(), grid.CellLabels())
        for row in range(rows) for column in range(columns))
    return grid.solve(grid.GridLayoutSpec(
        rows, columns, cells, runs=runs, geometry=geometry))


def test_profile_row_gap_uses_geometry_field():
    base = _profile_grid()
    taller = _profile_grid(replace(
        grid.DEFAULT_GEOMETRY,
        profile_row_gap=grid.DEFAULT_GEOMETRY.profile_row_gap + 20))
    assert taller.figure_size.height == base.figure_size.height + 20
    assert taller.figure_size.width == base.figure_size.width


def test_profile_edge_padding_uses_geometry_field():
    base = _profile_grid()
    wider = _profile_grid(replace(
        grid.DEFAULT_GEOMETRY,
        figure_edge_padding=grid.DEFAULT_GEOMETRY.figure_edge_padding + 10))
    assert wider.figure_size.width == base.figure_size.width + 20
    assert wider.figure_size.height == base.figure_size.height + 20


def test_profile_common_axis_label_gap_uses_geometry_field():
    run = grid.LabelRun('x', (0,), (_candidate('shared x', 30, 8),))
    base = _profile_grid(rows=1, columns=1, runs=(run,))
    wider = _profile_grid(replace(
        grid.DEFAULT_GEOMETRY,
        common_axis_label_to_tick_labels_gap=(
            grid.DEFAULT_GEOMETRY.common_axis_label_to_tick_labels_gap + 11)),
        rows=1, columns=1, runs=(run,))
    assert wider.figure_size.height == base.figure_size.height + 11
    assert wider.rects['label/x/1/1'].y == base.rects['label/x/1/1'].y


def test_stack_block_weights_preserve_total_height_and_order():
    cell = grid.CellLayout(
        0, 0, None, grid.stack_heights(
            _cell((StackBlock(0, 0), StackBlock(1, 0))),
            (1, 3), 200, grid.DEFAULT_GEOMETRY.heatmap_group_gap),
        120, grid.CellInsets(), grid.CellLabels(),
        blocks=(StackBlock(0, 0), StackBlock(1, 0)))
    solved = grid.solve(grid.GridLayoutSpec(1, 1, (cell,)))
    upper = solved.rects['cell/1/1/block/1']
    lower = solved.rects['cell/1/1/block/2']
    gap = grid.DEFAULT_GEOMETRY.heatmap_group_gap
    assert upper.width == lower.width == 120
    assert upper.height / lower.height == pytest.approx(1 / 3)
    assert upper.height + lower.height + gap == pytest.approx(200)
    assert upper.y0 - lower.y1 == gap


@pytest.mark.parametrize('location', ['above', 'below', 'right'])
def test_summary_legend_owns_nonoverlapping_figure_slot(location):
    legend = grid.Decoration('legend', 'figure', 0, location,
                             size=grid.Size(80, 24))
    solved = grid.solve(_stack_spec(profile=True,
                                    decorations=(legend,)))
    slot = solved.rects['legend/figure/1']
    profile = solved.rects['cell/1/1/profile']
    block = solved.rects['cell/1/1/block/1']
    if location == 'above':
        assert slot.y0 >= profile.y1
    elif location == 'below':
        assert slot.y1 <= block.y0
    else:
        assert slot.x0 >= profile.x1


def test_below_common_colorbars_pack_titles_inside_canvas():
    bars = tuple(grid.Decoration(
        'colorbar', 'figure', index, 'below',
        candidates=(_candidate(f'long title {index}', 100, 10),
                    _candidate(f'title {index}', 45, 20)),
        below_entry=grid.BelowColorbarEntry(tick_height=8))
        for index in range(3))
    solved = grid.solve(_stack_spec(columns=3, decorations=bars))
    assert all(f'colorbar/figure/{index}' in solved.rects
               for index in range(1, 4))
    for role, rect in solved.rects.items():
        if role.startswith('colorbar_title/'):
            assert 0 <= rect.x0 <= rect.x1 <= solved.figure_size.width
            assert 0 <= rect.y0 <= rect.y1 <= solved.figure_size.height


def test_title_and_x_candidates_are_selected_in_one_grid():
    labels = (grid.CellLabels(
        title_text='first title', title_order=0,
        title_candidates=(_candidate('first title', 100, 10),
                          _candidate('first\\ntitle', 40, 20))),
              grid.CellLabels(
                  title_text='second title', title_order=1,
                  title_candidates=(_candidate('second title', 100, 10),
                                    _candidate('second\\ntitle', 40, 20))))
    run = grid.LabelRun('x', (0, 1),
                        (_candidate('shared x', 100, 10),
                         _candidate('shared\\nx', 40, 20)))
    solved = grid.solve(replace(_stack_spec(columns=2, labels=labels),
                                runs=(run,)))
    assert {'cell/1/1/title', 'cell/1/2/title', 'label/x/1/1'} <= \
        solved.selections.keys()
    assert all(index in (0, 1) for index in solved.selections.values())


def test_side_colorbar_grid_widens_for_more_scales():
    def solve(count):
        bars = tuple(grid.Decoration(
            'colorbar', 'figure', index, 'right', tick_label_width=12)
            for index in range(count))
        spec = _stack_spec(columns=4, decorations=bars)
        return grid.solve(replace(
            spec, cells=tuple(replace(cell, stack=(80,))
                              for cell in spec.cells)))

    one, many = solve(1), solve(4)
    assert many.figure_size.width > one.figure_size.width
    bars = [many.rects[f'colorbar/figure/{index}']
            for index in range(1, 5)]
    assert all(bar.x0 >= many.rects['cell/1/4/block/1'].x1 for bar in bars)


def test_below_common_bars_are_unstretched_and_enlarge_canvas():
    plain = grid.Decoration('colorbar', 'cell', 0, 'below',
                            covered_cells=(0, 1))
    common = tuple(grid.Decoration(
        'colorbar', 'figure', index, 'below',
        below_entry=grid.BelowColorbarEntry(
            left_overhang=6, right_overhang=6, tick_height=10,
            label_height=10, label_width=30)) for index in range(2))
    spec = _stack_spec(columns=2)
    simple = grid.solve(replace(spec, decorations=(plain,)))
    packed = grid.solve(replace(spec, decorations=common))
    assert packed.figure_size.height > simple.figure_size.height
    assert {packed.rects[f'colorbar/figure/{i}'].width for i in (1, 2)} == {50}


def test_side_and_below_colorbar_titles_keep_their_alignment():
    for scope, location in (('figure', 'right'), ('cell', 'below')):
        bar = grid.Decoration(
            'colorbar', scope, 0, location,
            candidates=(_candidate('scale', 30, 10),),
            covered_cells=(0, 0) if scope == 'cell' else None)
        solved = grid.solve(_stack_spec(decorations=(bar,)))
        rect = solved.rects[f'colorbar/{scope}/1']
        title = solved.rects[f'colorbar_title/{scope}/1']
        if location == 'right':
            assert title.x0 == rect.x0
            assert title.y0 > rect.y1
        else:
            assert title.x0 + title.width / 2 == \
                pytest.approx(rect.x0 + rect.width / 2)
            assert title.y0 > rect.y1


def test_right_stack_labels_fit_with_each_colorbar_scope():
    region = grid.LabelRun('stack', (0,), (_candidate('region', 30, 14),),
                           block_index=0, location='right')
    for scope, location in (('cell', 'below'), ('figure', 'below'),
                            ('figure', 'right')):
        bar = grid.Decoration(
            'colorbar', scope, 0, location,
            covered_cells=(0, 0) if scope == 'cell' else None,
            below_entry=(grid.BelowColorbarEntry() if scope == 'figure' and
                         location == 'below' else None))
        solved = grid.solve(replace(_stack_spec(decorations=(bar,)),
                                    runs=(region,)))
        label = solved.rects['label/stack/1/1/1']
        assert label.x1 + grid.DEFAULT_GEOMETRY.figure_edge_padding <= \
            solved.figure_size.width


def test_disabling_title_gap_growth_keeps_base_column_gap():
    title = _candidate('wide title', 100, 10)
    labels = tuple(grid.CellLabels(
        title_text='wide title', title_order=index,
        title_candidates=(title,)) for index in range(2))
    spec = _stack_spec(columns=2, labels=labels)
    automatic = grid.solve(spec)
    literal = grid.solve(replace(spec, label_layout=replace(
        spec.label_layout, auto_panel_title_column_gap=False)))
    left, right = 'cell/1/1/block/1', 'cell/1/2/block/1'
    assert _gap(literal.rects[left], literal.rects[right]) == \
        grid.DEFAULT_GEOMETRY.heatmap_min_column_gap
    assert _gap(automatic.rects[left], automatic.rects[right]) > \
        _gap(literal.rects[left], literal.rects[right])


def test_below_common_pareto_pruning_matches_exhaustive_greedy_oracle():
    pairs = (
        ((120.0, 10.0), (60.0, 20.0)),
        ((60.0, 10.0), (30.0, 20.0)),
        ((60.0, 10.0), (30.0, 20.0)),
        ((90.0, 10.0), (45.0, 20.0)),
        ((120.0, 10.0), (60.0, 20.0)),
    )
    candidate_sets = tuple(
        tuple(LabelCandidate(
            lines=('bar{} line {}'.format(index, line_count),) * line_count,
            width=width, height=height,
            line_count=line_count,
            balance_penalty=0.0, orphan_penalty=0.0)
            for line_count, (width, height) in enumerate(pair, 1))
        for index, pair in enumerate(pairs))
    entries = tuple(grid.BelowColorbarEntry(tick_height=tick)
                    for tick in (0.0, 0.0, 30.0, 0.0, 30.0))
    label_layout = grid.DEFAULT_LABEL_LAYOUT
    penalty_scale = max(
        candidate.height / max(candidate.line_count, 1)
        for candidates in candidate_sets for candidate in candidates
        if candidate.height > 0.0)

    def exhaustive_cost(path):
        packed_entries = tuple(
            replace(entry, label_width=candidate.width,
                    label_height=candidate.height)
            for entry, candidate in zip(entries, path))
        _, _, packed_height = grid.pack_below_common(
            packed_entries, 40.0, 8.0, 180.0, 0.0, 0.0,
            horizontal_gap=8.0, geometry=grid.DEFAULT_GEOMETRY,
            include_label_widths=True)
        overflow = sum(max(0.0, max(40.0, candidate.width) - 180.0)
                       for candidate in path)
        penalty = sum(
            label_layout.horizontal_colorbar_label_extra_row_penalty *
            max(0, candidate.line_count - min(
                item.line_count for item in candidates)) +
            label_layout.horizontal_colorbar_label_balance_weight *
            candidate.balance_penalty +
            label_layout.horizontal_colorbar_label_orphan_weight *
            candidate.orphan_penalty
            for candidates, candidate in zip(candidate_sets, path))
        typography_height = penalty_scale * penalty / len(path)
        return (overflow, packed_height + typography_height,
                packed_height, penalty,
                tuple(candidate.text for candidate in path))

    expected = min(product(*candidate_sets), key=exhaustive_cost)
    selected = grid._select_below_common_label_candidates(
        candidate_sets, entries, 40.0, 8.0, 180.0, label_layout,
        8.0, grid.DEFAULT_GEOMETRY)

    assert tuple(candidate.text for candidate in selected) == tuple(
        candidate.text for candidate in expected)
    assert tuple(candidate.line_count for candidate in selected) == (
        1, 2, 2, 2, 2)


def test_below_common_selector_prefers_legal_break_over_equal_geometry_fallback():
    legal = LabelCandidate(
        lines=('long title',), width=40.0, height=10.0,
        line_count=1,
        balance_penalty=0.0, orphan_penalty=0.0)
    emergency = replace(legal, break_penalty=1.0)

    selected = grid._select_below_common_label_candidates(
        ((emergency, legal),), (grid.BelowColorbarEntry(),), 40.0, 8.0, 180.0,
        grid.DEFAULT_LABEL_LAYOUT, 8.0, grid.DEFAULT_GEOMETRY)

    assert selected == (legal,)
