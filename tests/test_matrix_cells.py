"""New matrix cells are resolved, drawn, and described with the same roles."""

from pathlib import Path

from matplotlib.text import Text
import pytest

from deeptoolsr import plotMatrix
from deeptoolsr.plotting.geometry import DEFAULT_GEOMETRY, cm_to_points
from deeptoolsr.describe import describe
from tests import test_plot_baselines as baselines


MATRIX = Path(__file__).parent / 'test_heatmapper' / 'master_multi.mat.gz'


def _render(tmp_path, monkeypatch, *flags):
    """Capture the real builder result while the command saves a real PNG."""
    built = {}
    original = plotMatrix.build_matrix_figure

    def capture(matrix, layout, labels, plan, spec, prepared, run, extents,
                **kwargs):
        figure, solution = original(
            matrix, layout, labels, plan, spec, prepared, run, extents,
            **kwargs)
        built.update(figure=figure, solution=solution, layout=layout,
                     plan=plan, prepared=prepared, spec=spec)
        return figure, solution

    argv = ['-m', str(MATRIX), '-o', str(tmp_path / 'matrix.png'),
            '-p', '1', '--dpi', '100', '--sortRegions', 'no', *flags]
    with monkeypatch.context() as patch:
        patch.setattr(plotMatrix, 'build_matrix_figure', capture)
        plotMatrix.main(argv)
    gids = [axis.get_gid() for axis in built['figure'].axes]
    assert all(gids)
    assert len(gids) == len(set(gids))
    assert set(gids) <= set(built['solution'].rects)
    built['png'] = (tmp_path / 'matrix.png').read_bytes()
    built['describe'] = describe('plotMatrixR', argv)
    return built


def _roles(result, prefix):
    return {role: rect for role, rect in result['solution'].rects.items()
            if role.startswith(prefix)}


def _assert_described_cells_are_drawn(result):
    roles = result['solution'].rects
    for cell in result['describe']['cells']:
        row, column = cell['row'], cell['column']
        if cell['profile'] is not None:
            assert f'cell/{row}/{column}/profile' in roles
        for index, _block in enumerate(cell['blocks'], 1):
            assert f'cell/{row}/{column}/block/{index}' in roles


def test_multi_sample_cell_has_one_height_and_sample_major_blocks(
        tmp_path, monkeypatch):
    result = _render(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4', '--colorbarLocation', 'below',
        '--heatmapHeight', '8', '--regionsLabel', '', '')
    cells = result['describe']['cells']
    assert len(cells) == 2
    assert [(block['group'], block['sample']) for block in cells[0]['blocks']] == [
        (1, 1), (2, 1), (1, 2), (2, 2)]
    assert [len(cell['blocks']) for cell in cells] == [4, 4]
    gap = DEFAULT_GEOMETRY.heatmap_group_gap
    for cell in cells:
        row, column = cell['row'], cell['column']
        blocks = [result['solution'].rects[f'cell/{row}/{column}/block/{i}']
                  for i in range(1, 5)]
        assert all(upper.y0 - lower.y1 == pytest.approx(gap)
                   for upper, lower in zip(blocks, blocks[1:]))
        assert sum(block.height for block in blocks) + 3 * gap == \
            pytest.approx(cm_to_points(8))
    assert len(_roles(result, 'colorbar/cell/')) == 2
    _assert_described_cells_are_drawn(result)


def test_explicit_pergroup_sets_equal_adjacent_cells(tmp_path, monkeypatch):
    grouped = _render(tmp_path, monkeypatch, '--heatmap', '--perGroup',
                      '--arrangeSamples', '1', '2', '3', '4')
    adjacent = _render(tmp_path, monkeypatch, '--heatmap',
                       '--arrangeSamples', '1', '2', '3', '4',
                       '--sampleSetGroupArrangement', 'adjacent')
    assert len(grouped['describe']['cells']) == 8
    assert grouped['describe'] == adjacent['describe']
    assert grouped['solution'].rects == adjacent['solution'].rects
    assert grouped['png'] == adjacent['png']
    _assert_described_cells_are_drawn(grouped)


@pytest.mark.parametrize('placement,shape,row_labels,stack_runs', [
    ('by_row', (4, 2), 4, 8),
    ('by_column', (2, 4), 2, 2),
])
def test_grid_placements_draw_described_roles(
        tmp_path, monkeypatch, placement, shape, row_labels, stack_runs):
    result = _render(tmp_path, monkeypatch, '--heatmap', '--profile',
                     '--sampleSetGroupArrangement', placement)
    cells = result['describe']['cells']
    assert (max(cell['row'] for cell in cells),
            max(cell['column'] for cell in cells)) == shape
    assert len(_roles(result, 'label/row/')) == row_labels
    assert len(_roles(result, 'label/stack/')) == stack_runs
    _assert_described_cells_are_drawn(result)


def test_single_column_grid_keeps_bottom_x_decoration(tmp_path, monkeypatch):
    result = _render(tmp_path, monkeypatch, '--heatmap', '--profile',
                     '--gridColumns', '1', '--xAxisLabel', 'distance',
                     '--yAxisVisibility', 'outer_merged')
    assert len(result['describe']['cells']) == 4
    assert {cell['row'] for cell in result['describe']['cells']} == \
        {1, 2, 3, 4}
    assert len(_roles(result, 'label/x/')) == 1
    assert len(_roles(result, 'label/heatmap_y/')) <= 1
    _assert_described_cells_are_drawn(result)


def test_different_reference_labels_make_two_x_runs(tmp_path, monkeypatch):
    result = _render(tmp_path, monkeypatch, '--heatmap',
                     '--gridColumns', '1', '--refPointLabel', 'A', 'A', 'B', 'B')
    assert len(_roles(result, 'label/x/')) == 2
    _assert_described_cells_are_drawn(result)


def test_merged_x_runs_continue_only_from_the_previous_column(
        tmp_path, monkeypatch):
    """A column without an X label breaks a one-row merged X run."""
    from deeptoolsr.plotting import matrix_figure

    def label(state, sample):
        return '' if sample == 1 else 'A'
    runs = []
    original = matrix_figure._heatmap_x_runs

    def capture(state, context):
        runs.extend(original(state, context))
        return runs
    monkeypatch.setattr(matrix_figure, '_heatmap_distance_axis_label', label)
    monkeypatch.setattr(matrix_figure, '_heatmap_x_runs', capture)
    _render(tmp_path, monkeypatch, '--heatmap',
            '--xAxisVisibility', 'outer_merged')
    assert [run.panel_indices for run, _style in runs] == [(0,), (2, 3)]


@pytest.mark.parametrize('location,scope,count', [
    ('right', 'row', 4),
    ('right_common', 'figure', 1),
    ('below', 'cell', 4),
    ('below_common', 'figure', 1),
])
def test_colorbar_scope_in_multirow_grid(
        tmp_path, monkeypatch, location, scope, count):
    result = _render(tmp_path, monkeypatch, '--heatmap', '--profile',
                     '--gridColumns', '1', '--colorbarLocation', location)
    assert len(_roles(result, f'colorbar/{scope}/')) == count
    _assert_described_cells_are_drawn(result)


def test_ci_summary_is_drawn_above_each_stack(tmp_path, monkeypatch):
    result = _render(tmp_path, monkeypatch, '--heatmap', '--profile',
                     '--plotType', 'ci')
    roles = result['solution'].rects
    for cell in result['describe']['cells']:
        profile = roles[f"cell/{cell['row']}/{cell['column']}/profile"]
        upper_stack = roles[f"cell/{cell['row']}/{cell['column']}/block/1"]
        assert profile.y0 >= upper_stack.y1
    assert any(axis.collections for axis in result['figure'].axes
               if (axis.get_gid() or '').endswith('/profile'))
    _assert_described_cells_are_drawn(result)


def test_heatmap_y_labels_form_two_runs(tmp_path, monkeypatch):
    result = _render(tmp_path, monkeypatch, '--heatmap',
                     '--heatmapYAxisLabel', 'A', 'A', 'B', 'B')
    assert len(_roles(result, 'label/heatmap_y/')) == 2
    texts = [axis.texts[0].get_text() for axis in result['figure'].axes
             if (axis.get_gid() or '').startswith('label/heatmap_y/')]
    assert texts == ['A', 'B']


def test_series_heatmap_can_use_a_two_by_two_grid(tmp_path, monkeypatch):
    result = _render(tmp_path, monkeypatch, '--profile',
                     '--plotType', 'heatmap', '--gridColumns', '2')
    cells = result['describe']['cells']
    assert len(cells) == 4
    assert {cell['row'] for cell in cells} == {1, 2}
    assert {cell['column'] for cell in cells} == {1, 2}
    assert len(_roles(result, 'colorbar/cell/')) == 4
    _assert_described_cells_are_drawn(result)


@pytest.mark.parametrize('visibility, shown', [
    ('show_all', {'1/1', '1/2', '2/1', '2/2'}),
    ('outer', {'1/1', '2/1'}),
    ('outer_merged', {'1/1', '2/1'}),
])
def test_series_heatmap_row_names_follow_the_y_axis_run_rule(
        tmp_path, monkeypatch, visibility, shown):
    result = _render(tmp_path, monkeypatch, '--profile',
                     '--plotType', 'heatmap', '--gridColumns', '2',
                     '--yAxisVisibility', visibility)
    labelled = {axis.get_gid()[len('cell/'):-len('/profile')]
                for axis in result['figure'].axes
                if (axis.get_gid() or '').endswith('/profile') and
                any(label.get_visible() and label.get_text()
                    for label in axis.get_yticklabels())}
    assert labelled == shown


def test_series_heatmap_colour_assignment_addresses_panels(
        tmp_path, monkeypatch):
    result = _render(tmp_path, monkeypatch, '--profile',
                     '--plotType', 'heatmap', '--colors', '2=viridis')
    maps = {axis.get_gid(): axis.images[0].get_cmap().name
            for axis in result['figure'].axes
            if (axis.get_gid() or '').endswith('/profile')}
    assert maps == {'cell/1/1/profile': 'RdYlBu_r',
                    'cell/1/2/profile': 'viridis',
                    'cell/1/3/profile': 'RdYlBu_r',
                    'cell/1/4/profile': 'RdYlBu_r'}


def test_series_heatmap_rows_are_named_by_what_varies(tmp_path, monkeypatch):
    # by_row puts one group's samples in each panel: rows are samples.
    result = _render(tmp_path, monkeypatch, '--profile',
                     '--plotType', 'heatmap', '--arrangeSamples', '1,2', '3,4',
                     '--samplesLabel', 'A', 'B', 'C', 'D',
                     '--sampleSetGroupArrangement', 'by_row',
                     '--yAxisVisibility', 'show_all')
    names = {axis.get_gid(): [label.get_text()
                              for label in axis.get_yticklabels()]
             for axis in result['figure'].axes
             if (axis.get_gid() or '').endswith('/profile')}
    assert sorted(map(sorted, names.values())) == [
        ['A', 'B'], ['A', 'B'], ['C', 'D'], ['C', 'D']]


@pytest.mark.parametrize('value, message', [
    ('7=viridis', '--colors: 7 names panel 7, but there are only 4 panels'),
    ('2=notamap', '--colors: notamap is not a colour map'),
])
def test_series_heatmap_colour_errors_name_the_option(
        tmp_path, capsys, value, message):
    with pytest.raises(SystemExit) as error:
        plotMatrix.main(['-m', str(MATRIX), '-o', str(tmp_path / 'm.png'),
                         '--profile', '--plotType', 'heatmap',
                         '--colors', value])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('extra', [
    ('--arrangeSamples', '1,2', '3,4'),
    ('--sampleSetGroupArrangement', 'adjacent'),
    ('--gridColumns', '2'),
    ('--sampleSetLabels', 'first', 'second', 'third', 'fourth'),
    ('--xAxisVisibility', 'show_all'),
    ('--yAxisVisibility', 'show_all'),
    ('--commonLegend',),
])
def test_series_heatmap_custom_layout_renders_and_describes(
        tmp_path, monkeypatch, extra):
    result = _render(tmp_path, monkeypatch, '--profile',
                     '--plotType', 'heatmap', *extra)
    _assert_described_cells_are_drawn(result)
    assert len(_roles(result, 'colorbar/cell/')) == \
        len(result['describe']['cells'])


def _intersects(a, b, clearance=0.5):
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > clearance and
            min(a.y1, b.y1) - max(a.y0, b.y0) > clearance)


def _is_data(role):
    return role.startswith('cell/') and (
        '/block/' in role or role.endswith('/profile'))


def _is_text_slot(role):
    return (role == 'title' or role.startswith('label/') or
            role.startswith('colorbar_title/') or
            role.startswith('cell/') and role.endswith('/title'))


def _visible_texts(figure):
    """Yield visible artist text with its owner and whether it is a tick."""
    seen = set()
    for axis in figure.axes:
        groups = [(False, axis.texts)]
        if axis.axison:
            groups.append((False, (axis.title,)))
            if axis.xaxis.get_visible():
                groups.append((False, (axis.xaxis.label,)))
                groups.append((True, _in_view_ticks(axis.xaxis, axis.get_xlim())))
            if axis.yaxis.get_visible():
                groups.append((False, (axis.yaxis.label,)))
                groups.append((True, _in_view_ticks(axis.yaxis, axis.get_ylim())))
        legend = axis.get_legend()
        if legend is not None and legend.get_visible():
            groups.append((False, legend.get_texts()))
        for marker, texts in groups:
            for item in texts:
                if (not isinstance(item, Text) or id(item) in seen or
                        not item.get_visible() or not item.get_text()):
                    continue
                seen.add(id(item))
                yield axis, item, marker
    for item in figure.texts:
        if (id(item) not in seen and item.get_visible() and item.get_text()):
            yield None, item, False


def _in_view_ticks(axis, limits):
    low, high = sorted(limits)
    tolerance = max(high - low, 1) * 1e-9
    for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
        if low - tolerance <= tick.get_loc() <= high + tolerance:
            yield tick.label1
            yield tick.label2


def test_grid_columns_profiles_keep_visible_y_ticks(tmp_path):
    """Each row in the one-column grid owns its profile Y tick labels."""
    _solution, figure = baselines._render_matrix(
        baselines.MATRIX_CASES['matrix_grid_columns'],
        tmp_path / 'matrix_grid_columns.png')
    roles = {axis.get_gid(): axis for axis in figure.axes}
    for row in range(1, 5):
        axis = roles[f'cell/{row}/1/profile']
        assert axis.yaxis.get_visible()
        assert any(tick.get_visible() and tick.get_text()
                   for tick in _in_view_ticks(axis.yaxis, axis.get_ylim())), row


@pytest.mark.parametrize('kind,name,case', baselines.ALL_CASES,
                         ids=[name for _, name, _ in baselines.ALL_CASES])
def test_rendered_text_and_roles_fit_without_overlap(kind, name, case,
                                                     tmp_path):
    """Check the final renderer, including tick text after savefig settles it."""
    render = {'heatmap': baselines._render_heatmap,
              'profile': baselines._render_profile,
              'matrix': baselines._render_matrix}[kind]
    _solution, figure = render(case, tmp_path / (name + '.png'))
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    roles = {axis.get_gid(): axis for axis in figure.axes}
    problems = []
    assert len(roles) == len(figure.axes)
    assert None not in roles
    checked_roles = [(role, axis) for role, axis in roles.items()
                     if _is_text_slot(role) or _is_data(role)]
    for index, (role, axis) in enumerate(checked_roles):
        for other_role, other in checked_roles[index + 1:]:
            if _intersects(axis.bbox, other.bbox):
                problems.append(f'{role} rect overlaps {other_role} rect')
    indicator = roles.get('indicator')
    if indicator is not None:
        for role, axis in roles.items():
            if _is_text_slot(role) and _intersects(
                    indicator.bbox, axis.bbox, clearance=1e-6):
                problems.append(f'indicator rect overlaps {role} rect')
    for owner, item, tick in _visible_texts(figure):
        box = item.get_window_extent(renderer)
        if (indicator is not None and owner is not indicator and
                _intersects(indicator.bbox, box, clearance=1e-6)):
            problems.append(f'indicator overlaps '
                            f'{owner.get_gid() if owner else "figure"} '
                            f'text {item.get_text()!r}')
        if box.x0 < canvas.x0 - 1 or box.y0 < canvas.y0 - 1 or \
                box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1:
            problems.append(f'{owner.get_gid() if owner else "figure"} '
                            f'text {item.get_text()!r} outside canvas: '
                            f'{tuple(round(x, 2) for x in box.bounds)}')
        if not tick:
            continue
        for role, axis in roles.items():
            if axis is owner or not (_is_data(role) or _is_text_slot(role)):
                continue
            if _intersects(box, axis.bbox):
                problems.append(f'{owner.get_gid()} tick '
                                f'{item.get_text()!r} overlaps {role}')
    assert not problems, f'{name} ({kind}): ' + '; '.join(problems)


@pytest.mark.parametrize('location', ['left', 'right'])
@pytest.mark.parametrize('flags_y', [
    (), ('--heatmapYAxisLabel', 'first signal', 'second signal')],
    ids=['no_heatmap_y', 'heatmap_y'])
@pytest.mark.parametrize('flags', [
    ('--profile', '--arrangeSamples', '1,2', '3,4',
     '--sampleSetGroupArrangement', 'by_row'),
    ('--arrangeSamples', '1,2', '3,4', '--samplesLabel', 'A', 'B', 'C', 'D',
     '--sampleSetGroupArrangement', 'by_column'),
    ('--perGroup', '--arrangeSamples', '1,2', '3,4',
     '--samplesLabel', 'A', 'B', 'C', 'D'),
], ids=['by_row', 'by_column', 'overlay_differing_labels'])
def test_region_labels_clear_every_heatmap_and_heatmap_y_label(
        tmp_path, monkeypatch, location, flags_y, flags):
    result = _render(tmp_path, monkeypatch, '--heatmap', *flags,
                     '--regionLabelLocation', location, *flags_y)
    labels = _roles(result, 'label/stack/')
    blocks = {role: rect for role, rect in _roles(result, 'cell/').items()
              if '/block/' in role}
    y_labels = _roles(result, 'label/heatmap_y/')
    assert labels and bool(y_labels) == bool(flags_y)
    assert not [(label, block) for label, rect in labels.items()
                for block, other in blocks.items() if _intersects(rect, other)]
    # A heatmap Y label spans its row's stacks: no region label of that row
    # may share its horizontal extent.
    assert not [(label, y_label) for label, rect in labels.items()
                for y_label, other in y_labels.items()
                if label.split('/')[2] == y_label.split('/')[2] and
                rect.x0 < other.x1 and other.x0 < rect.x1]


def test_column_legends_below_clear_the_last_rows_colorbars(
        tmp_path, monkeypatch):
    result = _render(tmp_path, monkeypatch, '--profile', '--heatmap',
                     '--arrangeSamples', '1,2', '3,4',
                     '--sampleSetGroupArrangement', 'by_column',
                     '--legendLocation', 'below',
                     '--colorbarLocation', 'below')
    legends = _roles(result, 'legend/column/')
    bars = _roles(result, 'colorbar/')
    assert legends and bars
    for legend in legends.values():
        for bar in bars.values():
            assert (legend.y1 <= bar.y0 or bar.y1 <= legend.y0 or
                    legend.x1 <= bar.x0 or bar.x1 <= legend.x0)


def _collision_cases():
    arrangements = ('overlay', 'adjacent', 'by_row', 'by_column')
    legends = ('below', 'above', 'right', 'best', 'upper-right')
    bars = ('below', 'right', 'right_common')
    for arrangement in arrangements:
        for legend in legends:
            yield ('--profile',), arrangement, ('--legendLocation', legend)
            for bar in bars:
                yield (('--profile', '--heatmap'), arrangement,
                       ('--legendLocation', legend, '--colorbarLocation', bar))
        for bar in bars:
            yield ('--heatmap',), arrangement, ('--colorbarLocation', bar)
        for plots in (('--profile',), ('--profile', '--heatmap')):
            yield plots, arrangement, ('--commonLegend', '--legendLocation',
                                       'below')


@pytest.mark.parametrize(
    'plots, arrangement, extra', tuple(_collision_cases()),
    ids=lambda value: (value if isinstance(value, str) else
                       '_'.join(item.lstrip('-') for item in value)))
def test_decorations_never_collide(tmp_path, monkeypatch, plots, arrangement,
                                   extra):
    """No two solved rects overlap; a cell's own profile and blocks nest."""
    result = _render(tmp_path, monkeypatch, *plots,
                     '--arrangeSamples', '1,2', '3,4', '--quantiles', '2',
                     '--sortRegions', 'descend', '--showRegionCounts',
                     '--cellWidth', '3', '--samplesLabel', 'POINTseq_rep1',
                     'POINTseq_rep2', 'Control_rep1', 'Control_rep2',
                     '--sampleSetGroupArrangement', arrangement, *extra)
    rects = sorted(result['solution'].rects.items())
    assert not [(a, b) for index, (a, left) in enumerate(rects)
                for b, right in rects[index + 1:]
                if _intersects(left, right)]
