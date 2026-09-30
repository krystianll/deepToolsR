"""Focused regressions for shared label and colorbar clearance."""

from deeptoolsr.plotting.geometry import DEFAULT_GEOMETRY
from tests.plotting.test_matrix_figure import _render
from tests.test_matrix_cells import _render as _render_command
import pytest


def _axes(result, prefix):
    return {axis.get_gid(): axis for axis in result['figure'].axes
            if (axis.get_gid() or '').startswith(prefix)}


def test_side_colorbars_pack_final_height_tick_labels():
    figure, _ = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar',
        colorbarLocation='right', heatmapWidth=3, heatmapHeight=6,
        zMin=['1=-2', '2=-2.5', '3=-1.5', '4=-1'], zMax=[1],
        colorbarLabels=['abcd', 'efgh', 'ijkl', 'mnop'])
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bars = sorted((axis for axis in figure.axes
                   if (axis.get_gid() or '').startswith('colorbar/row/1/')),
                  key=lambda axis: axis.get_gid())
    assert len(bars) == 4
    assert '-0.75' in [tick.get_text() for tick in bars[1].get_yticklabels()]
    first_column_right = max(
        axis.get_tightbbox(renderer).x1 for axis in bars[:2])
    second_column_left = min(
        axis.get_position().x0 * figure.bbox.width for axis in bars[2:])
    gap_points = (second_column_left - first_column_right) * 72 / figure.dpi
    assert gap_points >= DEFAULT_GEOMETRY.colorbar_long_edge_gap - 1e-7


def test_heatmap_by_row_decorations_follow_sample_sets(tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4', '--sampleSetLabels', 'First',
        'Second', '--sampleSetGroupArrangement', 'by_row',
        '--sortRegions', 'descend', '--legendLocation', 'right',
        '--colorbarLocation', 'right', '--colorbarLabels',
        '1=First bar', '2=Second bar', '--samplesLabel',
        'One', 'Two', 'Three', 'Four', '--colors',
        'red', 'green', 'blue', 'black', '--colorMap', 'Reds', 'Blues',
        '--zMin', '1=-2', '2=-1')
    assert set(_axes(result, 'indicator/')) == {'indicator/1', 'indicator/2'}
    assert set(_axes(result, 'colorbar/')) == {
        'colorbar/row/1/1', 'colorbar/row/2/1'}
    titles = _axes(result, 'colorbar_title/')
    assert {role: axis.texts[0].get_text() for role, axis in
            titles.items()} == {
                'colorbar_title/row/1/1': 'First bar',
                'colorbar_title/row/2/1': 'Second bar'}
    legends = _axes(result, 'legend/')
    assert {role: tuple(text.get_text() for text in axis.get_legend().texts)
            for role, axis in legends.items()} == {
                'legend/row/1': ('One', 'Two'),
                'legend/row/2': ('Three', 'Four')}
    labels = _axes(result, 'label/row/')
    assert {role: axis.texts[0].get_text() for role, axis in labels.items()} \
        == {'label/row/1': 'First', 'label/row/2': 'Second'}
    assert all(axis.texts[0].get_rotation() == 270 for axis in
               labels.values())
    assert result['prepared'].color_options['colorMap'] == ['Reds', 'Blues']
    assert result['prepared'].z_min == (-2.0, -1.0)
    profiles = _axes(result, 'cell/')
    assert [line.get_color() for line in
            profiles['cell/1/1/profile'].lines[:2]] == ['red', 'green']
    assert [line.get_color() for line in
            profiles['cell/2/1/profile'].lines[:2]] == ['blue', 'black']


def test_heatmap_by_column_uses_cell_stack_labels(tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4', '--sampleSetLabels', 'First',
        'Second', '--samplesLabel', 'A', 'B', 'C', 'D',
        '--sampleSetGroupArrangement', 'by_column',
        '--legendLocation', 'right', '--colorbarLocation', 'right')
    labels = _axes(result, 'label/stack/')
    # Each block names its own sample (the group is the row), per cell.
    assert len(labels) == sum(len(cell.blocks) for cell in
                              result['plan'].cells)
    assert {axis.texts[0].get_text() for axis in
            labels.values()} == {'A', 'B', 'C', 'D'}
    # A right legend repeats beside each cell's profile (not one per column).
    assert set(_axes(result, 'legend/')) == {
        f'legend/cell/{index}' for index in range(1, 5)}
    rows = _axes(result, 'label/row/')
    # by_column rows are group sets, as on a profile-only figure.
    assert [rows[f'label/row/{row}'].texts[0].get_text()
            for row in (1, 2)] == list(result['plan'].row_labels)
    assert all(axis.texts[0].get_rotation() == 270 for axis in rows.values())


def test_by_row_legends_do_not_share_one_slot(tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4',
        '--sampleSetGroupArrangement', 'by_row',
        '--legendLocation', 'below')
    rects = result['solution'].rects
    first = rects['legend/row/1']
    second = rects['legend/row/2']
    assert first.y0 > second.y1


def test_right_common_bar_and_common_legend_are_figure_scope(
        tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4',
        '--sampleSetGroupArrangement', 'by_row',
        '--colorbarLocation', 'right_common',
        '--commonLegend', '--legendLocation', 'below')
    assert set(_axes(result, 'colorbar/')) == {'colorbar/figure/1'}
    assert set(_axes(result, 'legend/')) == {'legend/figure/1'}
    legend = result['solution'].rects['legend/figure/1']
    cells = [rect for role, rect in result['solution'].rects.items()
             if role.startswith('cell/') and
             ('/block/' in role or role.endswith('/profile'))]
    assert legend.y1 < min(rect.y0 for rect in cells)
    bar = result['solution'].rects['colorbar/figure/1']
    assert bar.y0 == pytest.approx(min(rect.y0 for rect in cells))
    assert bar.y1 == pytest.approx(max(rect.y1 for rect in cells))


def test_profile_legend_sits_between_profile_and_stack(tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4', '--legendLocation', 'below')
    rects = result['solution'].rects
    for column in (1, 2):
        legend = rects[f'legend/cell/{column}']
        profile = rects[f'cell/1/{column}/profile']
        block = rects[f'cell/1/{column}/block/1']
        assert block.y1 < legend.y0 and legend.y1 < profile.y0


def test_row_legend_sits_between_profile_and_stack(tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4',
        '--sampleSetGroupArrangement', 'by_row',
        '--legendLocation', 'below')
    rects = result['solution'].rects
    for row in (1, 2):
        legend = rects[f'legend/row/{row}']
        profile = rects[f'cell/{row}/1/profile']
        block = rects[f'cell/{row}/1/block/1']
        assert block.y1 < legend.y0 and legend.y1 < profile.y0
    # The row gap no longer holds the legend or an empty title band.
    gap = rects['cell/1/1/block/2'].y0 - rects['cell/2/1/profile'].y1
    assert gap == pytest.approx(DEFAULT_GEOMETRY.profile_row_gap)


def test_column_legend_stays_outside_the_grid(tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4',
        '--sampleSetGroupArrangement', 'by_column',
        '--legendLocation', 'below')
    rects = result['solution'].rects
    cells = [rect for role, rect in rects.items()
             if role.startswith('cell/') and
             ('/block/' in role or role.endswith('/profile'))]
    for column in (1, 2):
        assert rects[f'legend/column/{column}'].y1 < min(
            rect.y0 for rect in cells)


def test_stacked_rows_do_not_reserve_x_labels_twice(tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4',
        '--sampleSetGroupArrangement', 'by_row')
    rects = result['solution'].rects
    upper = min(rect.y0 for role, rect in rects.items()
                if role.startswith('label/x/') and role.endswith('/1'))
    lower = max(rect.y1 for role, rect in rects.items()
                if role.startswith('cell/2/') and role.endswith('/profile'))
    assert upper - lower <= (DEFAULT_GEOMETRY.profile_row_gap +
                             DEFAULT_GEOMETRY.column_header_to_panel_gap +
                             1e-6)


def test_row_labels_clear_stacks_and_row_colorbars(tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4',
        '--sampleSetGroupArrangement', 'by_row', '--colorbarLocation',
        'right')
    rects = result['solution'].rects
    for row in (1, 2):
        label = rects[f'label/row/{row}']
        others = [rect for role, rect in rects.items()
                  if role.startswith((f'cell/{row}/', f'colorbar/row/{row}/'))]
        assert label.x0 > max(rect.x1 for rect in others)


def test_adjacent_region_labels_keep_minimum_clearance(tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--arrangeSamples', '1,2',
        '--samplesLabel', 'a long sample label', 'another long label',
        'C', 'D')
    rects = sorted((rect for role, rect in result['solution'].rects.items()
                    if role.startswith('label/stack/1/1/')),
                   key=lambda rect: -rect.y1)
    assert len(rects) > 1
    for upper, lower in zip(rects, rects[1:]):
        assert upper.y0 - lower.y1 >= (
            DEFAULT_GEOMETRY.label_min_clearance - 1e-6)


def test_best_legend_that_does_not_fit_moves_above_panels(
        tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile', '--cellWidth', '2',
        '--legendLocation', 'best', '--samplesLabel',
        'a sample label far wider than its narrow panel', 'B', 'C', 'D')
    figure, rects = result['figure'], result['solution'].rects
    assert not any(axis.get_legend() for axis in figure.axes
                   if (axis.get_gid() or '').endswith('/profile'))
    for column in range(1, 5):
        legend = rects[f'legend/cell/{column}']
        assert legend.y0 > rects[f'cell/1/{column}/profile'].y1


def test_above_legends_share_one_baseline_over_uneven_titles(
        tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--samplesLabel', 'a much longer first sample title that has to wrap '
        'over several lines', 'B', 'C', 'D',
        '--legendLocation', 'above')
    rects = result['solution'].rects
    titles = [rect for role, rect in rects.items()
              if role.startswith('cell/1/') and role.endswith('/title')]
    assert len({round(rect.height, 6) for rect in titles}) > 1
    legends = [rect for role, rect in rects.items()
               if role.startswith('legend/cell/')]
    assert len(legends) == 4
    assert len({round(rect.y, 6) for rect in legends}) == 1
    assert min(rect.y for rect in legends) > max(rect.y1 for rect in titles)


def test_one_group_keeps_one_colour_across_sample_panels(
        tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--regionsLabel', 'A', 'A', '--sameGroupLabels', 'merge',
        '--commonLegend', '--legendLocation', 'below')
    profiles = _axes(result, 'cell/')
    colours = {line.get_color() for role, axis in profiles.items()
               if role.endswith('/profile') for line in axis.lines
               if not line.get_label().startswith('_')}
    assert len(colours) == 1
    legend = _axes(result, 'legend/')['legend/figure/1'].get_legend()
    assert len(legend.texts) == 1


def test_row_right_legend_sits_beside_profiles_before_row_label(
        tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4',
        '--sampleSetGroupArrangement', 'by_row', '--sampleSetLabels',
        'First', 'Second', '--legendLocation', 'right')
    rects = result['solution'].rects
    geometry = DEFAULT_GEOMETRY
    for row in (1, 2):
        legend = rects[f'legend/row/{row}']
        profile = rects[f'cell/{row}/2/profile']
        gap = legend.x - profile.x1
        assert geometry.right_legend_to_content_gap <= gap < \
            geometry.right_legend_to_content_gap + 30
        assert profile.y <= legend.y + legend.height / 2 <= profile.y1
        assert rects[f'label/row/{row}'].x > legend.x1


def test_by_column_right_legend_repeats_per_cell_with_heatmaps(
        tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4',
        '--sampleSetGroupArrangement', 'by_column',
        '--legendLocation', 'right')
    legends = _axes(result, 'legend/')
    assert set(legends) == {f'legend/cell/{index}' for index in range(1, 5)}


def test_common_right_legend_centres_on_the_whole_envelope(
        tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4',
        '--sampleSetGroupArrangement', 'by_row',
        '--commonLegend', '--legendLocation', 'right')
    rects = result['solution'].rects
    data = [rect for role, rect in rects.items()
            if role.startswith('cell/') and
            ('/block/' in role or role.endswith('/profile'))]
    legend = rects['legend/figure/1']
    centre = (min(rect.y0 for rect in data) + max(rect.y1 for rect in data)) / 2
    assert legend.y + legend.height / 2 == pytest.approx(centre)
    assert legend.x > max(rect.x1 for role, rect in rects.items()
                          if role.startswith('label/row/'))


@pytest.mark.parametrize('placement', ['by_row', 'by_column'])
def test_row_labels_share_one_line_clear_of_every_legend(
        tmp_path, monkeypatch, placement):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--profile',
        '--arrangeSamples', '1,2', '3,4',
        '--sampleSetGroupArrangement', placement, '--sampleSetLabels',
        'First', 'Second', '--samplesLabel', 'a long first sample label',
        'B', 'a long third sample label', 'D', '--legendLocation', 'right')
    rects = result['solution'].rects
    labels = [rect for role, rect in rects.items()
              if role.startswith('label/row/')]
    assert len(labels) == 2
    assert len({round(rect.x, 6) for rect in labels}) == 1
    legends = [rect for role, rect in rects.items()
               if role.startswith('legend/')]
    assert min(rect.x for rect in labels) > max(rect.x1 for rect in legends)


def test_below_colorbar_titles_follow_their_bars_in_every_row(
        tmp_path, monkeypatch):
    result = _render_command(
        tmp_path, monkeypatch, '--heatmap', '--gridColumns', '2',
        '--colorbarLocation', 'below', '--colorbarLabels', 'a', 'b', 'c', 'd')
    titles = _axes(result, 'colorbar_title/')
    rects = result['solution'].rects
    assert [titles[role].texts[0].get_text() for role in sorted(titles)] == \
        ['a', 'b', 'c', 'd']
    for role in titles:
        bar = rects[role.replace('colorbar_title/', 'colorbar/')]
        title = rects[role]
        assert title.y >= bar.y1
        assert bar.x <= title.x + title.width / 2 <= bar.x1
