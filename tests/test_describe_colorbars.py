"""Describe protocol 2: heatmap colour bars, planned once for plot and describe."""

import pytest

from deeptoolsr import plotHeatmap
from deeptoolsr.describe import describe
from deeptoolsr.plotting import matrix_figure

from tests.test_describe_protocol2 import _write


@pytest.fixture
def plain(tmp_path):
    return _write(tmp_path / 'plain.gz')


def _bars(record):
    return [(bar['position'], bar['row'], bar['cells'], bar['sample_sets'])
            for bar in record['colorbars']]


def _drawn(monkeypatch, tmp_path, argv):
    """Plot and return the drawn bars and the resolved state."""
    seen = {}
    original = matrix_figure._draw_cell_colorbars

    def capture(state):
        original(state)
        seen['state'] = state
    monkeypatch.setattr(matrix_figure, '_draw_cell_colorbars', capture)
    plotHeatmap.main([*argv, '-o', str(tmp_path / 'plot.png')])
    state = seen['state']
    titles = {axis.get_gid(): axis.texts[0].get_text()
              for axis in state.figure.axes
              if (axis.get_gid() or '').startswith('colorbar_title/')}
    return state, [
        (start, role, titles.get(role.replace('colorbar/',
                                              'colorbar_title/', 1)))
        for start, _, _, _, role in state.colorbar_records]


def test_default_plot_has_one_side_bar(plain):
    record = describe('plotHeatmapR', ['-m', plain])
    assert _bars(record) == [('side', 1, [1, 2, 3], [1, 2, 3])]
    assert record['colorbars'][0]['scale'] == {
        'color_map': 'RdYlBu', 'color_list': None, 'z_min': 'auto',
        'z_max': 'auto', 'z_mid': None}
    assert record['colorbars'][0]['title'] == {
        'auto': None, 'value': None, 'auto_text': 'Sample A,\nSample B'}


def test_no_heatmap_has_no_bars(plain):
    record = describe('plotMatrixR', ['-m', plain, '--profile'])
    assert record['colorbars'] == []


@pytest.mark.parametrize('location, expected', [
    ('right', [('side', 1, [1, 2, 3], [1, 2, 3])]),
    ('right_common', [('side_common', None, [1, 2, 3], [1, 2, 3])]),
    ('below_common', [('below_common', None, [1, 2, 3], [1, 2, 3])]),
    ('below', [('below', 1, [1], [1]), ('below', 1, [2], [2]),
               ('below', 1, [3], [3])]),
    ('best', [('side', 1, [1, 2, 3], [1, 2, 3])])])
def test_locations(plain, location, expected):
    record = describe('plotHeatmapR', ['-m', plain,
                                       '--colorbarLocation', location])
    assert _bars(record) == expected


def test_best_with_several_scales_shares_a_bar_per_sample_set(plain):
    record = describe('plotMatrixR', [
        '-m', plain, '--heatmap', '--perGroup', '--arrangeSamples', '1,2',
        '3', '--colorMap', 'Reds', 'Blues'])
    assert [(bar['cells'], bar['sample_sets'], bar['scale']['color_map'])
            for bar in record['colorbars']] == [
        ([1, 2], [1], 'Reds'), ([3, 4], [2], 'Blues')]


def test_per_group_and_groups_as_sample_sets(plain):
    assert _bars(describe('plotHeatmapR', ['-m', plain, '--perGroup'])) == [
        ('side', 1, [1, 2], [1])]
    record = describe('plotHeatmapR', [
        '-m', plain, '--perGroup', '--groupsAsSampleSets',
        '--colorMap', 'Reds', 'Blues'])
    assert [(bar['position'], bar['cells'], bar['sample_sets'],
             bar['scale']['color_map']) for bar in record['colorbars']] == [
        ('below', [1], [1], 'Reds'), ('below', [2], [2], 'Blues')]


def test_per_set_colour_map_gives_distinct_bars(plain):
    record = describe('plotHeatmapR', ['-m', plain, '--colorMap', '2=Blues',
                                       '--colorbarLocation', 'right'])
    assert [(bar['cells'], bar['scale']['color_map'])
            for bar in record['colorbars']] == [
        ([1, 3], 'RdYlBu'), ([2], 'Blues')]


def test_colour_list_and_given_limits(plain):
    record = describe('plotHeatmapR', [
        '-m', plain, '--colorList', '2=white, red', '--zMin', '0', 'auto',
        '--zMid', '2=1.5', '--colorbarLocation', 'right_common'])
    assert [bar['scale'] for bar in record['colorbars']] == [
        {'color_map': 'RdYlBu', 'color_list': None, 'z_min': 0.0,
         'z_max': 'auto', 'z_mid': None},
        {'color_map': None, 'color_list': ['white', 'red'],
         'z_min': 'auto', 'z_max': 'auto', 'z_mid': 1.5}]
    assert [bar['cells'] for bar in record['colorbars']] == [[1, 3], [2]]


def test_automatic_limit_equal_to_a_typed_one_keeps_two_bars(
        plain, monkeypatch, tmp_path):
    argv = ['-m', plain, '--colorbarLocation', 'right']
    state, _ = _drawn(monkeypatch, tmp_path, argv)
    automatic = state.prepared.z_max[0]
    argv += ['--zMax', '1=auto', f'2={automatic!r}', '3=auto']
    record = describe('plotHeatmapR', argv)
    assert [(bar['cells'], bar['scale']['z_max'])
            for bar in record['colorbars']] == [
        ([1, 3], 'auto'), ([2], automatic)]
    state, drawn = _drawn(monkeypatch, tmp_path, argv)
    assert state.prepared.z_max[0] == state.prepared.z_max[1]
    assert [start for start, _, _ in drawn] == [0, 1]


def test_colorbar_labels_number_the_bars(plain):
    record = describe('plotHeatmapR', [
        '-m', plain, '--colorMap', 'Reds', 'Blues', 'Greens',
        '--colorbarLabels', '2=Second', 'auto'])
    assert [bar['index'] for bar in record['colorbars']] == [1, 2, 3]
    assert [bar['title']['value'] for bar in record['colorbars']] == [
        'Sample A', 'Second', 'Sample B']
    assert [bar['title']['auto_text'] for bar in record['colorbars']] == [
        'Sample A', 'Sample A', 'Sample B']


def test_out_of_range_colorbar_label_is_a_problem(plain):
    record = describe('plotHeatmapR', ['-m', plain,
                                       '--colorbarLabels', '2=x', 'Kept'])
    assert [(item['code'], item['option'], item['target'], item['slot'])
            for item in record['problems']] == [
        ('assignment_out_of_range', '--colorbarLabels', 'colorbar', 2)]
    assert record['colorbars'][0]['title']['value'] == 'Kept'


def test_plot_raises_the_out_of_range_colorbar_label(plain, tmp_path,
                                                     capsys):
    with pytest.raises(SystemExit) as error:
        plotHeatmap.main(['-m', plain, '--colorbarLabels', '2=x',
                          '-o', str(tmp_path / 'plot.png')])
    assert error.value.code == 2
    assert ('--colorbarLabels: 2 names colorbar 2, but there are only 1 '
            'colorbars') in capsys.readouterr().err


@pytest.mark.parametrize('extra', [
    [],
    ['--colorbarLocation', 'below', '--colorbarLabels', 'auto'],
    ['--colorbarLocation', 'below_common', '--colorMap', '3=Blues',
     '--colorbarLabels', 'auto'],
    ['--colorbarLocation', 'right_common', '--zMax', '2=5'],
    ['--colorMap', 'Reds', 'Blues', '--colorbarLabels', 'One', 'auto'],
    ['--perGroup', '--colorbarLocation', 'below', '--colorbarLabels', 'P'],
    ['--perGroup', '--groupsAsSampleSets', '--colorMap', 'Reds', 'Blues'],
    ['--xAxisVisibility', 'outer_merged', '--colorbarLocation', 'below']])
def test_plot_and_describe_agree(plain, monkeypatch, tmp_path, extra):
    argv = ['-m', plain, *extra]
    record = describe('plotHeatmapR', argv)
    state, drawn = _drawn(monkeypatch, tmp_path, argv)
    assert state.colorbar_position == record['colorbars'][0]['position']
    assert [(start + 1, title) for start, _, title in drawn] == [
        (bar['cells'][0], bar['title']['value'])
        for bar in record['colorbars']]
