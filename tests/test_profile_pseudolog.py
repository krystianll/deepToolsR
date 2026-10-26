"""--yScale pseudolog: a ggplot-style pseudo-log Y axis on profile panels."""

import json

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io, plotMatrix
from deeptoolsr.describe import describe

BINS = 20


def _write(path, ranges, groups=1, rows=4):
    """One sample per (low, high) range, each row a ramp from low to high."""
    values = np.hstack([np.tile(np.linspace(low, high, BINS), (rows, 1))
                        for low, high in ranges]).astype(np.float32)
    values = np.vstack([values] * groups)
    count = len(ranges)
    regions = [['chr1', [(index * 5000, index * 5000 + 10)],
                f'region{index}', index, '+', '0']
               for index in range(rows * groups)]
    header = {
        'sample_labels': [f'Sample {chr(65 + index)}'
                          for index in range(count)],
        'sample_boundaries': [index * BINS for index in range(count + 1)],
        'group_labels': [f'GENE{index + 1}' for index in range(groups)],
        'group_boundaries': [index * rows for index in range(groups + 1)],
        'bin size': [100] * count, 'upstream': [1000] * count,
        'body': [0] * count, 'downstream': [1000] * count,
        'unscaled 5 prime': [0] * count, 'unscaled 3 prime': [0] * count,
        'ref point': ['TSS'] * count, 'min threshold': None,
        'max threshold': None, 'sort regions': 'no', 'sort using': 'mean',
        'proc number': 1}
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header), regions, values, 1)
    return str(path)


def _render(tmp_path, monkeypatch, tool, matrix, *flags, name='plot.png'):
    """Run ``tool`` and return its profile axes in drawing order."""
    built = {}
    original = plotMatrix.build_matrix_figure

    def capture(*args, **kwargs):
        figure, solution = original(*args, **kwargs)
        built['figure'] = figure
        return figure, solution

    argv = ['-m', matrix, '-o', str(tmp_path / name), '-p', '1',
            '--dpi', '50', *flags]
    if tool == 'plotMatrixR':
        argv.append('--profile')
    with monkeypatch.context() as patch:
        patch.setattr(plotMatrix, 'build_matrix_figure', capture)
        plotMatrix.matrix_main(tool)(argv)
    figure = built['figure']
    figure.canvas.draw()
    profiles = [axis for axis in figure.axes
                if (axis.get_gid() or '').endswith('/profile')]
    return sorted(profiles, key=lambda axis: axis.get_gid())


def _labelled_ticks(axis):
    """The visible Y tick values inside the view and their label texts."""
    low, high = axis.get_ylim()
    return [(tick.get_loc(), tick.label1.get_text())
            for tick in axis.yaxis.get_major_ticks()
            if low <= tick.get_loc() <= high and tick.label1.get_visible()
            and tick.label1.get_text()]


def _number(text):
    return float(text.replace('\N{MINUS SIGN}', '-'))


@pytest.mark.parametrize('tool', ['plotProfileR', 'plotMatrixR'])
def test_pseudolog_keeps_sign_and_labels_ticks_in_data_units(
        tmp_path, monkeypatch, tool):
    matrix = _write(tmp_path / 'm.gz', [(-3, 1000)])
    (axis,) = _render(tmp_path, monkeypatch, tool, matrix,
                      '--yScale', 'pseudolog')
    assert axis.get_yscale() == 'asinh'
    to_screen = axis.transData.transform
    negative, zero, positive = (to_screen([(0, value)])[0, 1]
                                for value in (-1, 0, 1))
    assert negative < zero < positive
    ticks = _labelled_ticks(axis)
    assert [_number(text) for _, text in ticks] == pytest.approx(
        [loc for loc, _ in ticks])
    assert any(loc < 0 for loc, _ in ticks) or axis.get_ylim()[0] < 0
    # Logarithmic beyond sigma: equal screen steps per decade.
    steps = np.diff(to_screen([(0, 10), (0, 100), (0, 1000)])[:, 1])
    assert steps[0] == pytest.approx(steps[1], rel=0.01)


def test_narrow_ranges_show_at_least_two_labelled_ticks(tmp_path, monkeypatch):
    ranges = [(2, 8), (0, 0.5), (-3, 1000), (0.9, 1.3)]
    matrix = _write(tmp_path / 'm.gz', ranges)
    axes = _render(tmp_path, monkeypatch, 'plotProfileR', matrix,
                   '--yScale', 'pseudolog', '--yAxisLimits', 'per_panel')
    for axis in axes:
        ticks = _labelled_ticks(axis)
        assert len(ticks) >= 2
        assert [_number(text) for _, text in ticks] == pytest.approx(
            [loc for loc, _ in ticks])


def test_explicit_limits_stay_in_data_units(tmp_path, monkeypatch):
    matrix = _write(tmp_path / 'm.gz', [(-3, 1000)])
    (axis,) = _render(tmp_path, monkeypatch, 'plotProfileR', matrix,
                      '--yScale', 'pseudolog', '--yMin', '-2',
                      '--yMax', '500')
    assert axis.get_ylim() == pytest.approx((-2, 500))


def test_assignment_and_recycling_pick_sample_sets(tmp_path, monkeypatch):
    matrix = _write(tmp_path / 'm.gz', [(0, 100)] * 3)
    assigned = _render(tmp_path, monkeypatch, 'plotProfileR', matrix,
                       '--yScale', '2=pseudolog')
    assert [axis.get_yscale() for axis in assigned] == [
        'linear', 'asinh', 'linear']
    recycled = _render(tmp_path, monkeypatch, 'plotProfileR', matrix,
                       '--yScale', 'pseudolog', 'linear')
    assert [axis.get_yscale() for axis in recycled] == [
        'asinh', 'linear', 'asinh']


def test_group_panels_count_as_sample_sets(tmp_path, monkeypatch):
    matrix = _write(tmp_path / 'm.gz', [(0, 100)], groups=2)
    axes = _render(tmp_path, monkeypatch, 'plotProfileR', matrix,
                   '--perGroup', '--groupsAsSampleSets',
                   '--yScale', '2=pseudolog')
    assert [axis.get_yscale() for axis in axes] == ['linear', 'asinh']


def test_assignment_to_a_missing_sample_set_is_rejected(tmp_path, capsys):
    matrix = _write(tmp_path / 'm.gz', [(0, 100)] * 2)
    with pytest.raises(SystemExit):
        plotMatrix.matrix_main('plotProfileR')(
            ['-m', matrix, '-o', str(tmp_path / 'plot.png'),
             '--yScale', '3=pseudolog'])
    assert '--yScale: 3 names sample set 3' in capsys.readouterr().err


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_sigma_must_be_finite_and_positive(tmp_path, capsys, value):
    matrix = _write(tmp_path / 'm.gz', [(0, 100)])
    with pytest.raises(SystemExit):
        plotMatrix.matrix_main('plotProfileR')(
            ['-m', matrix, '-o', str(tmp_path / 'plot.png'),
             '--yScale', 'pseudolog', '--yPseudologSigma', value])
    assert '--yPseudologSigma' in capsys.readouterr().err


def test_sigma_sets_the_linear_zone(tmp_path, monkeypatch):
    matrix = _write(tmp_path / 'm.gz', [(0, 1000)])
    (axis,) = _render(tmp_path, monkeypatch, 'plotProfileR', matrix,
                      '--yScale', 'pseudolog', '--yPseudologSigma', '5')
    assert axis.yaxis.get_transform().linear_width == 10


def test_sigma_without_a_pseudolog_panel_warns(tmp_path, monkeypatch, capsys):
    matrix = _write(tmp_path / 'm.gz', [(0, 100)])
    _render(tmp_path, monkeypatch, 'plotProfileR', matrix,
            '--yPseudologSigma', '2')
    assert ('--yPseudologSigma is unused unless --yScale pseudolog'
            in capsys.readouterr().err)
    _render(tmp_path, monkeypatch, 'plotProfileR', matrix,
            '--yScale', 'pseudolog', '--yPseudologSigma', '2')
    assert 'unused' not in capsys.readouterr().err


def test_series_heatmap_reports_y_scale_unused(tmp_path, capsys):
    matrix = _write(tmp_path / 'm.gz', [(0, 100)])
    plotMatrix.matrix_main('plotProfileR')(
        ['-m', matrix, '-o', str(tmp_path / 'plot.png'),
         '--plotType', 'heatmap', '--yScale', 'pseudolog'])
    assert ('--yScale is unused with --plotType heatmap'
            in capsys.readouterr().err)


def _tick_labels_shown(axis):
    return any(text for _, text in _labelled_ticks(axis))


@pytest.mark.parametrize('scales, second_shows_ticks', [
    (['pseudolog'], False), (['2=pseudolog'], True)])
def test_outer_merged_merges_ticks_only_on_equal_scales(
        tmp_path, monkeypatch, scales, second_shows_ticks):
    matrix = _write(tmp_path / 'm.gz', [(0, 100)] * 2)
    first, second = _render(
        tmp_path, monkeypatch, 'plotProfileR', matrix,
        '--yAxisVisibility', 'outer_merged', '--yAxisLimits', 'common',
        '--yAxisLabel', 'signal', '--yMin', '0', '--yMax', '100',
        '--yScale', *scales)
    assert first.get_ylim() == second.get_ylim()
    assert _tick_labels_shown(first)
    assert _tick_labels_shown(second) is second_shows_ticks


def test_linear_default_draws_like_no_option(tmp_path, monkeypatch):
    matrix = _write(tmp_path / 'm.gz', [(-3, 100), (0, 5)])
    common = ('--outFileNameData',)
    _render(tmp_path, monkeypatch, 'plotProfileR', matrix,
            *common, str(tmp_path / 'plain.tsv'), name='plain.png')
    _render(tmp_path, monkeypatch, 'plotProfileR', matrix,
            *common, str(tmp_path / 'linear.tsv'), '--yScale', 'linear',
            name='linear.png')
    assert ((tmp_path / 'plain.png').read_bytes() ==
            (tmp_path / 'linear.png').read_bytes())
    assert ((tmp_path / 'plain.tsv').read_bytes() ==
            (tmp_path / 'linear.tsv').read_bytes())


def test_profile_table_is_unchanged_by_the_scale(tmp_path, monkeypatch):
    matrix = _write(tmp_path / 'm.gz', [(-3, 100)])
    scales = {'linear': (), 'pseudolog': ('--yScale', 'pseudolog')}
    for name, flags in scales.items():
        _render(tmp_path, monkeypatch, 'plotProfileR', matrix,
                '--outFileNameData', str(tmp_path / f'{name}.tsv'), *flags,
                name=f'{name}.png')
    assert ((tmp_path / 'linear.tsv').read_bytes() ==
            (tmp_path / 'pseudolog.tsv').read_bytes())


def test_heatmap_summary_row_honours_its_scale(tmp_path, monkeypatch, capsys):
    matrix = _write(tmp_path / 'm.gz', [(0, 100)] * 2)
    axes = _render(tmp_path, monkeypatch, 'plotHeatmapR', matrix,
                   '--yScaleSummaryPlot', '1=pseudolog',
                   '--yPseudologSigmaSummaryPlot', '3')
    assert [axis.get_yscale() for axis in axes] == ['asinh', 'linear']
    assert axes[0].yaxis.get_transform().linear_width == 6
    assert 'unused' not in capsys.readouterr().err


def test_describe_reports_each_sample_set_scale(tmp_path):
    matrix = _write(tmp_path / 'm.gz', [(0, 100)] * 2)
    record = describe('plotProfileR', [
        '-m', matrix, '--yScale', '2=pseudolog', '--yPseudologSigma', '4'])
    assert [item['y_scale'] for item in record['sample_sets']] == [
        {'scale': 'linear', 'sigma': None},
        {'scale': 'pseudolog', 'sigma': 4.0}]
    option = next(item for item in record['options']
                  if item['option'] == '--yScale')
    assert option['counts'] == 'sample set' and option['recycles']


@pytest.mark.parametrize('limits', ['common', 'per_y_label'])
def test_mixed_scale_group_shares_the_raw_data_range(
        tmp_path, monkeypatch, limits):
    matrix = _write(tmp_path / 'm.gz', [(0, 100), (0, 50)])
    linear, pseudolog = _render(
        tmp_path, monkeypatch, 'plotProfileR', matrix, '--yAxisLimits',
        limits, '--yScale', '2=pseudolog')
    # The linear panel pads the shared 0..100 by Matplotlib's 5 % only.
    assert linear.get_ylim() == pytest.approx((-5, 105))
    low, high = pseudolog.get_ylim()
    # The pseudolog panel pads on its own scale: little room below 0.
    assert -1 < low < 0 and high > 100


def test_mixed_scale_group_keeps_typed_limits(tmp_path, monkeypatch):
    matrix = _write(tmp_path / 'm.gz', [(0, 100), (0, 50)])
    axes = _render(tmp_path, monkeypatch, 'plotProfileR', matrix,
                   '--yAxisLimits', 'common', '--yScale', '2=pseudolog',
                   '--yMin', '-2', '--yMax', '300')
    assert [axis.get_ylim() for axis in axes] == [
        pytest.approx((-2, 300))] * 2
