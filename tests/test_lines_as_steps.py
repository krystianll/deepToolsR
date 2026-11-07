"""--linesAsSteps: one flat segment per bin for profile lines and bands."""

import json

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io, plotMatrix
from deeptoolsr.describe import describe

ROWS, BINS, GAP = 6, 10, 4


@pytest.fixture
def matrix(tmp_path):
    """One sample of 1 bp bins; bin GAP is NaN in every row."""
    header = {
        'sample_labels': ['Sample A'], 'sample_boundaries': [0, BINS],
        'group_labels': ['GENE1', 'GENE2'],
        'group_boundaries': [0, ROWS // 2, ROWS],
        'bin size': [1], 'upstream': [BINS // 2], 'body': [0],
        'downstream': [BINS // 2], 'unscaled 5 prime': [0],
        'unscaled 3 prime': [0], 'ref point': ['TSS'],
        'min threshold': None, 'max threshold': None, 'sort regions': 'no',
        'sort using': 'mean', 'proc number': 1}
    regions = [['chr1', [(i * 100, i * 100 + 10)], f'region{i}', i, '+', '0']
               for i in range(ROWS)]
    values = np.random.default_rng(3).random((ROWS, BINS)).astype(np.float32)
    values[:, GAP] = np.nan
    path = tmp_path / 'matrix.gz'
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header), regions, values, 1)
    return str(path)


def _profiles(tmp_path, monkeypatch, tool, matrix, *flags, name='plot.png'):
    """Run ``tool`` and return its profile axes in drawing order."""
    built = {}
    original = plotMatrix.build_matrix_figure

    def capture(*args, **kwargs):
        built['figure'], solution = original(*args, **kwargs)
        return built['figure'], solution
    argv = ['-m', matrix, '-o', str(tmp_path / name), '-p', '1',
            '--dpi', '50', *flags]
    with monkeypatch.context() as patch:
        patch.setattr(plotMatrix, 'build_matrix_figure', capture)
        plotMatrix.matrix_main(tool)(argv)
    return [axis for axis in built['figure'].axes
            if (axis.get_gid() or '').endswith('/profile')]


def _bin_values(xs, ys):
    """Check every bin is one flat segment over its edges; return values."""
    assert np.allclose(xs, np.repeat(np.arange(BINS + 1) - .5, 2)[1:-1])
    left, right = np.asarray(ys[::2], float), np.asarray(ys[1::2], float)
    assert np.array_equal(left, right, equal_nan=True)
    return left


def _polygon_xs(axis):
    return np.concatenate([path.vertices[:, 0] for collection
                           in axis.collections
                           for path in collection.get_paths()])


def test_line_steps_over_bin_edges_and_keeps_nan_gap(
        tmp_path, monkeypatch, matrix):
    axis, = _profiles(tmp_path, monkeypatch, 'plotProfileR', matrix,
                      '--linesAsSteps')[:1]
    line = axis.lines[0]
    values = _bin_values(line.get_xdata(), line.get_ydata())
    plain, = _profiles(tmp_path, monkeypatch, 'plotProfileR', matrix)[:1]
    assert np.array_equal(values, np.asarray(plain.lines[0].get_ydata(),
                                             float), equal_nan=True)
    assert np.isnan(values[GAP]) and np.isfinite(np.delete(values, GAP)).all()
    assert (line.get_color(), line.get_alpha(), line.get_linewidth()) == (
        plain.lines[0].get_color(), plain.lines[0].get_alpha(),
        plain.lines[0].get_linewidth())


@pytest.mark.parametrize('plot_type', ['fill', 'se', 'std', 'ci',
                                       'bootstrap'])
def test_bands_follow_the_steps(tmp_path, monkeypatch, matrix, plot_type):
    axis = _profiles(tmp_path, monkeypatch, 'plotProfileR', matrix,
                     '--plotType', plot_type, '--linesAsSteps')[0]
    xs = _polygon_xs(axis)
    assert xs.size
    # Every band corner sits on a bin edge, and none inside the NaN bin.
    assert np.allclose(xs, np.round(xs - .5) + .5)
    assert xs.min() == -.5 and xs.max() == BINS - .5
    assert not ((xs > GAP - .5) & (xs < GAP + .5)).any()


def test_xlim_spans_bin_edges_and_ticks_stay(tmp_path, monkeypatch, matrix):
    flags = ('--linesAtTickMarks', '--minorTickMarks', '1')
    plain = _profiles(tmp_path, monkeypatch, 'plotProfileR', matrix, *flags)
    steps = _profiles(tmp_path, monkeypatch, 'plotProfileR', matrix, *flags,
                      '--linesAsSteps')
    for before, after in zip(plain, steps, strict=True):
        assert before.get_xlim() == (0, BINS - 1)
        assert after.get_xlim() == (-.5, BINS - .5)
        for minor in (False, True):
            assert list(after.get_xticks(minor=minor)) == list(
                before.get_xticks(minor=minor))
        guides = [[tuple(line.get_xdata()) for line in axis.lines
                   if len(line.get_xdata()) == 2] for axis in (before, after)]
        assert guides[0] == guides[1] and guides[0]


def test_series_heatmap_ignores_the_flag_with_a_warning(
        tmp_path, matrix, capsys):
    outputs = []
    for flags in ((), ('--linesAsSteps',)):
        out = tmp_path / f'plot{len(flags)}.png'
        plotMatrix.matrix_main('plotProfileR')(
            ['-m', matrix, '-o', str(out), '-p', '1', '--dpi', '50',
             '--plotType', 'heatmap', *flags])
        outputs.append(out.read_bytes())
        warned = ('--linesAsSteps is unused with --plotType heatmap'
                  in capsys.readouterr().err)
        assert warned is bool(flags)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('tool, flags', [
    ('plotHeatmapR', ('--linesAsStepsSummaryPlot',)),
    ('plotMatrixR', ('--profile', '--heatmap', '--plotType', 'se',
                     '--linesAsSteps')),
])
def test_other_tools_draw_steps(tmp_path, monkeypatch, matrix, tool, flags):
    axes = _profiles(tmp_path, monkeypatch, tool, matrix, *flags)
    assert axes
    for axis in axes:
        assert axis.get_xlim() == (-.5, BINS - .5)
        for line in axis.lines[:1]:
            _bin_values(line.get_xdata(), line.get_ydata())


def test_numbers_do_not_change(tmp_path, monkeypatch, matrix):
    tables = []
    for flags in ((), ('--linesAsSteps',)):
        table = tmp_path / f'data{len(flags)}.tsv'
        _profiles(tmp_path, monkeypatch, 'plotProfileR', matrix,
                  '--plotType', 'ci', '--outFileNameData', str(table), *flags)
        tables.append(table.read_bytes())
    assert tables[0] == tables[1]
    records = [describe('plotProfileR', ['-m', matrix, *flags],
                        force_load=True)
               for flags in ((), ('--linesAsSteps',))]
    assert records[0] == records[1]
