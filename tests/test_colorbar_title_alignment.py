"""Colorbar titles: left-aligned beside right bars, centred over bars below."""

import json

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io, plotHeatmap, plotMatrix

ROWS, BINS, GROUP_ROWS = 40, 10, 20


@pytest.fixture
def matrix(tmp_path):
    path = tmp_path / 'matrix.gz'
    header = {
        'sample_labels': ['Sample A', 'Sample B'],
        'sample_boundaries': [0, BINS, 2 * BINS],
        'group_labels': ['GENE1', 'GENE2'],
        'group_boundaries': [0, GROUP_ROWS, ROWS],
        'bin size': [10, 10], 'upstream': [50, 50], 'body': [0, 0],
        'downstream': [50, 50], 'unscaled 5 prime': [0, 0],
        'unscaled 3 prime': [0, 0], 'ref point': ['TSS', 'TSS'],
        'min threshold': None, 'max threshold': None, 'sort regions': 'no',
        'sort using': 'mean', 'proc number': 1}
    regions = [['chr1', [(i * 100, i * 100 + 50)], f'region{i}', i, '+', '0']
               for i in range(ROWS)]
    values = np.random.default_rng(5).random((ROWS, 2 * BINS)).astype(
        np.float32)
    values[:, BINS:] *= 100
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header), regions, values, 1)
    return str(path)


def _titles(monkeypatch, tmp_path, matrix, *extra):
    """(title text, its bar's left edge in pixels) for every bar title."""
    seen = {}
    original = plotMatrix.build_matrix_figure

    def capture(*args, **kwargs):
        seen['figure'], solution = original(*args, **kwargs)
        return seen['figure'], solution
    monkeypatch.setattr(plotMatrix, 'build_matrix_figure', capture)
    plotHeatmap.main(['-m', matrix, '--sortRegions', 'no', '--dpi', '50',
                      *extra, '-o', str(tmp_path / 'plot.png')])
    figure = seen['figure']
    renderer = figure.canvas.get_renderer()
    axes = {axis.get_gid(): axis for axis in figure.axes if axis.get_gid()}
    found = []
    for gid, axis in axes.items():
        if gid.startswith('colorbar_title/'):
            bar = axes[gid.replace('colorbar_title/', 'colorbar/', 1)]
            text, = axis.texts
            found.append((text, text.get_window_extent(renderer),
                          bar.get_window_extent(renderer)))
    assert found
    return found


def test_right_bar_titles_are_left_aligned(matrix, tmp_path, monkeypatch):
    titles = _titles(monkeypatch, tmp_path, matrix,
                     '--zAxisLimits', 'per_heatmap',
                     '--colorbarLocation', 'right')
    for text, extent, bar in titles:
        # Unset multialignment follows ha, so every line starts there too.
        assert text.get_horizontalalignment() == 'left'
        assert extent.x0 == pytest.approx(bar.x0, abs=1.0)


def test_below_bar_titles_stay_centred(matrix, tmp_path, monkeypatch):
    titles = _titles(monkeypatch, tmp_path, matrix,
                     '--zAxisLimits', 'per_sample_set',
                     '--colorbarLocation', 'below',
                     '--colorbarLabels', 'First\nlonger line', 'Second')
    for text, extent, bar in titles:
        assert text.get_horizontalalignment() == 'center'
        assert (extent.x0 + extent.x1) / 2 == pytest.approx(
            (bar.x0 + bar.x1) / 2, abs=1.0)
