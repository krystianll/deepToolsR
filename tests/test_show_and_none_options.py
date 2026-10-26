"""--showSummaryPlot, --colorbarLocation none and --regionLabelLocation none."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io, plotHeatmap, plotMatrix
from deeptoolsr.describe import describe


@pytest.fixture
def matrix(tmp_path):
    """Two samples and two region groups of two regions each."""
    path = tmp_path / 'matrix.gz'
    values = np.arange(24, dtype=np.float32).reshape(4, 6) / 10 + 1
    regions = [['chr1', [(i * 20, i * 20 + 10)], f'region{i}', i, '+', '0']
               for i in range(4)]
    header = {
        'sample_labels': ['Sample A', 'Sample B'],
        'sample_boundaries': [0, 3, 6],
        'group_labels': ['Group 1', 'Group 2'],
        'group_boundaries': [0, 2, 4],
        'bin size': [1, 1], 'upstream': [1, 1], 'body': [0, 0],
        'downstream': [2, 2], 'unscaled 5 prime': [0, 0],
        'unscaled 3 prime': [0, 0], 'ref point': ['TSS', 'TSS'],
        'min threshold': None, 'max threshold': None, 'sort regions': 'no',
        'sort using': 'mean', 'proc number': 1}
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header), regions, values, 1)
    return str(path)


def _figure(module, matrix, tmp_path, *extra):
    seen = {}

    def save(figure, path, **_kwargs):
        seen['figure'] = figure
        Path(path).write_bytes(b'')
    with patch.object(plotMatrix, 'save_figure_atomic', save):
        module.main(['-m', matrix, '-o', str(tmp_path / 'plot.png'),
                     '-p', '1', *extra])
    return seen['figure']


def _gids(figure, prefix):
    return [axis.get_gid() for axis in figure.axes
            if (axis.get_gid() or '').startswith(prefix)]


def _profiles(figure):
    return [gid for gid in _gids(figure, 'cell/')
            if gid.endswith('/profile')]


def test_summary_plot_off_draws_heatmaps_only(matrix, tmp_path):
    shown = _figure(plotHeatmap, matrix, tmp_path)
    hidden = _figure(plotHeatmap, matrix, tmp_path, '--showSummaryPlot', 'off')
    assert _profiles(shown)
    assert not _profiles(hidden)
    assert _gids(hidden, 'colorbar/')
    heatmap_only = _figure(plotMatrix, matrix, tmp_path, '--heatmap')
    assert _gids(hidden, '') == _gids(heatmap_only, '')
    assert list(hidden.get_size_inches()) == list(
        heatmap_only.get_size_inches())
    record = describe('plotHeatmapR', ['-m', matrix,
                                       '--showSummaryPlot', 'off'])
    assert not [panel for panel in record['panels']
                if panel['kind'] == 'profile']
    assert all(cell['profile'] is None for cell in record['cells'])
    on = describe('plotHeatmapR', ['-m', matrix, '--showSummaryPlot', 'on'])
    assert [panel for panel in on['panels'] if panel['kind'] == 'profile']


def test_what_to_show_is_rejected(capsys):
    with pytest.raises(SystemExit) as stopped:
        plotHeatmap.process_args(['-m', 'unused', '-o', 'unused.png',
                                  '--whatToShow', 'heatmap and colorbar'])
    assert stopped.value.code == 2
    assert '--whatToShow' in capsys.readouterr().err


@pytest.mark.parametrize('module, tool, extra', [
    (plotHeatmap, 'plotHeatmapR', []),
    (plotMatrix, 'plotMatrixR', ['--heatmap', '--profile'])])
def test_colorbar_location_none_draws_no_bars(matrix, tmp_path, module,
                                              tool, extra):
    drawn = _figure(module, matrix, tmp_path, *extra)
    bare = _figure(module, matrix, tmp_path, *extra,
                   '--colorbarLocation', 'none')
    assert _gids(drawn, 'colorbar')
    assert not _gids(bare, 'colorbar')
    # The space the bars took is given back.
    assert bare.get_size_inches()[0] < drawn.get_size_inches()[0]
    record = describe(tool, ['-m', matrix, *extra,
                             '--colorbarLocation', 'none'])
    assert record['colorbars'] == []


def test_colorbar_labels_unused_without_bars(matrix, tmp_path, capsys):
    # The N= target is not checked against bars that are not drawn.
    figure = _figure(plotHeatmap, matrix, tmp_path, '--colorbarLocation',
                     'none', '--colorbarLabels', '5=Scale')
    assert not _gids(figure, 'colorbar')
    assert ('--colorbarLabels is unused with --colorbarLocation none'
            in capsys.readouterr().err)
    record = describe('plotHeatmapR', ['-m', matrix, '--colorbarLocation',
                                       'none', '--colorbarLabels', 'Scale'])
    assert any('--colorbarLabels is unused' in warning
               for warning in record['warnings'])


@pytest.mark.parametrize('module, extra', [
    (plotHeatmap, []), (plotMatrix, ['--heatmap', '--profile'])])
def test_region_label_location_none_hides_group_labels(matrix, tmp_path,
                                                       module, extra):
    shown = _figure(module, matrix, tmp_path, *extra)
    hidden = _figure(module, matrix, tmp_path, *extra,
                     '--regionLabelLocation', 'none')
    assert _gids(shown, 'label/stack/')
    assert not _gids(hidden, 'label/stack/')
    assert hidden.get_size_inches()[0] < shown.get_size_inches()[0]
    # Legend entries and sample titles stay.
    legends = [axis.get_legend() for axis in hidden.axes
               if axis.get_legend() is not None]
    assert legends
    assert {text.get_text() for legend in legends
            for text in legend.get_texts()} == {'Group 1', 'Group 2'}
    titles = [text.get_text() for axis in hidden.axes
              if (axis.get_gid() or '').endswith('/title')
              for text in axis.texts]
    assert titles == ['Sample A', 'Sample B']
