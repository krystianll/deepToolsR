"""Heatmap rows are read from the bottom within each group.

Owner ruling: on plotHeatmapR and plotMatrixR heatmaps the sorted order is
drawn bottom-up inside each group (ascend puts the biggest value at the top
of the group), groups keep their top-to-bottom order, keep/no draw the input
order top-down, and written files keep the sort order.

The tests read what is drawn: each heatmap block's image is decoded back to
row values through a grey colour map, top row first.
"""

import matplotlib
import numpy as np
import pytest

from deeptoolsr import plotMatrix
from deeptoolsr.matrix import OwnedMatrix

matplotlib.use('Agg')

Z_MAX = 255.0
UPSTREAM = 10


def _write(path, groups):
    """One sample of reference-point rows; region value = 10 * length.

    ``groups`` maps a group label to its row values in input order.
    """
    values = [value for rows in groups.values() for value in rows]
    columns = 2 * UPSTREAM
    data = np.repeat(np.array(values, dtype=np.float32)[:, None], columns, 1)
    regions = [['chr1', [(i * 100, i * 100 + (int(value // 10)
                                              if value == value else 1))],
                f'GENE{int(value)}' if value == value else 'GENE0',
                0, '+', '0']
               for i, value in enumerate(values)]
    boundaries = [0]
    for rows in groups.values():
        boundaries.append(boundaries[-1] + len(rows))
    parameters = {
        'upstream': [UPSTREAM], 'downstream': [UPSTREAM], 'body': [0],
        'unscaled 5 prime': [0], 'unscaled 3 prime': [0],
        'ref point': ['TSS'], 'bin size': [1],
        'sort regions': 'keep', 'sort using': 'mean',
        'sample_labels': ['Sample A'], 'group_labels': list(groups),
        'sample_boundaries': [0, columns], 'group_boundaries': boundaries,
        'min threshold': None, 'max threshold': None,
    }
    OwnedMatrix.from_compute(parameters, data, regions).save(
        str(path), compressed=True, threads=1)
    return values


def _draw(tmp_path, monkeypatch, groups, *flags, tool='plotHeatmapR',
          source=None):
    """Plot ``groups``, or the saved matrix ``source`` holding those rows."""
    if source is None:
        source = tmp_path / 'input.gz'
        values = _write(source, groups)
    else:
        values = [value for rows in groups.values() for value in rows]
    built = {}
    original = plotMatrix.build_matrix_figure

    def capture(*args, **kwargs):
        figure, solution = original(*args, **kwargs)
        built['figure'] = figure
        return figure, solution

    argv = ['-m', str(source), '-o',
            str(tmp_path / 'plot.png'), '-p', '1', '--dpi', '100',
            '--colorMap', 'gray', '--zMin', '0', '--zMax', str(Z_MAX),
            '--interpolationMethod', 'nearest', *flags]
    if tool == 'plotMatrixR':
        argv.append('--heatmap')
        main = plotMatrix.main
    else:
        from deeptoolsr.plotHeatmap import main
    with monkeypatch.context() as patch:
        patch.setattr(plotMatrix, 'build_matrix_figure', capture)
        main(argv)
    return built['figure'], values


def _blocks(figure):
    """Block axes of the single heatmap column, top group first."""
    axes = [axis for axis in figure.axes
            if (axis.get_gid() or '').split('/')[-2:-1] == ['block']]
    return sorted(axes, key=lambda axis: -axis.get_position().y1)


def _drawn_rows(axis, values):
    """Decode the block's pixel rows (top first) to the known row values.

    A pixel that is not grey is the --missingDataColor and reads 'NaN'.
    """
    image = axis.get_images()[0]
    pixels = np.asarray(image.get_array())[:, 0, :3].astype(float)
    known = np.array(sorted(value for value in set(values) if value == value))
    decoded = ['NaN' if pixel[0] != pixel[1] else
               float(known[np.argmin(np.abs(known - pixel[0] / 255 * Z_MAX))])
               for pixel in pixels]
    rows = [decoded[0]]
    for value in decoded[1:]:
        if value != rows[-1]:
            rows.append(value)
    return rows


def _drawn(figure, values):
    return [_drawn_rows(axis, values) for axis in _blocks(figure)]


ONE_GROUP = {'GENE group': [30.0, 10.0, 40.0, 20.0]}
QUANTILE_INPUT = {'GENE group': [50.0, 20.0, 80.0, 10.0, 60.0, 40.0,
                                 70.0, 30.0]}
KMEANS_INPUT = {'GENE group': [220.0, 10.0, 200.0, 30.0, 230.0, 20.0,
                               210.0, 40.0]}
WITH_NAN = {'GENE group': [30.0, float('nan'), 10.0, 20.0]}
TWO_GROUPS = {'First': [20.0, 40.0, 10.0, 30.0],
              'Second': [70.0, 50.0, 80.0, 60.0]}


@pytest.mark.parametrize('tool', ['plotHeatmapR', 'plotMatrixR'])
@pytest.mark.parametrize('groups,flags,expected', [
    pytest.param(ONE_GROUP, ['--sortRegions', 'ascend'],
                 [[40.0, 30.0, 20.0, 10.0]], id='ascend_biggest_on_top'),
    pytest.param(ONE_GROUP, ['--sortRegions', 'descend'],
                 [[10.0, 20.0, 30.0, 40.0]], id='descend_smallest_on_top'),
    pytest.param(QUANTILE_INPUT, ['--sortRegions', 'ascend',
                                  '--quantileSortedRegions', '2'],
                 [[40.0, 30.0, 20.0, 10.0], [80.0, 70.0, 60.0, 50.0]],
                 id='quantiles_flipped_in_order'),
    pytest.param(TWO_GROUPS, ['--sortRegions', 'ascend'],
                 [[40.0, 30.0, 20.0, 10.0], [80.0, 70.0, 60.0, 50.0]],
                 id='region_groups_flipped_in_order'),
    pytest.param(KMEANS_INPUT, ['--sortRegions', 'ascend', '--kmeans', '2'],
                 [[230.0, 220.0, 210.0, 200.0], [40.0, 30.0, 20.0, 10.0]],
                 id='kmeans_clusters_flipped_in_order'),
    pytest.param(WITH_NAN, ['--sortRegions', 'ascend',
                            '--missingDataColor', 'red'],
                 [[30.0, 20.0, 10.0, 'NaN']], id='ascend_nan_row_at_bottom'),
    pytest.param(WITH_NAN, ['--sortRegions', 'descend',
                            '--missingDataColor', 'red'],
                 [[10.0, 20.0, 30.0, 'NaN']], id='descend_nan_row_at_bottom'),
    pytest.param(ONE_GROUP, ['--sortRegions', 'keep'],
                 [[30.0, 10.0, 40.0, 20.0]], id='keep_input_order'),
])
def test_heatmap_rows_are_drawn_bottom_up_within_each_group(
        tmp_path, monkeypatch, tool, groups, flags, expected):
    figure, values = _draw(tmp_path, monkeypatch, groups, *flags, tool=tool)
    assert _drawn(figure, values) == expected


def _write_profile(tmp_path, groups, tool, *flags):
    """Run a profile-only command (no heatmap) on ``groups``."""
    _write(tmp_path / 'input.gz', groups)
    argv = ['-m', str(tmp_path / 'input.gz'), '-o',
            str(tmp_path / 'plot.png'), '-p', '1', '--dpi', '100', *flags]
    if tool == 'plotMatrixR':
        plotMatrix.main(argv + ['--profile'])
    else:
        from deeptoolsr.plotProfile import main
        main(argv)


@pytest.mark.parametrize('option', ['--outFileSortedRegions',
                                    '--outFileNameMatrix'])
@pytest.mark.parametrize('direction', ['ascend', 'descend'])
@pytest.mark.parametrize('clustering,groups,order', [
    pytest.param((), ONE_GROUP, [[10.0, 20.0, 30.0, 40.0]], id='no_clusters'),
    pytest.param(('--kmeans', '2'), KMEANS_INPUT,
                 [[200.0, 210.0, 220.0, 230.0], [10.0, 20.0, 30.0, 40.0]],
                 id='kmeans_sorted_within_clusters'),
    pytest.param(('--hclust', '2'), KMEANS_INPUT,
                 [[200.0, 210.0, 220.0, 230.0], [10.0, 20.0, 30.0, 40.0]],
                 id='hclust_sorted_within_clusters'),
])
@pytest.mark.parametrize('tool', ['plotHeatmapR', 'plotProfileR',
                                  'plotMatrixR'])
def test_written_files_keep_the_sort_order(tmp_path, monkeypatch, direction,
                                           option, clustering, groups, order,
                                           tool):
    # A profile-only command writes the same sorted rows as a heatmap.
    written = tmp_path / 'written.tab'
    flags = ('--sortRegions', direction, *clustering, option, str(written))
    if tool == 'plotHeatmapR':
        _draw(tmp_path, monkeypatch, groups, *flags)
    else:
        _write_profile(tmp_path, groups, tool, *flags)
    rows = [line.split('\t') for line in written.read_text().splitlines()
            if not line.startswith(('#', '@'))]
    if option == '--outFileSortedRegions':
        drawn = [float(row[3][len('GENE'):]) for row in rows]
    else:
        # The matrix: six BED columns, then the row's bin values.
        drawn = [float(row[6]) for row in rows]
    assert drawn == [value for cluster in order for value in
                     (cluster if direction == 'ascend' else cluster[::-1])]


def test_region_length_line_follows_the_drawn_rows(tmp_path, monkeypatch):
    figure, values = _draw(tmp_path, monkeypatch, ONE_GROUP,
                           '--sortRegions', 'ascend',
                           '--sortUsing', 'region_length')
    (axis,) = _blocks(figure)
    drawn = _drawn_rows(axis, values)
    assert drawn == [40.0, 30.0, 20.0, 10.0]
    (line,) = [line for line in axis.get_lines()
               if line.get_linestyle() == '--']
    # TSS reference: the dashed line marks upstream + region length (bins)
    # on each row; region length = value / 10.
    for x, y in line.get_xydata():
        assert x == UPSTREAM + drawn[int(y)] / 10


@pytest.mark.parametrize('direction,wide_end', [
    ('ascend', 1.0), ('descend', 0.0)])
def test_sort_wedge_is_wide_at_the_biggest_values(
        tmp_path, monkeypatch, direction, wide_end):
    figure, _ = _draw(tmp_path, monkeypatch, ONE_GROUP,
                      '--sortRegions', direction)
    (indicator,) = [axis for axis in figure.axes
                    if (axis.get_gid() or '').startswith('indicator/')]
    (wedge,) = indicator.patches
    vertices = wedge.get_xy()
    widths = {y: np.ptp(vertices[vertices[:, 1] == y][:, 0])
              for y in (0.0, 1.0)}
    assert max(widths, key=widths.get) == wide_end


def _wedges(figure):
    return [patch for axis in figure.axes
            if (axis.get_gid() or '').startswith('indicator/')
            for patch in axis.patches]


@pytest.mark.parametrize('tool', ['plotHeatmapR', 'plotMatrixR'])
@pytest.mark.parametrize('using', ['mean', 'region_length'])
@pytest.mark.parametrize('replot,drawn,wedge', [
    pytest.param('keep', [10.0, 20.0, 30.0, 40.0], False,
                 id='keep_file_order_no_wedge'),
    pytest.param('no', [10.0, 20.0, 30.0, 40.0], False,
                 id='no_file_order_no_wedge'),
    pytest.param('ascend', [40.0, 30.0, 20.0, 10.0], True,
                 id='ascend_resorts_with_wedge'),
    pytest.param('descend', [10.0, 20.0, 30.0, 40.0], True,
                 id='descend_resorts_with_wedge'),
])
def test_replotted_sorted_matrix_wedge_shows_only_this_plots_sort(
        tmp_path, monkeypatch, tool, using, replot, drawn, wedge):
    # Owner ruling: the wedge and the region_length end line show only a
    # sort this plot made. A saved ascending matrix (file rows 10, 20, 30,
    # 40) re-plotted with keep/no draws its file order top-down, no wedge
    # and no end line.
    saved = tmp_path / 'sorted.gz'
    _draw(tmp_path, monkeypatch, ONE_GROUP, '--sortRegions', 'ascend',
          '--sortUsing', using, '--outFileNameMatrix', str(saved))
    figure, values = _draw(tmp_path, monkeypatch, ONE_GROUP,
                           '--sortRegions', replot, '--sortUsing', using,
                           tool=tool, source=saved)
    (axis,) = _blocks(figure)
    assert _drawn_rows(axis, values) == drawn
    assert bool(_wedges(figure)) == wedge
    lines = [line for line in axis.get_lines()
             if line.get_linestyle() == '--']
    if using == 'region_length' and wedge:
        # The dashed region-end line follows the drawn rows.
        (line,) = lines
        for x, y in line.get_xydata():
            assert x == UPSTREAM + drawn[int(y)] / 10
    else:
        assert lines == []
