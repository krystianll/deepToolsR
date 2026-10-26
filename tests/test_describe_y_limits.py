"""Describe reports the profile Y limits and pseudocounts the plot uses."""

import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io, _statistics, plotMatrix
from deeptoolsr.describe import describe

# Two region groups of four rows, two samples of two bins; the second group
# is wider, and the second sample's first bin is zero in every row.
VALUES = np.array([[1 + row + col for col in range(4)] for row in range(8)],
                  dtype=np.float32)
VALUES[4:] *= 9
VALUES[:, 2] = 0
GROUPS = [(0, 4), (4, 8)]
SAMPLES = [(0, 2), (2, 4)]


@pytest.fixture
def matrix(tmp_path):
    regions = [['chr1', [(i * 20, i * 20 + 10)], f'region{i}', i, '+', '0']
               for i in range(len(VALUES))]
    header = {
        'sample_labels': ['Sample A', 'Sample B'],
        'sample_boundaries': [0, 2, 4],
        'group_labels': ['Group A', 'Group B'],
        'group_boundaries': [0, 4, 8],
        'bin size': [1, 1], 'upstream': [1, 1], 'body': [0, 0],
        'downstream': [1, 1], 'unscaled 5 prime': [0, 0],
        'unscaled 3 prime': [0, 0], 'ref point': ['TSS', 'TSS'],
        'min threshold': None, 'max threshold': None,
        'sort regions': 'no', 'sort using': 'mean', 'proc number': 1}
    path = os.path.join(str(tmp_path), 'values.gz')
    _compute_matrix_io.write_matrix(
        path, json.dumps(header, separators=(',', ':')), regions, VALUES, 1)
    return path


def _drawn(tool, matrix, tmp_path, options):
    """Each profile panel's drawn Y limits, in drawing order."""
    captured = []

    def save(figure, path, **_kwargs):
        captured.extend(axis.get_ylim() for axis in figure.axes
                        if (axis.get_gid() or '').endswith('/profile'))
        with open(path, 'wb') as handle:
            handle.write(b'')

    with patch.object(plotMatrix, 'save_figure_atomic', save):
        plotMatrix.matrix_main(tool)(
            ['-m', matrix, '-o', os.path.join(str(tmp_path), 'out.png'),
             *options])
    return [tuple(float(value) for value in limits) for limits in captured]


def _described(tool, matrix, options, **kwargs):
    record = describe(tool, ['-m', matrix, *options], **kwargs)
    return record, {entry['panel']: entry for group in record['y_limits']
                    for entry in group['per_panel']}


def _check_drawn(tool, matrix, tmp_path, options):
    """Describe's per-panel ``value`` is what the plot draws."""
    record, panels = _described(tool, matrix, options, force_load=True)
    order = [cell['profile'] for cell in record['cells']]
    assert [tuple(panels[panel]['value']) for panel in order] == _drawn(
        tool, matrix, tmp_path, options)
    return record


TWO_SCALES = ['--arrangeSamples', '1', '2', '--yScale', 'linear', 'pseudolog']


@pytest.mark.parametrize('mode', ['common', 'per_panel', 'per_sample_set',
                                  'per_y_label'])
@pytest.mark.parametrize('scales', [['linear'], ['pseudolog'],
                                    ['linear', 'pseudolog']])
def test_value_is_the_drawn_limit_for_every_grouping(
        matrix, tmp_path, mode, scales):
    record = _check_drawn('plotProfileR', matrix, tmp_path, [
        '--arrangeSamples', '1', '2', '--perGroup', '--yScale', *scales,
        '--yAxisLimits', mode])
    for group in record['y_limits']:
        panel_scales = {record['sample_sets'][entry['sample_set'] - 1]
                        ['y_scale']['scale'] for entry in group['per_panel']}
        if len(panel_scales) > 1:
            # Each panel pads the shared data range on its own scale.
            assert group['auto'] is None
        else:
            assert all(entry['auto'] == group['auto']
                       for entry in group['per_panel'])
        assert all(entry['value'] == entry['auto']
                   for entry in group['per_panel'])


def test_groups_follow_the_limit_mode(matrix):
    options = ['--arrangeSamples', '1', '2', '--perGroup']

    def grouping(mode, *extra):
        record, _ = _described('plotProfileR', matrix, [
            *options, '--yAxisLimits', mode, *extra])
        return [(group['sample_sets'], group['panels'])
                for group in record['y_limits']]
    assert grouping('common') == [([1, 2], [1, 2, 3, 4])]
    assert grouping('per_panel') == [([1], [1]), ([1], [2]), ([2], [3]),
                                     ([2], [4])]
    assert grouping('per_sample_set') == [([1], [1, 2]), ([2], [3, 4])]
    assert grouping('per_y_label', '--yAxisLabel', 'same') == [
        ([1, 2], [1, 2, 3, 4])]


def test_typed_limits_replace_the_automatic_ones(matrix, tmp_path):
    record = _check_drawn('plotProfileR', matrix, tmp_path, [
        *TWO_SCALES, '--yAxisLimits', 'common', '--yMin', '1=-2',
        '--yMax', '2=1000'])
    first, second = record['y_limits'][0]['per_panel']
    assert first['value'] == [-2.0, first['auto'][1]]
    assert second['value'] == [second['auto'][0], 1000.0]


def test_a_typed_bound_past_the_data_widens_by_one(matrix, tmp_path):
    record = _check_drawn('plotProfileR', matrix, tmp_path, [
        *TWO_SCALES, '--yMin', '1=5000'])
    assert record['y_limits'][0]['per_panel'][0]['value'] == [5000.0, 5001.0]


def test_groups_as_sample_sets_limit_each_group_panel(matrix, tmp_path):
    record = _check_drawn('plotProfileR', matrix, tmp_path, [
        '--groupsAsSampleSets', '--perGroup', '--yScale', 'pseudolog',
        'linear', '--yAxisLimits', 'per_sample_set', '--yMax', '2=500'])
    assert [group['sample_sets'] for group in record['y_limits']] == [[1], [2]]
    assert record['y_limits'][1]['per_panel'][0]['value'][1] == 500.0


@pytest.mark.parametrize('extra', [
    [], ['--yScaleSummaryPlot', 'pseudolog', 'linear',
         '--yAxisLimitsSummaryPlot', 'common', '--yMinSummaryPlot', '2=1']])
def test_heatmap_summary_row_reports_its_limits(matrix, tmp_path, extra):
    _check_drawn('plotHeatmapR', matrix, tmp_path,
                 ['--perGroup', '--arrangeSamples', '1', '2', *extra])


def test_data_range_holds_the_fill_base_and_the_bands(matrix):
    def data_range(*options):
        record, _ = _described('plotProfileR', matrix, [
            '--arrangeSamples', '1', '--yAxisLimits', 'common', *options],
            force_load=True)
        return record['y_limits'][0]['data_range']
    lines = data_range()
    assert lines[0] > 0
    assert data_range('--plotType', 'fill') == [0.0, lines[1]]
    bands = data_range('--plotType', 'std')
    assert bands[0] < lines[0] and bands[1] > lines[1]


def test_header_only_keeps_the_grouping_without_numbers(matrix):
    options = [*TWO_SCALES, '--perGroup']
    header, _ = _described('plotProfileR', matrix, options)
    loaded, _ = _described('plotProfileR', matrix, options, force_load=True)

    def grouping(record):
        return [(group['index'], group['sample_sets'], group['panels'],
                 [(entry['panel'], entry['sample_set'])
                  for entry in group['per_panel']])
                for group in record['y_limits']]
    assert not header['values_read'] and grouping(header) == grouping(loaded)
    for group in header['y_limits']:
        assert group['data_range'] is None and group['auto'] is None
        assert all(entry['auto'] is None and entry['value'] is None
                   for entry in group['per_panel'])


@pytest.mark.parametrize('tool,options', [
    ('plotProfileR', ['--plotType', 'heatmap']),
    ('plotHeatmapR', ['--showSummaryPlot', 'off']),
    ('plotMatrixR', ['--heatmap']),
])
@pytest.mark.parametrize('force_load', [False, True])
def test_no_profile_axis_means_no_limits(matrix, tool, options, force_load):
    record = describe(tool, ['-m', matrix, *options], force_load=force_load)
    assert record['y_limits'] is None


def _native_pseudocount(group, sample):
    block = np.ascontiguousarray(VALUES[slice(*GROUPS[group]),
                                        slice(*SAMPLES[sample])])
    return _statistics.geom_mean_axis(block, 0, -1.0, 1)[1]


def _pseudocounts(tool, matrix, options, **kwargs):
    record = describe(tool, ['-m', matrix, *options], **kwargs)
    return {(item['group'], item['sample']): item['pseudocount']
            for item in record['series']}


@pytest.mark.parametrize('plot_type', ['lines', 'ci', 'bootstrap'])
def test_automatic_pseudocount_is_the_native_one(matrix, plot_type):
    found = _pseudocounts('plotProfileR', matrix, [
        '--averageType', 'geom_mean', '--plotType', plot_type],
        force_load=True)
    expected = {(group + 1, sample + 1): _native_pseudocount(group, sample)
                for group in range(2) for sample in range(2)}
    assert len(set(expected.values())) > 1
    assert found == {key: {'auto': value, 'value': value}
                     for key, value in expected.items()}


@pytest.mark.parametrize('tool,options', [
    ('plotProfileR', ['--averageType', 'geom_mean', '--pseudocount', '0.25']),
    ('plotHeatmapR', ['--averageTypeSummaryPlot', 'geom_mean',
                      '--pseudocountSummaryPlot', '0.25']),
])
def test_typed_pseudocount_is_the_value(matrix, tool, options):
    found = _pseudocounts(tool, matrix, options, force_load=True)
    assert found == {(group + 1, sample + 1): {
        'auto': _native_pseudocount(group, sample), 'value': 0.25}
        for group in range(2) for sample in range(2)}


@pytest.mark.parametrize('options,force_load', [
    (['--averageType', 'mean'], True),
    (['--averageType', 'geom_mean'], False),
])
def test_pseudocount_is_null_without_a_read_geometric_mean(
        matrix, options, force_load):
    found = _pseudocounts('plotProfileR', matrix, options,
                          force_load=force_load)
    assert set(found.values()) == {None}
