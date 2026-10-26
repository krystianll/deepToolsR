"""describe predicts value-dependent groups from the header alone."""

import json
from io import StringIO

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io
from deeptoolsr.describe import describe
from deeptoolsr.prepare import DataError
from deeptoolsr.session import PlotSession

GROUPS = ('GENE1', 'GENE2', 'GENE3')


def _write(path, values, groups=GROUPS, bounds=None):
    rows = len(values)
    bounds = bounds or [0, 10, 20, rows]
    regions = [['chr1', [(i * 20, i * 20 + 10)], f'region{i}', i, '+', '0']
               for i in range(rows)]
    header = {
        'sample_labels': ['Sample A', 'Sample B'],
        'sample_boundaries': [0, 2, 4],
        'group_labels': list(groups), 'group_boundaries': bounds,
        'bin size': [1], 'upstream': [1], 'body': [0], 'downstream': [1],
        'unscaled 5 prime': [0], 'unscaled 3 prime': [0],
        'ref point': ['TSS'], 'min threshold': None, 'max threshold': None,
        'sort regions': 'no', 'sort using': 'mean', 'proc number': 1}
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header), regions,
        np.asarray(values, dtype=np.float32), 1)
    return str(path)


@pytest.fixture
def matrix(tmp_path):
    rng = np.random.default_rng(7)
    # Three well-separated levels, so every cluster and quantile has rows.
    levels = np.repeat([0.0, 10.0, 20.0], 10)[:, None]
    return _write(tmp_path / 'm.gz', levels + rng.random((30, 4)))


@pytest.fixture
def unread(monkeypatch):
    """Fail if describe loads matrix values."""
    monkeypatch.setattr(PlotSession, 'matrix_for', _refuse)


def _without_values(record):
    record = dict(record, values_read=None)
    record['groups'] = [dict(group, regions=None)
                        for group in record['groups']]
    record['colorbars'] = [dict(bar, z_range=None)
                           for bar in record['colorbars']]
    record['series'] = [dict(item, pseudocount=None)
                        for item in record['series']]
    if record['y_limits'] is not None:
        record['y_limits'] = [
            dict(group, data_range=None, auto=None, per_panel=[
                dict(entry, auto=None, value=None)
                for entry in group['per_panel']])
            for group in record['y_limits']]
    return record


CASES = {
    'kmeans': ['plotHeatmapR', '--kmeans', '3'],
    'hclust': ['plotHeatmapR', '--hclust', '3'],
    'kmeans_profile': ['plotProfileR', '--kmeans', '2'],
    'quantiles': ['plotHeatmapR', '--quantileSortedRegions', '3'],
    'quantiles_profile': ['plotProfileR', '--quantileSortedRegions', '2'],
    'merge_quantiles': ['plotHeatmapR', '--sameGroupLabels', 'merge',
                        '--regionsLabel', 'A', 'B', 'A', 'B',
                        '--quantileSortedRegions', '4'],
    'merge': ['plotProfileR', '--sameGroupLabels', 'merge',
              '--regionsLabel', 'A', 'B', 'A'],
    'merge_kmeans': ['plotHeatmapR', '--kmeans', '3', '--sameGroupLabels',
                     'merge', '--regionsLabel', 'X', 'Y', 'X'],
    'filter': ['plotHeatmapR', '--filterNans', 'any_bin'],
    'arrange_clusters': ['plotHeatmapR', '--kmeans', '3',
                         '--arrangeGroups', '3', '1'],
    'arrange_named': ['plotHeatmapR', '--kmeans', '3', '--regionsLabel',
                      '2=middle', '--arrangeGroups', 'middle', 'cluster_1'],
    'arrange_quantiles': ['plotHeatmapR', '--quantileSortedRegions', '3',
                          '--arrangeGroups', 'GENE1_Q3', '1'],
    'arrange_merge': ['plotProfileR', '--sameGroupLabels', 'merge',
                      '--regionsLabel', 'A', 'B', 'A', '--arrangeGroups',
                      '2'],
    'relabel_clusters': ['plotHeatmapR', '--kmeans', '3', '--regionsLabel',
                         '1=low', '3=high', '--sortRegions', 'ascend'],
    'relabel_quantiles': ['plotHeatmapR', '--quantileSortedRegions', '2',
                          '--regionsLabel', '1=first'],
}


@pytest.mark.parametrize('name', CASES)
def test_header_only_record_equals_loaded(name, matrix, monkeypatch):
    tool, *options = CASES[name]
    with monkeypatch.context() as patched:
        patched.setattr(PlotSession, 'matrix_for', _refuse)
        header_only = describe(tool, ['-m', matrix, *options])
    loaded = describe(tool, ['-m', matrix, *options], force_load=True)
    assert header_only['values_read'] is False
    assert all(bar['z_range'] is None for bar in header_only['colorbars'])
    assert loaded['values_read'] is True
    assert all(group['regions'] for group in loaded['groups'])
    assert _without_values(header_only) == _without_values(loaded)
    # Clustering and filtering decide the counts; merging and quantile
    # pooling only regroup the header's rows, so their counts are exact.
    counted = '--kmeans' not in options and '--hclust' not in options and (
        '--filterNans' not in options)
    assert [group['regions'] for group in header_only['groups']] == (
        [group['regions'] for group in loaded['groups']] if counted else
        [None] * len(loaded['groups']))


def _refuse(*_args, **_kwargs):
    raise AssertionError('matrix values were read')


def test_session_reports_no_value_stages(matrix):
    stages = []
    with PlotSession() as session:
        record = session.describe(
            ['plotHeatmapR', '-m', matrix, '--kmeans', '3'],
            progress=stages.append)
        assert session.matrix is None
    assert record['values_read'] is False
    assert stages == ['parse']


def test_plain_command_keeps_header_counts(matrix, unread):
    record = describe('plotHeatmapR', ['-m', matrix])
    assert [group['regions'] for group in record['groups']] == [10, 10, 10]


def test_empty_cluster_diverges_from_the_run(tmp_path, unread):
    # Identical rows form one populated cluster; the header cannot know.
    path = _write(tmp_path / 'flat.gz', np.ones((30, 4)))
    record = describe('plotHeatmapR', ['-m', path, '--kmeans', '3'])
    assert [group['key'] for group in record['groups']] == [
        'cluster_1', 'cluster_2', 'cluster_3']


def test_empty_cluster_is_dropped_when_loaded(tmp_path):
    path = _write(tmp_path / 'flat.gz', np.ones((30, 4)))
    record = describe('plotHeatmapR', ['-m', path, '--kmeans', '3'],
                      force_load=True)
    assert len(record['groups']) < 3


def test_quantiles_beyond_regions_fail_header_only(matrix, unread):
    with pytest.raises(DataError, match='bigger than the number of regions'):
        describe('plotHeatmapR', ['-m', matrix,
                                  '--quantileSortedRegions', '31'])


def test_out_of_range_numbering_is_reported_header_only(matrix, unread):
    record = describe('plotHeatmapR', ['-m', matrix, '--kmeans', '2',
                                       '--regionsLabel', '3=extra'])
    assert record['problems']
    assert [group['label'] for group in record['groups']] == [
        'cluster_1', 'cluster_2']


def test_serve_load_returns_the_full_record(matrix):
    from deeptoolsr.serve import _Worker, _Writer

    request = _Worker._validate_request({
        'id': 'one', 'describe': ['plotHeatmapR', '-m', matrix, '--kmeans',
                                  '3'], 'load': True})
    worker = _Worker(_Writer(StringIO()), StringIO())
    with PlotSession() as session:
        reply = worker._run(session, request)
    assert reply['ok'] and reply['describe']['values_read'] is True
    assert all(group['regions'] for group in reply['describe']['groups'])


@pytest.mark.parametrize('request_data', [
    {'id': 'x', 'command': ['plotProfileR'], 'load': True},
    {'id': 'x', 'info': ['-m', 'matrix.gz'], 'load': True},
    {'id': 'x', 'describe': ['plotProfileR'], 'load': 'yes'},
])
def test_serve_rejects_load_outside_describe(request_data):
    from deeptoolsr.serve import _Worker

    with pytest.raises(ValueError, match='load'):
        _Worker._validate_request(request_data)


def test_merge_counts_are_exact_without_filtering(matrix, unread):
    merge = ['--sameGroupLabels', 'merge', '--regionsLabel', 'A', 'B', 'A']
    record = describe('plotProfileR', ['-m', matrix, *merge])
    assert [group['regions'] for group in record['groups']] == [20, 10]
    filtered = describe('plotProfileR', ['-m', matrix, *merge,
                                         '--filterNans', 'any_bin'])
    assert [group['regions'] for group in filtered['groups']] == [None, None]


def test_unknown_counts_show_a_placeholder(matrix, unread):
    record = describe('plotProfileR', ['-m', matrix, '--kmeans', '2',
                                       '--showRegionCounts'])
    assert [group['label'] for group in record['groups']] == [
        'cluster_1 [n = ???]', 'cluster_2 [n = ???]']
    assert all(group['regions'] is None for group in record['groups'])
    known = describe('plotProfileR', ['-m', matrix, '--quantileSortedRegions',
                                      '3', '--showRegionCounts'])
    assert [group['label'] for group in known['groups']] == [
        'GENE1_Q1 [n = 10]', 'GENE1_Q2 [n = 10]', 'GENE1_Q3 [n = 10]']
