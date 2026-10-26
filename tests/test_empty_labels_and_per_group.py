"""Empty per-set labels and --perGroup panel titles."""

import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io, plotMatrix
from deeptoolsr.describe import describe


def _write(path):
    """Three reference-point samples in two region groups."""
    values = np.arange(4 * 6, dtype=np.float32).reshape(4, 6) / 10 + 1
    regions = [['chr1', [(i * 20, i * 20 + 10)], f'region{i}', i, '+', '0']
               for i in range(4)]
    header = {
        'sample_labels': ['Sample A', 'Sample B', 'Sample C'],
        'sample_boundaries': [0, 2, 4, 6],
        'group_labels': ['GENE1', 'GENE2'], 'group_boundaries': [0, 2, 4],
        'bin size': [1, 1, 1], 'upstream': [1, 1, 1],
        'body': [0, 0, 0], 'downstream': [1, 1, 1],
        'unscaled 5 prime': [0, 0, 0], 'unscaled 3 prime': [0, 0, 0],
        'ref point': ['TSS', 'TSS', 'TSS'],
        'min threshold': None, 'max threshold': None, 'sort regions': 'no',
        'sort using': 'mean', 'proc number': 1}
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header, separators=(',', ':')),
        regions, values, 1)
    return str(path)


@pytest.fixture
def matrix(tmp_path):
    return _write(tmp_path / 'matrix.gz')


def _x_labels(record):
    return [item['texts']['x_axis_label'] for item in record['sample_sets']]


def _profile_axes(matrix, tmp_path, options):
    """(drawn X-axis label text, tick labels) of each profile column."""
    captured = {}

    def save(figure, path, **_kwargs):
        labels = {}
        for axis in figure.axes:
            role = (axis.get_gid() or '').split('/')
            if role[:2] == ['label', 'x']:
                labels[role[2]] = ' '.join(t.get_text() for t in axis.texts)
        captured['axes'] = [
            (labels.get(role.split('/')[2], ''),
             [t.get_text() for t in axis.get_xticklabels()])
            for axis in figure.axes
            for role in [axis.get_gid() or '']
            if role.endswith('/profile')]
        with open(path, 'wb') as handle:
            handle.write(b'')

    with patch.object(plotMatrix, 'save_figure_atomic', save):
        plotMatrix.main(['-m', matrix, '--profile', '-o',
                         os.path.join(str(tmp_path), 'out.png')] + options)
    return captured['axes']


SETS = ['--arrangeSamples', '1', '2']


def test_x_label_automatic_when_not_given(matrix):
    record = describe('plotMatrixR', ['-m', matrix, '--profile'] + SETS)
    assert _x_labels(record) == [
        {'auto': 'distance from TSS', 'value': 'distance from TSS'}] * 2


@pytest.mark.parametrize('given', [['1='], ['']])
def test_empty_x_label_means_no_label(matrix, tmp_path, given):
    options = SETS + ['--xAxisLabel'] + given
    record = describe('plotMatrixR', ['-m', matrix, '--profile'] + options)
    first = _x_labels(record)[0]
    assert first == {'auto': 'distance from TSS', 'value': ''}
    assert _profile_axes(matrix, tmp_path, options)[0][0] == ''


def test_empty_x_label_for_one_set_keeps_others_automatic(matrix, tmp_path):
    options = SETS + ['--xAxisLabel', '1=']
    record = describe('plotMatrixR', ['-m', matrix, '--profile'] + options)
    assert _x_labels(record) == [
        {'auto': 'distance from TSS', 'value': ''},
        {'auto': 'distance from TSS', 'value': 'distance from TSS'}]
    drawn = [label for label, _ticks in _profile_axes(matrix, tmp_path,
                                                      options)]
    assert drawn == ['', 'distance from TSS']


def test_empty_reference_label_falls_back_to_distance(matrix, tmp_path):
    options = SETS + ['--refPointLabel', '1=']
    record = describe('plotMatrixR', ['-m', matrix, '--profile'] + options)
    first, second = record['sample_sets']
    assert first['texts']['reference_label']['value'] == ''
    assert first['texts']['x_axis_label'] == {'auto': 'distance',
                                              'value': 'distance'}
    assert second['texts']['x_axis_label']['value'] == 'distance from TSS'
    axes = _profile_axes(matrix, tmp_path, options)
    assert axes[0][0] == 'distance'
    assert 'TSS' not in axes[0][1]
    assert 'TSS' in axes[1][1]


@pytest.mark.parametrize('option, key', [
    ('--sampleSetLabels', 'subplot_title'),
    ('--yAxisLabel', 'y_axis_label'),
    ('--heatmapYAxisLabel', 'heatmap_y_axis_label'),
    ('--refPointLabel', 'reference_label'),
])
def test_other_empty_set_texts_mean_empty(matrix, option, key):
    record = describe('plotMatrixR', ['-m', matrix, '--heatmap', '--profile']
                      + SETS + [option, '1='])
    assert record['sample_sets'][0]['texts'][key]['value'] == ''


@pytest.mark.parametrize('option, key', [
    ('--startLabel', 'start_label'), ('--endLabel', 'end_label')])
def test_empty_region_labels_mean_empty(tmp_path, option, key):
    path = tmp_path / 'scaled.gz'
    values = np.ones((2, 4), dtype=np.float32)
    regions = [['chr1', [(i * 20, i * 20 + 10)], f'region{i}', i, '+', '0']
               for i in range(2)]
    header = {
        'sample_labels': ['Sample A', 'Sample B'],
        'sample_boundaries': [0, 2, 4], 'group_labels': ['GENE1'],
        'group_boundaries': [0, 2], 'bin size': [1, 1],
        'upstream': [0, 0], 'body': [2, 2], 'downstream': [0, 0],
        'unscaled 5 prime': [0, 0], 'unscaled 3 prime': [0, 0],
        'ref point': [None, None], 'min threshold': None,
        'max threshold': None, 'sort regions': 'no', 'sort using': 'mean',
        'proc number': 1}
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header, separators=(',', ':')),
        regions, values, 1)
    record = describe('plotMatrixR', ['-m', str(path), '--profile'] + SETS
                      + [option, '1='])
    assert record['sample_sets'][0]['texts'][key]['value'] == ''
    assert record['sample_sets'][1]['texts'][key]['value'] != ''


def _titles(record):
    return [panel['title'] for panel in record['panels']]


@pytest.mark.parametrize('arrange', [[], ['--arrangeSamples', '1,3']])
def test_per_group_one_set_titles_by_group(matrix, arrange):
    record = describe('plotProfileR', ['-m', matrix, '--perGroup'] + arrange)
    assert _titles(record) == ['GENE1', 'GENE2']
    assert [cell['title'] for cell in record['cells']] == ['GENE1', 'GENE2']


def test_per_group_several_sets_keep_set_titles(matrix):
    record = describe('plotProfileR', ['-m', matrix, '--perGroup',
                                       '--arrangeSamples', '1', '3'])
    titles = _titles(record)
    assert all(' — GENE' in title for title in titles)
    assert any(title.startswith('Sample A') for title in titles)
    assert any(title.startswith('Sample C') for title in titles)


def _warned(record):
    return any('--sampleSetLabels is unused with --perGroup' in warning
               for warning in record['warnings'])


@pytest.mark.parametrize('arrange', [[], ['--arrangeSamples', '1,3']])
def test_set_labels_with_per_group_warn(matrix, arrange):
    record = describe('plotProfileR', ['-m', matrix, '--perGroup',
                                       '--sampleSetLabels', 'X'] + arrange)
    assert _warned(record)


@pytest.mark.parametrize('extra', [
    ['--arrangeSamples', '1', '3', '--sampleSetLabels', 'X', 'Y'],
    ['--groupsAsSampleSets', '--sampleSetLabels', 'X', 'Y'],
])
def test_set_labels_used_with_per_group_do_not_warn(matrix, extra):
    record = describe('plotProfileR', ['-m', matrix, '--perGroup'] + extra)
    assert not _warned(record)
