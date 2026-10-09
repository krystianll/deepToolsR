"""--arrangeGroups selects and orders the final region groups.

``write_input(distinct=True)`` has groups First (2 rows), Other (3) and Last
(4) with values rising row by row; ``_write_two_levels`` has groups First (4),
an empty group and Last (5) whose rows hold only two distinct values, so
clustering can never populate more than two clusters.
"""

import gzip
import json

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io, plotMatrix
from deeptoolsr.describe import describe
from deeptoolsr.prepare import DataError, OptionError, warning_text
from deeptoolsr.session import PlotSession
from tests.helpers.capture_series_goldens import write_input


def _write_two_levels(path):
    values = np.repeat(np.array([[1.0], [1.0], [5.0]], dtype=np.float32),
                       3, axis=0).repeat(6, axis=1)
    regions = [['chr1', [(i * 20, i * 20 + 10)], f'region{i}', i, '+', '0']
               for i in range(9)]
    header = {
        'sample_labels': ['A', 'B', 'C'], 'sample_boundaries': [0, 2, 4, 6],
        'group_labels': ['First', 'Empty', 'Last'],
        'group_boundaries': [0, 4, 4, 9],
        'bin size': [1, 1, 1], 'upstream': [1, 1, 1], 'body': [0, 0, 0],
        'downstream': [1, 1, 1], 'unscaled 5 prime': [0, 0, 0],
        'unscaled 3 prime': [0, 0, 0], 'ref point': [None, None, None],
        'min threshold': None, 'max threshold': None, 'sort regions': 'no',
        'sort using': 'mean', 'proc number': 1,
    }
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header), regions, values, 1)


def _describe(tmp_path, *flags, two_levels=False):
    matrix = tmp_path / 'input.gz'
    (_write_two_levels if two_levels else
     lambda path: write_input(path, distinct=True))(matrix)
    record = describe('plotProfileR', ['-m', str(matrix), '-o', 'x.png',
                                       *flags])
    return ([(group['original'], group['key'], group['regions'])
             for group in record['groups']], record['warnings'], record)


@pytest.mark.parametrize('flags,groups', [
    ((), [(1, 'First', 2), (2, 'Other', 3), (3, 'Last', 4)]),
    (('--arrangeGroups', '3', '1'), [(3, 'Last', 4), (1, 'First', 2)]),
    (('--arrangeGroups', 'Last', '2-1'),
     [(3, 'Last', 4), (2, 'Other', 3), (1, 'First', 2)]),
    (('--kmeans', '2', '--arrangeGroups', '2', '1'),
     [(2, 'cluster_2', 5), (1, 'cluster_1', 4)]),
    (('--quantiles', '3', '--sortRegions', 'descend', '--arrangeGroups',
      'First_Q3', '1'), [(3, 'First_Q3', 3), (1, 'First_Q1', 3)]),
    (('--regionsLabel', 'A', 'B', 'A', '--sameGroupLabels', 'merge',
      '--arrangeGroups', '2', 'A'), [(2, 'B', 3), (1, 'A', 6)]),
    # A name selects the whole equal-label set, drawn at its first member.
    (('--regionsLabel', 'A', 'B', 'A', '--sameGroupLabels', 'together',
      '--arrangeGroups', 'A', '2'),
     [(1, 'A', 2), (3, 'A', 4), (2, 'B', 3)]),
    (('--regionsLabel', 'A', 'B', 'A', '--sameGroupLabels', 'together',
      '--arrangeGroups', '3', '2', '1'),
     [(3, 'A', 4), (1, 'A', 2), (2, 'B', 3)]),
])
def test_selection_and_order(tmp_path, flags, groups):
    assert _describe(tmp_path, *flags)[0] == groups


@pytest.mark.parametrize('flags,message', [
    (('--arrangeGroups', '1,2'), "one group per value, not '1,2'"),
    (('--arrangeGroups', '4'), 'there is no group 4, only 3 groups'),
    (('--arrangeGroups', '2-5'), 'there is no group 4, only 3 groups'),
    (('--arrangeGroups', 'Nope'), "unknown group 'Nope'"),
    (('--arrangeGroups', '1', 'First'), 'lists group 1 more than once'),
    (('--kmeans', '2', '--arrangeGroups', '3'),
     'there is no group 3, only 2 clusters'),
    (('--quantiles', '2', '--sortRegions', 'descend', '--arrangeGroups',
      '3'), 'there is no group 3, only 2 quantiles'),
    (('--regionsLabel', 'A', 'B', 'A', '--sameGroupLabels', 'together',
      '--arrangeGroups', '1', '2'),
     "every group labelled 'A' or none; missing 3"),
])
def test_errors_need_no_values(tmp_path, monkeypatch, flags, message):
    def no_values(*_args):
        raise AssertionError('values loaded')
    monkeypatch.setattr(PlotSession, 'matrix_for', no_values)
    with pytest.raises(OptionError, match=message):
        _describe(tmp_path, *flags)


def test_unformed_cluster_is_skipped_with_a_warning(tmp_path):
    groups, warnings, _ = _describe(
        tmp_path, '--kmeans', '4', '--arrangeGroups', '3', '2', '1',
        two_levels=True)
    assert groups == [(2, 'cluster_2', 6), (1, 'cluster_1', 3)]
    assert warnings[-1] == ('WARNING: --arrangeGroups: group 3 was not formed '
                            '(2 of 4 clusters); drawing 2, 1.')


def test_empty_group_keeps_numbering(tmp_path):
    # Quantile groups are never empty (--quantiles must not exceed the
    # region count), so an empty matrix group stands for the rule.
    groups, warnings, _ = _describe(
        tmp_path, '--arrangeGroups', '3', '2', '1', two_levels=True)
    assert groups == [(3, 'Last', 5), (1, 'First', 4)]
    assert warnings[-1] == ('WARNING: --arrangeGroups: group 2 has no '
                            'regions; drawing 3, 1.')
    with pytest.raises(DataError, match='nothing to draw'):
        _describe(tmp_path, '--arrangeGroups', 'Empty', two_levels=True)


def test_colour_numbering_follows_arranged_order(tmp_path):
    record = _describe(tmp_path, '--arrangeGroups', '3', '1',
                       '--colors', '1=red')[2]
    assert [item['label'] for item in record['series']][:2] == [
        'Last', 'First']
    slot = record['color_domains'][0]['slots'][0]
    assert slot['series'][0] == record['series'][0]['key']


def _render(tmp_path, *flags):
    matrix = tmp_path / 'input.gz'
    write_input(matrix, distinct=True)
    built = {}
    original = plotMatrix.build_matrix_figure

    def capture(matrix, layout, labels, plan, spec, prepared, *args, **kwargs):
        built.update(plan=plan, prepared=prepared)
        return original(matrix, layout, labels, plan, spec, prepared, *args,
                        **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plotMatrix, 'build_matrix_figure', capture)
        plotMatrix.main(['-m', str(matrix), '-o', str(tmp_path / 'plot.png'),
                         '-p', '1', '--dpi', '50', '--sortRegions', 'no',
                         *flags])
    return built


def test_limits_scan_only_drawn_groups(tmp_path):
    full = _render(tmp_path, '--heatmap')['prepared'].z_max
    kept = _render(tmp_path, '--heatmap', '--arrangeGroups', '2', '1')
    assert all(low < high for low, high in zip(kept['prepared'].z_max, full))


def test_promoted_sample_sets_follow_arranged_order(tmp_path):
    plan = _render(tmp_path, '--heatmap', '--perGroup',
                   '--groupsAsSampleSets', '--arrangeGroups', '3', '1')['plan']
    assert [cell.title for cell in plan.cells] == ['Last', 'First']


def test_outputs_hold_only_drawn_groups(tmp_path):
    regions, values = tmp_path / 'regions.bed', tmp_path / 'matrix.gz'
    _render(tmp_path, '--profile', '--arrangeGroups', '3', '1',
            '--outFileSortedRegions', str(regions),
            '--outFileNameMatrix', str(values))
    groups = [line.split('\t')[12] for line in
              regions.read_text().splitlines()[1:]]
    assert groups == ['Last'] * 4 + ['First'] * 2
    with gzip.open(values, 'rt') as handle:
        header = json.loads(handle.readline()[1:])
    assert (header['group_labels'], header['group_boundaries']) == (
        ['Last', 'First'], [0, 4, 6])


def test_merged_cluster_names_resolve_after_the_merge(tmp_path):
    flags = ('--kmeans', '3', '--regionsLabel', 'A', 'A', 'B',
             '--sameGroupLabels', 'merge', '--arrangeGroups')
    groups = {token: _describe(tmp_path, *flags, *token.split())[0]
              for token in ('A', 'B', 'B A')}
    assert [key for _, key, _ in groups['A']] == ['A']
    assert [key for _, key, _ in groups['B']] == ['B']
    assert [key for _, key, _ in groups['B A']] == ['B', 'A']
    with pytest.raises(OptionError, match='there is no group 3, only 2 clusters'):
        _describe(tmp_path, *flags, '3')
    regions = tmp_path / 'regions.bed'
    _render(tmp_path, '--profile', *flags, 'A', '--outFileSortedRegions', str(regions))
    assert {line.split('\t')[12] for line in regions.read_text().splitlines()[1:]} == {'A'}


def test_no_group_drawn_warning_lists_none(tmp_path):
    with pytest.raises(DataError, match='nothing to draw') as error:
        _describe(tmp_path, '--kmeans', '4', '--arrangeGroups', '3',
                  two_levels=True)
    assert warning_text(error.value.warnings[-1], error.value.layout) == (
        'WARNING: --arrangeGroups: group 3 was not formed (2 of 4 clusters).\n')


def _describe_empty_first(tmp_path, *flags):
    matrix = tmp_path / 'empty_first.gz'
    values = np.arange(10, dtype=np.float32).repeat(2).reshape(10, 2)
    header = {
        'sample_labels': ['S'], 'sample_boundaries': [0, 2],
        'group_labels': ['Empty', 'Genes'], 'group_boundaries': [0, 0, 10],
        'bin size': [1], 'upstream': [1], 'body': [0], 'downstream': [1],
        'unscaled 5 prime': [0], 'unscaled 3 prime': [0], 'ref point': [None],
        'min threshold': None, 'max threshold': None, 'sort regions': 'no',
        'sort using': 'mean', 'proc number': 1,
    }
    _compute_matrix_io.write_matrix(
        str(matrix), json.dumps(header),
        [['chr1', [(i * 20, i * 20 + 10)], f'r{i}', 1, '+', '0'] for i in range(10)],
        values, 1)
    record = describe('plotProfileR', ['-m', str(matrix), '-o', 'x.png', '--quantiles', '2',
                                       '--sortRegions', 'descend', *flags])
    return [(group['original'], group['key']) for group in record['groups']]


def test_quantile_names_follow_the_surviving_group(tmp_path):
    assert _describe_empty_first(tmp_path, '--arrangeGroups', 'Genes_Q2') == [(2, 'Genes_Q2')]
    assert _describe_empty_first(tmp_path, '--regionsLabel', 'X', 'Y',
                                 '--arrangeGroups', 'Y_Q1') == [(1, 'Y_Q1')]
    with pytest.raises(DataError, match="unknown group 'Empty_Q1'; the quantiles are "
                                        "'Genes_Q1', 'Genes_Q2'"):
        _describe_empty_first(tmp_path, '--arrangeGroups', 'Empty_Q1')
