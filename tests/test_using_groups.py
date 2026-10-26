"""--clusterUsingGroups/--quantileUsingGroups pool only some matrix groups.

``write_input(distinct=True)`` has groups First (regions 0-1), Other (2-4)
and Last (5-8) whose values rise region by region.
"""

import gzip

import pytest

from deeptoolsr.describe import describe
from deeptoolsr.prepare import OptionError
from deeptoolsr.session import PlotSession
from tests.helpers.capture_series_goldens import write_input

QUANTILES = ('--quantileSortedRegions', '2')


@pytest.fixture
def matrix(tmp_path):
    path = tmp_path / 'input.gz'
    write_input(path, distinct=True)
    return str(path)


def _groups(matrix, *flags, force_load=True):
    record = describe('plotHeatmapR', ['-m', matrix, *flags],
                      force_load=force_load)
    return [(group['key'], group['regions']) for group in record['groups']]


def _run(matrix, tmp_path, *flags):
    with PlotSession(cache_bytes=0) as session:
        return session.run(['plotHeatmapR', '-m', matrix, '-o',
                            str(tmp_path / 'plot.png'), *flags])


def _bed_names(path):
    return [line.split('\t')[3] for line in
            path.read_text().splitlines() if not line.startswith('#')]


@pytest.mark.parametrize('selection', [
    ('2', '3'), ('3', '2'), ('2-3',), ('Other', 'Last'), ('Last', '2')])
def test_numbers_ranges_and_names_select_the_same_groups(matrix, selection):
    for spelling in ('--clusterUsingGroups', '--quantileUsingGroups'):
        assert _groups(matrix, *QUANTILES, spelling, *selection) == [
            ('Other_Q1', 3), ('Other_Q2', 4)]


def test_quantile_names_follow_the_first_selected_group(matrix):
    assert _groups(matrix, *QUANTILES, '--quantileUsingGroups', '3') == [
        ('Last_Q1', 2), ('Last_Q2', 2)]
    assert _groups(matrix, *QUANTILES, '--quantileUsingGroups', '1', '3',
                   '--regionsLabel', 'low', 'high') == [('low', 3), ('high', 3)]


def test_dropped_regions_leave_every_output(matrix, tmp_path):
    regions = tmp_path / 'sorted.bed'
    values = tmp_path / 'values.gz'
    result = _run(matrix, tmp_path, *QUANTILES, '--quantileUsingGroups', '1',
                  '3', '--outFileSortedRegions', str(regions),
                  '--outFileNameMatrix', str(values), '--showRegionCounts')
    kept = {f'region{i}' for i in (0, 1, 5, 6, 7, 8)}
    assert set(_bed_names(regions)) == kept
    with gzip.open(values, 'rt') as handle:
        lines = handle.read().splitlines()
    assert {line.split('\t')[3] for line in lines[1:]} == kept
    assert '"group_labels":["First_Q1 [n = 3]","First_Q2 [n = 3]"]' in lines[0]
    assert result.outputs == (str(tmp_path / 'plot.png'), str(values),
                              str(regions))


def test_clustering_uses_only_the_selected_regions(matrix, tmp_path):
    regions = tmp_path / 'sorted.bed'
    _run(matrix, tmp_path, '--kmeans', '2', '--clusterUsingGroups', 'Last',
         '--outFileSortedRegions', str(regions))
    assert sorted(_bed_names(regions)) == [f'region{i}' for i in range(5, 9)]
    assert sum(count for _, count in _groups(
        matrix, '--kmeans', '2', '--clusterUsingGroups', 'Last')) == 4
    # Header-only describe cannot know cluster sizes, as without a selection.
    assert _groups(matrix, '--kmeans', '2', '--clusterUsingGroups', 'Last',
                   force_load=False) == [('cluster_1', None),
                                         ('cluster_2', None)]


def test_silhouette_scores_only_the_selected_regions(matrix, tmp_path):
    regions = tmp_path / 'sorted.bed'
    _run(matrix, tmp_path, '--kmeans', '2', '--silhouette',
         '--clusterUsingGroups', '2-3', '--outFileSortedRegions', str(regions))
    rows = regions.read_text().splitlines()
    assert rows[0].endswith('\tsilhouette')
    assert len(rows) == 1 + 7


def test_header_only_quantile_counts_come_from_the_selected_groups(
        matrix, monkeypatch):
    flags = (*QUANTILES, '--quantileUsingGroups', '1', '2')

    def refuse(*_args, **_kwargs):
        raise AssertionError('header-only describe read the values')
    with monkeypatch.context() as patched:
        patched.setattr(PlotSession, 'matrix_for', refuse)
        predicted = _groups(matrix, *flags, force_load=False)
    assert predicted == _groups(matrix, *flags) == [
        ('First_Q1', 2), ('First_Q2', 3)]


@pytest.mark.parametrize('flags,message', [
    (('2', '2'), '--clusterUsingGroups lists group 2 more than once'),
    (('1-2', 'Other'), '--clusterUsingGroups lists group 2 more than once'),
    (('4',), '--clusterUsingGroups 4: there is no group 4, only 3 groups'),
    (('Nope',), "--clusterUsingGroups: unknown group 'Nope'; the groups are "
                "'First', 'Other', 'Last'"),
])
def test_bad_selections_are_option_errors(matrix, flags, message):
    with pytest.raises(OptionError, match=message):
        _groups(matrix, '--kmeans', '2', '--clusterUsingGroups', *flags)


def test_errors_use_the_spelling_given(matrix):
    with pytest.raises(OptionError, match="--quantileUsingGroups: unknown "
                                          "group 'Nope'"):
        _groups(matrix, *QUANTILES, '--quantileUsingGroups', 'Nope')


def test_without_clustering_or_quantiles_the_selection_is_unused(matrix):
    record = describe('plotHeatmapR', ['-m', matrix, '--clusterUsingGroups',
                                       '1'])
    assert [(group['key'], group['regions']) for group in record['groups']] == [
        ('First', 2), ('Other', 3), ('Last', 4)]
    assert any('--clusterUsingGroups is unused' in message and
               '--arrangeGroups' in message
               for message in record['warnings'])


def test_selection_changes_the_cached_layout(matrix, tmp_path):
    regions = tmp_path / 'sorted.bed'
    with PlotSession() as session:
        for selection in ('1', '3'):
            session.run(['plotHeatmapR', '-m', matrix, '-o',
                         str(tmp_path / 'plot.png'), *QUANTILES,
                         '--quantileUsingGroups', selection,
                         '--outFileSortedRegions', str(regions)])
            assert len(_bed_names(regions)) == (2 if selection == '1' else 4)
