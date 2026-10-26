"""--regionsLabel names the quantiles of --quantileSortedRegions, as clusters.

The matrix has two groups of three regions, so four quantiles hold 1, 2, 1
and 2 regions.
"""

from pathlib import Path

import pytest

from deeptoolsr.describe import describe
from deeptoolsr.prepare import OptionError
from deeptoolsr.session import PlotSession

MATRIX = str(Path(__file__).parent / 'test_data' / 'compute_matrix_expected'
             / 'multiple_bed.mat.gz')
FIRST = 'group1.bed'


def _refuse(*_args, **_kwargs):
    raise AssertionError('header-only describe read the values')


def _groups(*flags, force_load=True, tool='plotHeatmapR'):
    record = describe(tool, ['-m', MATRIX, '--quantileSortedRegions', '4',
                             *flags], force_load=force_load)
    return [(group['original'], group['key'], group['regions'])
            for group in record['groups']]


def _keys(*flags, **kwargs):
    return [key for _, key, _ in _groups(*flags, **kwargs)]


def test_default_names_follow_the_first_group():
    assert _keys() == [f'{FIRST}_Q{q}' for q in range(1, 5)]


def test_a_plain_list_names_every_quantile():
    assert _keys('--regionsLabel', 'Q1', 'Q2', 'Q3', 'Q4') == [
        'Q1', 'Q2', 'Q3', 'Q4']


def test_a_list_must_name_every_quantile(tmp_path):
    record = describe('plotHeatmapR', ['-m', MATRIX, '--quantileSortedRegions',
                                       '4', '--regionsLabel', 'Q1', 'Q2'])
    assert [problem['message'] for problem in record['problems']
            if problem['code'] == 'label_count'] == [
        '--regionsLabel: 2 labels given for 4 region groups']
    with pytest.raises(OptionError, match='2 labels given for 4 region '
                                          'groups'):
        with PlotSession(cache_bytes=0) as session:
            session.run(['plotHeatmapR', '-m', MATRIX, '-o',
                         str(tmp_path / 'plot.png'), '--quantileSortedRegions',
                         '4', '--regionsLabel', 'Q1', 'Q2'])
    assert not (tmp_path / 'plot.png').exists()


def test_an_assignment_renames_one_quantile():
    assert _keys('--regionsLabel', '2=mid') == [
        f'{FIRST}_Q1', 'mid', f'{FIRST}_Q3', f'{FIRST}_Q4']


def test_plain_entries_mixed_with_assignments_name_the_rest():
    assert _keys('--regionsLabel', '2=mid', 'a', 'b', 'c') == [
        'a', 'mid', 'b', 'c']


def test_arrange_groups_takes_quantile_numbers_and_new_names():
    labels = ('--regionsLabel', 'low', 'mid', 'high', 'top')
    assert _groups(*labels, '--arrangeGroups', 'top', '1') == [
        (4, 'top', 2), (1, 'low', 1)]
    assert _groups(*labels, '--arrangeGroups', '3', 'mid') == [
        (3, 'high', 1), (2, 'mid', 2)]
    with pytest.raises(OptionError, match="unknown group 'nope'; the "
                                          "quantiles are 'low', 'mid'"):
        _groups(*labels, '--arrangeGroups', 'nope')


def test_together_keeps_equal_quantile_labels_apart_but_drawn_together():
    flags = ('--regionsLabel', 'a', 'b', 'a', 'b',
             '--sameGroupLabels', 'together')
    assert _groups(*flags) == [(1, 'a', 1), (2, 'b', 2), (3, 'a', 1),
                               (4, 'b', 2)]
    assert _groups(*flags, '--arrangeGroups', 'b', 'a') == [
        (2, 'b', 2), (4, 'b', 2), (1, 'a', 1), (3, 'a', 1)]
    with pytest.raises(OptionError, match="every group labelled 'b' or "
                                          'none; missing 4'):
        _groups(*flags, '--arrangeGroups', '2')


def test_merge_joins_equal_quantile_labels(tmp_path):
    flags = ('--regionsLabel', 'a', 'b', 'a', 'b', '--sameGroupLabels', 'merge')
    assert _groups(*flags) == [(1, 'a', 2), (2, 'b', 4)]
    assert _groups(*flags, '--arrangeGroups', 'b') == [(2, 'b', 4)]
    # The merged groups are whole quantiles, so their rows stay sorted.
    regions = tmp_path / 'sorted.bed'
    with PlotSession(cache_bytes=0) as session:
        session.run(['plotHeatmapR', '-m', MATRIX, '-o',
                     str(tmp_path / 'plot.png'), '--quantileSortedRegions',
                     '4', *flags, '--outFileSortedRegions', str(regions)])
    alone = tmp_path / 'alone.bed'
    with PlotSession(cache_bytes=0) as session:
        session.run(['plotHeatmapR', '-m', MATRIX, '-o',
                     str(tmp_path / 'plot.png'), '--quantileSortedRegions',
                     '4', '--outFileSortedRegions', str(alone)])
    quantile = {line.split('\t')[3]: line.split('\t')[12]
                for line in alone.read_text().splitlines()[1:]}
    rows = [line.split('\t') for line in regions.read_text().splitlines()[1:]]
    assert [quantile[row[3]] for row in rows] == [
        f'{FIRST}_Q1', f'{FIRST}_Q3', f'{FIRST}_Q2', f'{FIRST}_Q2',
        f'{FIRST}_Q4', f'{FIRST}_Q4']
    assert [row[12] for row in rows] == ['a'] * 2 + ['b'] * 4


@pytest.mark.parametrize('flags', [
    (), ('--regionsLabel', 'Q1', 'Q2', 'Q3', 'Q4'), ('--regionsLabel', '2=mid'),
    ('--regionsLabel', 'a', 'b', 'a', 'b', '--sameGroupLabels', 'merge'),
    ('--regionsLabel', 'a', 'b', 'a', 'b', '--sameGroupLabels', 'together',
     '--arrangeGroups', 'b', 'a'),
    ('--regionsLabel', 'up', 'down', 'x', 'y', '--showRegionCounts'),
])
@pytest.mark.parametrize('tool', ['plotHeatmapR', 'plotProfileR'])
def test_header_only_names_match_the_loaded_names(flags, tool, monkeypatch):
    with monkeypatch.context() as patched:
        patched.setattr(PlotSession, 'matrix_for', _refuse)
        predicted = describe(tool, ['-m', MATRIX, '--quantileSortedRegions',
                                    '4', *flags])
    loaded = describe(tool, ['-m', MATRIX, '--quantileSortedRegions', '4',
                             *flags], force_load=True)
    assert predicted['values_read'] is False
    assert predicted['groups'] == loaded['groups']
    assert [series['label'] for series in predicted['series']] == [
        series['label'] for series in loaded['series']]
