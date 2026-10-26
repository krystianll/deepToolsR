"""--outFileSortedRegionsPerGroup writes one BED file per drawn group.

``write_input(distinct=True)`` has groups First (2 regions), Other (3) and
Last (4).
"""

import os

import pytest

from deeptoolsr.path_validation import portable_file_name
from deeptoolsr.prepare import OptionError
from deeptoolsr.session import PlotSession
from tests.helpers.capture_series_goldens import write_input


@pytest.fixture
def matrix(tmp_path):
    path = tmp_path / 'input.gz'
    write_input(path, distinct=True)
    return str(path)


def _run(matrix, tmp_path, *flags, tool='plotHeatmapR'):
    with PlotSession(cache_bytes=0) as session:
        return session.run([tool, '-m', matrix, '-o',
                            str(tmp_path / 'plot.png'), *flags])


def _written(directory):
    return sorted(name for name in os.listdir(directory)
                  if name.endswith('.bed'))


def test_files_hold_the_sorted_regions_lines_of_each_group(matrix, tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    sorted_regions = tmp_path / 'all.bed'
    result = _run(matrix, tmp_path, '--kmeans', '3', '--regionsLabel', 'up',
                  'flat', 'down', '--arrangeGroups', '3', '1', '2',
                  '--showRegionCounts', '--silhouette',
                  '--outFileSortedRegions', str(sorted_regions),
                  '--outFileSortedRegionsPerGroup', str(out / 'clusters'))
    order = ['down', 'up', 'flat']
    paths = [out / f'clusters_{label}.bed' for label in order]
    assert _written(out) == sorted(path.name for path in paths)
    assert result.outputs[-3:] == tuple(map(str, paths))
    lines = sorted_regions.read_text().splitlines(keepends=True)
    assert lines[0].startswith('#chrom')
    assert ''.join(path.read_text() for path in paths) == ''.join(lines[1:])
    for label, path in zip(order, paths):
        groups = {line.split('\t')[12] for line in path.read_text().splitlines()}
        assert len(groups) == 1 and groups.pop().startswith(f'{label} [n = ')


@pytest.mark.parametrize('prefix', ['regions', 'regions.bed', 'regions.BED'])
def test_one_bed_ending_of_the_prefix_is_dropped(matrix, tmp_path, prefix):
    _run(matrix, tmp_path, '--quantileSortedRegions', '4',
         '--outFileSortedRegionsPerGroup', str(tmp_path / prefix),
         tool='plotProfileR')
    assert _written(tmp_path) == [f'regions_First_Q{q}.bed'
                                  for q in range(1, 5)]


def test_only_bed_is_dropped(matrix, tmp_path):
    _run(matrix, tmp_path, '--outFileSortedRegionsPerGroup',
         str(tmp_path / 'regions.txt'))
    assert _written(tmp_path) == [f'regions.txt_{name}.bed'
                                  for name in ('First', 'Last', 'Other')]


def test_merged_groups_get_one_file_each(matrix, tmp_path):
    _run(matrix, tmp_path, '--regionsLabel', 'A', 'B', 'A',
         '--sameGroupLabels', 'merge', '--outFileSortedRegionsPerGroup',
         str(tmp_path / 'merged'))
    assert _written(tmp_path) == ['merged_A.bed', 'merged_B.bed']
    assert len((tmp_path / 'merged_A.bed').read_text().splitlines()) == 6


def test_only_drawn_groups_are_written(matrix, tmp_path):
    (tmp_path / 'old_First.bed').write_bytes(b'kept\n')
    _run(matrix, tmp_path, '--arrangeGroups', 'Last',
         '--outFileSortedRegionsPerGroup', str(tmp_path / 'old'))
    assert _written(tmp_path) == ['old_First.bed', 'old_Last.bed']
    assert (tmp_path / 'old_First.bed').read_bytes() == b'kept\n'


@pytest.mark.parametrize('label,name', [
    ('a/b\\c:d', 'a_b_c_d'), ('<>"|?*', '______'), ('tab\there', 'tab_here'),
    ('dots. . ', 'dots'), ('con', 'con_'), ('LPT1.x', 'LPT1_.x'),
    ('COM10', 'COM10'), ('plain name', 'plain name')])
def test_portable_file_names(label, name):
    assert portable_file_name(label) == name


def test_labels_become_portable_file_names(matrix, tmp_path):
    _run(matrix, tmp_path, '--regionsLabel', 'up/down', 'NUL', 'x?',
         '--outFileSortedRegionsPerGroup', str(tmp_path / 'p'))
    assert _written(tmp_path) == ['p_NUL_.bed', 'p_up_down.bed', 'p_x_.bed']


@pytest.mark.parametrize('labels,message', [
    (('A', 'B', 'A'), "region groups 'A' and 'A' would both be written to "),
    (('Up', 'B', 'up'), "region groups 'Up' and 'up' would both be written"),
    (('a/b', 'a_b', 'c'), "region groups 'a/b' and 'a_b' would both"),
])
def test_clashing_names_write_nothing(matrix, tmp_path, labels, message):
    with pytest.raises(OptionError, match=message + '.*--regionsLabel'):
        _run(matrix, tmp_path, '--regionsLabel', *labels,
             '--sameGroupLabels', 'together', '--outFileSortedRegions',
             str(tmp_path / 'all.bed'), '--outFileSortedRegionsPerGroup',
             str(tmp_path / 'p'))
    assert _written(tmp_path) == []
    assert not (tmp_path / 'plot.png').exists()


def test_an_empty_file_name_is_refused(matrix, tmp_path):
    with pytest.raises(OptionError, match="region group '..' gives an empty "
                                          'file name'):
        _run(matrix, tmp_path, '--regionsLabel', 'A', '..', 'C',
             '--outFileSortedRegionsPerGroup', str(tmp_path / 'p'))
    assert _written(tmp_path) == []


def test_profile_rows_are_sorted_as_for_sorted_regions(matrix, tmp_path):
    _run(matrix, tmp_path, '--sortRegions', 'descend',
         '--outFileSortedRegions', str(tmp_path / 'all.bed'),
         '--outFileSortedRegionsPerGroup', str(tmp_path / 'p'),
         tool='plotProfileR')
    lines = (tmp_path / 'all.bed').read_text().splitlines(keepends=True)
    assert lines[1].split('\t')[3] == 'region1'
    assert ''.join((tmp_path / f'p_{name}.bed').read_text()
                   for name in ('First', 'Other', 'Last')) == ''.join(lines[1:])
