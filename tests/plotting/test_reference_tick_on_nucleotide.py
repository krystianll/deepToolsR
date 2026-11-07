"""Reference-point ticks sit on the reference nucleotide's bin.

computeMatrix opens the downstream bins with the TSS nucleotide and closes
the upstream bins with the TES nucleotide, so the TSS tick sits on the first
downstream bin and the TES tick on the last upstream bin, as scale-regions
places its TSS and TES ticks. The header's reference point decides, never
the label.
"""

import csv
import json

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io, plotHeatmap, plotMatrix, plotProfile
from deeptoolsr.plotting.ticks import getProfileTicks


def _parameters(reference, upstream=10, downstream=20, bin_size=1):
    return {
        'upstream': [upstream], 'downstream': [downstream], 'body': [0],
        'bin size': [bin_size], 'unscaled 5 prime': [0],
        'unscaled 3 prime': [0], 'ref point': [reference],
    }


@pytest.mark.parametrize('reference, bin_size, expected', [
    ('TSS', 1, 10), ('TES', 1, 9), ('TSS', 10, 10), ('TES', 10, 9),
])
def test_tick_sits_on_the_reference_nucleotide_bin(reference, bin_size,
                                                   expected):
    ticks, labels = getProfileTicks(
        _parameters(reference, 10 * bin_size, 20 * bin_size, bin_size),
        'here', 'TSS', 'TES', 0)
    assert ticks[labels.index('here')] == expected


@pytest.mark.parametrize('reference', ['center', 'PAS', None])
def test_centre_or_unknown_reference_keeps_the_edge(reference):
    ticks, labels = getProfileTicks(_parameters(reference), 'here',
                                    'TSS', 'TES', 0)
    assert ticks == [0, 9.5, 29]


def test_label_does_not_decide_the_side():
    ticks, labels = getProfileTicks(_parameters('TES'), 'PAS', 'TSS', 'TES',
                                    0)
    assert ticks[labels.index('PAS')] == 9


@pytest.mark.parametrize('reference, upstream, downstream', [
    ('TSS', 10, 0), ('TES', 0, 20),
])
def test_missing_side_keeps_the_edge(reference, upstream, downstream):
    ticks, labels = getProfileTicks(
        _parameters(reference, upstream, downstream), 'here', 'TSS', 'TES', 0)
    assert ticks[labels.index('here')] == max(upstream - 0.5, 0)


@pytest.mark.parametrize('reference, upstream, downstream, expected', [
    ('TSS', 1, 1, ([0, 1], ['-1 bp', 'here'])),
    ('TES', 1, 1, ([0, 1], ['here', '1 bp'])),
])
def test_outer_tick_on_the_reference_bin_gives_way(reference, upstream,
                                                   downstream, expected):
    ticks, labels = getProfileTicks(
        _parameters(reference, upstream, downstream), 'here', 'TSS', 'TES', 0)
    assert (ticks, labels) == expected


def test_each_sample_uses_its_own_reference_point():
    parameters = _parameters('TSS')
    parameters = {key: value * 2 for key, value in parameters.items()}
    parameters['ref point'] = ['TSS', 'TES']
    assert getProfileTicks(parameters, 'TSS', 'TSS', 'TES', 0)[0][1] == 10
    assert getProfileTicks(parameters, 'TES', 'TSS', 'TES', 1)[0][1] == 9


@pytest.fixture
def tes_matrix(tmp_path):
    path = tmp_path / 'matrix.gz'
    header = {
        'sample_labels': ['Sample A'], 'sample_boundaries': [0, 30],
        'group_labels': ['GENE1'], 'group_boundaries': [0, 4],
        'bin size': [1], 'upstream': [10], 'body': [0], 'downstream': [20],
        'unscaled 5 prime': [0], 'unscaled 3 prime': [0], 'ref point': ['TES'],
        'min threshold': None, 'max threshold': None, 'sort regions': 'no',
        'sort using': 'mean', 'proc number': 1}
    regions = [['chr1', [(i * 100, i * 100 + 50)], f'region{i}', i, '+', '0']
               for i in range(4)]
    values = np.random.default_rng(3).random((4, 30)).astype(np.float32)
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header), regions, values, 1)
    return str(path)


def test_profile_table_records_the_moved_tick(tes_matrix, tmp_path):
    table = tmp_path / 'profile.tsv'
    plotProfile.main(['-m', tes_matrix, '--refPointLabel', 'PAS',
                      '-o', str(tmp_path / 'plot.png'),
                      '--outFileNameData', str(table)])
    with open(table, newline='', encoding='utf-8') as handle:
        row = next(csv.DictReader(handle, delimiter='\t'))
    assert json.loads(row['breakpoint_labels'])[1] == 'PAS'
    assert json.loads(row['breakpoint_bins']) == [0.0, 9.0, 29.0]


def test_heatmap_tick_matches_the_profile_tick(tes_matrix, tmp_path,
                                               monkeypatch):
    seen = {}
    original = plotMatrix.build_matrix_figure

    def capture(*args, **kwargs):
        seen['figure'], solution = original(*args, **kwargs)
        return seen['figure'], solution
    monkeypatch.setattr(plotMatrix, 'build_matrix_figure', capture)
    plotHeatmap.main(['-m', tes_matrix, '--sortRegions', 'no', '--dpi', '50',
                      '-o', str(tmp_path / 'plot.png')])
    axes = {axis.get_gid(): axis for axis in seen['figure'].axes
            if axis.get_gid()}
    profile = next(axis for gid, axis in axes.items()
                   if gid.endswith('/profile'))
    block = next(axis for gid, axis in axes.items() if '/block/' in gid)
    # Heatmap pixel i spans [i, i + 1]; the profile knot i sits at i.
    assert list(block.get_xticks()) == [
        value + .5 for value in profile.get_xticks()]
    assert 9.5 in list(block.get_xticks())
