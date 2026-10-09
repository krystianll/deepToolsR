"""computeMatrixR --blackListFileName with --blackListMode and --featureTouchStrand."""
import gzip

import numpy as np
import pytest

from deeptoolsr import computeMatrix
from tests.helpers.bigwig import pyBigWig
from tests.helpers.parity import read_matrix

NAN = np.nan
SIZE = 10000


@pytest.fixture
def ramp(tmp_path):
    """chr1 with value = position, except the gap 8000-8100 (no data)."""
    path = tmp_path / 'ramp.bw'
    starts = [i for i in range(SIZE) if not 8000 <= i < 8100]
    with pyBigWig.open(str(path), 'w') as bw:
        bw.addHeader([('chr1', SIZE)])
        bw.addEntries(['chr1'] * len(starts), starts, ends=[i + 1 for i in starts],
                      values=[float(i) for i in starts])
    return path


def write(path, lines, opener=open):
    with opener(path, 'wt') as handle:
        handle.write(''.join(line + '\n' for line in lines))
    return path


def compute(tmp_path, bigwig, regions, blacklist, mode=None, extra=(), command='scale-regions'):
    bed = write(tmp_path / 'regions.bed',
                [f'chr1\t{start}\t{end}\t{name}\t0\t{strand}'
                 for name, start, end, strand in regions])
    output, sorted_bed = tmp_path / 'out.gz', tmp_path / 'sorted.bed'
    args = [command, '-S', str(bigwig), '-R', str(bed), '-o', str(output),
            '--sortRegions', 'no', '--outFileSortedRegions', str(sorted_bed), *extra]
    if command == 'scale-regions' and '-m' not in extra:
        args += ['-b', '0', '-a', '0', '-m', '1000', '-bs', '100']
    if blacklist is not None:
        args += ['--blackListFileName', str(blacklist)]
    if mode is not None:
        args += ['--blackListMode', mode]
    computeMatrix.main(args)
    _, rows, values = read_matrix(output)
    names = [row[3] for row in rows]
    assert names == [line.split('\t')[3] for line in sorted_bed.read_text().splitlines()
                     if not line.startswith('#')]
    return dict(zip(names, values))


def bins(start, end, n=10):
    """Mean of the ramp over each of n equal bins of [start, end)."""
    edges = [start + i * (end - start) // n for i in range(n + 1)]
    return np.array([(a + b - 1) / 2 for a, b in zip(edges, edges[1:])])


# The cases of the plan's table: blacklist chr1:5000-6000, scale-regions, 10 bins.
TABLE = [('clear', 1000, 2000, '+'), ('ends_in_bl', 4500, 5500, '+'),
         ('inside_bl', 5200, 5800, '+'), ('starts_in_bl', 5500, 6500, '+'),
         ('spans_bl', 4000, 7000, '+')]


def test_soft_masks_blacklisted_bases_only(tmp_path, ramp):
    blacklist = write(tmp_path / 'bl.bed', ['chr1\t5000\t6000'])
    rows = compute(tmp_path, ramp, TABLE, blacklist)  # soft is the default
    assert list(rows) == ['clear', 'spans_bl', 'ends_in_bl', 'inside_bl', 'starts_in_bl']
    np.testing.assert_allclose(rows['clear'], bins(1000, 2000))
    np.testing.assert_allclose(rows['ends_in_bl'], [*bins(4500, 5500)[:5], *[NAN] * 5])
    assert np.isnan(rows['inside_bl']).all()
    np.testing.assert_allclose(rows['starts_in_bl'], [*[NAN] * 5, *bins(5500, 6500)[5:]])
    # 300 bp bins: 4900-5200 and 5800-6100 keep 100 bases each (mean of those).
    expected = bins(4000, 7000)
    expected[3], expected[4], expected[5], expected[6] = 4949.5, NAN, NAN, 6049.5
    np.testing.assert_allclose(rows['spans_bl'], expected)


def test_hard_drops_regions_with_a_blacklisted_base(tmp_path, ramp, capsys):
    blacklist = write(tmp_path / 'bl.bed', ['chr1\t5000\t6000'])
    rows = compute(tmp_path, ramp, TABLE, blacklist, 'hard')
    assert list(rows) == ['clear']
    assert 'Skipped 4 regions overlapping the blacklist' in capsys.readouterr().err
    compute(tmp_path, ramp, TABLE, blacklist, 'hard', extra=['--quiet'])
    assert 'blacklist' not in capsys.readouterr().err
    compute(tmp_path, ramp, TABLE, blacklist, 'hard', extra=['--verbose'])
    assert 'Region inside_bl chr1:5200-5800 overlaps the blacklist' in capsys.readouterr().err
    # The plotted span includes the flanks; a flank base is enough.
    rows = compute(tmp_path, ramp, [('a', 3000, 4000, '+'), ('b', 2000, 3000, '+')], blacklist,
                   'hard', extra=['-b', '0', '-a', '1001', '-m', '1000', '-bs', '1'])
    assert list(rows) == ['b']


def test_no_blacklist_entry_on_the_span(tmp_path, ramp):
    blacklist = write(tmp_path / 'bl.bed', ['chr1\t2000\t2500'])
    for mode in ('soft', 'hard'):
        rows = compute(tmp_path, ramp, [('a', 1000, 2000, '+'), ('b', 2500, 3500, '-')], blacklist, mode)
        np.testing.assert_allclose(rows['a'], bins(1000, 2000))
        np.testing.assert_allclose(rows['b'], bins(2500, 3500)[::-1])


# Entry strands + / - / . against regions + / - / . for each --featureTouchStrand.
HITS = {
    'sense': {'+': {'+', '.'}, '-': {'-', '.'}, '.': {'+', '-', '.'}},
    'antisense': {'+': {'-', '.'}, '-': {'+', '.'}, '.': {'+', '-', '.'}},
    'ignore': {'+': {'+', '-', '.'}, '-': {'+', '-', '.'}, '.': {'+', '-', '.'}},
}


@pytest.mark.parametrize('touch', ['sense', 'antisense', 'ignore'])
@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_feature_touch_strand(tmp_path, ramp, touch, mode):
    entries = {'+': 1000, '-': 3000, '.': 5000}  # one entry per strand
    blacklist = write(tmp_path / 'bl.bed', [f'chr1\t{start}\t{start + 1000}\te\t0\t{strand}'
                                            for strand, start in entries.items()])
    regions = [(f'{entry}{region}', start, start + 1000, region)
               for entry, start in entries.items() for region in '+-.']
    regions = [(n.replace('.', 'u'), s, e, r) for n, s, e, r in regions] + [('clear', 7000, 8000, '+')]
    rows = compute(tmp_path, ramp, regions, blacklist, mode, extra=['--featureTouchStrand', touch])
    np.testing.assert_allclose(rows['clear'], bins(7000, 8000))
    for entry, start in entries.items():
        for region in '+-.':
            name = f'{entry}{region}'.replace('.', 'u')
            hit = region in HITS[touch] and entry in HITS[touch][region]
            if mode == 'hard':
                assert (name in rows) is not hit, name
            else:
                assert bool(np.isnan(rows[name]).all()) is hit, name


def test_bed3_blacklist_hits_every_strand_by_default(tmp_path, ramp):
    blacklist = write(tmp_path / 'bl.bed', ['chr1\t1000\t2000'])
    rows = compute(tmp_path, ramp, [('p', 1000, 2000, '+'), ('m', 1000, 2000, '-')], blacklist)
    assert np.isnan(rows['p']).all() and np.isnan(rows['m']).all()


def test_missing_data_as_zero_keeps_blacklisted_bases_missing(tmp_path, ramp):
    blacklist = write(tmp_path / 'bl.bed', ['chr1\t7500\t7600', 'chr1\t8050\t8100'])
    rows = compute(tmp_path, ramp, [('a', 7000, 8000, '+'), ('gap', 7600, 8600, '+')], blacklist,
                   extra=['--missingDataAsZero'])
    expected = bins(7000, 8000)
    expected[5] = NAN
    np.testing.assert_allclose(rows['a'], expected)
    # The data gap 8000-8100 is zero where it is not blacklisted.
    expected = bins(7600, 8600)
    expected[4] = 0
    np.testing.assert_allclose(rows['gap'], expected)
    blacklist = write(tmp_path / 'bl.bed', ['chr1\t8000\t8100'])
    rows = compute(tmp_path, ramp, [('gap', 7600, 8600, '+')], blacklist, extra=['--missingDataAsZero'])
    expected[4] = NAN
    np.testing.assert_allclose(rows['gap'], expected)


def test_overlapping_nested_entries_and_gzip(tmp_path, ramp):
    blacklist = write(tmp_path / 'bl.bed.gz', ['chr1\t1100\t1300', 'chr1\t1150\t1200',
                                               'chr1\t1250\t1450', 'chr1\t1900\t2000'], gzip.open)
    rows = compute(tmp_path, ramp, [('a', 1000, 2000, '+')], blacklist)
    expected = bins(1000, 2000)
    expected[[1, 2, 3, 9]] = NAN
    expected[4] = 1474.5  # 1450-1500 remain
    np.testing.assert_allclose(rows['a'], expected)


def test_metagene_masks_exon_bases(tmp_path, ramp):
    bed = write(tmp_path / 'genes.bed',
                ['chr1\t1000\t3000\tg\t0\t+\t1000\t3000\t0\t2\t500,500,\t0,1500,'])
    blacklist = write(tmp_path / 'bl.bed', ['chr1\t1200\t2600'])  # exon 1 tail, intron, exon 2 head
    output = tmp_path / 'out.gz'
    computeMatrix.main(['scale-regions', '-S', str(ramp), '-R', str(bed), '-o', str(output),
                        '-m', '1000', '-bs', '100', '-b', '0', '-a', '0', '--metagene',
                        '--blackListFileName', str(blacklist)])
    _, _, values = read_matrix(output)
    exons = np.r_[1000:1500, 2500:3000]
    expected = exons.reshape(10, 100).mean(axis=1).astype(float)
    expected[[2, 3, 4, 5]] = NAN
    np.testing.assert_allclose(values[0], expected)


def test_reference_point_flanks_and_nan_after_end(tmp_path, ramp):
    blacklist = write(tmp_path / 'bl.bed', ['chr1\t900\t1000'])
    rows = compute(tmp_path, ramp, [('p', 1000, 1100, '+'), ('m', 600, 700, '-')], blacklist,
                   command='reference-point',
                   extra=['-b', '200', '-a', '300', '-bs', '100', '--nanAfterEnd'])
    np.testing.assert_allclose(rows['p'], [849.5, NAN, 1049.5, NAN, NAN])
    # Minus strand: TSS at 700, upstream 700-900 (blacklist-free), 300 downstream past the end.
    np.testing.assert_allclose(rows['m'], [849.5, 749.5, 649.5, NAN, NAN])


@pytest.mark.parametrize('antisense', ['as_samples', 'as_groups'])
def test_stranded_pairs_and_antisense_use_the_region_strand(tmp_path, ramp, antisense):
    blacklist = write(tmp_path / 'bl.bed', ['chr1\t1000\t2000\te\t0\t+'])
    rows = compute(tmp_path, ramp, [('p', 1000, 2000, '+'), ('m', 1000, 2000, '-')], blacklist,
                   extra=['--scoreFileNameMinus', str(ramp), '--antisense', antisense])
    for name, values in rows.items():
        masked = np.isnan(values)
        assert masked.all() if name.startswith('p') else not masked.any(), name


def test_blacklist_chromosome_aliases_merge(tmp_path, ramp):
    blacklist = write(tmp_path / 'bl.bed', ['chr1\t1000\t2000', '1\t3000\t4000'])
    rows = compute(tmp_path, ramp, [('a', 1000, 2000, '+'), ('b', 3000, 4000, '+'),
                                    ('c', 5000, 6000, '+')], blacklist)
    assert np.isnan(rows['a']).all() and np.isnan(rows['b']).all()
    np.testing.assert_allclose(rows['c'], bins(5000, 6000))


def test_regions_shorter_than_a_bin_are_dropped(tmp_path, ramp, capsys):
    # 3.5.6 kept such regions as all-NaN (or all-0) rows.
    regions = [('short', 1000, 1050, '+'), ('long', 2000, 3000, '+'), ('short2', 4000, 4099, '-')]
    for extra in ([], ['--missingDataAsZero']):
        rows = compute(tmp_path, ramp, regions, None, extra=extra)
        assert list(rows) == ['long']
        err = capsys.readouterr().err
        assert "Skipped 2 regions shorter than the bin size (after the unscaled 5'/3' parts)." in err
        assert 'Skipping...' not in err
    compute(tmp_path, ramp, regions, None, extra=['--quiet'])
    assert 'shorter' not in capsys.readouterr().err
    compute(tmp_path, ramp, regions, None, extra=['--verbose'])
    assert 'short2 chr1:4000:4099. Skipping...' in capsys.readouterr().err
    # --unscaled5prime counts: 1000 bp minus 900 unscaled leaves 100 bp, 950 leaves 50.
    rows = compute(tmp_path, ramp, [('a', 1000, 2000, '+'), ('b', 3000, 3950, '+')], None,
                   extra=['-b', '0', '-a', '0', '-m', '1000', '-bs', '100', '--unscaled5prime', '900'])
    assert list(rows) == ['a']


def test_all_short_regions_of_a_group_or_run(tmp_path, ramp, capsys):
    bed = write(tmp_path / 'g.bed', ['chr1\t1000\t1050\ta\t0\t+', '#tiny', 'chr1\t2000\t3000\tb\t0\t+', '#genes'])
    output = tmp_path / 'out.gz'
    computeMatrix.main(['scale-regions', '-S', str(ramp), '-R', str(bed), '-o', str(output),
                        '-m', '1000', '-bs', '100', '-b', '0', '-a', '0'])
    header, rows, _ = read_matrix(output)
    assert [row[3] for row in rows] == ['b']
    assert header['group_labels'] == ['genes'] and header['group_boundaries'] == [0, 1]
    bed = write(tmp_path / 'all.bed', ['chr1\t1000\t1050\ta'])
    with pytest.raises(SystemExit):
        computeMatrix.main(['scale-regions', '-S', str(ramp), '-R', str(bed), '-o', str(output),
                            '-m', '1000', '-bs', '100'])
    assert 'none remaining after filtering' in capsys.readouterr().err
