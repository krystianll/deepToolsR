"""Compute construction helpers keep their historical filename handling."""
from deeptoolsr.matrix import Matrix
from deeptoolsr import computeMatrix
import numpy as np
from tests.helpers.parity import original
from tests.helpers.bigwig import pyBigWig
from tests.helpers.parity import read_matrix
import bz2
from deeptoolsr import computeMatrixOperations as cmo
import gzip
from deeptoolsr import _compute_matrix_native
import itertools
import math
from scipy import stats

import pytest

from deeptoolsr.compute import smartLabel, smartLabels


@pytest.mark.parametrize(('path', 'expected'), [
    ('/work/sample.bw', 'sample'),
    ('/work/sample.special.bw', 'sample.special'),
    ('/work/.hidden', '.hidden'),
    ('/work/.hidden.bw', '.hidden'),
])
def test_smart_label(path, expected):
    assert smartLabel(path) == expected


def test_smart_labels_preserve_input_order():
    assert smartLabels(['/a/one.bw', '/b/.hidden']) == ['one', '.hidden']


# From audit followup 2026 09 12.

@pytest.fixture()
def coverage_isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path / 'config'))


def write_track(path, values):
    with pyBigWig.open(str(path), 'w') as bw:
        bw.addHeader([('chr1', len(values))], maxZooms=0)
        starts = [i for i, value in enumerate(values) if np.isfinite(value)]
        bw.addEntries(['chr1'] * len(starts), starts, ends=[i + 1 for i in starts], values=[float(values[i]) for i in starts])


def matrix(path):
    return Matrix.load(str(path), threads=1)


@pytest.mark.usefixtures('coverage_isolated_config')
def test_sparse_finite_track_does_not_warn_of_nonfinite_input(tmp_path, capfd):
    track, bed = (tmp_path / 'sparse.bw', tmp_path / 'regions.bed')
    values = np.full(100, np.nan)
    values[20:30] = 7
    write_track(track, values)
    bed.write_text('chr1\t10\t40\tr\t0\t+\n')
    computeMatrix.main(['reference-point', '-S', str(track), '-R', str(bed), '-o', str(tmp_path / 'matrix.gz'), '-b', '0', '-a', '30', '-bs', '10', '--quiet'])
    assert 'non-finite' not in capfd.readouterr().err


@pytest.mark.usefixtures('coverage_isolated_config')
@pytest.mark.parametrize('statistic', ['mean', 'median', 'min', 'max', 'sum'])
@pytest.mark.parametrize('mode', ['TSS', 'TES', 'center', 'scaled'])
def test_scaled_and_reference_options_match_original_with_sparse_signed_signal(tmp_path, statistic, mode):
    track, bed = (tmp_path / 'signal.bw', tmp_path / 'regions.bed')
    values = np.sin(np.arange(500) * 0.31) * 10
    values[::11] = np.nan
    write_track(track, values)
    bed.write_text('chr1\t50\t160\tplus\t2\t+\nchr1\t230\t360\tminus\t1\t-\n')
    common = ['scale-regions' if mode == 'scaled' else 'reference-point', '-S', str(track), '-R', str(bed), '-a', '30', '-b', '20', '-bs', '5', '--averageTypeBins', statistic, '--missingDataAsZero', '--scale', '-2', '--sortRegions', 'descend', '--sortUsing', 'sum', '--sortUsingSamples', '1', '--minThreshold', '-1000', '--maxThreshold', '1000', '--skipZeros', '--quiet', '-p', '1']
    if mode == 'scaled':
        common += ['--regionBodyLength', '40', '--unscaled5prime', '10', '--unscaled3prime', '15']
    else:
        common += ['--referencePoint', mode, '--nanAfterEnd']
    plus, upstream = (tmp_path / 'plus.gz', tmp_path / 'upstream.gz')
    computeMatrix.main(common + ['-o', str(plus)])
    original('computeMatrix', common + ['-o', str(upstream)])
    left, right = (matrix(plus), matrix(upstream))
    assert left.regions == right.regions
    assert left.header.sample_boundaries == right.header.sample_boundaries
    np.testing.assert_allclose(left.values, right.values, rtol=3e-07, atol=2.01e-06, equal_nan=True)


# From audit regressions.

@pytest.fixture()
def io_native_backends(monkeypatch):
    monkeypatch.delenv('DTP_BAM_BACKEND', raising=False)


def sparse_inputs(root):
    source = root / 'source.bw'
    with pyBigWig.open(str(source), 'w') as bw:
        bw.addHeader([('chr1', 100)])
        bw.addEntries(['chr1', 'chr1'], [10, 50], ends=[20, 70], values=[100.0, 1.0])
    bed = root / 'regions.bed'
    bed.write_text('chr1\t10\t30\tbad\t0\t+\nchr1\t50\t70\tgood\t0\t+\n')
    return ['reference-point', '-S', str(source), '-R', str(bed), '-a', '20', '-b', '0', '--binSize', '10', '--sortRegions', 'keep']


@pytest.mark.usefixtures('io_native_backends')
def test_max_threshold_applies_to_partly_missing_rows(tmp_path):
    output = tmp_path / 'matrix.gz'
    computeMatrix.main(sparse_inputs(tmp_path) + ['--maxThreshold', '10', '-o', str(output)])
    matrix = Matrix.load(output, threads=1)
    assert [r[2] for r in matrix.regions] == ['good']


# From general audit regressions.

def alias_inputs(tmp_path, names, reverse):
    paths = [tmp_path / 'first.bw', tmp_path / 'second.bw']
    if reverse:
        names = names[::-1]
    for path, chroms, value in zip(paths, names, [2.0, 7.0]):
        with pyBigWig.open(str(path), 'w') as bw:
            bw.addHeader([(name, 100) for name in chroms])
            bw.addEntries(list(chroms), [0] * len(chroms), ends=[100] * len(chroms), values=[value] * len(chroms))
    return paths


def compute_aliases(tmp_path, names, queries, reverse=False):
    paths = alias_inputs(tmp_path, names, reverse)
    bed, out = (tmp_path / 'regions.bed', tmp_path / 'out.gz')
    bed.write_text(''.join((f'{c}\t10\t20\tr{i}\t0\t+\n' for i, c in enumerate(queries))))
    computeMatrix.main(['reference-point', '-S', *map(str, paths), '-R', str(bed), '-o', str(out), '-b', '0', '-a', '10', '-bs', '10', '--sortRegions', 'keep', '--quiet'])
    _, rows, values = read_matrix(out)
    assert [row[3] for row in rows] == [f'r{i}' for i in range(len(queries))]
    np.testing.assert_array_equal(values, [[2, 7]] * len(queries))


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('kind', ['mixed', 'mitochondria'])
def test_g04_task_planning_keeps_all_aliased_chromosomes(tmp_path, kind, reverse):
    if kind == 'mixed':
        names, queries = ([('chr1', 'chr2'), ('1', 'chr2')], ['chr1', 'chr2'])
    else:
        names, queries = ([('chrM',), ('MT',)], ['chrM'])
    compute_aliases(tmp_path, names, queries, reverse)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('kind', ['exact', 'uniform_alias'])
def test_exact_and_uniform_chromosome_aliases_work(tmp_path, kind, reverse):
    names = [('chr1', 'chr2'), ('chr1', 'chr2') if kind == 'exact' else ('1', '2')]
    compute_aliases(tmp_path, names, ['chr1', 'chr2'], reverse)


def test_chromosome_planner_reads_tables_once_and_checks_exact_lengths():
    from deeptoolsr.compute import _get_chrom_sizes
    tables = {'a': [('chr1', 100), ('chr2', 100)], 'b': [('chr1', 90), ('1', 100), ('2', 100)]}
    calls = []

    def reader(path):
        calls.append(path)
        return tables[path]
    common, unmatched = _get_chrom_sizes(['a', 'b', 'a'], reader)
    assert calls == ['a', 'b']
    assert common == [('chr2', 100)]
    assert ('chr1', 100) in unmatched


def test_chromosome_planner_preserves_distinct_exact_names():
    from deeptoolsr.compute import _get_chrom_sizes
    tables = {'a': [('chr1', 100), ('1', 100)], 'b': [('1', 100)]}
    common, unmatched = _get_chrom_sizes(['a', 'b'], tables.__getitem__)
    assert common == [('1', 100), ('chr1', 100)]
    assert unmatched == set()


@pytest.mark.parametrize('kind', ['absent', 'different_length'])
def test_no_common_chromosomes_still_fail(kind):
    from deeptoolsr.compute import _get_chrom_sizes
    tables = {'a': [('chr1', 100)], 'b': [('chr2', 100)] if kind == 'absent' else [('1', 99)]}
    with pytest.raises(SystemExit, match='No common chromosomes'):
        _get_chrom_sizes(['a', 'b'], tables.__getitem__)


def test_alias_task_does_not_duplicate_bed_rows_or_override_exact_signal(tmp_path):
    first, second = (tmp_path / 'first.bw', tmp_path / 'second.bw')
    with pyBigWig.open(str(first), 'w') as bw:
        bw.addHeader([('1', 100), ('chr1', 100)])
        bw.addEntries(['1', 'chr1'], [0, 0], ends=[100, 100], values=[100.0, 2.0])
    with pyBigWig.open(str(second), 'w') as bw:
        bw.addHeader([('1', 100)])
        bw.addEntries(['1'], [0], ends=[100], values=[7.0])
    bed, out = (tmp_path / 'regions.bed', tmp_path / 'out.gz')
    bed.write_text('chr1\t10\t20\tr\t0\t+\n')
    computeMatrix.main(['reference-point', '-S', str(first), str(second), '-R', str(bed), '-o', str(out), '-b', '0', '-a', '10', '-bs', '10', '--sortRegions', 'keep', '--quiet'])
    _, rows, values = read_matrix(out)
    assert len(rows) == 1 and rows[0][0] == 'chr1'
    np.testing.assert_array_equal(values, [[2, 7]])


# From general second audit regressions.

def signal(tmp_path):
    path = tmp_path / 'signal.bw'
    with pyBigWig.open(str(path), 'w') as bw:
        bw.addHeader([('chr1', 100)])
        bw.addEntries(['chr1', 'chr1'], [0, 50], ends=[50, 100], values=[2.0, 7.0])
    return path


def matrix_args(bw, regions, output, extra=()):
    return ['reference-point', '-S', str(bw), '-R', str(regions), '-o', str(output), '-b', '0', '-a', '10', '-bs', '10', *extra]


def assert_rows(path, names=('one', 'two'), values=(2.0, 7.0)):
    header, rows, observed = read_matrix(path)
    assert [row[3] for row in rows] == list(names)
    np.testing.assert_array_equal(observed, np.asarray(values).reshape(-1, 1))
    return header


def compressed_bed(tmp_path, compression):
    text = b'chr1\t60\t70\ttwo\t0\t+\nchr1\t10\t20\tone\t0\t+\n'
    path = tmp_path / ('regions.bed' + ('.' + compression if compression else ''))
    path.write_bytes(gzip.compress(text) if compression == 'gz' else bz2.compress(text) if compression == 'bz2' else text)
    return path


@pytest.mark.parametrize('compression', ['gz', 'bz2'])
@pytest.mark.parametrize('route', ['compute', 'sort'])
def test_h02_compressed_bed_input_order(tmp_path, compression, route):
    bw, regions = (signal(tmp_path), compressed_bed(tmp_path, compression))
    out = tmp_path / 'out.gz'
    if route == 'compute':
        computeMatrix.main(matrix_args(bw, regions, out))
    else:
        source = tmp_path / 'source.gz'
        computeMatrix.main(matrix_args(bw, regions, source, ['--sortRegions', 'no']))
        cmo.main(['sort', '-m', str(source), '-R', str(regions), '-o', str(out)])
    assert_rows(out, ['two', 'one'], [7, 2])


@pytest.mark.parametrize('compression', ['gz', 'bz2'])
def test_compressed_bed_native_extraction_succeeds_without_order_restoration(tmp_path, compression):
    bw, regions = (signal(tmp_path), compressed_bed(tmp_path, compression))
    out = tmp_path / 'out.gz'
    computeMatrix.main(matrix_args(bw, regions, out, ['--sortRegions', 'no']))
    assert_rows(out)


@pytest.mark.parametrize('route', ['compute', 'sort'])
def test_plain_bed_input_order_control(tmp_path, route):
    bw, regions = (signal(tmp_path), compressed_bed(tmp_path, None))
    out = tmp_path / 'out.gz'
    if route == 'compute':
        computeMatrix.main(matrix_args(bw, regions, out))
    else:
        source = tmp_path / 'source.gz'
        computeMatrix.main(matrix_args(bw, regions, source, ['--sortRegions', 'no']))
        cmo.main(['sort', '-m', str(source), '-R', str(regions), '-o', str(out)])
    assert_rows(out, ['two', 'one'], [7, 2])


def transcript_gtf(tmp_path, leading_gene=False):
    path = tmp_path / 'regions.gtf'
    lines = []
    if leading_gene:
        lines.append('chr1\ttest\tgene\t1\t100\t.\t+\t.\tgene_id "gene";\n')
    for name, start in [('one', 11), ('two', 61)]:
        lines.append(f'chr1\ttest\tmRNA\t{start}\t{start + 9}\t.\t+\t.\tgene_id "gene"; transcript_id "{name}";\n')
    path.write_text(''.join(lines))
    return path


@pytest.mark.parametrize('feature', ['mRNA', 'MRNA'])
@pytest.mark.parametrize('leading_gene', [False, True])
@pytest.mark.parametrize('route', ['compute', 'sort'])
def test_h03_mixed_case_gtf_feature_preserves_all_transcripts(tmp_path, feature, leading_gene, route):
    bw, regions = (signal(tmp_path), transcript_gtf(tmp_path, leading_gene))
    out = tmp_path / 'out.gz'
    if route == 'compute':
        computeMatrix.main(matrix_args(bw, regions, out, ['--transcriptID', feature]))
    else:
        source = tmp_path / 'source.gz'
        computeMatrix.main(matrix_args(bw, regions, source, ['--transcriptID', feature, '--sortRegions', 'no']))
        cmo.main(['sort', '-m', str(source), '-R', str(regions), '-o', str(out), '--transcriptID', feature])
    assert_rows(out)


@pytest.mark.parametrize('leading_gene', [False, True])
@pytest.mark.parametrize('sort', ['keep', 'no'])
def test_gtf_case_control(tmp_path, leading_gene, sort):
    bw, regions = (signal(tmp_path), transcript_gtf(tmp_path, leading_gene))
    out = tmp_path / 'out.gz'
    computeMatrix.main(matrix_args(bw, regions, out, ['--transcriptID', 'mrna' if sort == 'keep' else 'mRNA', '--sortRegions', sort]))
    assert_rows(out)


@pytest.mark.parametrize('compression', ['', 'gz', 'bz2'])
@pytest.mark.parametrize('grouping', ['markers', 'column'])
@pytest.mark.parametrize('route', ['compute', 'sort'])
def test_region_text_iterator_handles_headers_comments_and_groups(tmp_path, compression, grouping, route):
    if grouping == 'markers':
        text = '# preliminary comment\ntrack name=example\nchr1\t60\t70\ttwo\t0\t+\nchr1\t10\t20\tone\t0\t+\n#first\nchr1\t20\t30\tthree\t0\t+\n#second\n'
    else:
        text = '#chrom\tstart\tend\tname\tscore\tstrand\tdeepTools_group\nchr1\t60\t70\ttwo\t0\t+\tfirst\nchr1\t20\t30\tthree\t0\t+\tsecond\n# ignored comment\nchr1\t10\t20\tone\t0\t+\tfirst\n'
    bed = tmp_path / ('regions.bed' + ('.' + compression if compression else ''))
    data = text.encode()
    bed.write_bytes(gzip.compress(data) if compression == 'gz' else bz2.compress(data) if compression == 'bz2' else data)
    bw, out = (signal(tmp_path), tmp_path / 'out.gz')
    if route == 'compute':
        computeMatrix.main(matrix_args(bw, bed, out))
    else:
        source = tmp_path / 'source.gz'
        computeMatrix.main(matrix_args(bw, bed, source, ['--sortRegions', 'no']))
        cmo.main(['sort', '-m', str(source), '-R', str(bed), '-o', str(out)])
    header = assert_rows(out, ['two', 'one', 'three'], [7, 2, 2])
    assert header['group_labels'] == ['first', 'second']
    assert header['group_boundaries'] == [0, 2, 3]


@pytest.mark.parametrize('compression', ['gz', 'bz2'])
@pytest.mark.parametrize('route', ['compute', 'sort'])
def test_compressed_gtf_uses_one_feature_predicate(tmp_path, compression, route):
    original_gtf = transcript_gtf(tmp_path, leading_gene=True)
    gtf = tmp_path / ('compressed.gtf.' + compression)
    data = b'# annotation\n' + original_gtf.read_bytes()
    gtf.write_bytes(gzip.compress(data) if compression == 'gz' else bz2.compress(data))
    bw, out = (signal(tmp_path), tmp_path / 'out.gz')
    if route == 'compute':
        computeMatrix.main(matrix_args(bw, gtf, out, ['--transcriptID', 'MRNA']))
    else:
        source = tmp_path / 'source.gz'
        computeMatrix.main(matrix_args(bw, gtf, source, ['--transcriptID', 'MRNA', '--sortRegions', 'no']))
        cmo.main(['sort', '-m', str(source), '-R', str(gtf), '-o', str(out), '--transcriptID', 'MRNA'])
    assert_rows(out)


@pytest.mark.parametrize('verbose', [False, True])
@pytest.mark.parametrize('mismatch', ['missing_group', 'missing_row'])
def test_explicit_sort_validation_does_not_depend_on_verbosity(tmp_path, verbose, mismatch):
    from deeptoolsr.matrix import OwnedMatrix
    bw, bed = (signal(tmp_path), compressed_bed(tmp_path, None))
    source = tmp_path / 'source.gz'
    computeMatrix.main(matrix_args(bw, bed, source))
    selector = tmp_path / 'selector.bed'
    selector.write_text('chr1\t60\t70\tabsent\t0\t+\n' if mismatch == 'missing_row' else bed.read_text() + '#genes\nchr1\t20\t30\tabsent\t0\t+\n#missing\n')
    matrix = OwnedMatrix.load(source, 1)
    before = matrix.values.copy()
    with pytest.raises(ValueError, match='missing.*region group|no matching entries'):
        cmo.sortMatrix(matrix, [str(selector)], 'transcript', 'transcript_id', verbose=verbose)
    np.testing.assert_array_equal(matrix.values, before)
    out = tmp_path / 'preserved.gz'
    out.write_bytes(b'existing result')
    with pytest.raises(SystemExit):
        cmo.main(['sort', '-m', str(source), '-R', str(selector), '-o', str(out)])
    assert out.read_bytes() == b'existing result'


def test_full_order_restoration_keeps_native_values_allocation(tmp_path):
    from deeptoolsr.matrix import OwnedMatrix
    bw, bed, source = (signal(tmp_path), compressed_bed(tmp_path, None), tmp_path / 'source.gz')
    computeMatrix.main(matrix_args(bw, bed, source, ['--sortRegions', 'no']))
    matrix = OwnedMatrix.load(source, 1)
    before = matrix.values
    order = cmo.sortMatrix(matrix, [str(bed)], 'transcript', 'transcript_id')
    assert matrix.values is before
    np.testing.assert_array_equal(before[:, 0], [2, 7])
    np.testing.assert_array_equal(before[order, 0], [7, 2])
    assert [matrix.regions[index][2] for index in order] == ['two', 'one']


@pytest.mark.parametrize('quiet', [False, True])
def test_skipzeros_all_empty_keeps_valid_empty_matrix(tmp_path, quiet):
    bw, out = (tmp_path / 'zero.bw', tmp_path / 'out.gz')
    with pyBigWig.open(str(bw), 'w') as handle:
        handle.addHeader([('chr1', 100)])
        handle.addEntries(['chr1'], [0], ends=[100], values=[0.0])
    bed = compressed_bed(tmp_path, None)
    computeMatrix.main(matrix_args(bw, bed, out, ['--skipZeros', *(['--quiet'] if quiet else [])]))
    header, rows, _ = read_matrix(out)
    assert rows == []
    assert header['group_boundaries'] == [0, 0]


def test_no_overlapping_region_reports_empty_bed(tmp_path, capsys):
    bw, bed, out = (tmp_path / 'signal.bw', tmp_path / 'regions.bed',
                    tmp_path / 'out.gz')
    with pyBigWig.open(str(bw), 'w') as handle:
        handle.addHeader([('chr1', 100)])
        handle.addEntries(['chr1'], [0], ends=[100], values=[1.0])
    bed.write_text('chr2\t10\t20\tone\t0\t+\n')
    with pytest.raises(SystemExit) as error:
        computeMatrix.main(matrix_args(bw, bed, out, []))
    assert error.value.code == 1
    assert 'does not contain any valid regions' in capsys.readouterr().err
    assert not out.exists()


# From equivalence regressions.

def write_bigwig(path, chrom, values):
    values = np.asarray(values, dtype=np.float32)
    starts = np.flatnonzero(np.isfinite(values)).tolist()
    with pyBigWig.open(str(path), 'w') as bw:
        bw.addHeader([(chrom, len(values))])
        if starts:
            bw.addEntries([chrom] * len(starts), starts, ends=[x + 1 for x in starts], values=values[starts].tolist())


@pytest.mark.parametrize('order', [('chr1', '1'), ('1', 'chr1')])
@pytest.mark.parametrize('zeros', [False, True])
def test_e01_mixed_sample_chromosome_aliases(tmp_path, order, zeros):
    paths = [tmp_path / f'{chrom}.bw' for chrom in order]
    for path, chrom, value in zip(paths, order, [2.0, 7.0]):
        write_bigwig(path, chrom, np.full(100, value))
    bed = tmp_path / 'regions.bed'
    bed.write_text('chr1\t20\t40\tr\t0\t+\n')
    args = ['reference-point', '-S', *map(str, paths), '-R', str(bed), '-b', '0', '-a', '20', '--binSize', '10', '--sortRegions', 'keep', '--quiet', '-p', '1']
    if zeros:
        args += ['--missingDataAsZero']
    reference, output = (tmp_path / 'reference.gz', tmp_path / 'plus.gz')
    original('computeMatrix', args + ['-o', str(reference)])
    computeMatrix.main(args + ['-o', str(output)])
    expected = read_matrix(reference)[2]
    np.testing.assert_array_equal(expected, [[2, 2, 7, 7]])
    np.testing.assert_array_equal(read_matrix(output)[2], expected)


def geom_reference(values, pc):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    sign = np.sign(math.fsum(values) / len(values))
    shifted = sign * values + pc
    shifted = shifted[shifted > 0]
    if not len(shifted):
        return np.nan
    return sign * (math.exp(math.fsum(map(math.log, shifted)) / len(shifted)) - pc)


def statistic_reference(values, statistic, pc=1.0, trim=0.2):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    if statistic == 'geom_mean':
        return geom_reference(values, pc)
    if statistic == 'trim_mean':
        return stats.trim_mean(values, trim)
    return getattr(np, statistic)(values)


@pytest.mark.parametrize('statistic', ['mean', 'median', 'sum', 'min', 'max'])
@pytest.mark.parametrize('point', ['TSS', 'TES', 'center', 'scale'])
@pytest.mark.parametrize('zero', [False, True])
def test_multiexon_and_chromosome_edge_matrix_parity(tmp_path, statistic, point, zero):
    rng = np.random.default_rng(721)
    signal = rng.normal(1, 2, size=401).astype(np.float32)
    signal[::3] = np.nan
    signal[160:240] = np.nan
    bw, bed = (tmp_path / 'signal.bw', tmp_path / 'regions.bed')
    write_bigwig(bw, 'chr1', signal)
    bed.write_text(''.join((f'chr1\t{start}\t{start + 43}\tr{i}\t0\t{strand}\t{start}\t{start + 43}\t0\t2\t7,19\t0,24\n' for i, (start, strand) in enumerate(itertools.product([0, 31, 153, 355], ['+', '-'])))))
    mode = 'scale-regions' if point == 'scale' else 'reference-point'
    args = [mode, '-S', str(bw), '-R', str(bed), '-b', '14', '-a', '21', '--binSize', '7', '--averageTypeBins', statistic, '--metagene', '--sortRegions', 'keep', '--quiet', '-p', '1']
    args += ['--regionBodyLength', '63', '--unscaled5prime', '7', '--unscaled3prime', '7'] if point == 'scale' else ['--referencePoint', point, '--nanAfterEnd']
    if zero:
        args += ['--missingDataAsZero']
    ref, out = (tmp_path / 'reference.gz', tmp_path / 'plus.gz')
    original('computeMatrix', args + ['-o', str(ref)])
    computeMatrix.main(args + ['-o', str(out)])
    _, ref_regions, expected = read_matrix(ref)
    _, out_regions, actual = read_matrix(out)
    assert ref_regions == out_regions
    np.testing.assert_allclose(actual, expected, rtol=3e-07, atol=1.01e-06, equal_nan=True)


def test_e06_bed_export_preserves_relative_exon_coordinates(tmp_path):
    bw, bed, matrix, exported = [tmp_path / name for name in ['s.bw', 'r.bed', 'm.gz', 'export.bed']]
    write_bigwig(bw, 'chr1', np.arange(300))
    bed.write_text('chr1\t120\t190\tr\t0\t+\t120\t190\t0\t2\t20,30\t0,40\n')
    args = ['scale-regions', '-S', str(bw), '-R', str(bed), '-o', str(matrix), '--metagene', '--regionBodyLength', '50', '--binSize', '10', '--sortRegions', 'keep', '--quiet', '--outFileSortedRegions', str(exported)]
    computeMatrix.main(args)
    row = [line.split('\t') for line in exported.read_text().splitlines() if not line.startswith('#')][0]
    assert row[10] == '20,30'
    assert row[11] == '0,40'
    roundtrip = tmp_path / 'roundtrip.gz'
    args = args[:-2]
    args[args.index('-R') + 1] = str(exported)
    args[args.index('-o') + 1] = str(roundtrip)
    computeMatrix.main(args)
    _, expected_rows, expected_values = read_matrix(matrix)
    _, actual_rows, actual_values = read_matrix(roundtrip)
    assert actual_rows == expected_rows
    np.testing.assert_array_equal(actual_values, expected_values)


@pytest.mark.parametrize('statistic', ['mean', 'median', 'sum', 'min', 'max'])
def test_randomized_multizone_binning_matches_exact_integer_partition(tmp_path, statistic):
    signal = np.arange(1000, dtype=np.float32) % 73 - 20
    signal[::7] = np.nan
    bw = tmp_path / 'signal.bw'
    write_bigwig(bw, 'chr1', signal)
    reader = _compute_matrix_native.NativeBigWigReader([str(bw)])
    rng = np.random.default_rng(113)
    for _ in range(250):
        lengths, bins = rng.integers(1, 85, size=(2, 3))
        zones, expected = ([], [])
        for start, length, count in zip([10, 100, 300], lengths, bins):
            zones.append(([(start, int(start + length))], int(count)))
            for i in range(count):
                begin, end = (i * length // count, (i + 1) * length // count)
                end = max(end, begin + 1)
                values = signal[start + begin:start + end].astype(np.float64)
                expected.append(statistic_reference(values, statistic))
        actual = reader.bin_row('chr1', zones, [[0]], [[1.0]], False, 0, 0, statistic, False)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14, equal_nan=True)


@pytest.mark.parametrize('source,query', [('chr1', '1'), ('1', 'chr1'), ('chrM', 'MT'), ('MT', 'chrM')])
@pytest.mark.parametrize('zeros', [False, True])
def test_native_source_aliases_include_mitochondria(tmp_path, source, query, zeros):
    path = tmp_path / 's.bw'
    write_bigwig(path, source, np.full(30, 7))
    reader = _compute_matrix_native.NativeBigWigReader([str(path)])
    result = reader.bin_row(query, [([(0, 20)], 2)], [[0]], [[1.0]], False, 0, 0, 'mean', zeros)
    np.testing.assert_array_equal(result, [7, 7])


def test_native_chromosome_exact_match_precedes_alias(tmp_path):
    path = tmp_path / 's.bw'
    with pyBigWig.open(str(path), 'w') as bw:
        bw.addHeader([('chr1', 30), ('1', 30)])
        bw.addEntries(['chr1', '1'], [0, 0], ends=[30, 30], values=[2.0, 7.0])
    reader = _compute_matrix_native.NativeBigWigReader([str(path)])
    for chrom, expected in [('chr1', 2), ('1', 7)]:
        result = reader.bin_row(chrom, [([(0, 20)], 2)], [[0]], [[1.0]], False, 0, 0, 'mean', False)
        np.testing.assert_array_equal(result, [expected, expected])


# From numerical audit regressions.

@pytest.mark.parametrize('overflow', ['sum', 'scale', 'strands'])
@pytest.mark.parametrize('threads', [1, 2])
def test_n04_compute_overflow_does_not_replace_saved_result_with_nan(tmp_path, overflow, threads):
    source, bed, output = (tmp_path / 'large.bw', tmp_path / 'regions.bed', tmp_path / 'matrix.gz')
    with pyBigWig.open(str(source), 'w') as bw:
        bw.addHeader([('chr1', 2)])
        bw.addEntries(['chr1'], [0], ends=[2], values=[2e+38])
    bed.write_text(''.join((f'chr1\t0\t2\tr{i}\t0\t.\n' for i in range(8))))
    sentinel = b'PREVIOUS VALID RESULT\n'
    output.write_bytes(sentinel)
    extra = {'sum': ['--averageTypeBins', 'sum'], 'scale': ['--scalePlus', '2'], 'strands': ['--scoreFileNameMinus', str(source)]}[overflow]
    with pytest.raises(SystemExit, match='exceeds.*range'):
        computeMatrix.main(['reference-point', '-S', str(source), '-R', str(bed), '-a', '2', '-b', '0', '-bs', '2', '-o', str(output), '--quiet', '-p', str(threads)] + extra)
    assert output.read_bytes() == sentinel


def numerical_write_bigwig(path, values):
    with pyBigWig.open(str(path), 'w') as bw:
        bw.addHeader([('chr1', len(values))])
        bw.addEntries(['chr1'] * len(values), list(range(len(values))), ends=list(range(1, len(values) + 1)), values=list(map(float, values)))


def native_bins(path, indices, scales, owned=False, statistic='mean', zero_missing=False):
    operation = _compute_matrix_native.bin_bigwig_batch_owned if owned else _compute_matrix_native.bin_bigwig_batch
    return np.asarray(operation([str(path)], ['chr1'] * 4, [[([(0, 2)], 1)]] * 4, [[indices]] * 4, [[scales]] * 4, [False] * 4, [0] * 4, [0] * 4, statistic, zero_missing, 2))


@pytest.mark.parametrize('stage', ['scaling', 'combination'])
@pytest.mark.parametrize('owned', [False, True])
def test_n04_double_intermediate_overflow_propagates(tmp_path, stage, owned):
    path = tmp_path / 'source.bw'
    numerical_write_bigwig(path, [2 if stage == 'scaling' else 1] * 2)
    indices = [0] if stage == 'scaling' else [0, 0]
    scales = [np.finfo(np.float64).max] * len(indices)
    with pytest.raises(OverflowError, match=stage):
        native_bins(path, indices, scales, owned)
    reader = _compute_matrix_native.NativeBigWigReader([str(path)])
    with pytest.raises(OverflowError, match=stage):
        reader.bin_row('chr1', [([(0, 2)], 1)], [indices], [scales], False, 0, 0)


@pytest.mark.parametrize('owned', [False, True])
def test_n04_finite_cancellation_happens_before_float32_narrowing(tmp_path, owned):
    path = tmp_path / 'source.bw'
    numerical_write_bigwig(path, [2e+38] * 2)
    np.testing.assert_array_equal(native_bins(path, [0, 0], [2.0, -2.0], owned), 0.0)


@pytest.mark.parametrize('missing', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('zero_missing', [False, True])
def test_n04_nonfinite_input_remains_missing_or_zero(tmp_path, missing, zero_missing):
    path = tmp_path / 'missing.bw'
    numerical_write_bigwig(path, [missing] * 2)
    result = native_bins(path, [0], [2.0], zero_missing=zero_missing)
    if zero_missing:
        np.testing.assert_array_equal(result, 0.0)
    else:
        assert np.isnan(result).all()


# From parity audit regressions.

def values(path, chrom='chr1'):
    with pyBigWig.open(str(path)) as bw:
        return np.asarray(bw.values(chrom, 0, bw.chroms(chrom)))


@pytest.fixture
def parity_matrix_inputs(tmp_path):
    paths = []
    for sample in range(2):
        path = tmp_path / f's{sample}.bw'
        with pyBigWig.open(str(path), 'w') as bw:
            bw.addHeader([('chr1', 200)])
            starts = [i for i in range(200) if not 85 <= i < 100]
            bw.addEntries(['chr1'] * len(starts), starts, ends=[i + 1 for i in starts], values=[0.0 if 40 <= i < 55 else float((i % 11 + 1) * (sample + 1)) for i in starts])
        paths.append(str(path))
    bed = tmp_path / 'regions.bed'
    bed.write_text(''.join((f'chr1\t{s}\t{e}\tr{i}\t{i}\t{strand}\n' for i, (s, e, strand) in enumerate([(2, 23, '+'), (35, 53, '-'), (80, 95, '.'), (130, 151, '-'), (183, 198, '+')]))))
    return (paths, str(bed))


MATRIX_CASES = [(f'{point}-nan{nan}-zero{zero}', 'reference-point', ['--referencePoint', point] + (['--nanAfterEnd'] if nan else []) + (['--missingDataAsZero'] if zero else [])) for point in ('TSS', 'TES', 'center') for nan in (False, True) for zero in (False, True)]


MATRIX_CASES += [
    (
        f"unscaled{a}-{b}",
        "scale-regions",
        ["--unscaled5prime", str(a), "--unscaled3prime", str(b)],
    )
    for a, b in ((0, 0), (5, 10), (10, 5))
]
MATRIX_CASES += [
    (
        f"sort-{stat}-{order}",
        "reference-point",
        ["--sortRegions", order, "--sortUsing", stat, "--sortUsingSamples", "2"],
    )
    for stat in ("mean", "median", "min", "max", "sum", "region_length")
    for order in ("ascend", "descend")
]
MATRIX_CASES += [
    ("skipZeros", "reference-point", ["--skipZeros"]),
    ("scale", "reference-point", ["--scale", "-2"]),
]


@pytest.mark.parametrize('name,mode,extra', MATRIX_CASES, ids=[case[0] for case in MATRIX_CASES])
def test_compute_matrix_option_parity(tmp_path, parity_matrix_inputs, name, mode, extra):
    paths, bed = parity_matrix_inputs
    common = [mode, '-S', *paths, '-R', bed, '-b', '10', '-a', '30', '-bs', '5', '--sortRegions', 'keep', '--quiet', '-p', '1']
    if mode == 'scale-regions':
        common += ['--regionBodyLength', '20']
    common += extra
    upstream, plus = (tmp_path / 'original.gz', tmp_path / 'plus.gz')
    original('computeMatrix', common + ['-o', str(upstream)])
    computeMatrix.main(common + ['-o', str(plus)])
    _, orows, ov = read_matrix(upstream)
    _, prows, pv = read_matrix(plus)
    lookup = {tuple(row): i for i, row in enumerate(prows)}
    assert set(map(tuple, orows)) == set(lookup)
    np.testing.assert_allclose(pv[[lookup[tuple(row)] for row in orows]], ov, rtol=3e-07, atol=1.01e-06, equal_nan=True)
    if orows != prows:
        assert name.startswith('sort-mean-'), 'unexpected region reordering'
        expected_key = {tuple(row): np.nanmean(values[8:]) for row, values in zip(orows, ov)}
        np.testing.assert_allclose([expected_key[tuple(row)] for row in prows], [expected_key[tuple(row)] for row in orows], rtol=3e-07, atol=1e-06)


# Argument contract.

@pytest.mark.parametrize('extra,message', [(['--scaleMinus', '1'], 'unused without --scoreFileNameMinus'), (['--scaleAntisense', '1'], 'unused with --antisense skip'), (['--sortUsing', 'max'], 'unused unless --sortRegions is ascend or descend'), (['--sortUsingSamples', '1'], 'unused unless --sortRegions is ascend or descend'), (['--quantiles', '1'], 'unused unless --sortRegions is ascend or descend')])
def test_explicit_inert_compute_matrix_options_warn(tmp_path, capsys, extra, message):
    from deeptoolsr import computeMatrix
    args = computeMatrix.process_args(['reference-point', '-S', str(tmp_path / 'signal.bw'), '-R', str(tmp_path / 'regions.bed'), '-o', str(tmp_path / 'matrix.gz')] + extra)
    assert args is not None
    assert message in capsys.readouterr().err
