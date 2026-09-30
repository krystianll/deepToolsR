from deeptoolsr.matrix import Matrix
from deeptoolsr.computeMatrix import main, process_args
import numpy as np
import pytest

from tests.helpers.bigwig import pyBigWig

try:
    from deeptoolsr import _compute_matrix_native
except ImportError:
    _compute_matrix_native = None


def _write_constant_bigwig(path, value):
    bw = pyBigWig.open(str(path), 'w')
    bw.addHeader([('chr1', 100)])
    bw.addEntries(['chr1'], [0], ends=[100], values=[value])
    bw.close()


def test_paired_minus_bigwig_selection_and_scaling(tmp_path):
    plus = tmp_path / 'plus.bw'
    minus = tmp_path / 'minus.bw'
    regions = tmp_path / 'regions.bed'
    output = tmp_path / 'matrix.gz'
    _write_constant_bigwig(plus, 2.0)
    _write_constant_bigwig(minus, 5.0)
    regions.write_text('chr1\t20\t30\tplus\t0\t+\n'
                       'chr1\t20\t30\tminus\t0\t-\n')

    main(['reference-point', '-S', str(plus),
          '--scoreFileNameMinus', str(minus), '-R', str(regions),
          '-o', str(output), '-b', '0', '-a', '10', '--binSize', '10',
          '--scalePlus', '3', '--scaleMinus', '4', '--scale', '2',
          '--sortRegions', 'keep', '--quiet'])

    hm = Matrix.load(str(output), threads=1)
    rows = {region[2]: np.ma.asarray(hm.values[index]).filled(np.nan)
            for index, region in enumerate(hm.regions)}
    np.testing.assert_allclose(rows['plus'], [12.0])
    np.testing.assert_allclose(rows['minus'], [40.0])
    assert list(hm.header.sample_labels) == ['plus']


def test_track_and_global_scales_multiply_calculated_bins_last(tmp_path):
    signal = tmp_path / 'signal.bw'
    regions = tmp_path / 'regions.bed'
    output = tmp_path / 'matrix.gz'
    with pyBigWig.open(str(signal), 'w') as bw:
        bw.addHeader([('chr1', 2)])
        bw.addEntries(['chr1', 'chr1'], [0, 1], ends=[1, 2],
                      values=[1., 10.])
    regions.write_text('chr1\t0\t1\tregion\t0\t+\n')

    main([
        'reference-point', '-S', str(signal), '-R', str(regions),
        '-o', str(output), '-b', '0', '-a', '2', '--binSize', '2',
        '--averageTypeBins', 'min', '--scalePlus', '-1', '--scale', '2',
        '--sortRegions', 'keep', '--quiet',
    ])

    hm = Matrix.load(str(output), threads=1)
    # Binning yields min([1, 10]) == 1; scalePlus and global scale then
    # multiply the completed bin. Scaling bases before min would yield -20.
    np.testing.assert_array_equal(
        np.ma.filled(hm.values, np.nan), [[-2.]])


def test_minus_bigwig_count_must_match_primary_count(tmp_path):
    output = tmp_path / 'matrix.gz'
    with pytest.raises(SystemExit):
        process_args(['reference-point', '-S', 'one.bw', 'two.bw',
                      '--scoreFileNameMinus', 'minus.bw', '-R', 'regions.bed',
                      '-o', str(output)])


def test_strand_scale_options(tmp_path):
    output = tmp_path / 'matrix.gz'
    args = process_args(['reference-point', '-S', 'plus.bw',
                         '--scoreFileNameMinus', 'minus.bw',
                         '-R', 'regions.bed', '-o', str(output),
                         '--scalePlus', '2', '--scaleMinus', '3'])
    assert args.scalePlus == 2
    assert args.scaleMinus == 3


@pytest.mark.parametrize(
    'extra_args, expected_message',
    [(['--antisense', 'as_samples'], '--scoreFileNameMinus'),
     (['--unstranded', 'as_plus'], '--scoreFileNameMinus'),
     (['--scaleMinus', '-1'], '--scoreFileNameMinus'),
     (['--scaleAntisense', '-1'], '--antisense')])
def test_paired_track_options_require_valid_context(extra_args,
                                                    expected_message,
                                                    tmp_path,
                                                    capsys):
    output = tmp_path / 'matrix.gz'
    with pytest.raises(SystemExit):
        process_args(['reference-point', '-S', 'plus.bw',
                      '-R', 'regions.bed', '-o', str(output)] + extra_args)
    assert expected_message in capsys.readouterr().err


def test_antisense_as_samples_and_unstranded_sum(tmp_path):
    plus = tmp_path / 'plus.bw'
    minus = tmp_path / 'minus.bw'
    regions = tmp_path / 'regions.bed'
    output = tmp_path / 'matrix.gz'
    _write_constant_bigwig(plus, 2.0)
    _write_constant_bigwig(minus, -5.0)
    regions.write_text('chr1\t20\t30\tplus\t0\t+\n'
                       'chr1\t20\t30\tminus\t0\t-\n'
                       'chr1\t20\t30\tunstranded\t0\t*\n')

    main(['reference-point', '-S', str(plus),
          '--scoreFileNameMinus', str(minus), '-R', str(regions),
          '-o', str(output), '-b', '0', '-a', '10', '--binSize', '10',
          '--scaleMinus', '-1', '--scaleAntisense', '-1',
          '--antisense', 'as_samples', '--unstranded', 'sum_strands',
          '--sortRegions', 'keep', '--quiet'])

    hm = Matrix.load(str(output), threads=1)
    rows = {region[2]: np.ma.asarray(hm.values[index]).filled(np.nan)
            for index, region in enumerate(hm.regions)}
    np.testing.assert_allclose(rows['plus'], [2.0, -5.0])
    np.testing.assert_allclose(rows['minus'], [5.0, -2.0])
    np.testing.assert_allclose(rows['unstranded'], [7.0, -7.0])
    assert list(hm.header.sample_labels) == ['plus', 'plus_antisense']


def test_antisense_as_groups_duplicates_without_flipping_strand(tmp_path):
    plus = tmp_path / 'plus.bw'
    minus = tmp_path / 'minus.bw'
    regions = tmp_path / 'regions.bed'
    output = tmp_path / 'matrix.gz'
    _write_constant_bigwig(plus, 2.0)
    _write_constant_bigwig(minus, -5.0)
    regions.write_text('chr1\t20\t30\tplus\t0\t+\n'
                       'chr1\t20\t30\tminus\t0\t-\n'
                       'chr1\t20\t30\tunstranded\t0\t*\n')

    main(['reference-point', '-S', str(plus),
          '--scoreFileNameMinus', str(minus), '-R', str(regions),
          '-o', str(output), '-b', '0', '-a', '10', '--binSize', '10',
          '--scaleMinus', '-1', '--scaleAntisense', '-1',
          '--antisense', 'as_groups', '--unstranded', 'sum_strands',
          '--sortRegions', 'keep', '--quiet'])

    hm = Matrix.load(str(output), threads=1)
    rows = {region[2]: np.ma.asarray(hm.values[index]).filled(np.nan)
            for index, region in enumerate(hm.regions)}
    assert len(hm.regions) == 6, [region[2] for region in hm.regions]
    assert len(rows) == 6, list(rows)
    # The antisense group reads the opposite bigwig strand (scaled by
    # --scaleAntisense), exactly as before.
    np.testing.assert_allclose(rows['plus'], [2.0])
    np.testing.assert_allclose(rows['minus'], [5.0])
    np.testing.assert_allclose(rows['unstranded'], [7.0])
    np.testing.assert_allclose(rows['plus_antisense'], [-5.0])
    np.testing.assert_allclose(rows['minus_antisense'], [-2.0])
    np.testing.assert_allclose(rows['unstranded_antisense'], [-7.0])
    assert list(hm.header.group_labels) == ['genes', 'genes_antisense']
    # The antisense group's regions keep their ORIGINAL strand: flipping it
    # would swap which end is treated as TSS vs TES for the antisense copy.
    # Only the bigwig source selection differs, not the region's orientation.
    strands = {region[2]: region[4] for region in hm.regions}
    assert strands['plus_antisense'] == '+'
    assert strands['minus_antisense'] == '-'
    # deeptoolsintervals normalizes non-+/- BED strands to ".".
    assert strands['unstranded_antisense'] == '.'


def test_antisense_as_groups_preserves_bin_orientation(tmp_path):
    """The antisense copy must use the same upstream/downstream direction
    as the coding copy, so TSS/TES are not swapped and bins are not
    reversed -- only the bigwig strand read is flipped."""
    plus = tmp_path / 'plus.bw'
    minus = tmp_path / 'minus.bw'
    regions = tmp_path / 'regions.bed'
    output = tmp_path / 'matrix.gz'
    # Two distinct, position-dependent bins on each strand so a reversal
    # would be detectable: bin 0 differs from bin 1 on both bigwigs.
    bw_plus = pyBigWig.open(str(plus), 'w')
    bw_plus.addHeader([('chr1', 100)])
    bw_plus.addEntries(['chr1'], [10], ends=[20], values=[1.0])
    bw_plus.addEntries(['chr1'], [20], ends=[30], values=[9.0])
    bw_plus.close()
    bw_minus = pyBigWig.open(str(minus), 'w')
    bw_minus.addHeader([('chr1', 100)])
    bw_minus.addEntries(['chr1'], [10], ends=[20], values=[3.0])
    bw_minus.addEntries(['chr1'], [20], ends=[30], values=[7.0])
    bw_minus.close()
    # A minus-strand region at [10, 20) with reference point TSS at 20:
    # upstream (toward higher coordinates, bin covering [20,30)) comes
    # first in the profile, then downstream (bin covering [10,20)).
    regions.write_text('chr1\t10\t20\tminus\t0\t-\n')

    main(['reference-point', '-S', str(plus),
          '--scoreFileNameMinus', str(minus), '-R', str(regions),
          '-o', str(output), '-b', '10', '-a', '10', '--binSize', '10',
          '--antisense', 'as_groups',
          '--sortRegions', 'keep', '--quiet'])

    hm = Matrix.load(str(output), threads=1)
    rows = {region[2]: np.ma.asarray(hm.values[index]).filled(np.nan)
            for index, region in enumerate(hm.regions)}
    # Coding copy reads the minus bigwig (matches the region's own '-'
    # strand): upstream bin is [20,30)=7.0, downstream bin is [10,20)=3.0.
    np.testing.assert_allclose(rows['minus'], [7.0, 3.0])
    # Antisense copy reads the opposite (plus) bigwig, but keeps the SAME
    # upstream/downstream bin order as the coding copy -- not reversed.
    np.testing.assert_allclose(rows['minus_antisense'], [9.0, 1.0])


@pytest.mark.parametrize('mode, expected', [('as_plus', 2.0),
                                            ('as_minus', 5.0)])
def test_unstranded_orientation_source(mode, expected, tmp_path):
    plus = tmp_path / 'plus.bw'
    minus = tmp_path / 'minus.bw'
    regions = tmp_path / 'regions.bed'
    output = tmp_path / 'matrix.gz'
    _write_constant_bigwig(plus, 2.0)
    _write_constant_bigwig(minus, -5.0)
    regions.write_text('chr1\t20\t30\tunstranded\t0\t*\n')

    main(['reference-point', '-S', str(plus),
          '--scoreFileNameMinus', str(minus), '-R', str(regions),
          '-o', str(output), '-b', '0', '-a', '10', '--binSize', '10',
          '--scaleMinus', '-1', '--unstranded', mode,
          '--sortRegions', 'keep', '--quiet'])

    hm = Matrix.load(str(output), threads=1)
    np.testing.assert_allclose(np.ma.filled(hm.values, np.nan), [[expected]])


@pytest.mark.skipif(_compute_matrix_native is None,
                    reason='native compute extension is not built')
def test_native_batched_libbigwig_binning(tmp_path):
    plus = tmp_path / 'plus.bw'
    minus = tmp_path / 'minus.bw'
    _write_constant_bigwig(plus, 2.0)
    _write_constant_bigwig(minus, -5.0)

    matrix = _compute_matrix_native.bin_bigwig_batch(
        [str(plus), str(minus)],
        ['chr1', 'chr1'],
        [[([(20, 30)], 1)], [([(20, 30)], 1)]],
        [[[0], [1], [0, 1]], [[1], [0], [0, 1]]],
        [[[1.0], [-1.0], [1.0, -1.0]],
         [[-1.0], [1.0], [1.0, -1.0]]],
        [False, True], [0, 0], [0, 0],
        'mean', False, 2)
    np.testing.assert_allclose(matrix,
                               [[2.0, 5.0, 7.0],
                                [5.0, 2.0, 7.0]])


@pytest.mark.skipif(_compute_matrix_native is None,
                    reason='native compute extension is not built')
def test_native_bigwig_chromosome_length_above_signed_int32(tmp_path):
    path = tmp_path / 'huge-chromosome.bw'
    with pyBigWig.open(str(path), 'w') as bw:
        bw.addHeader([('chrHuge', 3_000_000_000)])
        bw.addEntries(['chrHuge'], [10], ends=[20], values=[7.0])

    value = _compute_matrix_native.NativeBigWigReader([str(path)]).bin_row(
        'chrHuge', [([(10, 20)], 1)], [[0]], [[1.0]], False, 0, 0,
        'mean', False)
    np.testing.assert_allclose(value, [7.0])


@pytest.mark.skipif(_compute_matrix_native is None,
                    reason='native compute extension is not built')
def test_native_scaled_region_binning_matches_python_floor_boundaries(tmp_path):
    path = tmp_path / 'varying.bw'
    bw = pyBigWig.open(str(path), 'w')
    bw.addHeader([('chr1', 10)])
    bw.addEntries(['chr1'] * 10, list(range(10)),
                  ends=list(range(1, 11)),
                  values=[float(value) for value in range(10)])
    bw.close()

    native = _compute_matrix_native.NativeBigWigReader([str(path)]).bin_row(
        'chr1', [([(0, 5)], 3)], [[0]], [[1.0]], False, 0, 0,
        'mean', False)
    np.testing.assert_allclose(native, [0.0, 1.5, 3.5])


@pytest.mark.skipif(_compute_matrix_native is None,
                    reason='native compute extension is not built')
def test_legacy_disable_setting_cannot_change_strand_semantics(tmp_path, monkeypatch):
    plus = tmp_path / 'plus.bw'
    minus = tmp_path / 'minus.bw'
    regions = tmp_path / 'regions.bed'
    native_output = tmp_path / 'native.gz'
    python_output = tmp_path / 'python.gz'
    _write_constant_bigwig(plus, 2.0)
    _write_constant_bigwig(minus, -5.0)
    regions.write_text('chr1\t20\t60\tplus\t0\t+\n'
                       'chr1\t20\t60\tminus\t0\t-\n'
                       'chr1\t20\t60\tunstranded\t0\t*\n')

    common = ['scale-regions', '-S', str(plus),
              '--scoreFileNameMinus', str(minus), '-R', str(regions),
              '-b', '10', '-a', '10', '--regionBodyLength', '20',
              '--binSize', '10', '--scalePlus', '3', '--scaleMinus', '-2',
              '--scaleAntisense', '-1', '--antisense', 'as_samples',
              '--unstranded', 'sum_strands', '--sortRegions', 'keep',
              '--quiet']
    main(common + ['-o', str(native_output)])
    monkeypatch.setenv('DEEPTOOLSR_DISABLE_NATIVE_COMPUTE', '1')
    main(common + ['-o', str(python_output)])

    native_hm = Matrix.load(str(native_output), threads=1)
    python_hm = Matrix.load(str(python_output), threads=1)
    np.testing.assert_allclose(np.ma.filled(native_hm.values, np.nan),
                               np.ma.filled(python_hm.values, np.nan),
                               equal_nan=True)
    assert native_hm.header.sample_labels == python_hm.header.sample_labels
