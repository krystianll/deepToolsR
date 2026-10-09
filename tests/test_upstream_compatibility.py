"""Comparisons against pinned ORIGINAL deepTools, never a R fallback.

Install the test-only extra ``deepToolsR[reference]`` or put the unpacked
3.5.6 wheel on PYTHONPATH. The normal suite skips this module without deepTools.
"""
from pathlib import Path
import gzip
import json
import subprocess
import sys

import numpy as np
import pytest
from tests.helpers.bigwig import pyBigWig

if sys.platform == 'win32':
    pytest.skip('Original deepTools is unsupported on Windows', allow_module_level=True)
pytest.importorskip('deeptools')
from deeptoolsr import computeMatrix as plus_compute, bamCoverage as plus_coverage, stats as kernels  # noqa: E402


def original(task, payload):
    # Separate interpreters prevent the two forks' matplotlib colormap
    # registration and process-global settings from contaminating one another.
    code = """
import json, sys, importlib.metadata
from deeptools import heatmapper, computeMatrix
import numpy as np
assert importlib.metadata.version('deepTools') == '3.5.6'
assert 'deeptoolsr' not in heatmapper.__file__
task, payload = json.loads(sys.stdin.read())
if task == 'profile':
    values = heatmapper.heatmapper.matrix_avg(np.array(payload[0], dtype=np.float64), payload[1])
    print(json.dumps(np.ma.filled(values, 0 if payload[1] == 'sum' else np.nan).tolist()))
elif task == 'compute':
    sys.argv = ['computeMatrix'] + payload
    computeMatrix.main(payload)
elif task == 'coverage':
    from deeptools import bamCoverage
    sys.argv = ['bamCoverage'] + payload
    bamCoverage.main(payload)
else:
    print(heatmapper.__file__)
"""
    result = subprocess.run([sys.executable, '-c', code], input=json.dumps([task, payload]),
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_reference_is_original_pinned_release():
    assert '/deeptools/heatmapper.py' in original('identity', None)


@pytest.mark.parametrize('operation', ['mean', 'median', 'min', 'max', 'sum', 'std'])
@pytest.mark.parametrize('shape', [(5, 9), (25, 7), (1, 6)])
def test_profile_statistics_match_original_deeptools(operation, shape):
    rng = np.random.default_rng(201)
    x = rng.normal(size=shape).astype(np.float32)
    x[::4, 2] = np.nan
    # Upstream matrix_avg performs masked reductions on float64 matrices.
    expected = np.array(json.loads(original('profile', [x.tolist(), operation])))
    actual = kernels.reduce(x, operation, 0, threads=1)
    # Native sum retains its documented all-missing identity; profile masks
    # remain checked separately through finite counts.
    expected = np.ma.filled(expected, 0 if operation == 'sum' else np.nan)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=2e-15, equal_nan=True)


def read_matrix(path):
    with gzip.open(path, 'rt') as handle:
        header = json.loads(handle.readline()[1:])
        rows = [line.rstrip().split('\t') for line in handle if line.strip()]
    return header, [row[:6] for row in rows], np.array([[float(v) for v in row[6:]] for row in rows])


@pytest.fixture
def genomic_inputs(tmp_path):
    paths = []
    for sample in range(2):
        path = tmp_path / f'sample{sample}.bw'
        starts = [i for i in range(400) if (i // 7 + sample) % 6 != 0]
        values = [float((i % 17 - 4) * (sample + 1) / 3) for i in starts]
        with pyBigWig.open(str(path), 'w') as bw:
            bw.addHeader([('chr1', 400)])
            bw.addEntries(['chr1'] * len(starts), starts, ends=[i + 1 for i in starts], values=values)
        paths.append(str(path))
    bed = tmp_path / 'regions.bed'
    bed.write_text(''.join(f'chr1\t{s}\t{s + 60}\tr{i}\t0\t{strand}\n'
                           for i, (s, strand) in enumerate([(0, '+'), (40, '-'), (110, '+'), (170, '-'), (280, '+')])))
    return paths, str(bed)


@pytest.mark.parametrize('operation', ['mean', 'median', 'min', 'max', 'sum'])
@pytest.mark.parametrize('mode', ['reference-point', 'scale-regions'])
@pytest.mark.parametrize('missing_zero', [False, True])
def test_compute_matrix_matches_original_values_and_missingness(tmp_path, genomic_inputs, operation, mode, missing_zero):
    paths, bed = genomic_inputs
    common = [mode, '-S', *paths, '-R', bed, '-b', '10', '-a', '30', '--binSize', '5',
              '--averageTypeBins', operation, '--sortRegions', 'keep', '--quiet', '-p', '1']
    if mode == 'scale-regions':
        common += ['--regionBodyLength', '40']
    if missing_zero:
        common += ['--missingDataAsZero']
    original = tmp_path / 'original.gz'
    plus = tmp_path / 'plus.gz'
    globals()['original']('compute', common + ['-o', str(original)])
    plus_compute.main(common + ['-o', str(plus)])
    oh, oreg, ov = read_matrix(original)
    ph, preg, pv = read_matrix(plus)
    assert preg == oreg
    assert ph['sample_boundaries'] == oh['sample_boundaries']
    assert ph['group_boundaries'] == oh['group_boundaries']
    np.testing.assert_array_equal(np.isnan(pv), np.isnan(ov))
    # Both CLI formats serialize six decimal places; R stores float32.
    np.testing.assert_allclose(pv, ov, rtol=2 * np.finfo(np.float32).eps, atol=1.01e-6, equal_nan=True)


@pytest.mark.parametrize('threshold', ['--minThreshold', '--maxThreshold'])
def test_partial_missing_thresholds_match_original_deeptools(tmp_path, genomic_inputs, threshold):
    paths, bed = genomic_inputs
    value = '-10' if threshold == '--minThreshold' else '9'
    common = ['reference-point', '-S', *paths, '-R', bed, '-a', '30', '-b', '10', '--binSize', '5',
              '--sortRegions', 'keep', '--quiet', '-p', '1', threshold, value]
    original = tmp_path / 'original.gz'
    plus = tmp_path / 'plus.gz'
    globals()['original']('compute', common + ['-o', str(original)])
    plus_compute.main(common + ['-o', str(plus)])
    _, oreg, ov = read_matrix(original)
    _, preg, pv = read_matrix(plus)
    assert preg == oreg
    np.testing.assert_allclose(pv, ov, rtol=2 * np.finfo(np.float32).eps, atol=1.01e-6, equal_nan=True)


@pytest.mark.parametrize('normalization', ['None', 'CPM', 'RPKM'])
def test_bam_coverage_matches_original_deeptools(tmp_path, normalization):
    import pysam
    bam = Path(__file__).parent / 'test_data' / 'test1.bam'
    reference_path = tmp_path / 'reference.bw'
    plus_path = tmp_path / 'plus.bw'
    arguments = ['-b', str(bam), '--binSize', '10', '-p', '1']
    original('coverage', arguments + ['--normalizeUsing', normalization, '-o', str(reference_path)])
    plus_normalization = 'read-count' if normalization == 'None' else normalization
    plus_coverage.main(arguments + ['--normalizeUsing', plus_normalization, '-o', str(plus_path)])
    # Independent per-read/bin oracle. This fixture contains only primary,
    # unspliced 51 bp alignments, so upstream and R placement agree.
    with pysam.AlignmentFile(str(bam)) as reads:
        mapped = reads.mapped
        expected = {chrom: np.zeros(length) for chrom, length in zip(reads.references, reads.lengths)}
        for read in reads.fetch():
            assert read.flag in (0, 16) and read.query_length == 51
            touched = {b for start, end in read.get_blocks() for b in range(start // 10, (end - 1) // 10 + 1)}
            for b in touched:
                expected[read.reference_name][10 * b:10 * (b + 1)] += 1
    factor = {'None': 1, 'CPM': 1e6 / mapped, 'RPKM': 1e9 / (mapped * 10)}[normalization]
    with pyBigWig.open(str(reference_path)) as reference_bw, pyBigWig.open(str(plus_path)) as plus_bw:
        assert reference_bw.chroms() == plus_bw.chroms()
        for chrom, length in reference_bw.chroms().items():
            reference_values = np.array(reference_bw.values(chrom, 0, length))
            plus_values = np.array(plus_bw.values(chrom, 0, length))
            np.testing.assert_array_equal(np.isnan(plus_values), np.isnan(reference_values))
            # Original writes {:g} (six significant digits) to its intermediate
            # bedGraph. R writes the more accurate float32 value directly.
            np.testing.assert_allclose(plus_values, reference_values, rtol=5.2e-6, atol=1e-6, equal_nan=True)
            oracle = expected[chrom] * factor
            np.testing.assert_allclose(plus_values, oracle, rtol=np.finfo(np.float32).eps, atol=0)
            assert np.max(np.abs(plus_values - oracle)) <= np.max(np.abs(reference_values - oracle)) + 1e-14


def test_bpm_definition_difference_is_explicit(tmp_path):
    bam = Path(__file__).parent / 'test_data' / 'test1.bam'
    paths = {name: tmp_path / f'{name}.bw' for name in ('upstream_bpm', 'upstream_cpm', 'plus_bpm')}
    arguments = ['-b', str(bam), '--binSize', '10', '-p', '1']
    original('coverage', arguments + ['--normalizeUsing', 'BPM', '-o', str(paths['upstream_bpm'])])
    original('coverage', arguments + ['--normalizeUsing', 'CPM', '-o', str(paths['upstream_cpm'])])
    plus_coverage.main(arguments + ['--normalizeUsing', 'BPM', '-o', str(paths['plus_bpm'])])
    values = {}
    for name, path in paths.items():
        with pyBigWig.open(str(path)) as bw:
            values[name] = np.concatenate([bw.values(chrom, 0, length)[::10] for chrom, length in bw.chroms().items()])
    # Upstream's documented implementation divides by mapped alignments;
    # R's BPM divides by total placed bin signal (TPM-like).
    np.testing.assert_array_equal(values['upstream_bpm'], values['upstream_cpm'])
    np.testing.assert_allclose(np.sum(values['plus_bpm']), 1e6, rtol=np.finfo(np.float32).eps)
    assert not np.allclose(values['plus_bpm'], values['upstream_bpm'])
