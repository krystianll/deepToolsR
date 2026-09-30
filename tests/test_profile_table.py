"""Profile table bytes captured from the pre-extraction CLI."""
from deeptoolsr.stats import calc_avg
from deeptoolsr import computeMatrix
import numpy as np
from tests.helpers.bigwig import pyBigWig
from tests.helpers.parity import read_matrix

import csv
import json
import math
from pathlib import Path

import pytest

from deeptoolsr import plotProfile


ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / 'tests' / 'contract' / 'profile_tables'
CASES = json.loads((GOLDENS / 'cases.json').read_text())
# The t-based CI band differs in the last bits between x86-64 and aarch64
# (FMA contraction). Its bounds are mean - t*se, which cancels near zero, so
# bins are compared at 1e-12 of the row's largest magnitude; text is exact.
ARCHITECTURE_ROUNDED = {'ci'}


def _rows(data):
    return list(csv.reader(data.decode().splitlines(), delimiter='\t'))


def _assert_close_table(actual, expected):
    actual, expected = _rows(actual), _rows(expected)
    assert len(actual) == len(expected)
    assert actual[0] == expected[0]
    first_bin = expected[0].index('bin_0')
    for got, want in zip(actual[1:], expected[1:]):
        assert got[:first_bin] == want[:first_bin]
        assert len(got) == len(want)
        scale = max(abs(float(value)) for value in want[first_bin:])
        for a, b in zip(got[first_bin:], want[first_bin:]):
            assert a == b or math.isclose(
                float(a), float(b), rel_tol=0, abs_tol=1e-12 * scale)


@pytest.mark.parametrize('name,command', CASES.items())
def test_profile_table_matches_golden(name, command, tmp_path):
    matrix, *options = command
    table = tmp_path / (name + '.tsv')
    plotProfile.main([
        '-m', str(ROOT / matrix), '-o', str(tmp_path / (name + '.png')),
        '--outFileNameData', str(table), *options])
    expected = (GOLDENS / (name + '.tsv')).read_bytes()
    if name in ARCHITECTURE_ROUNDED:
        _assert_close_table(table.read_bytes(), expected)
    else:
        assert table.read_bytes() == expected


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


@pytest.fixture
def parity_input_matrix(tmp_path, parity_matrix_inputs):
    paths, bed = parity_matrix_inputs
    output = tmp_path / 'input.gz'
    computeMatrix.main(['reference-point', '-S', *paths, '-R', bed, '-b', '10', '-a', '30', '-bs', '5', '--quiet', '-p', '1', '-o', str(output)])
    return str(output)


@pytest.mark.parametrize('statistic', ['mean', 'median', 'min', 'max', 'sum', 'std'])
def test_profile_export_matches_independent_display_series(tmp_path, parity_input_matrix, statistic):
    output = tmp_path / 'plus.txt'
    plotProfile.main(['-m', parity_input_matrix, '-o', str(tmp_path / 'plus.png'), '--outFileNameData', str(output), '--averageType', statistic, '--plotType', 'lines'])
    with open(output, newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    header, _regions, values = read_matrix(parity_input_matrix)
    sample_bounds = header['sample_boundaries']
    group_bounds = header['group_boundaries']
    assert len(rows) == (len(sample_bounds) - 1) * (len(group_bounds) - 1)
    for row in rows:
        sample = int(row['panel_index']) - 1
        group = int(row['series_index']) - 1
        expected = calc_avg(values[group_bounds[group]:group_bounds[group + 1], sample_bounds[sample]:sample_bounds[sample + 1]], statistic, axis=0, threads=1)
        observed = np.array([float(row['bin_{}'.format(index)]) for index in range(int(row['valid_bin_count']))])
        assert row['component'] == statistic
        np.testing.assert_allclose(observed, expected, rtol=3e-07, atol=1e-06, equal_nan=True)
