"""Independent row-policy oracles shared by resident and streaming adapters."""
import gzip
import json
import math

import numpy as np
import pytest

from deeptoolsr import _statistics, _compute_matrix_stream, stats as kernels
from deeptoolsr import computeMatrixOperations as cmo
from deeptoolsr.matrix import Matrix


BOUNDS = [0, 1, 4, 6]
GROUPS = [0, 2, 2, 8]
DATA = np.array([
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [2, np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan, 1, 2, 3, 2, 2],
    [2, 1, np.nan, 3, np.inf, 2],
    [-1, 1, 2, 3, 2, 5],
    [4, 1e20, 1, -1e20, 0, 4],
    [0, 0, 0, 0, 0, 0],
    [2, -np.inf, np.nan, np.inf, 2, 2],
], dtype=np.float32)


def reference(data, samples, statistic, nan_mode, action, low=0, high=4):
    clean = np.ma.filled(data, np.nan).copy()
    clean[~np.isfinite(clean)] = np.nan
    output = clean.astype(np.float32)
    keep = np.ones(len(data), dtype=bool)
    for r, row in enumerate(clean):
        failures, empty = [], []
        for sample in samples:
            values = [float(x) for x in row[BOUNDS[sample]:BOUNDS[sample + 1]] if np.isfinite(x)]
            empty.append(not values)
            failed = False
            if values:
                ordered = sorted(values)
                middle = len(values) // 2
                estimates = {
                    'mean': math.fsum(values) / len(values),
                    'sum': math.fsum(values), 'min': min(values), 'max': max(values),
                    'median': ordered[middle] if len(values) % 2 else
                    (ordered[middle - 1] + ordered[middle]) / 2,
                }
                tested = values if statistic == 'perBin' else [estimates[statistic]]
                failed = any(x < low or x > high for x in tested)
            failed |= nan_mode == 'any_bin' and len(values) < BOUNDS[sample + 1] - BOUNDS[sample]
            failed |= nan_mode == 'any_sample' and not values
            failures.append(failed)
        whole_missing = nan_mode == 'all_bins' and bool(samples) and all(empty)
        if action == 'removeRegion':
            keep[r] = not (any(failures) or whole_missing)
        else:
            for sample, failed in zip(samples, failures):
                if failed or whole_missing:
                    output[r, BOUNDS[sample]:BOUNDS[sample + 1]] = np.nan
    return keep, output if action == 'maskSample' else clean[keep]


def write_fixture(path):
    header = {
        'group_boundaries': GROUPS, 'group_labels': ['first', 'empty', 'last'],
        'sample_boundaries': BOUNDS, 'sample_labels': ['a', 'b', 'c'],
        'upstream': [0] * 3, 'downstream': [0] * 3, 'body': [1, 3, 2],
        'unscaled 5 prime': [0] * 3, 'unscaled 3 prime': [0] * 3,
        'ref point': [None] * 3, 'bin size': [1] * 3,
        'sort regions': 'keep', 'sort using': 'mean',
    }
    with gzip.open(path, 'wt') as handle:
        handle.write('@' + json.dumps(header) + '\n')
        for i, row in enumerate(DATA):
            handle.write(f'chr1\t{i}\t{i + 1}\tr{i}\t0\t+\t' +
                         '\t'.join(format(float(x), '.17g') for x in row) + '\n')


@pytest.mark.parametrize('statistic', ['perBin', 'mean', 'median', 'sum', 'min', 'max'])
@pytest.mark.parametrize('nan_mode', ['keep', 'any_bin', 'any_sample', 'all_bins'])
@pytest.mark.parametrize('action', ['removeRegion', 'maskSample'])
@pytest.mark.parametrize('samples', [[], [1], [2, 1], [0, 1, 2]])
def test_row_policy_matches_independent_oracle(tmp_path, monkeypatch, statistic, nan_mode, action, samples):
    # Alternate serial/parallel adapters across the option grid.
    workers = 3 if len(samples) % 2 else 1
    src, out = tmp_path / 'input.gz', tmp_path / 'stream.gz'
    write_fixture(src)
    expected_keep, expected_data = reference(DATA, samples, statistic, nan_mode, action)
    names = [str(i + 1) for i in samples]
    cmo.stream_filter_values(str(src), str(out), 0, 4, names, statistic,
                             action, nan_mode, threads=workers)
    streamed = Matrix.load(str(out), threads=1)
    expected_names = [f'r{i}' for i, selected in enumerate(expected_keep) if selected]
    expected_bounds = [int(expected_keep[:end].sum()) for end in GROUPS]
    np.testing.assert_array_equal(
        streamed.values, expected_data)
    assert [r[2] for r in streamed.regions] == expected_names
    assert streamed.header.group_boundaries == tuple(expected_bounds)
    assert streamed.header.group_labels == ('first', 'empty', 'last')
    assert streamed.header.sample_boundaries == tuple(BOUNDS)
    assert streamed.header.sample_labels == ('a', 'b', 'c')


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('layout', ['readonly', 'reversed', 'masked'])
@pytest.mark.parametrize('action', ['removeRegion', 'maskSample'])
def test_resident_policy_borrows_input_without_mutation(dtype, layout, action):
    data = DATA.astype(dtype)
    if layout == 'reversed':
        data = data[::-1, ::-1]
    if layout == 'masked':
        data[4, 0] = np.nan
    data.flags.writeable = False
    before = data.copy()
    expected_keep, expected_data = reference(data, [2, 0], 'median', 'all_bins', action)
    keep, masked = kernels.filter_matrix(data, BOUNDS, [2, 0], 'median', 0, 4,
                                         nan_mode=3, on_fail=action,
                                         threads=3)
    np.testing.assert_array_equal(keep, expected_keep)
    if action == 'maskSample':
        assert masked.dtype == np.float32
        np.testing.assert_array_equal(masked, expected_data)
        assert not np.shares_memory(masked, data)
    else:
        assert masked is None
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize('bounds,samples,stat,nans,low,high', [
    ([], [], 0, 0, 0, 4), ([1, 6], [0], 0, 0, 0, 4),
    ([0, 0, 6], [0], 0, 0, 0, 4), ([0, 4, 3, 6], [0], 0, 0, 0, 4),
    (BOUNDS, [3], 0, 0, 0, 4), (BOUNDS, [1, 1], 0, 0, 0, 4),
    (BOUNDS, [0], 0, 4, 0, 4), (BOUNDS, [0], 0, -1, 0, 4),
    (BOUNDS, [0], 0, 0, np.nan, 4), (BOUNDS, [0], 0, 0, 0, np.nan),
    (BOUNDS, [0], 0, 0, np.inf, np.inf), (BOUNDS, [0], 0, 0, -np.inf, -np.inf),
    (BOUNDS, [0], 0, 0, 5, 4),
])
def test_adapters_share_validation_before_output(tmp_path, bounds, samples, stat, nans, low, high):
    out = tmp_path / 'untouched.gz'
    out.write_bytes(b'preserve existing output')
    for action in ('removeRegion', 'maskSample'):
        with pytest.raises(RuntimeError):
            _statistics.filter_matrix(DATA, bounds, samples, 'perBin', low, high, nans, action, 2)
    args = (bounds, samples, stat, nans, True, low, True, high, 2)
    with pytest.raises(RuntimeError):
        _compute_matrix_stream.filter_values('unused', str(out), GROUPS, *args)
    with pytest.raises(RuntimeError):
        _compute_matrix_stream.filter_values_mask('unused', str(out), '{}', *args)
    assert out.read_bytes() == b'preserve existing output'


def test_resident_policy_validates_width_and_action():
    with pytest.raises(RuntimeError, match='width'):
        _statistics.filter_matrix(DATA, [0, 7], [0], 'mean', 0, 4)
    with pytest.raises(ValueError, match='action'):
        _statistics.filter_matrix(DATA, BOUNDS, [], 'mean', 0, 4, 0, 'typo')


@pytest.mark.parametrize('action', ['removeRegion', 'maskSample'])
def test_float64_decisions_precede_output_narrowing(action):
    data = np.array([[4], [np.nextafter(4., np.inf)]])
    keep, masked = kernels.filter_matrix(data, [0, 1], [0], 'mean', 0, 4,
                                         on_fail=action, threads=1)
    if action == 'removeRegion':
        np.testing.assert_array_equal(keep, [True, False])
    else:
        np.testing.assert_array_equal(masked[:, 0], [4, np.nan])


@pytest.mark.parametrize('action', ['removeRegion', 'maskSample'])
def test_empty_resident_matrix(action):
    keep, masked = kernels.filter_matrix(np.empty((0, 6), np.float32), BOUNDS,
                                         [0, 1, 2], 'mean', -np.inf, np.inf,
                                         3, action, threads=1)
    assert keep.shape == (0,)
    if action == 'maskSample':
        assert masked.shape == (0, 6)
    else:
        assert masked is None
