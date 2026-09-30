"""Shared kernels preserve independent numerical and public option contracts."""
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from deeptoolsr import _statistics, stats as kernels
from deeptoolsr.stats import bootstrap_ci, calc_avg
from tests.test_statistical_contracts import statistic_reference


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('layout', ['plain', 'reversed', 'masked'])
def test_shared_moments_against_decimal(dtype, layout):
    maximum = np.finfo(dtype).max
    small = np.nextafter(dtype(0), dtype(1))
    adjacent = np.nextafter(dtype(1), dtype(2))
    data = np.array([[maximum, maximum, 1, small, 0, np.nan],
                     [-maximum, maximum, adjacent, small, 0, np.nan],
                     [0, np.nan, np.nan, 0, np.nan, np.inf]], dtype=dtype)
    if layout == 'reversed':
        data = data[::-1, ::-1]
    if layout == 'masked':
        data[0, 0] = np.nan
    before = np.ma.filled(data, np.nan).copy()
    values = kernels.buffers(data)
    mean, std, count, _, _ = _statistics.summarize_columns(
        values, False, 0, 2)
    for c, column in enumerate(before.T):
        finite = [float(x) for x in column if np.isfinite(x)]
        assert count[c] == len(finite)
        if not finite:
            assert np.isnan(mean[c]) and np.isnan(std[c])
            continue
        with localcontext() as ctx:
            ctx.prec = 800
            numbers = [Decimal.from_float(x) for x in finite]
            expected_mean = sum(numbers) / len(numbers)
            expected_std = (sum((x - expected_mean) ** 2 for x in numbers) / len(numbers)).sqrt()
        np.testing.assert_allclose(mean[c], float(expected_mean), rtol=3e-15, atol=0)
        np.testing.assert_allclose(std[c], float(expected_std), rtol=3e-15, atol=small)
    np.testing.assert_array_equal(np.ma.filled(data, np.nan), before)


@pytest.mark.parametrize('statistic', kernels.STATISTIC_CHOICES['profile'])
@pytest.mark.parametrize('layout', ['plain', 'masked', 'reversed'])
@pytest.mark.parametrize('processors', [1, 3])
def test_bootstrap_center_and_interval_share_preparation(statistic, layout, processors):
    data = np.array([[0, -4, np.nan, 0], [2, -2, np.nan, 0],
                     [8, 0, np.nan, np.nan], [np.nan, -8, np.nan, 0]], dtype=np.float64)
    if layout == 'masked':
        data[0, 1] = np.nan
    elif layout == 'reversed':
        data = data[::-1, ::-1]
    before = np.ma.filled(data, np.nan).copy()
    expected = [statistic_reference(column, statistic, pc=1, trim=.25)
                for column in before.T]
    kwargs = dict(pseudocount=-1, trim_perc=.25, n_resamples=80, seed=9,
                  threads=processors)
    center, lower, upper = bootstrap_ci(data, statistic, include_center=True, **kwargs)
    standalone = bootstrap_ci(data, statistic, **kwargs)
    np.testing.assert_allclose(np.ma.filled(center, np.nan), expected,
                               rtol=3e-14, atol=3e-14, equal_nan=True)
    np.testing.assert_array_equal([lower, upper], standalone)
    np.testing.assert_array_equal(np.ma.filled(data, np.nan), before)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('statistic', kernels.STATISTIC_CHOICES['filter'])
@pytest.mark.parametrize('nan_mode', [0, 1, 2, 3])
def test_shared_filter_threshold_and_missing_counts(dtype, statistic, nan_mode):
    data = np.array([[0, 2, 4], [2, np.nan, 2], [np.nan, np.inf, -np.inf],
                     [1e20, 1, -1e20], [2, 2, 2]], dtype=dtype)
    expected = []
    for row in data:
        finite = [float(x) for x in row if np.isfinite(x)]
        fail = False
        if finite:
            if statistic == 'perBin':
                fail = min(finite) < 2 or max(finite) > 2
            else:
                value = math.fsum(finite) if statistic in ('sum', 'mean') else None
                if statistic == 'mean':
                    value /= len(finite)
                if statistic == 'min':
                    value = min(finite)
                if statistic == 'max':
                    value = max(finite)
                if statistic == 'median':
                    value = float(np.median(finite))
                fail = value < 2 or value > 2
        fail |= nan_mode == 1 and len(finite) < len(row)
        fail |= nan_mode == 2 and not finite
        expected.append(fail)
    failed, counts = kernels.filter_rows(data, statistic, 2, 2,
                                         nan_mode=nan_mode, threads=1)
    np.testing.assert_array_equal(failed, expected)
    np.testing.assert_array_equal(counts, [3, 2, 0, 3, 3])


def test_profile_sum_preserves_missing_without_second_count_reduction(monkeypatch):
    data = np.array([[np.nan, 0, 1], [np.inf, 0, 2]], np.float32)
    original = _statistics.reduce_axis
    calls = []

    def reduce_once(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(_statistics, 'reduce_axis', reduce_once)
    np.testing.assert_array_equal(np.ma.filled(calc_avg(data, 'sum', threads=1), np.nan), [np.nan, 0, 3])
    assert calls == ['sum']
    np.testing.assert_array_equal(kernels.reduce(data, 'sum', threads=1), [0, 0, 3])


def test_bin_and_filter_capabilities_keep_deliberately_removed_std_out():
    assert set(kernels.STATISTIC_CHOICES['bin']) == {'mean', 'median', 'min', 'max', 'sum'}
    assert set(kernels.STATISTIC_CHOICES['filter']) == set(_statistics.filter_statistics)
    assert 'std' in kernels.STATISTIC_CHOICES['profile']
    assert 'std' not in kernels.STATISTIC_CHOICES['filter']
