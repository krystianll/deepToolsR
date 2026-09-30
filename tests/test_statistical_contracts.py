"""Independent numerical contracts, including adversarial/missing input."""
from deeptoolsr.stats import center_and_analytic_band
import itertools
import shutil
from scipy import stats
import subprocess
from deeptoolsr import stats as profile
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from deeptoolsr import _statistics, _transform, stats as kernels
from deeptoolsr.stats import calc_avg, bootstrap_ci


def reference(values, operation):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return 0.0 if operation == 'sum' else math.nan
    if operation == 'min':
        return min(values)
    if operation == 'max':
        return max(values)
    if operation == 'median':
        values.sort()
        n = len(values)
        return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    if operation == 'sum':
        return math.fsum(values)
    if operation == 'mean':
        return math.fsum(values) / len(values)
    with localcontext() as ctx:
        ctx.prec = 100
        numbers = [Decimal.from_float(x) for x in values]
        mean = sum(numbers) / len(numbers)
        return float((sum((x - mean)**2 for x in numbers) / len(numbers)).sqrt())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('operation', ['mean', 'sum', 'min', 'max', 'median', 'std'])
@pytest.mark.parametrize('layout', ['contiguous', 'reversed', 'strided', 'fortran'])
def test_reductions_against_independent_reference(dtype, axis, operation, layout):
    data = np.array([[1, 2, np.nan, 0], [3, np.nan, 8, 0],
                     [-1, 10, 12, np.nan], [np.inf, 20, -np.inf, np.nan]], dtype=dtype)
    if layout == 'reversed':
        data = data[::-1, ::-1]
    if layout == 'strided':
        data = data[:, ::2]
    if layout == 'fortran':
        data = np.asfortranarray(data)
    before = np.ma.filled(data, np.nan).copy()
    scan = before.T if axis == 0 else before
    expected = [reference(row, operation) for row in scan]
    actual = kernels.reduce(data, operation, axis, threads=1)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-14, equal_nan=True)
    np.testing.assert_array_equal(np.ma.filled(data, np.nan), before)


@pytest.mark.parametrize('operation,expected', [('sum', 1), ('mean', 1 / 3)])
def test_cancellation_retains_small_contribution(operation, expected):
    x = np.array([[1e20, 1, -1e20]], np.float32)
    np.testing.assert_allclose(kernels.reduce(x, operation, 1, threads=1),
                               [expected], rtol=1e-15)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_extreme_representable_mean_and_std(dtype):
    maximum = np.finfo(dtype).max
    x = np.array([[maximum, maximum], [maximum, -maximum]], dtype=dtype)
    np.testing.assert_allclose(kernels.reduce(x, 'mean', 1, threads=1),
                               [maximum, 0], rtol=2e-15)
    np.testing.assert_allclose(kernels.reduce(x, 'std', 1, threads=1),
                               [0, maximum], rtol=2e-15)


@pytest.mark.parametrize('value', [1., 1e308, -1e308, np.nextafter(0., 1.)])
def test_std_of_adjacent_float64_values_retains_spread(value):
    x = np.array([[value, np.nextafter(value, np.inf)]])
    expected = (x[0, 1] - x[0, 0]) / 2
    actual = kernels.reduce(x, 'std', 1, threads=1)[0]
    assert actual == expected


def test_unaligned_read_buffer():
    x = np.ndarray((2, 2), dtype=np.float32, buffer=bytearray(17), offset=1)
    x[:] = [[1, 2], [3, 4]]
    np.testing.assert_array_equal(_statistics.reduce_axis(x, 0, 'sum', 1), [4, 6])


@pytest.mark.parametrize('axis', [-2, 2, 100])
def test_invalid_axis_is_rejected(axis):
    with pytest.raises(ValueError):
        kernels.reduce(np.ones((2, 2), np.float32), 'mean', axis, threads=1)


def test_invalid_statistic_projection_and_permutation_are_rejected():
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    with pytest.raises(ValueError):
        kernels.reduce(x, 'unknown', threads=1)
    with pytest.raises(ValueError):
        _statistics.reduce_axis(x, 0, 'mean', 1,
                                rows=np.zeros((2, 2), bool))
    with pytest.raises(ValueError):
        _statistics.permute_rows_inplace(x, 0, 3, [0, 0, 1])
    np.testing.assert_array_equal(x, np.arange(6).reshape(3, 2))


@pytest.mark.parametrize('statistic', ['perBin', 'mean', 'sum', 'min', 'max', 'median', 'std'])
@pytest.mark.parametrize('inclusive', [False, True])
def test_filters_use_finite_statistics_and_explicit_missing_counts(statistic, inclusive):
    x = np.array([[100, np.nan, 2], [1, 2, 3], [np.nan, np.inf, np.nan],
                  [-100, np.nan, 2], [0, 0, 0], [10, 10, np.nan]], np.float32)
    failed, counts = kernels.filter_rows(x, statistic, -10, 10, inclusive,
                                         threads=1)
    expected = []
    for row in x:
        finite = [float(v) for v in row if math.isfinite(v)]
        if not finite:
            expected.append(False)
            continue
        values = finite if statistic == 'perBin' else [reference(finite, statistic)]
        expected.append(any(v <= -10 or v >= 10 for v in values) if inclusive
                        else any(v < -10 or v > 10 for v in values))
    np.testing.assert_array_equal(failed, expected)
    np.testing.assert_array_equal(counts, [2, 3, 0, 2, 3, 2])


def test_trimmed_means_use_finite_tail_counts_and_floor():
    x = np.array([[0, 1], [10, 2], [100, 3], [np.nan, 100], [np.nan, np.nan]], np.float32)
    np.testing.assert_allclose(calc_avg(x, 'trim_mean', trim_perc=.25, threads=1), [110 / 3, 2.5])


def test_geometric_bootstrap_rejects_mixed_signs():
    with pytest.raises(ValueError, match='mixed positive and negative'):
        bootstrap_ci(np.array([[-1.], [1.]], np.float32), 'geom_mean',
                     pseudocount=1, n_resamples=1000, threads=1)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_exact_radix_quantiles_match_order_statistics(dtype):
    rng = np.random.default_rng(731)
    x = rng.normal(size=(300, 9)).astype(dtype)[::-1, ::-1]
    x[::7, 2] = np.nan
    probabilities = [0, .1, 1, 10, 50, 90, 99, 100]
    actual = _statistics.nan_quantiles(x, probabilities, max_exact=0, exact=True)
    expected = np.nanpercentile(x.astype(np.float64), probabilities)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-14)


@pytest.mark.parametrize('operation,args', [('add_scalar', (1,)), ('scale_scalar', (2,))])
def test_inplace_transforms_reject_implicit_copies_and_readonly(operation, args):
    function = getattr(_transform, operation)
    with pytest.raises(TypeError):
        function(np.ones((2, 2), np.float64), *args)
    x = np.ones((2, 2), np.float32)
    x.flags.writeable = False
    with pytest.raises((ValueError, BufferError)):
        function(x, *args)


def test_row_scaling_keeps_wide_divisor():
    x = np.full((1, 2), np.finfo(np.float32).max, np.float32)
    _transform.scale_rows(x, [0, 2], [0], [2], 0, 1)
    np.testing.assert_array_equal(x, [[.5, .5]])


def test_overflow_is_reported_instead_of_becoming_missing():
    x = np.full((1, 2), np.finfo(np.float32).max, np.float32)
    with pytest.raises(OverflowError):
        _transform.scale_scalar(x, 2, 1)


def test_profile_sum_keeps_entirely_missing_bins_missing():
    result = calc_avg(np.array([[1, np.nan], [2, np.nan]], np.float32), 'sum', threads=1)
    assert result[0] == 3
    assert np.ma.is_masked(result[1])


def test_binary_transform_accepts_readonly_source_and_rejects_overlap():
    numerator = np.ones((2, 2), np.float32)
    denominator = np.full((2, 2), 2, np.float32)
    denominator.flags.writeable = False
    _transform.binary_combine(numerator, denominator, [0, 2], [0], 1, 0, 1)
    np.testing.assert_array_equal(numerator, .5)
    data = np.ones((3, 2), np.float32)
    with pytest.raises(ValueError, match='overlapping'):
        _transform.binary_combine(data[:-1], data[1:], [0, 2], [0], 1, 0, 2)


def test_geometric_mean_and_band_retain_signal_below_pseudocount_precision():
    from deeptoolsr.stats import center_and_analytic_band
    x = np.array([[1.], [1.]], np.float32)
    np.testing.assert_allclose(calc_avg(x, 'geom_mean', pseudocount=1e30, threads=1), [1.], rtol=1e-14)
    center, low, high = center_and_analytic_band(x, 'geom_mean', 'std', pseudocount=1e30, threads=1)
    np.testing.assert_allclose([center[0], low[0], high[0]], [1, 1, 1], rtol=1e-14)


def test_geometric_inverse_handles_unaligned_reversed_buffers_and_invalid_signs():
    values = np.ndarray((3,), dtype=np.float64, buffer=bytearray(25), offset=1)
    values[:] = [0, np.log(2), np.log(3)]
    actual = _statistics.geometric_inverse(values[::-1], np.array([1., -1., 0.]), 1)
    np.testing.assert_allclose(actual, [2, -1, 0], rtol=1e-14)
    for sign in (np.nan, np.inf, 2.):
        with pytest.raises(ValueError, match='sign'):
            _statistics.geometric_inverse(np.array([0.]), np.array([sign]), 1)


@pytest.mark.parametrize('data', [[[1 + 2j]], [['3']]])
def test_kernel_buffers_reject_non_real_numeric_data(data):
    with pytest.raises(TypeError, match='real numeric'):
        kernels.reduce(data, 'mean', threads=1)


# From equivalence regressions.

@pytest.fixture(scope='module')
def equivalence_bootstrap_draws(tmp_path_factory):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('audit draw fixture requires a C++ compiler')
    path = tmp_path_factory.mktemp('bootstrap-draws')
    source, executable = (path / 'draws.cpp', path / 'draws')
    source.write_text("\n#include <iostream>\n#include <random>\n#include <cstdlib>\nint main(int argc, char** argv) {\n    const size_t rows = std::strtoull(argv[1], nullptr, 10);\n    const size_t replicates = std::strtoull(argv[2], nullptr, 10);\n    std::mt19937_64 rng(std::strtoull(argv[3], nullptr, 10));\n    std::uniform_int_distribution<size_t> pick(0, rows - 1);\n    for (size_t r = 0; r < replicates; ++r) {\n        for (size_t i = 0; i < rows; ++i) std::cout << pick(rng) << ' ';\n        std::cout << '\\n';\n    }\n}\n")
    subprocess.run([compiler, '-std=c++17', str(source), '-o', str(executable)], check=True, capture_output=True, text=True)

    def draw(rows, replicates, seed):
        result = subprocess.check_output([str(executable), str(rows), str(replicates), str(seed)], text=True)
        return np.fromstring(result, sep=' ', dtype=int).reshape(replicates, rows)
    return draw


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


@pytest.mark.parametrize('statistic', ['mean', 'median', 'sum', 'min', 'max', 'trim_mean', 'geom_mean', 'std'])
@pytest.mark.parametrize('layout', ['dense', 'missing', 'masked'])
@pytest.mark.parametrize('threads', [1, 3])
def test_bootstrap_each_estimator_against_exact_draws(equivalence_bootstrap_draws, statistic, layout, threads):
    matrix = np.array([[1, -1, 0, 4], [3, -3, 0, -4], [8, -8, 0, 2], [2, -2, 0, -3], [7, -7, 0, 6]], dtype=np.float32)
    if statistic == 'geom_mean':
        matrix[:, 3] = np.abs(matrix[:, 3])
    if layout != 'dense':
        matrix[1, 0] = np.nan
        matrix[3:, 1] = np.nan
        matrix[:, 2] = np.nan
    if layout == 'masked':
        matrix[0, 3] = np.nan
        matrix = matrix[::-1, ::-1]
    data = np.ma.filled(matrix, np.nan)
    seed, replicates, level, pc = (319, 137, 0.8, 1.0)
    draws = equivalence_bootstrap_draws(len(data), replicates, seed)
    estimates = np.array([[statistic_reference(data[draw, c], statistic, pc) for c in range(data.shape[1])] for draw in draws])
    expected = np.full((2, data.shape[1]), np.nan)
    for c in range(data.shape[1]):
        finite = estimates[:, c][np.isfinite(estimates[:, c])]
        if len(finite):
            expected[:, c] = np.quantile(finite, [(1 - level) / 2, (1 + level) / 2])
    actual = bootstrap_ci(matrix, statistic, pseudocount=pc, trim_perc=0.2, n_resamples=replicates, ci_level=level, seed=seed, threads=threads)
    np.testing.assert_allclose(np.ma.filled(np.ma.vstack(actual), np.nan), expected, rtol=3e-14, atol=3e-14, equal_nan=True)


@pytest.mark.parametrize('path', ['center', 'std', 'ci', 'bootstrap'])
def test_e04_default_geometric_mean_accepts_zero_only_series(path):
    matrix = np.zeros((4, 3), dtype=np.float32)
    if path == 'center':
        result = calc_avg(matrix, 'geom_mean', threads=1)
    elif path == 'bootstrap':
        result = np.ma.vstack(bootstrap_ci(matrix, 'geom_mean', n_resamples=20, threads=1))
    else:
        result = np.ma.vstack(center_and_analytic_band(matrix, 'geom_mean', path, threads=1))
    np.testing.assert_array_equal(np.ma.filled(result, np.nan), 0.0)


@pytest.mark.parametrize('pc', [0.0, 0.25, 2.0, 100000000.0])
@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('axis', [0, 1])
def test_geometric_centers_and_analytic_bands_follow_shifted_log_formula(pc, sign, axis):
    matrix = sign * np.array([[1.0, 4.0, np.nan], [3.0, 6.0, 2.0], [9.0, 2.0, 7.0], [2.0, 8.0, 4.0]])
    matrix[1, 1] = np.nan
    if axis == 1:
        matrix = matrix.T
    data = np.ma.filled(matrix, np.nan)
    scan = data.T if axis == 0 else data
    expected, expected_bands = ([], {key: [] for key in ['std', 'se', 'ci']})
    for row in scan:
        magnitudes = sign * row[np.isfinite(row)]
        logs = np.log1p(magnitudes / pc) if pc else np.log(magnitudes)
        inverse = (lambda value: sign * pc * np.expm1(value)) if pc else lambda value: sign * np.exp(value)
        expected.append(inverse(logs.mean()))
        for kind in expected_bands:
            margin = logs.std(ddof=0) if kind == 'std' else logs.std(ddof=1) / np.sqrt(len(logs))
            if kind == 'ci':
                margin *= stats.t.ppf(0.975, len(logs) - 1)
            expected_bands[kind].append(sorted([inverse(logs.mean() - margin), inverse(logs.mean() + margin)]))
    np.testing.assert_allclose(calc_avg(matrix, 'geom_mean', pc, axis=axis, threads=1), expected, rtol=2e-14)
    if axis == 0:
        for kind in expected_bands:
            center, lower, upper = center_and_analytic_band(matrix, 'geom_mean', kind, pc, threads=1)
            np.testing.assert_allclose(center, expected, rtol=2e-14)
            np.testing.assert_allclose(np.column_stack([lower, upper]), expected_bands[kind], rtol=3e-14, atol=2e-14)


@pytest.mark.parametrize('sign', [-1, 1])
def test_e04_zero_pseudocount_preserves_true_zero_observations(sign):
    values = sign * np.array([[0.0], [4.0]])
    expected = sign * stats.gmean(np.abs(values), axis=0)
    np.testing.assert_array_equal(calc_avg(values, 'geom_mean', pseudocount=0, threads=1), expected)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_auto_pseudocount_ignores_zeros_but_keeps_them_in_mean(axis, dtype):
    from deeptoolsr import _statistics
    data = np.array([[0, 0, 0, np.nan], [2, -8, 0, np.nan], [8, -2, 0, np.nan]], dtype=dtype)
    if axis:
        data = data.T
    actual, pc = _statistics.geom_mean_axis(data, axis, -1, 2)
    assert pc == 1
    expected = [(1 * 3 * 9) ** (1 / 3) - 1, -((1 * 9 * 3) ** (1 / 3) - 1), 0, np.nan]
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-14, equal_nan=True)


@pytest.mark.parametrize('pc', [-1, 0, 1, 100])
@pytest.mark.parametrize('entry', ['center0', 'center1', 'summary', 'bootstrap'])
def test_native_geometric_paths_reject_mixed_signs(pc, entry):
    from deeptoolsr import _statistics
    values = np.array([[-1.0], [1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match='mixed positive and negative'):
        if entry.startswith('center'):
            axis = int(entry[-1])
            _statistics.geom_mean_axis(values.T if axis else values, axis, pc, 2)
        elif entry == 'summary':
            _statistics.summarize_columns(values, True, pc, 2)
        else:
            _statistics.bootstrap_ci(values, 'geom_mean', pc, 0.05, 10, 0.95, 0, 2, False)


@pytest.mark.parametrize('path', ['center', 'std', 'se', 'ci', 'bootstrap'])
@pytest.mark.parametrize('fortran', [False, True])
def test_entirely_missing_geometric_profile_remains_missing(path, fortran):
    data = np.full((4, 2), np.nan)
    if fortran:
        data = np.asfortranarray(data)
    if path == 'center':
        result = calc_avg(data, 'geom_mean', threads=1)
    elif path == 'bootstrap':
        result = np.ma.vstack(bootstrap_ci(data, 'geom_mean', n_resamples=10, threads=1))
    else:
        result = np.ma.vstack(center_and_analytic_band(data, 'geom_mean', path, threads=1))
    assert np.ma.getmaskarray(result).all()


@pytest.mark.parametrize('kind', ['std', 'se', 'ci'])
def test_geometric_zero_pseudocount_bands_require_positive_shift_for_zeros(kind):
    data = np.array([[0.0], [4.0]])
    with pytest.raises(ValueError, match='positive pseudocount'):
        center_and_analytic_band(data, 'geom_mean', kind, pseudocount=0, threads=1)
    center, lo, hi = center_and_analytic_band(data, 'geom_mean', kind, threads=1)
    assert np.isfinite([center[0], lo[0], hi[0]]).all()


@pytest.mark.parametrize('statistic', ['mean', 'median', 'min', 'max', 'sum', 'std', 'trim_mean', 'geom_mean'])
@pytest.mark.parametrize('sign', [-1, 1])
def test_bootstrap_two_observation_exact_distribution_portable(statistic, sign):
    data = sign * np.array([[0.0], [4.0]])
    estimates = [statistic_reference(data[list(indices), 0], statistic, pc=1) for indices in itertools.product([0, 1], repeat=2)]
    actual = bootstrap_ci(data, statistic, pseudocount=1, n_resamples=2000, seed=45, threads=1)
    np.testing.assert_allclose([actual[0][0], actual[1][0]], [min(estimates), max(estimates)], atol=2e-14)


# From numerical audit regressions.

@pytest.mark.parametrize('value,pc', [(1e-300, 1e+30), (1e-30, 1e+300), (np.finfo(np.float64).max, 1e+30), (np.finfo(np.float64).max, 1e+300)])
def test_n08_geometric_mean_of_identical_values_is_that_value(value, pc):
    values = np.full((2, 1), value, np.float64)
    result = profile.calc_avg(values, 'geom_mean', pseudocount=pc, threads=1)
    assert not np.ma.is_masked(result[0])
    assert math.isclose(float(result[0]), value, rel_tol=5e-13, abs_tol=0)
