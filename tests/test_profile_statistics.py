from pathlib import Path
from deeptoolsr import stats
import numpy as np
import pytest

from deeptoolsr import _statistics
from deeptoolsr.stats import (StatisticsSpec, bootstrap_ci,
                              calc_avg, parametric_ci)
from deeptoolsr.plotProfile import process_args
from deeptoolsr import stats as statistics_module


@pytest.mark.parametrize('df', [1, 2, 5, 30, 1e3, 1e6])
@pytest.mark.parametrize('level', [0.8, 0.9, 0.95, 0.99])
def test_student_t_quantile_matches_scipy(df, level):
    ss = pytest.importorskip('scipy.stats', reason='SciPy oracle absent')
    probability = (1.0 + level) / 2.0
    observed = _statistics.student_t_quantile(
        np.array([df, df], dtype=float), probability)
    np.testing.assert_allclose(observed, ss.t.ppf(probability, df), rtol=1e-12)


def test_student_t_quantile_invalid_degrees_of_freedom():
    result = _statistics.student_t_quantile(
        np.array([0.0, -1.0, np.nan, 2.0]), 0.975)
    assert np.isnan(result[:3]).all()
    assert np.isfinite(result[3])
    assert np.isnan(_statistics.student_t_quantile(2.0, np.nan))


def test_signed_geometric_mean():
    matrix = np.array([[1.0, -1.0], [4.0, -4.0]])
    observed = calc_avg(matrix, 'geom_mean', pseudocount=1.0, threads=1)
    expected = np.array([np.sqrt(10.0) - 1.0, -(np.sqrt(10.0) - 1.0)])
    np.testing.assert_allclose(observed, expected)


def test_signed_geometric_mean_auto_pseudocount():
    matrix = np.array([[1.0, -1.0], [4.0, -4.0]])
    observed = calc_avg(matrix, 'geom_mean', pseudocount=-1, threads=1)
    # The smallest sign-adjusted magnitude is 1, so auto selects 0.5.
    expected = np.array([np.sqrt(6.75) - 0.5, -(np.sqrt(6.75) - 0.5)])
    np.testing.assert_allclose(observed, expected)


def test_trim_mean_trims_each_column_independently():
    matrix = np.array([[0.0, 1000.0], [10.0, 30.0], [20.0, 20.0],
                       [30.0, 10.0], [1000.0, 0.0]])
    observed = calc_avg(matrix, 'trim_mean', trim_perc=0.2, threads=1)
    np.testing.assert_allclose(observed, [20.0, 20.0])


def test_bootstrap_ci_is_reproducible_and_contains_mean():
    matrix = np.arange(40.0).reshape(10, 4)
    first = bootstrap_ci(matrix, 'mean', threads=1)
    second = bootstrap_ci(matrix, 'mean', threads=1)
    np.testing.assert_allclose(first, second)
    mean = matrix.mean(axis=0)
    assert np.all(first[0] < mean)
    assert np.all(mean < first[1])


def test_higher_ci_level_produces_wider_bootstrap_interval():
    matrix = np.arange(80.0).reshape(20, 4)
    lower_80, upper_80 = bootstrap_ci(matrix, 'mean', ci_level=0.80, threads=1)
    lower_95, upper_95 = bootstrap_ci(matrix, 'mean', ci_level=0.95, threads=1)
    assert np.all(lower_95 <= lower_80)
    assert np.all(upper_80 <= upper_95)
    assert np.any((upper_95 - lower_95) > (upper_80 - lower_80))


def test_parallel_bootstrap_matches_serial_bootstrap():
    matrix = np.arange(80.0).reshape(20, 4)
    serial = bootstrap_ci(matrix, 'mean', n_resamples=40, threads=1)
    parallel = bootstrap_ci(matrix, 'mean', n_resamples=40, threads=2)
    np.testing.assert_allclose(serial, parallel)


def test_bootstrap_progress_is_reported(capfd):
    matrix = np.arange(40.0).reshape(10, 4)
    bootstrap_ci(matrix, 'mean', n_resamples=5, show_progress=True, threads=1)
    progress = capfd.readouterr().err
    assert 'Bootstrap [' in progress
    assert '100% (4/4)' in progress


# def test_static_uncertainty_bands_are_half_opaque():
#     matrix = np.arange(40.0).reshape(10, 4)
#     for plot_type in ('se', 'std', 'ci', 'bootstrap'):
#         fig, axis = plt.subplots()
#         plot_single(axis, matrix, 'mean', 'blue', 'sample',
#                     plot_type=plot_type, bootstrap_replicates=10)
#         assert all(np.isclose(collection.get_facecolor()[0, 3], 0.5)
#                    for collection in axis.collections)
#         plt.close(fig)


def test_geom_mean_parametric_ci_back_transforms_positive_profile():
    ss = pytest.importorskip('scipy.stats', reason='SciPy oracle absent')
    matrix = np.array([[1.0], [2.0], [4.0], [8.0]])
    lower, upper = parametric_ci(matrix, 'geom_mean', pseudocount=0,
                                 ci_level=0.95, threads=1)
    logs = np.log(matrix[:, 0])
    margin = (logs.std(ddof=1) / np.sqrt(len(logs)) *
              ss.t.ppf(0.975, len(logs) - 1))
    expected = np.exp([logs.mean() - margin, logs.mean() + margin])
    np.testing.assert_allclose([lower[0], upper[0]], expected)


def test_geom_mean_parametric_ci_orders_negative_profile_bounds():
    matrix = -np.array([[1.0], [2.0], [4.0], [8.0]])
    lower, upper = parametric_ci(matrix, 'geom_mean', pseudocount=0,
                                 ci_level=0.95, threads=1)
    assert lower[0] < calc_avg(matrix, 'geom_mean', pseudocount=0, threads=1)[0] < upper[0]


def test_bootstrap_and_parametric_geom_ci_agree():
    # The bootstrap and analytical intervals should nearly coincide for a large
    # sample. This guards the fix that fixes the geom_mean pseudocount globally
    # rather than recomputing an unstable min-based one per resample.
    rng = np.random.default_rng(5)
    matrix = np.abs(rng.standard_normal((800, 12)) * 2 + 6)
    lo_a, hi_a = parametric_ci(matrix, 'geom_mean', pseudocount=-1, ci_level=0.95, threads=1)
    lo_b, hi_b = bootstrap_ci(matrix, 'geom_mean', pseudocount=-1,
                              ci_level=0.95, n_resamples=1500, seed=1,
                              show_progress=False, threads=1)
    width_a = np.ma.filled(hi_a - lo_a, np.nan)
    width_b = np.ma.filled(hi_b - lo_b, np.nan)
    ratio = width_b / width_a
    np.testing.assert_allclose(np.nanmean(ratio), 1.0, atol=0.1)


@pytest.mark.parametrize('average_type', ['mean', 'geom_mean'])
def test_fused_center_and_ci_matches_separate_calls(average_type):
    # Positive, negative, mixed-sign and partly masked columns, in the float32
    # layout the native summary path expects.
    rng = np.random.default_rng(11)
    positive = np.abs(rng.standard_normal((40, 1)) * 2 + 6)
    negative = -np.abs(rng.standard_normal((40, 1)) * 2 + 6)
    mixed = rng.standard_normal((40, 1)) * 5
    data = np.hstack([positive, negative, mixed]).astype(np.float32)
    if average_type == 'geom_mean':
        data[:, 2] = np.abs(data[:, 2])
    data[0, 2] = np.nan  # a masked observation in the mixed-sign column
    matrix = np.ma.masked_invalid(data)

    center, lower, upper = statistics_module.center_and_parametric_ci(
        matrix, average_type, pseudocount=-1, ci_level=0.95, threads=1)
    expected_center = calc_avg(matrix, average_type, pseudocount=-1, axis=0, threads=1)
    expected_lower, expected_upper = parametric_ci(
        matrix, average_type, pseudocount=-1, ci_level=0.95, threads=1)

    np.testing.assert_allclose(np.ma.filled(center, np.nan),
                               np.ma.filled(expected_center, np.nan))
    np.testing.assert_allclose(np.ma.filled(lower, np.nan),
                               np.ma.filled(expected_lower, np.nan))
    np.testing.assert_allclose(np.ma.filled(upper, np.nan),
                               np.ma.filled(expected_upper, np.nan))


def test_fused_center_and_ci_uses_one_native_pass(monkeypatch):
    matrix = np.ma.masked_invalid(
        (np.abs(np.random.default_rng(3).standard_normal((30, 4))) + 1)
        .astype(np.float32))
    summarize_calls = []
    original = statistics_module._native_summarize

    def counting_summarize(block, geom, pseudocount, threads):
        summarize_calls.append(geom)
        return original(block, geom, pseudocount, threads)

    monkeypatch.setattr(statistics_module, '_native_summarize',
                        counting_summarize)
    # calc_avg's native helpers must not run on the fused ci path.
    monkeypatch.setattr(statistics_module, 'reduce',
                        lambda *a, **k: pytest.fail('centre used a second pass'))
    monkeypatch.setattr(statistics_module, '_native_geom_mean',
                        lambda *a, **k: pytest.fail('centre used a second pass'))
    statistics_module.center_and_parametric_ci(matrix, 'mean', threads=1)
    assert summarize_calls == [False]


def test_fused_center_and_ci_accepts_float64_inputs(monkeypatch):
    matrix = np.array([[1.0, -8.0], [4.0, -2.0], [9.0, -18.0], [2.0, -5.0]])
    center, lower, upper = statistics_module.center_and_parametric_ci(
        matrix, 'geom_mean', pseudocount=-1, ci_level=0.95, threads=1)
    expected_center = calc_avg(matrix, 'geom_mean', pseudocount=-1, axis=0, threads=1)
    expected_lower, expected_upper = parametric_ci(
        matrix, 'geom_mean', pseudocount=-1, ci_level=0.95, threads=1)
    np.testing.assert_allclose(np.ma.filled(center, np.nan),
                               np.ma.filled(expected_center, np.nan))
    np.testing.assert_allclose(np.ma.filled(lower, np.nan),
                               np.ma.filled(expected_lower, np.nan))
    np.testing.assert_allclose(np.ma.filled(upper, np.nan),
                               np.ma.filled(expected_upper, np.nan))


@pytest.mark.parametrize('plot_type', ['std', 'se'])
def test_geometric_band_is_multiplicative_on_positive_data(plot_type):
    # Positive data with pc=0 is the textbook case: the band is GM x/ (G)SD,
    # i.e. exp(logmean +/- margin), not the arithmetic centre +/- spread.
    matrix = np.array([[1.0], [2.0], [4.0], [8.0], [16.0]])
    logs = np.log(matrix[:, 0])
    n = len(logs)
    logstd_pop = logs.std(ddof=0)
    margin = logstd_pop if plot_type == 'std' else logstd_pop / np.sqrt(n - 1)
    gm = np.exp(logs.mean())
    expected = np.exp([logs.mean() - margin, logs.mean() + margin])

    center, lower, upper = statistics_module.center_and_analytic_band(
        matrix, 'geom_mean', plot_type, pseudocount=0, threads=1)
    np.testing.assert_allclose(center[0], gm)
    np.testing.assert_allclose([lower[0], upper[0]], expected)
    # Multiplicative: the two half-ratios match rather than the two half-widths.
    np.testing.assert_allclose(upper[0] / gm, gm / lower[0])


def test_geometric_band_differs_from_arithmetic_and_stays_ordered():
    # With a positive pseudocount the band is asymmetric in linear space and
    # must not coincide with an arithmetic centre +/- raw-SD band.
    rng = np.random.default_rng(7)
    matrix = np.abs(rng.standard_normal((60, 3)) * 3 + 8)
    center, lower, upper = statistics_module.center_and_analytic_band(
        matrix, 'geom_mean', 'std', pseudocount=-1, threads=1)
    assert np.all(lower < center) and np.all(center < upper)
    arithmetic = np.ma.std(np.ma.masked_invalid(matrix), axis=0, ddof=0)
    # The lower arm of a multiplicative band is closer to the centre than the
    # upper arm, so it cannot equal a symmetric arithmetic spread.
    assert not np.allclose(center - lower, arithmetic)


def test_analytic_band_rejects_incoherent_centre():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    for plot_type in ('se', 'std', 'ci'):
        with pytest.raises(ValueError, match='bootstrap'):
            statistics_module.center_and_analytic_band(
                matrix, 'median', plot_type, threads=1)


def test_profile_rejects_spread_band_for_median_centre():
    from deeptoolsr.plotProfile import process_args
    with pytest.raises(SystemExit):
        process_args(['-m', 'x.gz', '-o', 'o.png',
                      '--plotType', 'se', '--averageType', 'median'])


@pytest.mark.parametrize('plot_type', ['std', 'ci'])
def test_heatmap_rejects_summary_spread_band_for_median_centre(plot_type):
    from deeptoolsr.plotHeatmap import process_args
    with pytest.raises(SystemExit):
        process_args(['-m', 'x.gz', '-o', 'o.png',
                      '--plotTypeSummaryPlot', plot_type,
                      '--averageTypeSummaryPlot', 'median'])


@pytest.mark.parametrize('plot_type', ['ci', 'bootstrap'])
def test_heatmap_summary_plot_draws_interval_band(tmp_path, plot_type):
    from deeptoolsr.plotHeatmap import main
    matrix = Path(__file__).parent / 'test_heatmapper' / 'master.mat.gz'
    data = tmp_path / 'profile.tab'
    main(['-m', str(matrix), '-o', str(tmp_path / 'out.png'),
          '--plotTypeSummaryPlot', plot_type, '--bootstrapReplicates', '20',
          '--outFileNameData', str(data)])
    assert (tmp_path / 'out.png').stat().st_size > 0


def _batch_specs(count, average_type='geom_mean'):
    from deeptoolsr.plotting.profile import ProfileSeriesSpec
    rng = np.random.default_rng(4)
    return [ProfileSeriesSpec(
        np.ma.masked_invalid((np.abs(rng.standard_normal((200, 12))) + 1)
                             .astype(np.float32)),
        'g{}'.format(k), '#123456') for k in range(count)]


@pytest.mark.parametrize('plot_type', ['ci', 'std', 'se', 'lines'])
def test_series_batch_matches_sequential(plot_type):
    from deeptoolsr.plotting.profile import (
        prepare_profile_series,
        prepare_profile_series_batch)

    specs = _batch_specs(9)
    options = StatisticsSpec(average_type='geom_mean',
                             plot_type=plot_type)
    sequential = [prepare_profile_series(s, options, threads=1) for s in specs]

    batched = prepare_profile_series_batch(specs, options, threads=4)

    assert len(batched) == len(sequential)
    for want, got in zip(sequential, batched):
        np.testing.assert_allclose(
            np.ma.filled(got.statistics.center, np.nan),
            np.ma.filled(want.statistics.center, np.nan))
        if want.statistics.lower is not None:
            np.testing.assert_allclose(
                np.ma.filled(got.statistics.lower, np.nan),
                np.ma.filled(want.statistics.lower, np.nan))
            np.testing.assert_allclose(
                np.ma.filled(got.statistics.upper, np.nan),
                np.ma.filled(want.statistics.upper, np.nan))


def test_series_batch_uses_one_inner_thread_when_fanned_out(monkeypatch):
    from deeptoolsr.plotting.profile import (
        prepare_profile_series_batch)

    seen = []
    real = statistics_module._native_summarize

    def spy(matrix, geom, pseudocount, threads):
        seen.append(threads)
        return real(matrix, geom, pseudocount, threads)

    monkeypatch.setattr(statistics_module, '_native_summarize', spy)
    prepare_profile_series_batch(
        _batch_specs(6),
        StatisticsSpec(average_type='mean', plot_type='ci'),
        threads=4)
    # Every native summary inside the fan-out ran with a single inner thread.
    assert seen and set(seen) == {1}


def test_series_batch_bootstrap_stays_sequential(monkeypatch):
    from deeptoolsr import stats as stats_module
    from deeptoolsr.plotting.profile import (
        prepare_profile_series_batch)
    # The bootstrap path carries its own parallelism; the batch must not fan it
    # out onto the shared thread pool.

    def fail(*_a, **_k):
        raise AssertionError('bootstrap must not run on the series thread pool')
    monkeypatch.setattr(stats_module, 'parallel_map', fail)
    result = prepare_profile_series_batch(
        _batch_specs(4),
        StatisticsSpec(average_type='mean', plot_type='bootstrap',
                       bootstrap_replicates=10), threads=4)
    assert len(result) == 4


def test_geom_bootstrap_resolves_auto_in_native_without_discarded_center(monkeypatch):
    matrix = np.array([[0.0, -8.0], [4.0, -2.0], [9.0, -18.0]])
    expected = statistics_module.bootstrap_ci(
        matrix, 'geom_mean', pseudocount=1, n_resamples=40, include_center=True, threads=1)
    monkeypatch.setattr(statistics_module._statistics, 'geom_mean_axis',
                        lambda *a, **k: pytest.fail('computed a discarded center'))
    actual = statistics_module.bootstrap_ci(
        matrix, 'geom_mean', pseudocount=-1, n_resamples=40, include_center=True, threads=1)
    np.testing.assert_array_equal(actual, expected)


def test_standard_error_band_uses_finite_count_per_bin():
    ss = pytest.importorskip('scipy.stats', reason='SciPy oracle absent')
    matrix = np.array([
        [1.0, 2.0, 5.0],
        [3.0, 4.0, np.nan],
        [np.nan, 8.0, np.nan],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    expected = ss.sem(matrix[:, :2], axis=0, nan_policy='omit')

    _, lower, upper = statistics_module.center_and_analytic_band(
        matrix, 'mean', 'se', threads=1)
    np.testing.assert_allclose(((upper - lower) / 2)[:2], expected[:2])
    assert np.ma.is_masked(lower[2]) and np.ma.is_masked(upper[2])


def test_parametric_ci_rejects_unsupported_average_type():
    matrix = np.arange(20.0).reshape(5, 4)
    try:
        parametric_ci(matrix, 'median', threads=1)
    except ValueError as error:
        assert 'only mean and geom_mean' in str(error)
    else:
        raise AssertionError('median parametric CI should be rejected')


def test_profile_arguments_expose_new_statistics_and_ci(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                         '--averageType', 'trim_mean', '--trim_perc', '0.1',
                         '--plotType', 'bootstrap', '--ci_level', '0.8',
                         '--bootstrapReplicates', '500', '-p', '2'])
    assert args.averageType == 'trim_mean'
    assert args.trim_perc == 0.1
    assert args.plotType == 'bootstrap'
    assert args.ci_level == 0.8
    assert args.bootstrapReplicates == 500
    assert args.numberOfProcessors == 2


def test_profile_arguments_accept_parametric_ci(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                         '--averageType', 'geom_mean', '--plotType', 'ci'])
    assert args.plotType == 'ci'


@pytest.mark.parametrize('removed_name',
                         ['trimmean_gene', 'trimmean_bin', 'logmean'])
def test_removed_average_type_names_are_rejected(tmp_path, removed_name):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    with pytest.raises(SystemExit):
        process_args(['-m', str(matrix),
                      '-out', str(tmp_path / 'plot.png'),
                      '--averageType', removed_name])


def test_zero_pseudocount_is_explicit():
    matrix = np.array([[1.0], [4.0]])
    np.testing.assert_allclose(calc_avg(matrix, 'geom_mean', pseudocount=0, threads=1), [2.0])


def test_profile_default_pseudocount_is_auto(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                         '--averageType', 'geom_mean'])
    assert args.pseudocount == -1


# From audit followup 2026 09 12.

@pytest.fixture()
def coverage_isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path / 'config'))


@pytest.mark.usefixtures('coverage_isolated_config')
@pytest.mark.parametrize('pc', [-1, 0, 0.5, 100000000.0])
@pytest.mark.parametrize('sign', [-1, 1])
def test_geometric_bootstrap_constant_columns_and_missingness(pc, sign):
    data = sign * np.array([[2.0, 0.0, np.nan], [2.0, 0.0, np.nan], [2.0, np.nan, np.nan]])
    center, lower, upper = stats.bootstrap_ci(data, 'geom_mean', pseudocount=pc, n_resamples=41, seed=912, threads=2, include_center=True, show_progress=False)
    for result in (center, lower, upper):
        np.testing.assert_array_equal(np.ma.filled(result, np.nan), [sign * 2.0, 0.0, np.nan])


# From audit regressions.

@pytest.fixture()
def io_native_backends(monkeypatch):
    monkeypatch.delenv('DTP_BAM_BACKEND', raising=False)


@pytest.mark.usefixtures('io_native_backends')
def test_trimmed_mean_uses_finite_count_per_bin():
    data = np.array([[1], [2], [3], [100], [np.nan], [np.nan], [np.nan], [np.nan]], np.float32)
    np.testing.assert_allclose(stats.calc_avg(data, 'trim_mean', trim_perc=0.25, threads=1), [2.5])


@pytest.mark.usefixtures('io_native_backends')
def test_geometric_bootstrap_rejects_invalid_signed_estimator():
    data = np.array([[-1.0], [1.0]], np.float32)
    with pytest.raises(ValueError, match='mixed positive and negative'):
        stats.bootstrap_ci(data, 'geom_mean', pseudocount=1, n_resamples=1000, threads=1)


# From equivalence regressions.

def test_e03_bootstrap_processor_argument_is_honored(monkeypatch):
    seen = []

    def capture(*args):
        seen.append(args[-2])
        return (np.zeros(2), np.zeros(2))
    monkeypatch.setattr(statistics_module._statistics, 'bootstrap_ci', capture)
    for processors in [1, 2, 4]:
        bootstrap_ci(np.ones((3, 2)), 'mean', threads=processors)
    assert seen == [1, 2, 4]


def test_bootstrap_explicit_processors_are_forwarded(monkeypatch):
    observed = []

    def capture(*args):
        observed.append(args[-2])
        return (np.zeros(1), np.zeros(1))
    monkeypatch.setattr(statistics_module._statistics, 'bootstrap_ci', capture)
    bootstrap_ci(np.ones((4, 1)), 'mean', threads=4)
    assert observed == [4]
