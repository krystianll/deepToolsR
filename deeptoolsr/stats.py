"""Projected native statistics and numeric profile calculations."""

import sys
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from deeptoolsr import _statistics
from deeptoolsr.matrix import Block, projection_kwargs
from deeptoolsr.parallel import parallel_map

STATISTIC_CHOICES = _statistics.statistic_choices
FILTER_NAN_MODES = _statistics.filter_nan_modes


def _projection(matrix):
    return projection_kwargs(matrix) if isinstance(matrix, Block) else {}


def buffers(matrix):
    data = np.asarray(matrix.values if isinstance(matrix, Block) else matrix)
    # Preserve both production float32 buffers and caller-supplied float64.
    # Integer/list inputs are small compatibility inputs, converted explicitly.
    if data.dtype.kind not in 'buif':
        raise TypeError('matrix must contain real numeric values')
    if data.dtype not in (np.dtype('float32'), np.dtype('float64')):
        data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError('matrix must be two-dimensional')
    return data


def reduce(matrix, operation, axis=0, trim_perc=0.05, *, threads,
           empty_sum_missing=False):
    data = buffers(matrix)
    return _statistics.reduce_axis(data, axis, operation,
                                   threads,
                                   trim_perc, empty_sum_missing,
                                   **_projection(matrix))


def quantiles(matrix, probabilities, exact=False, *, threads):
    data = buffers(matrix)
    return _statistics.nan_quantiles(data, probabilities,
                                     num_threads=threads, exact=exact,
                                     **_projection(matrix))


def filter_rows(matrix, statistic, low, high, inclusive=False, nan_mode=0,
                *, threads):
    data = buffers(matrix)
    return _statistics.filter_rows(data, statistic, low, high, inclusive,
                                   threads, nan_mode, **_projection(matrix))


def filter_matrix(matrix, sample_boundaries, filter_samples, statistic, low, high,
                  nan_mode=0, on_fail='removeRegion', *, threads):
    data = buffers(matrix)
    return _statistics.filter_matrix(
        data, sample_boundaries, filter_samples, statistic, low, high,
        nan_mode, on_fail, threads, **_projection(matrix))


def copy_values(matrix, zero_missing=False, *, threads):
    data = buffers(matrix)
    return _statistics.copy_values(data, zero_missing, threads,
                                   **_projection(matrix))


def _native_geom_mean(matrix, pseudocount, axis, threads):
    data = buffers(matrix)
    result, _ = _statistics.geom_mean_axis(data, axis, float(pseudocount),
                                           threads, **_projection(matrix))
    return result


def calc_avg(matrix, average_type, pseudocount=-1, trim_perc=0.05, axis=0,
             *, threads):
    if average_type == 'geom_mean':
        value = _native_geom_mean(matrix, pseudocount, axis, threads)
    else:
        value = reduce(matrix, average_type, axis, trim_perc, threads=threads,
                       empty_sum_missing=average_type == 'sum')
    result = np.ma.masked_invalid(value, copy=False)
    return result


def _native_bootstrap_ci(matrix, average_type, pseudocount, trim_perc,
                         ci_level, n_resamples, seed, processors, show_progress,
                         include_center=False):
    data = buffers(matrix)
    processors = int(processors)
    if processors < 1:
        raise ValueError('processors must be at least 1')
    extra = {'include_center': True} if include_center else {}
    results = _statistics.bootstrap_ci(
        data, average_type,
        float(pseudocount), float(trim_perc), int(n_resamples), float(ci_level),
        int(seed), processors, bool(show_progress),
        **extra, **_projection(matrix))
    return tuple(np.ma.masked_invalid(value, copy=False) for value in results)


def bootstrap_ci(matrix, average_type, pseudocount=-1, trim_perc=0.05,
                 ci_level=0.95, n_resamples=300, seed=0,
                 show_progress=None, *, threads, include_center=False):
    data = buffers(matrix)
    if (matrix.shape if isinstance(matrix, Block) else data.shape)[0] == 0:
        raise ValueError('bootstrap confidence intervals require a non-empty 2D matrix')
    if n_resamples < 1:
        raise ValueError('n_resamples must be at least 1')
    if not 0 < ci_level < 1:
        raise ValueError('ci_level must be greater than 0 and less than 1')
    if show_progress is None:
        show_progress = sys.stderr.isatty()
    return _native_bootstrap_ci(matrix, average_type, pseudocount, trim_perc,
                                ci_level, n_resamples, seed, threads, show_progress,
                                include_center=include_center)


def _native_summarize(matrix, geom, pseudocount, threads):
    data = buffers(matrix)
    mean, std, count, sign, pc = _statistics.summarize_columns(
        data, bool(geom),
        float(pseudocount), threads, **_projection(matrix))
    return mean, std, count, sign if geom else None, pc


ANALYTIC_BAND_TYPES = ('se', 'std', 'ci')


def _analytic_summary(matrix, is_geom, pseudocount, threads):
    return _native_summarize(matrix, is_geom, pseudocount, threads)


def center_and_analytic_band(matrix, average_type, plot_type, pseudocount=-1,
                             ci_level=0.95, *, threads):
    """Centre and ``(lower, upper)`` for an analytic band from one native summary.

    Supports ``plot_type`` in :data:`ANALYTIC_BAND_TYPES` for ``mean`` and
    ``geom_mean`` only. The three bands share a single per-column summary and
    differ only in the margin applied to the mean:

    - ``std``: one population standard deviation;
    - ``se``:  the standard error ``std_pop / sqrt(count - 1)``;
    - ``ci``:  ``se`` scaled by Student's t for ``ci_level``.

    For ``geom_mean`` the summary is taken in log space, so the margin is a
    *geometric* standard deviation / standard error and the band is
    back-transformed multiplicatively with the same sign and pseudocount the
    centre uses: ``endpoint = sign*pc*expm1(logmean +/- margin)`` for positive
    ``pc``, and ``sign*exp(logmean +/- margin)`` for zero ``pc``. When
    ``pc = 0`` this is the textbook ``GM x/ GSD``; a positive pseudocount makes
    the band asymmetric in linear space, which is the only form consistent with
    the shifted-log model shared by the geometric centre and CI. Pairing these
    bands with median/sum/etc. is rejected -- use bootstrap for those centres.
    """
    if plot_type not in ANALYTIC_BAND_TYPES:
        raise ValueError(
            'analytic bands support only {}'.format(ANALYTIC_BAND_TYPES))
    if average_type not in ('mean', 'geom_mean'):
        raise ValueError(
            "the '{}' band supports only mean and geom_mean; use bootstrap "
            "for {}".format(plot_type, average_type))
    if plot_type == 'ci' and not 0 < ci_level < 1:
        raise ValueError('ci_level must be greater than 0 and less than 1')

    is_geom = average_type == 'geom_mean'
    mean, std_pop, count, sign, pc = _analytic_summary(
        matrix, is_geom, pseudocount, threads)
    # 'ci'/'se' divide by (count - 1); a whole series with no bin of at least
    # two observations cannot form a spread. 'std' still yields a zero-width
    # band for single-observation bins, matching the raw reduction.
    if plot_type in ('ci', 'se') and np.any(count > 0) and np.max(count) < 2:
        raise ValueError(
            "the '{}' band requires at least two observations".format(plot_type))

    with np.errstate(invalid='ignore', divide='ignore'):
        if plot_type == 'std':
            margin = std_pop
        else:
            margin = std_pop / np.sqrt(count - 1.0)
            if plot_type == 'ci':
                margin = margin * _statistics.student_t_quantile(
                    count - 1.0, (1.0 + ci_level) / 2.0)
        if is_geom:
            center = _statistics.geometric_inverse(mean, sign, pc)
            endpoint_a = _statistics.geometric_inverse(mean - margin, sign, pc)
            endpoint_b = _statistics.geometric_inverse(mean + margin, sign, pc)
            lower = np.minimum(endpoint_a, endpoint_b)
            upper = np.maximum(endpoint_a, endpoint_b)
        else:
            center = mean
            lower, upper = mean - margin, mean + margin
    return (np.ma.masked_invalid(center),
            np.ma.masked_invalid(lower),
            np.ma.masked_invalid(upper))


def center_and_parametric_ci(matrix, average_type, pseudocount=-1,
                             ci_level=0.95, *, threads):
    """Centre and parametric interval from one native summary.

    Thin wrapper over :func:`center_and_analytic_band` for ``plot_type='ci'``,
    kept as the named entry point for the confidence-interval path.
    """
    return center_and_analytic_band(matrix, average_type, 'ci', pseudocount,
                                    ci_level, threads=threads)


def parametric_ci(matrix, average_type, pseudocount=-1, ci_level=0.95,
                  *, threads):
    _, lower, upper = center_and_parametric_ci(
        matrix, average_type, pseudocount, ci_level, threads=threads)
    return lower, upper


@dataclass(frozen=True)
class StatisticsSpec:
    """Effective numeric profile settings, independent of presentation."""
    average_type: str = 'mean'
    plot_type: str = 'lines'
    pseudocount: float = -1
    trim_perc: float = 0.05
    ci_level: float = 0.95
    bootstrap_replicates: int = 300


def statistics_spec(spec):
    """Project a figure request onto only the settings used by its numbers."""
    average = spec.average_type
    band = (spec.plot_type if spec.plot_type in
            ('se', 'std', 'ci', 'bootstrap') else 'lines')
    return StatisticsSpec(
        average_type=average, plot_type=band,
        pseudocount=spec.pseudocount if average == 'geom_mean' else -1,
        trim_perc=spec.trim_perc if average == 'trim_mean' else 0.05,
        ci_level=spec.ci_level if band in ('ci', 'bootstrap') else 0.95,
        bootstrap_replicates=(spec.bootstrap_replicates
                              if band == 'bootstrap' else 300))


def calculate_group_statistics_batch(data, pairs, spec, *, threads):
    """Compute projected group/sample statistics in the requested order."""
    blocks = (data.layout.block(data.matrix, group, sample)
              for group, sample in pairs)
    return calculate_profile_batch(blocks, spec, threads=threads)


def scan_prepared_matrix(data, probabilities, exact=False, *, threads):
    """Scan the full prepared row order for figure-wide percentiles."""
    block = Block(data.matrix.values, data.layout.rows,
                  (0, data.layout.nrows), None,
                  (0, data.matrix.values.shape[1]))
    return quantiles(block, probabilities, exact=exact, threads=threads)


@dataclass(frozen=True)
class ProfileStatistics:
    """Numeric centre and optional interval for one projected block."""

    x: np.ndarray
    center: np.ndarray
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None


class NumericProvider(Protocol):
    """Numeric cache boundary shared by profile and heatmap preparation."""

    def statistics_batch(self, data, pairs, spec, threads):
        """Return statistics in ``pairs`` order, batching cache misses."""

    def scan(self, data, probabilities, exact, threads):
        """Return figure-wide quantiles in the prepared row order."""


def calculate_profile_statistics(block, options, *, threads):
    """Calculate the centre and optional interval exactly once per series."""
    lower = upper = None

    # Each branch fills in the same (lower, upper) envelope around `center`
    # by a different method, so the drawing code stays agnostic of which was
    # used. The analytic bands (se/std/ci) all derive centre and interval from
    # one per-column summary -- arithmetic for `mean`, geometric (log-space
    # margin, back-transformed) for `geom_mean` -- so an SD/SE/CI band is always
    # coherent with its centre. 'bootstrap' instead resamples rows of the matrix
    # (expensive; count/processors are caller-tunable) and works for any centre.
    if options.plot_type in ANALYTIC_BAND_TYPES:
        center, lower, upper = center_and_analytic_band(
            block, options.average_type, options.plot_type,
            options.pseudocount, options.ci_level, threads=threads)
    elif options.plot_type == 'bootstrap':
        center, lower, upper = bootstrap_ci(
            block, options.average_type, options.pseudocount,
            options.trim_perc, options.ci_level,
            n_resamples=options.bootstrap_replicates,
            threads=threads, include_center=True)
    else:
        center = np.ma.asarray(calc_avg(
            block, options.average_type, options.pseudocount,
            options.trim_perc, axis=0, threads=threads))

    center = np.ma.asarray(center)
    x = np.arange(center.size)
    return ProfileStatistics(
        x=x, center=center,
        lower=None if lower is None else np.ma.asarray(lower),
        upper=None if upper is None else np.ma.asarray(upper))


def calculate_profile_batch(blocks, options, *, threads):
    """Schedule all numeric series in one bounded figure-wide batch."""
    blocks = list(blocks)
    if options.plot_type == 'bootstrap':
        return [calculate_profile_statistics(block, options, threads=threads)
                for block in blocks]
    return parallel_map(
        lambda block, inner: calculate_profile_statistics(
            block, options, threads=inner), blocks, threads)
