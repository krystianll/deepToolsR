"""Direct tests for the native C++ kernels added for the memory work.

Covers the correctness-critical pieces that were previously only exercised
indirectly: the reductions, the in-place sort permutation, the t-digest
quantiles, the heatmap rasteriser, the moments used for confidence intervals,
and the finite -> NaN normalisation invariant.
"""
from deeptoolsr import stats as kernels
import math
from deeptoolsr import _statistics as io__statistics
from deeptoolsr.plotting.heatmap import _resolve_color_limits
from deeptoolsr import _raster as general__raster
from deeptoolsr import _statistics as numerical__statistics

import gzip
import json
import subprocess
import sys

import numpy as np
import pytest

from deeptoolsr.matrix import Block

_statistics = pytest.importorskip('deeptoolsr._statistics')
_io = pytest.importorskip('deeptoolsr._compute_matrix_io')
_raster = pytest.importorskip('deeptoolsr._raster')


# --------------------------------------------------------------------------- #
# reduce_axis
# --------------------------------------------------------------------------- #

_REDUCE_OPS = {
    'mean': np.nanmean,
    'min': np.nanmin,
    'max': np.nanmax,
    'median': np.nanmedian,
    'std': lambda a, axis: np.nanstd(a, axis=axis),
}


@pytest.fixture
def noisy_matrix():
    rng = np.random.default_rng(1234)
    a = (rng.standard_normal((300, 64)) * 7).astype(np.float32)
    a[a > 12] = np.nan          # scattered missing
    a[17, :] = np.nan           # an all-missing row
    a[:, 40] = np.nan           # an all-missing column
    return a


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('op', sorted(_REDUCE_OPS))
@pytest.mark.parametrize('axis', [0, 1])
def test_reduce_axis_matches_numpy(noisy_matrix, op, axis):
    got = np.asarray(_statistics.reduce_axis(noisy_matrix, axis, op, 4))
    with np.errstate(all='ignore'):
        expected = _REDUCE_OPS[op](noisy_matrix, axis=axis)
    finite = np.isfinite(got) & np.isfinite(expected)
    assert np.allclose(got[finite], expected[finite], rtol=1e-4, atol=1e-4)
    # NaN placement must agree (all-missing lines produce NaN).
    assert np.array_equal(np.isnan(got), np.isnan(expected))


def test_reduce_axis_sum_of_all_missing_is_zero(noisy_matrix):
    # NumPy parity: nansum of an all-missing line is 0, not NaN.
    got = np.asarray(_statistics.reduce_axis(noisy_matrix, 1, 'sum', 4))
    expected = np.nansum(noisy_matrix, axis=1)
    assert np.allclose(got, expected, rtol=1e-4, atol=1e-3)
    assert got[17] == 0.0        # the all-missing row


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_reduce_axis_handles_strided_view(noisy_matrix):
    view = noisy_matrix[:, ::2]   # non-contiguous column slice
    assert not view.flags.c_contiguous
    got = np.asarray(_statistics.reduce_axis(view, 0, 'mean', 4))
    with np.errstate(all='ignore'):
        expected = np.nanmean(view, axis=0)
    finite = np.isfinite(got) & np.isfinite(expected)
    assert np.allclose(got[finite], expected[finite], rtol=1e-4, atol=1e-4)


def test_reduce_axis_respects_nan_missing(noisy_matrix):
    masked = np.ma.masked_greater(np.ma.masked_invalid(noisy_matrix), 5.0)
    data = np.ma.filled(masked, np.nan).astype(np.float32)
    got = np.asarray(_statistics.reduce_axis(data, 0, 'mean', 4))
    expected = np.ma.mean(masked, axis=0).filled(np.nan)
    finite = np.isfinite(got) & np.isfinite(expected)
    assert np.allclose(got[finite], expected[finite], rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
# permute_rows_inplace (sort data integrity)
# --------------------------------------------------------------------------- #

def test_permute_rows_matches_gather():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((40, 9)).astype(np.float32)
    order = np.argsort(a[:, 0])
    expected = a[order, :]
    _statistics.permute_rows_inplace(a, 0, a.shape[0], [int(i) for i in order])
    assert np.array_equal(a, expected)


def test_permute_rows_partial_range_and_bool():
    rng = np.random.default_rng(8)
    a = rng.standard_normal((30, 5)).astype(np.float32)
    mask = rng.random((30, 5)) > 0.5
    a_exp, m_exp = a.copy(), mask.copy()
    order = np.argsort(a[10:20, 1])
    a_exp[10:20, :] = a[10:20, :][order, :]
    m_exp[10:20, :] = mask[10:20, :][order, :]
    order_list = [int(i) for i in order]
    _statistics.permute_rows_inplace(a, 10, 20, order_list)
    _statistics.permute_rows_inplace(mask, 10, 20, order_list)
    assert np.array_equal(a, a_exp)
    assert np.array_equal(mask, m_exp)


@pytest.mark.parametrize('using', ['mean', 'median', 'max', 'min', 'sum'])
@pytest.mark.parametrize('method', ['ascend', 'descend'])
def test_sort_groups_native_matches_numpy(using, method):
    from deeptoolsr.matrix import Matrix, MatrixHeader, RowLayout, sort

    rng = np.random.default_rng(3)
    data = rng.standard_normal((12, 6)).astype(np.float32)
    data[data > 1.5] = np.nan
    regions = [['c', [(i, i + 1)], 'r%d' % i, 6 if i < 6 else 12, '+', str(i)]
               for i in range(12)]

    m = Matrix(MatrixHeader.from_parameters({
        'group_boundaries': [0, 6, 12], 'sample_boundaries': [0, 3, 6],
        'group_labels': ['g0', 'g1'], 'sample_labels': ['s0', 's1']}),
        data.copy(), [r[:] for r in regions], None)
    keys = getattr(np, 'nan' + using)(data.astype(np.float64), axis=1)
    expected = []
    for start in (0, 6):
        order = np.argsort(keys[start:start + 6], kind='stable') + start
        expected.extend(order[::-1] if method == 'descend' else order)
    layout = sort(m, RowLayout.identity(m), using=using, method=method,
                  threads=1)
    assert [m.regions[index][2] for index in layout.row_indices()] == [
        regions[i][2] for i in expected]
    np.testing.assert_array_equal(m.values[layout.row_indices()], data[expected])


# --------------------------------------------------------------------------- #
# nan_quantiles (t-digest)
# --------------------------------------------------------------------------- #

def test_nan_quantiles_exact_matches_numpy():
    rng = np.random.default_rng(11)
    a = (rng.standard_normal((400, 30)) * 4 + 3).astype(np.float32)
    a[a > 9] = np.nan
    got = np.asarray(_statistics.nan_quantiles(a, [1.0, 50.0, 98.0],
                                               25_000_000, 0))
    expected = np.nanpercentile(a, [1.0, 50.0, 98.0])
    assert np.allclose(got, expected, rtol=0, atol=1e-4)


def test_nan_quantiles_tdigest_is_close():
    rng = np.random.default_rng(12)
    a = (rng.standard_normal((2000, 200)) * 10 + 50).astype(np.float32)
    # Force the streaming t-digest path with a tiny max_exact.
    got = np.asarray(_statistics.nan_quantiles(a, [2.0, 98.0], 0, 1000))
    expected = np.nanpercentile(a, [2.0, 98.0])
    # Approximate but tight relative to the data's spread (~10).
    assert np.allclose(got, expected, rtol=0, atol=0.5)


def test_nan_quantiles_all_missing_is_nan():
    a = np.full((10, 10), np.nan, dtype=np.float32)
    got = np.asarray(_statistics.nan_quantiles(a, [1.0, 98.0], 25_000_000, 0))
    assert np.all(np.isnan(got))


def test_nan_quantiles_parallel_tdigest_reproducible_and_close():
    matrix = np.random.default_rng(503).lognormal(0, 1, (3000, 600)).astype(np.float32)
    probs = [0.1, 1.0, 50.0, 99.0, 99.9]
    observed = [np.asarray(_statistics.nan_quantiles(
        matrix, probs, 1048576, 1000, threads))
        for threads in (1, 2, 4, 4)]
    assert all(np.array_equal(observed[0], values) for values in observed[1:])
    assert np.array_equal(observed[0], np.asarray(_statistics.nan_quantiles(
        matrix, probs, 1048576, 1000)))
    assert np.allclose(observed[0], np.percentile(matrix, probs), rtol=0,
                       atol=0.6)


def test_nan_quantiles_digest_memory_is_bounded():
    # A fresh process isolates peak RSS from pytest's other allocations.
    code = '''
import json
import platform
import numpy as np
from deeptoolsr import _statistics

if platform.system() == 'Windows':
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t)]

    psapi = ctypes.WinDLL('psapi')
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    process = ctypes.windll.kernel32.GetCurrentProcess()

    def peak_mib():
        counters = Counters(cb=ctypes.sizeof(Counters))
        assert psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / (1024 * 1024)
else:
    import resource

    def peak_mib():
        scale = 1024 * 1024 if platform.system() == 'Darwin' else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale

matrix = np.random.default_rng(504).random((10000, 2000), dtype=np.float32)
# Run every pool worker once first: thread stacks (and, under x64 emulation
# on Windows arm64, per-thread translation state) are not digest memory.
_statistics.parallel_worker_count(4)
_statistics.nan_quantiles(matrix[:64], [50], 1048576, 1000, 4)
before = peak_mib()
result = _statistics.nan_quantiles(matrix, [1, 50, 99], 1048576, 1000, 4)
after = peak_mib()
print(json.dumps([before, after, np.asarray(result).tolist()]))
'''
    output = subprocess.run([sys.executable, '-c', code], check=True,
                            capture_output=True, text=True)
    before, after, values = json.loads(output.stdout)
    assert np.all(np.isfinite(values))
    # 17 compression-1000 digests reserve about 1.1 MiB; 4 MiB permits
    # allocator and thread-pool overhead without allowing one per input chunk.
    assert after - before <= 4.0


# --------------------------------------------------------------------------- #
# summarize_columns (moments behind CI)
# --------------------------------------------------------------------------- #

def test_summarize_columns_moments_match_numpy():
    rng = np.random.default_rng(21)
    a = (rng.standard_normal((500, 16)) * 3 + 5).astype(np.float32)
    a[a > 11] = np.nan
    mean, std, count, sign, pc = _statistics.summarize_columns(
        a, False, -1.0, 4)
    ref_mean = np.nanmean(a, axis=0)
    ref_std = np.nanstd(a, axis=0)                       # ddof=0 (population)
    ref_count = np.sum(np.isfinite(a), axis=0)
    assert np.allclose(mean, ref_mean, rtol=1e-4, atol=1e-4)
    assert np.allclose(std, ref_std, rtol=1e-4, atol=1e-4)
    assert np.array_equal(np.asarray(count), ref_count)
    assert np.asarray(sign).size == 0                    # empty for non-geom


def test_summarize_columns_geom_centre_matches_geom_mean():
    rng = np.random.default_rng(22)
    a = np.abs(rng.standard_normal((300, 8)) * 2 + 4).astype(np.float32)
    mean, std, count, sign, pc = _statistics.summarize_columns(
        a, True, -1.0, 4)
    sign = np.asarray(sign)
    centre = _statistics.geometric_inverse(np.asarray(mean), sign, pc)
    ref = np.asarray(_statistics.geom_mean_axis(a, 0, -1.0, 4)[0])
    assert np.allclose(centre, ref, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# bootstrap_ci
# --------------------------------------------------------------------------- #

@pytest.fixture
def boot_matrix():
    rng = np.random.default_rng(101)
    a = (rng.standard_normal((1500, 20)) * 3 + 12).astype(np.float32)
    a[a > 18] = np.nan
    a[:, 5] = np.nan            # an all-missing column -> NaN CI
    return a


def _boot(a, stat, seed=1, threads=4, R=400, trim=0.1):
    lo, up = _statistics.bootstrap_ci(a, stat, -1.0, trim,
                                      R, 0.95, seed, threads, False)
    return np.asarray(lo), np.asarray(up)


@pytest.mark.parametrize('stat', ['mean', 'median', 'trim_mean'])
def test_bootstrap_ci_brackets_point_estimate(boot_matrix, stat):
    lo, up = _boot(boot_matrix, stat)
    assert np.all(lo[np.isfinite(lo)] <= up[np.isfinite(up)])
    # The point estimate should sit inside the CI for almost every bin.
    valid = np.isfinite(boot_matrix).any(axis=0)
    centre = np.full(boot_matrix.shape[1], np.nan)
    if stat == 'median':
        centre[valid] = np.nanmedian(boot_matrix[:, valid], axis=0)
    else:
        centre[valid] = np.nanmean(boot_matrix[:, valid], axis=0)  # trim_mean ~ mean here
    ok = (lo <= centre) & (centre <= up)
    assert np.mean(ok[np.isfinite(centre)]) > 0.9
    assert np.isnan(lo[5]) and np.isnan(up[5])      # all-missing column


def test_bootstrap_ci_geom_brackets_geom_centre(boot_matrix):
    a = np.abs(boot_matrix)                          # keep positive for geom
    lo, up = _boot(a, 'geom_mean')
    centre = np.asarray(_statistics.geom_mean_axis(a, 0, -1.0, 4)[0])
    ok = (lo <= centre) & (centre <= up)
    finite = np.isfinite(centre) & np.isfinite(lo)
    assert np.mean(ok[finite]) > 0.9


def test_bootstrap_ci_is_gene_stable():
    # Shared seed across bins means identical columns must draw the same genes
    # and therefore get identical CIs (the R-script behaviour). Under per-bin
    # independent seeding they would differ.
    rng = np.random.default_rng(3)
    base = (rng.standard_normal((800, 1)) * 3 + 10).astype(np.float32)
    other = (rng.standard_normal((800, 1)) * 3 + 10).astype(np.float32)
    a = np.hstack([base, base, other])
    lo, up = _boot(a, 'mean', R=300)
    assert lo[0] == lo[1] and up[0] == up[1]     # identical columns -> identical CI
    assert lo[0] != lo[2]                          # a different column differs


def test_bootstrap_ci_reproducible_and_thread_independent(boot_matrix):
    a = _boot(boot_matrix, 'mean', seed=42, threads=1)[0]
    b = _boot(boot_matrix, 'mean', seed=42, threads=1)[0]
    assert np.array_equal(a, b, equal_nan=True)         # same seed -> identical
    c = _boot(boot_matrix, 'mean', seed=42, threads=8)[0]
    assert np.array_equal(a, c, equal_nan=True)         # thread-count independent
    d = _boot(boot_matrix, 'mean', seed=99, threads=1)[0]
    assert not np.array_equal(a, d, equal_nan=True)     # different seed differs


# --------------------------------------------------------------------------- #
# render_heatmap (colour delta)
# --------------------------------------------------------------------------- #

def _lut_and_bad(cmap):
    import matplotlib.colors as mc
    lut = np.ascontiguousarray(
        cmap(np.linspace(0, 1, 256), bytes=True)[:, :4], dtype=np.uint8)
    bad = (np.array(mc.to_rgba(cmap.get_bad())) * 255).round().astype(np.uint8)
    return lut, [int(x) for x in bad]


def test_render_heatmap_matches_matplotlib_colours():
    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.colors as mc
    rng = np.random.default_rng(31)
    a = (rng.random((60, 40)) * 10).astype(np.float32)
    a[a > 9] = np.nan
    vmin, vmax = 1.0, 8.0
    cmap = matplotlib.colormaps['viridis'].copy()
    cmap.set_bad((0.1, 0.2, 0.3, 1.0))
    lut, bad = _lut_and_bad(cmap)
    # out == src -> no resize, so we compare the raw colour mapping.
    got = np.asarray(_raster.render_heatmap(
        a, lut, bad, vmin, vmax,
        a.shape[0], a.shape[1], 'triangle', 8, 4))
    ref = cm.ScalarMappable(norm=mc.Normalize(vmin, vmax), cmap=cmap).to_rgba(
        a, bytes=True)
    finite = np.isfinite(a)
    # LUT is 256 entries + 8-bit rounding -> at most a few levels off.
    assert np.abs(got[finite].astype(int) - ref[finite].astype(int)).max() <= 3
    # NaN pixels take the bad colour exactly.
    assert np.array_equal(got[~finite][0], np.array(bad, dtype=np.uint8))


def test_render_heatmap_clamps_out_of_range():
    a = np.array([[-100.0, 0.5, 100.0]], dtype=np.float32)
    cmap_lut = np.zeros((256, 4), dtype=np.uint8)
    cmap_lut[:, 0] = np.arange(256)          # red ramp = index
    cmap_lut[:, 3] = 255
    got = np.asarray(_raster.render_heatmap(
        a, cmap_lut, [0, 0, 0, 255], 0.0, 1.0,
        1, 3, 'triangle', 8, 1))
    assert got[0, 0, 0] == 0                  # below vmin -> first entry
    assert got[0, 2, 0] == 255                # above vmax -> last entry


def test_deferred_heatmap_rasters_parallel_matches_sequential(monkeypatch):
    # Parallel across-block rasterisation must place pixels identical to the
    # sequential path (each block renders with a single inner native thread).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from deeptoolsr.plotting import heatmap as PH
    from deeptoolsr.options import RasterOptions

    lut = np.ascontiguousarray(
        matplotlib.colormaps['viridis'](np.linspace(0, 1, 256), bytes=True)[:, :4],
        dtype=np.uint8)

    def build(n, rows, cols):
        fig, axes = plt.subplots(1, n)
        axes = np.atleast_1d(axes)
        reqs = []
        for k in range(n):
            data = np.random.default_rng(500 + k).standard_normal(
                (rows, cols)).astype(np.float32)
            block = Block(data, None, (0, rows), None, (0, cols))
            reqs.append({
                'ax': axes[k], 'data': block,
                'rows': rows,
                'cols': cols, 'lut': lut, 'bad': [0, 0, 0, 255],
                'vmin': -2.0, 'vmax': 2.0, 'out_h': min(rows, 2000),
                'out_w': min(cols, 2000), 'extent': [0, cols, rows, 0],
                'interpolation': 'antialiased', 'alpha': 1.0,
                'raster': RasterOptions('cubic', 8),
                'threads': 1})
        return fig, reqs

    fig_seq, seq = build(10, 800, 240)
    PH.flush_deferred_heatmap_rasters(seq, threads=1)
    seq_pixels = [r['ax'].images[0].get_array().copy() for r in seq]

    fig_par, par = build(10, 800, 240)
    PH.flush_deferred_heatmap_rasters(par, threads=4)
    par_pixels = [r['ax'].images[0].get_array().copy() for r in par]

    for a, b in zip(seq_pixels, par_pixels):
        assert np.array_equal(a, b)
    plt.close(fig_seq)
    plt.close(fig_par)


def test_deferred_heatmap_image_establishes_axis_limits():
    # Regression: deferring the block's imshow must still set the axis data
    # limits immediately, so downstream code that reads/sets limits or draws in
    # data coordinates (the gene-length boundary curve, x ticks, region borders)
    # sees the image extent. Without this the axis keeps its default (0, 1)
    # limits and every block renders as a mis-scaled, flat-looking sliver.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from deeptoolsr.plotting import heatmap as PH
    from deeptoolsr.options import RasterOptions

    rows, cols = 2000, 200
    data = np.random.default_rng(0).standard_normal((rows, cols)).astype(np.float32)
    cmap = matplotlib.colormaps['viridis']
    fig, ax = plt.subplots()
    deferred = []
    PH.draw_heatmap_image(ax, Block(data, None, (0, rows), None, (0, cols)),
                          cmap, -1.0, 1.0, 1.0,
                          'bilinear', rows, cols, 200, deferred=deferred,
                          raster=RasterOptions('cubic', 8),
                          threads=1)
    # Coordinate system established before any pixels are placed.
    assert tuple(ax.get_xlim()) == (0, cols)
    assert tuple(ax.get_ylim()) == (rows, 0)
    assert len(deferred) == 1 and not ax.images

    PH.flush_deferred_heatmap_rasters(deferred, 1)
    assert len(ax.images) == 1
    assert list(ax.images[0].get_extent()) == [0, cols, rows, 0]
    plt.close(fig)


@pytest.mark.parametrize('filt', ['triangle', 'mitchell', 'cubic', 'box'])
@pytest.mark.parametrize('bits', [8, 16])
def test_render_heatmap_variants_downsample(filt, bits):
    rng = np.random.default_rng(33)
    a = (rng.random((200, 120)) * 5).astype(np.float32)
    lut = np.tile(np.arange(256, dtype=np.uint8)[:, None], (1, 4))
    lut[:, 3] = 255
    out = np.asarray(_raster.render_heatmap(
        a, lut, [0, 0, 0, 255], 0.0, 5.0,
        50, 60, filt, bits, 4))
    assert out.shape == (50, 60, 4)
    assert out.dtype == np.uint8


# --------------------------------------------------------------------------- #
# finite -> NaN normalisation invariant (reader + writer)
# --------------------------------------------------------------------------- #

def _header(rows, bins):
    return {
        'sample_labels': ['s'], 'sample_boundaries': [0, bins],
        'group_labels': ['g'], 'group_boundaries': [0, rows],
        'verbose': False, 'scale': 1, 'skip zeros': False,
        'nan after end': False, 'proc number': 1, 'sort regions': 'keep',
        'sort using': 'mean', 'unscaled 5 prime': [0], 'unscaled 3 prime': [0],
        'body': [0], 'downstream': [0], 'upstream': [0], 'ref point': ['TSS'],
        'bin size': [1], 'missing data as zero': False, 'min threshold': None,
        'max threshold': None, 'scalar': [1], 'bin avg type': 'mean',
    }


def test_reader_normalises_non_finite_to_nan(tmp_path):
    path = tmp_path / 'infs.gz'
    header = json.dumps(_header(1, 4), separators=(',', ':'))
    line = 'chr1\t0\t4\tregion\t.\t+\t1.0\tinf\t-inf\tnan\n'
    with gzip.open(path, 'wt') as handle:
        handle.write('@' + header + '\n')
        handle.write(line)
    _hdr, _regions, matrix = _io.read_matrix(
        str(path), 1, 4, [0, 1], 1)
    assert matrix.dtype == np.float32
    assert not np.isinf(matrix).any()               # inf collapsed
    assert matrix[0, 0] == np.float32(1.0)
    assert np.isnan(matrix[0, 1]) and np.isnan(matrix[0, 2])
    assert np.isnan(matrix[0, 3])


def test_writer_emits_nan_for_non_finite(tmp_path):
    path = tmp_path / 'out.gz'
    values = np.array([[1.0, np.inf, -np.inf, np.nan, 2.0]], dtype=np.float32)
    regions = [['chr1', [(0, 5)], 'region', 0, '+', '0']]
    _io.write_matrix(str(path), json.dumps(_header(1, 5), separators=(',', ':')),
                     regions, values, 1, 6, 256)
    with gzip.open(path, 'rt') as handle:
        fields = handle.readlines()[1].rstrip().split('\t')[6:]
    assert fields[1] == 'nan' and fields[2] == 'nan' and fields[3] == 'nan'
    assert float(fields[0]) == 1.0 and float(fields[4]) == 2.0


def test_native_pool_grows_past_first_participation_cap():
    import os
    import subprocess
    import sys

    if (os.cpu_count() or 1) < 4:
        pytest.skip('requires at least four CPUs')
    code = '''
from deeptoolsr import _statistics
assert _statistics.parallel_worker_count(2) == 2
assert _statistics.parallel_worker_count(4) == 4
assert _statistics.pool_workers() >= 4
'''
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


# From audit followup 2026 09 12.

@pytest.fixture()
def coverage_isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path / 'config'))


@pytest.mark.usefixtures('coverage_isolated_config')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('operation', ['mean', 'sum', 'std', 'min', 'max', 'median', 'trim_mean'])
def test_native_statistics_against_scalar_finite_oracle(dtype, axis, operation):
    rng = np.random.default_rng(912)
    data = rng.normal(size=(23, 17)).astype(dtype)
    data[::3, ::2] = np.nan
    data[1, 3] = np.inf
    data[3, 5] = -np.inf
    data[:, 0] = np.nan
    data[0, :] = np.nan
    mask = rng.random(data.shape) < 0.15
    data[mask] = np.nan
    data = data[::-1, ::-1]
    observed = kernels.reduce(data, operation, axis, trim_perc=0.15, threads=3)
    expected = []
    for line in data.T if axis == 0 else data:
        values = sorted((float(x) for x in line if np.isfinite(x)))
        n = len(values)
        if not n:
            expected.append(0 if operation == 'sum' else np.nan)
        elif operation == 'sum':
            expected.append(math.fsum(values))
        elif operation == 'mean':
            expected.append(math.fsum(values) / n)
        elif operation == 'std':
            center = math.fsum(values) / n
            expected.append(math.sqrt(math.fsum(((x - center) ** 2 for x in values)) / n))
        elif operation == 'median':
            expected.append(values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2)
        elif operation == 'trim_mean':
            trim = int(n * 0.15)
            selected = values[trim:n - trim]
            expected.append(math.fsum(selected) / len(selected))
        else:
            expected.append(values[0] if operation == 'min' else values[-1])
    np.testing.assert_allclose(observed, expected, rtol=4e-14, atol=2e-15, equal_nan=True)


# From audit regressions.

@pytest.fixture()
def io_native_backends(monkeypatch):
    monkeypatch.delenv('DTP_BAM_BACKEND', raising=False)


@pytest.mark.usefixtures('io_native_backends')
@pytest.mark.parametrize('kernel', ['reduce_axis', 'summarize_columns'])
def test_variance_is_stable_for_low_spread_float32_values(kernel):
    data = np.full((100000, 1), 1000000.0, np.float32)
    data[::2, 0] = np.nextafter(data[0, 0], np.float32(np.inf))
    if kernel == 'reduce_axis':
        observed = io__statistics.reduce_axis(data, 0, 'std', 1)
    else:
        observed = io__statistics.summarize_columns(data, False, 0, 1)[1]
    np.testing.assert_allclose(observed, data.astype(float).std(axis=0), rtol=1e-07)


# From general audit regressions.

@pytest.mark.parametrize('missing', [False, True])
def test_large_native_percentile_fallback_includes_extrema(missing):
    from deeptoolsr import stats as kernels
    data = np.full((1100, 1000), np.nan if missing else 0.0, dtype=np.float32)
    if not missing:
        data[-1, -1] = 10.0
    q = kernels.quantiles(data, [0.0, 1.0, 98.0, 100.0], threads=1)
    if missing:
        assert np.isnan(q).all()
    else:
        np.testing.assert_array_equal(q[[0, 3]], [0, 10])
    lower, upper = _resolve_color_limits(None, None, q)
    assert lower[0] == 0 and upper[0] > 0


# From general second audit regressions.

def native_raster(values, lut, bit_depth=8):
    return general__raster.render_heatmap(values, lut=lut, bad=[3, 5, 7, 255], vmin=0.0, vmax=1.0, filter='nearest', bit_depth=bit_depth)


def strided_lut(kind):
    if kind == 'rows':
        backing = np.full((512, 4), 99, dtype=np.uint8)
        lut = backing[::2]
    else:
        backing = np.full((256, 8), 99, dtype=np.uint8)
        lut = backing[:, ::2]
    lut[:] = [12, 34, 56, 255]
    return lut


@pytest.mark.parametrize('kind', ['rows', 'channels'])
@pytest.mark.parametrize('bit_depth', [8, 16])
def test_h04_strided_color_tables_match_the_same_contiguous_colors(kind, bit_depth):
    lut = strided_lut(kind)
    actual = native_raster(np.array([[0.0, 1 / 255, 1.0]], np.float32), lut, bit_depth)
    np.testing.assert_array_equal(actual, [[[12, 34, 56, 255]] * 3])


@pytest.mark.parametrize('kind', ['rows', 'channels'])
@pytest.mark.parametrize('bit_depth', [8, 16])
def test_contiguous_color_table_control(kind, bit_depth):
    actual = native_raster(np.array([[0.0, 1 / 255, 1.0]], np.float32), np.ascontiguousarray(strided_lut(kind)), bit_depth)
    np.testing.assert_array_equal(actual, [[[12, 34, 56, 255]] * 3])


@pytest.mark.parametrize('layout', ['contiguous', 'row_stride', 'column_stride', 'reversed'])
@pytest.mark.parametrize('extra_missing', [False, True])
@pytest.mark.parametrize('bit_depth', [8, 16])
def test_native_raster_value_views_preserve_missingness(layout, extra_missing, bit_depth):
    values = np.array([[0.0, 0.5, np.nan], [1.0, 0.25, 0.75]], np.float32)
    if extra_missing:
        values[0, 1] = values[1, 2] = np.nan
    if layout == 'row_stride':
        backing = np.zeros((4, 3), np.float32)
        backing[::2] = values
        values = backing[::2]
    elif layout == 'column_stride':
        backing = np.zeros((2, 6), np.float32)
        backing[:, ::2] = values
        values = backing[:, ::2]
    elif layout == 'reversed':
        values = values[::-1, ::-1]
    lut = np.zeros((256, 4), np.uint8)
    lut[:, 0] = np.arange(256, dtype=np.uint8)
    lut[:, 3] = 255
    expected = np.empty((*values.shape, 4), np.uint8)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = float(values[row, col])
            expected[row, col] = [3, 5, 7, 255] if not np.isfinite(value) else [int(np.floor(value * 255 + 0.5)), 0, 0, 255]
    np.testing.assert_array_equal(native_raster(values, lut, bit_depth), expected)


@pytest.mark.parametrize('layout', ['reversed_rows', 'reversed_channels', 'transposed'])
@pytest.mark.parametrize('bit_depth', [8, 16])
def test_native_lut_views_follow_logical_colors_without_modifying_input(layout, bit_depth):
    lut = np.arange(1024, dtype=np.uint16).astype(np.uint8).reshape(256, 4)
    if layout == 'reversed_rows':
        lut = lut[::-1]
    elif layout == 'reversed_channels':
        lut = lut[:, ::-1]
    else:
        lut = np.ascontiguousarray(lut.T).T
    before = lut.copy()
    values = np.array([[0.0, 1 / 255, 1.0]], np.float32)
    expected = before[[0, 1, 255]][None, :, :]
    np.testing.assert_array_equal(native_raster(values, lut, bit_depth), expected)
    np.testing.assert_array_equal(lut, before)


# From numerical audit regressions.

@pytest.mark.parametrize('max_exact', [100, 0])
def test_n09_exact_percentile_keeps_constant_subnormal(max_exact):
    tiny = np.nextafter(0.0, 1.0)
    result = numerical__statistics.nan_quantiles(np.full((2, 1), tiny), [50], max_exact=max_exact, exact=True, num_threads=1)
    assert result[0] == tiny


def test_n10_tdigest_keeps_finite_constant_with_default_dispatch():
    maximum = np.finfo(np.float64).max
    values = np.broadcast_to(np.array([[maximum]]), (1048577, 1))
    result = numerical__statistics.nan_quantiles(values, [0, 1, 25, 50, 75, 99, 100], num_threads=1)
    np.testing.assert_array_equal(result, maximum)
