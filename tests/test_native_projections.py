"""Projected native inputs must behave like materialised matrices."""

import json

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io as matrix_io
from deeptoolsr import _raster as raster
from deeptoolsr import _statistics as statistics


def _same(actual, expected):
    if isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _same(left, right)
    elif expected is None:
        assert actual is None
    elif isinstance(expected, np.ndarray):
        assert np.asarray(actual).dtype == expected.dtype
        assert np.asarray(actual).shape == expected.shape
        assert np.asarray(actual).tobytes() == expected.tobytes()
    else:
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('threads', [1, 4])
def test_projected_statistics_match_materialised(dtype, threads):
    rng = np.random.default_rng(713)
    source = rng.uniform(0.2, 10, (7, 8)).astype(dtype)
    source[1, 3] = np.nan
    source[4, 6] = np.nan
    rows = np.array([4, 1, 4, 0], dtype=np.int64)
    cols = np.array([6, 1, 3, 1], dtype=np.int64)
    materialised = source[np.ix_(rows, cols)]
    calls = [
        (statistics.reduce_axis, (0, 'median', threads)),
        (statistics.reduce_axis, (1, 'mean', threads)),
        (statistics.nan_quantiles, ([10, 50, 90], 1048576, 0, threads, True)),
        (statistics.summarize_columns, (False, -1, threads)),
        (statistics.bootstrap_ci, ('mean', -1, 0, 20, .95, 7, threads, False, True)),
        (statistics.geom_mean_axis, (0, -1, threads)),
        (statistics.nonzero_extreme, (True,)),
        (statistics.copy_values, (True, threads)),
        (statistics.filter_rows, ('mean', 0, 20, False, threads, 0)),
        (statistics.filter_matrix, ([0, 2, 4], [0, 1], 'mean', 0, 20,
                                    0, 'maskSample', threads)),
        (statistics.finite_values, ()),
        (statistics.silhouette_scores, ([0, 0, 1, 1],)),
    ]
    for function, args in calls:
        kwargs = {'num_threads': threads} if function is statistics.silhouette_scores else {}
        expected = function(materialised, *args, **kwargs)
        actual = function(source, *args, rows=rows, cols=cols, **kwargs)
        _same(actual, expected)

    # A range is used only when there is no index list on that axis.
    subset = source[1:6, 2:7]
    _same(statistics.reduce_axis(source, 0, 'median', threads,
                                 row_range=(1, 6), col_range=(2, 7)),
          statistics.reduce_axis(subset, 0, 'median', threads))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('threads', [1, 4])
def test_projected_tdigest_and_raster(dtype, threads):
    rng = np.random.default_rng(41)
    source = rng.lognormal(size=(90, 65)).astype(dtype)
    rows = rng.permutation(90)[:70].astype(np.int64)
    cols = np.r_[np.arange(12), np.arange(40, 55), [4, 4]].astype(np.int64)
    projected = source[np.ix_(rows, cols)]
    args = ([0.1, 50, 99.9], 100, 1000, threads, False)
    _same(statistics.nan_quantiles(source, *args, rows=rows, cols=cols),
          statistics.nan_quantiles(projected, *args))

    lut = np.tile(np.arange(256, dtype=np.uint8)[:, None], (1, 4))
    bad = np.array([0, 0, 0, 0], dtype=np.uint8)
    render_args = (lut, bad, 0, 15, 35, 20, 'triangle', 8, threads)
    _same(raster.render_heatmap(source, *render_args, rows=rows, cols=cols),
          raster.render_heatmap(projected, *render_args))


def test_empty_and_invalid_projections():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    empty = np.array([], dtype=np.int64)
    _same(statistics.copy_values(source, rows=empty), source[:0].copy())
    _same(statistics.reduce_axis(source, 0, 'sum', cols=empty),
          statistics.reduce_axis(source[:, :0], 0, 'sum'))
    _same(statistics.copy_values(source, cols=empty), source[:, :0].copy())
    for bad in (np.array([3], dtype=np.int64), np.array([-1], dtype=np.int64)):
        with pytest.raises(ValueError, match='out of bounds'):
            statistics.copy_values(source, rows=bad)
    for bad in (np.array([1], dtype=np.int32), np.array([[1]], dtype=np.int64),
                np.arange(4, dtype=np.int64)[::2]):
        with pytest.raises((ValueError, TypeError), match='int64'):
            statistics.copy_values(source, rows=bad)
    with pytest.raises((ValueError, TypeError), match='int64'):
        statistics.copy_values(source, rows=[0, 1])
    with pytest.raises(ValueError, match='conflict'):
        statistics.copy_values(source, rows=empty, row_range=(0, 1))
    with pytest.raises(ValueError, match='conflict'):
        statistics.copy_values(source, cols=empty, col_range=(0, 1))
    with pytest.raises(ValueError, match='out of bounds'):
        statistics.copy_values(source, row_range=(0, 4))
    with pytest.raises(ValueError, match='labels must match rows'):
        statistics.silhouette_scores(source, [0, 1], rows=np.array([0, 1, 2]))


def test_silhouette_zero_missing_matches_imputed_copy():
    source = np.array([[1, np.nan, 3], [2, 4, np.nan],
                       [8, 3, 2], [9, np.nan, 2]], dtype=np.float32)
    labels = [0, 0, 1, 1]
    imputed = statistics.copy_values(source, True)
    _same(statistics.silhouette_scores(source, labels, zero_missing=True,
                                       num_threads=4),
          statistics.silhouette_scores(imputed, labels, zero_missing=False,
                                       num_threads=4))


def test_composed_permutations_and_distance_projection():
    source = np.arange(42, dtype=np.float64).reshape(6, 7)[::-1, ::-1]
    first = np.array([5, 1, 3, 0], dtype=np.int64)
    second = np.array([2, 0, 2], dtype=np.int64)
    columns = np.array([6, 0, 4, 0], dtype=np.int64)
    composed = first[second]
    _same(statistics.copy_values(source, rows=composed, cols=columns),
          statistics.copy_values(source[np.ix_(first, columns)][second]))

    distances = np.abs(np.subtract.outer(np.arange(5), np.arange(5))).astype(np.float64)
    order = np.array([4, 2, 0, 3], dtype=np.int64)
    labels = [0, 0, 1, 1]
    _same(statistics.silhouette_score(distances, 0, labels,
                                      rows=order, cols=order),
          statistics.silhouette_score(distances[np.ix_(order, order)], 0, labels))


@pytest.mark.parametrize('threads', [1, 4])
def test_projected_writer_matches_materialised(tmp_path, threads):
    values = np.arange(20, dtype=np.float32).reshape(4, 5)
    regions = [['chr1', [(i, i + 1)], str(i), 0, '+', '0'] for i in range(4)]
    rows = np.array([3, 1, 2], dtype=np.int64)
    cols = np.array([4, 0, 2, 1], dtype=np.int64)
    header = json.dumps({'sample_labels': ['sample'], 'sample_boundaries': [0, 4],
                         'group_labels': ['group'], 'group_boundaries': [0, 3]},
                        separators=(',', ':'))
    projected = tmp_path / 'projected.gz'
    materialised = tmp_path / 'materialised.gz'
    matrix_io.write_matrix(str(projected), header, regions, values, threads,
                           row_order=rows, column_order=cols)
    matrix_io.write_matrix(str(materialised), header,
                           [regions[i] for i in rows], values[np.ix_(rows, cols)], threads)
    assert projected.read_bytes() == materialised.read_bytes()

    empty_rows = tmp_path / 'empty.gz'
    empty_header = json.dumps({'sample_labels': ['sample'], 'sample_boundaries': [0, 4],
                               'group_labels': ['group'], 'group_boundaries': [0, 0]},
                              separators=(',', ':'))
    matrix_io.write_matrix(str(empty_rows), empty_header, regions, values, threads,
                           row_order=np.array([], dtype=np.int64), column_order=cols)
    assert len(empty_rows.read_bytes()) > 0
    with pytest.raises(ValueError, match='at least one column'):
        matrix_io.write_matrix(str(tmp_path / 'empty_cols.gz'), header, regions,
                               values, threads, column_order=np.array([], dtype=np.int64))

    for row_order, column_order in ((np.array([1, 1], dtype=np.int64), None),
                                    (None, np.array([2, 2], dtype=np.int64))):
        with pytest.raises(ValueError, match='distinct indices'):
            matrix_io.write_matrix(str(tmp_path / 'repeated.gz'), header, regions,
                                   values, threads, row_order=row_order,
                                   column_order=column_order)
    with pytest.raises(ValueError, match='float32'):
        matrix_io.write_matrix(str(tmp_path / 'float64.gz'), header, regions,
                               values.astype(np.float64), threads)
