"""Transient clustering workspace and native cluster ordering."""

import numpy as np

from deeptoolsr import _cluster, _statistics
from deeptoolsr.matrix import (
    GroupOrigin, PreparationWarning, RowLayout, _readonly_indices,
)


def cluster(matrix, layout, k, *, method='kmeans', cols=None, seed=0,
            threads, ward_distance_budget_bytes=2 << 30):
    if not 1 <= k <= layout.nrows:
        raise ValueError('number of clusters must be between 1 and the number of regions')
    columns = _readonly_indices(cols)
    workspace = _statistics.copy_values(
        matrix.values, True, threads, rows=layout.rows,
        row_range=(0, layout.nrows) if layout.rows is None else None,
        cols=columns)
    if layout.nrows == 1:
        cluster_labels = np.zeros(1, dtype=np.int64)
    elif method == 'kmeans':
        cluster_labels, _, _ = _cluster.kmeans(workspace, k, seed=seed,
                                               threads=threads)
    elif method == 'hierarchical':
        cluster_labels = _cluster.cut_maxclust(
            _cluster.ward_linkage(workspace, threads=threads,
                                  distance_budget_bytes=ward_distance_budget_bytes),
            k) - 1
    else:
        raise ValueError(f'unknown clustering method: {method}')
    populated = np.unique(cluster_labels)
    warning = (PreparationWarning('unpopulated_clusters',
                                  (k, len(populated)))
               if len(populated) < k else None)
    cluster_ids = [np.flatnonzero(cluster_labels == number)
                   for number in populated]
    means = [_statistics.reduce_axis(
        workspace[ids, :].reshape(1, -1), 1, 'mean', threads)[0]
        for ids in cluster_ids]
    order = np.argsort(means)[::-1]
    current = layout.row_indices()
    rows = np.concatenate([current[cluster_ids[i]] for i in order])
    bounds = [0]
    for i in order:
        bounds.append(bounds[-1] + len(cluster_ids[i]))
    names = tuple(f'cluster_{i}' for i in range(1, len(order) + 1))
    result = RowLayout(rows, tuple(bounds), names,
                       tuple(GroupOrigin(i) for i in range(len(names))),
                       layout.sort and layout.sort.reordered(), columns)
    return result, warning
