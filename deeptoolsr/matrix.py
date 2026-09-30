"""Frozen plot matrix, projected row layouts and byte-compatible writers."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import blake2b
import json
import os
from types import MappingProxyType

import numpy as np

from deeptoolsr import _compute_matrix_io, _statistics
from deeptoolsr.matrix_file import open_matrix_text
from deeptoolsr.matrix_validation import (
    SPECIAL_SAMPLE_PARAMETERS, normalize_sample_parameters,
    validate_matrix_geometry, validate_matrix_header,
)
from deeptoolsr.path_validation import atomic_output_path

SPECIAL_PARAMS = SPECIAL_SAMPLE_PARAMETERS


class MatrixFormatError(ValueError):
    """Malformed matrix content, distinct from unrelated numeric errors."""


@dataclass(frozen=True)
class MatrixHeader:
    parameters: Mapping
    sample_labels: tuple[str, ...]
    sample_boundaries: tuple[int, ...]
    group_labels: tuple[str, ...]
    group_boundaries: tuple[int, ...]

    @classmethod
    def from_parameters(cls, parameters):
        return cls(MappingProxyType(dict(parameters)),
                   tuple(parameters['sample_labels']),
                   tuple(parameters['sample_boundaries']),
                   tuple(parameters['group_labels']),
                   tuple(parameters['group_boundaries']))


def read_header(path, *, normalize=False, scan=False, missing_message=None):
    """Read and validate a JSON header without loading matrix rows."""
    with open_matrix_text(path, 'rt') as handle:
        if scan:
            line = ''
            for line in handle:
                if line.startswith('@') or line.strip():
                    break
        else:
            line = handle.readline()
    if not line.startswith('@'):
        raise ValueError(missing_message or
                         'Matrix file has no computeMatrix JSON header')
    parameters = json.loads(line[1:].strip())
    validate_matrix_header(parameters)
    if normalize:
        parameters = normalize_sample_parameters(parameters)
    return MatrixHeader.from_parameters(parameters)


@dataclass(frozen=True)
class SourceKey:
    realpath: str
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int
    st_ctime_ns: int

    @classmethod
    def of(cls, path):
        realpath = os.path.realpath(path)
        stat = os.stat(realpath)
        return cls(realpath, stat.st_dev, stat.st_ino, stat.st_size,
                   stat.st_mtime_ns, stat.st_ctime_ns)


def _load(path, threads):
    header = read_header(path)
    parameters = header.parameters
    native_header, regions, values = _compute_matrix_io.read_matrix(
        str(path), header.group_boundaries[-1],
        header.sample_boundaries[-1], parameters['group_boundaries'], threads)
    parameters = json.loads(native_header)
    validate_matrix_header(parameters, rows=values.shape[0],
                           bins=values.shape[1])
    validate_matrix_geometry(
        values, parameters['group_boundaries'],
        parameters['sample_boundaries'],
        group_labels=parameters['group_labels'],
        sample_labels=parameters['sample_labels'], regions=regions)
    return (MatrixHeader.from_parameters(
        normalize_sample_parameters(parameters)), values, regions)


@dataclass(frozen=True, eq=False)
class Matrix:
    header: MatrixHeader
    values: np.ndarray
    regions: list
    source: SourceKey | None

    @classmethod
    def load(cls, path, threads):
        header, values, regions = _load(path, threads)
        values.flags.writeable = False
        return cls(header, values, regions, SourceKey.of(path))


@dataclass(eq=False)
class OwnedMatrix:
    """Writable batch buffer; plotting receives only the frozen form."""

    header: MatrixHeader
    values: np.ndarray
    regions: list
    source: SourceKey | None = None

    @classmethod
    def load(cls, path, threads):
        header, values, regions = _load(path, threads)
        return cls(header, values, regions, SourceKey.of(path))

    @classmethod
    def from_compute(cls, parameters, values, regions):
        validate_matrix_header(parameters, rows=values.shape[0],
                               bins=values.shape[1])
        validate_matrix_geometry(
            values, parameters['group_boundaries'],
            parameters['sample_boundaries'],
            group_labels=parameters['group_labels'],
            sample_labels=parameters['sample_labels'], regions=regions)
        return cls(MatrixHeader.from_parameters(parameters), values, regions)

    def freeze(self):
        self.values.flags.writeable = False
        return Matrix(self.header, self.values, self.regions, self.source)

    def save(self, path, *, compressed, threads, row_order=None,
             column_order=None):
        """Write this owned batch buffer, optionally through index orders."""
        if row_order is not None:
            row_order = np.asarray(row_order, dtype=np.int64, order='C')
        if column_order is not None:
            column_order = np.asarray(column_order, dtype=np.int64, order='C')
        _write_matrix(self, path, dict(self.header.parameters), compressed,
                      threads, row_order, column_order)


@dataclass(frozen=True)
class GroupOrigin:
    source: int
    quantile: int | None = None


@dataclass(frozen=True)
class SortState:
    method: str
    using: str

    def reordered(self):
        """The state after rows are reordered by something other than sort."""
        return replace(self, method='no')


@dataclass(frozen=True)
class Labels:
    group_keys: tuple[str, ...]
    groups: tuple[str, ...]
    samples: tuple[str, ...]


@dataclass(frozen=True)
class PreparationWarning:
    kind: str
    data: tuple = ()


@dataclass(frozen=True, eq=False)
class Block:
    values: np.ndarray
    rows: np.ndarray | None
    row_range: tuple[int, int]
    cols: np.ndarray | None
    col_range: tuple[int, int]

    @property
    def shape(self):
        return ((len(self.rows) if self.rows is not None else
                 self.row_range[1] - self.row_range[0]),
                (len(self.cols) if self.cols is not None else
                 self.col_range[1] - self.col_range[0]))


def projection_kwargs(block):
    """Translate a Block to the mutually exclusive native projections."""
    return {
        'rows': block.rows,
        'row_range': block.row_range if block.rows is None else None,
        'cols': block.cols,
        'col_range': block.col_range if block.cols is None else None,
    }


def _readonly_indices(indices):
    if indices is None:
        return None
    result = np.asarray(indices, dtype=np.int64)
    if not result.flags.c_contiguous or result.flags.writeable:
        result = np.array(result, dtype=np.int64, order='C', copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, eq=False)
class RowLayout:
    rows: np.ndarray | None
    group_bounds: tuple[int, ...]
    base_names: tuple[str, ...]
    origins: tuple[GroupOrigin, ...]
    sort: SortState | None = None
    cluster_columns: np.ndarray | None = None
    silhouette: np.ndarray | None = None
    _digests: dict = field(default_factory=dict, init=False, compare=False,
                           repr=False)

    __hash__ = None

    def __eq__(self, other):
        if not isinstance(other, RowLayout):
            return NotImplemented

        def same(left, right):
            if left is None or right is None:
                return left is right
            return np.array_equal(left, right, equal_nan=True)

        return (self.group_bounds == other.group_bounds and
                self.base_names == other.base_names and
                self.origins == other.origins and self.sort == other.sort and
                same(self.rows, other.rows) and
                same(self.cluster_columns, other.cluster_columns) and
                same(self.silhouette, other.silhouette))

    def __post_init__(self):
        object.__setattr__(self, 'rows', _readonly_indices(self.rows))
        object.__setattr__(self, 'cluster_columns',
                           _readonly_indices(self.cluster_columns))
        if self.silhouette is not None:
            values = np.asarray(self.silhouette)
            values.flags.writeable = False
            object.__setattr__(self, 'silhouette', values)
        if len(self.group_bounds) != len(self.origins) + 1:
            raise ValueError('group bounds and origins disagree')

    @classmethod
    def identity(cls, matrix):
        return cls.from_header(matrix.header)

    @classmethod
    def from_header(cls, header):
        """Create identity membership from validated metadata, without values."""
        sort_state = None
        if 'sort regions' in header.parameters:
            sort_state = SortState(header.parameters['sort regions'],
                                   header.parameters['sort using'])
        return cls(None, header.group_boundaries, header.group_labels,
                   tuple(GroupOrigin(i) for i in range(len(header.group_labels))),
                   sort_state)

    @property
    def nrows(self):
        return self.group_bounds[-1]

    def row_indices(self):
        if self.rows is None:
            return np.arange(self.nrows, dtype=np.int64)
        return self.rows

    def block(self, matrix, group, sample):
        row_start, row_stop = self.group_bounds[group:group + 2]
        col_start, col_stop = matrix.header.sample_boundaries[sample:sample + 2]
        rows = None if self.rows is None else self.rows[row_start:row_stop]
        return Block(matrix.values, rows, (row_start, row_stop), None,
                     (col_start, col_stop))

    def group_digest(self, group):
        if group not in self._digests:
            start, stop = self.group_bounds[group:group + 2]
            indices = (np.arange(start, stop, dtype='<i8') if self.rows is None
                       else np.asarray(self.rows[start:stop], dtype='<i8'))
            self._digests[group] = blake2b(indices.tobytes(),
                                           digest_size=32).digest()
        return self._digests[group]

    def digest(self):
        digest = blake2b(digest_size=32)
        digest.update(np.asarray(self.group_bounds, dtype='<i8').tobytes())
        for group in range(len(self.origins)):
            digest.update(self.group_digest(group))
        return digest.digest()


def filter_values(matrix, layout, low, high, *, nan_mode='keep', cols=None,
                  threads):
    """Keep rows passing the native per-bin policy without copying values."""
    low = -np.inf if low is None else float(low)
    high = np.inf if high is None else float(high)
    columns = _readonly_indices(cols)
    if columns is None:
        sample_bounds = matrix.header.sample_boundaries
    else:
        # Retain sample breaks in a selected, possibly disjoint projection.
        sample_ids = np.searchsorted(
            matrix.header.sample_boundaries[1:], columns, side='right')
        sample_bounds = [0]
        sample_bounds.extend(int(index) for index in
                             np.flatnonzero(np.diff(sample_ids)) + 1)
        sample_bounds.append(len(columns))
    keep, _ = _statistics.filter_matrix(
        matrix.values, sample_bounds, list(range(len(sample_bounds) - 1)),
        'perBin', low, high, _statistics.filter_nan_modes[nan_mode],
        'removeRegion', threads, rows=layout.rows,
        row_range=(0, layout.nrows) if layout.rows is None else None,
        cols=columns)
    indices = layout.row_indices()[np.flatnonzero(keep)]
    bounds = [0]
    for start, stop in zip(layout.group_bounds, layout.group_bounds[1:]):
        bounds.append(bounds[-1] + int(np.count_nonzero(keep[start:stop])))
    return replace(layout, rows=indices, group_bounds=tuple(bounds),
                   silhouette=None)


def remove_empty_groups(layout):
    keep = [i for i, (start, stop) in enumerate(zip(
        layout.group_bounds, layout.group_bounds[1:])) if stop > start]
    bounds = (0,) + tuple(layout.group_bounds[i + 1] for i in keep)
    return (replace(layout, group_bounds=bounds,
                    origins=tuple(layout.origins[i] for i in keep)),
            len(layout.origins) - len(keep))


def merge_groups_by_key(layout, keys):
    if len(keys) != len(layout.origins):
        raise ValueError('group key count does not match groups')
    key_order = list(dict.fromkeys(keys))
    source_rows = layout.row_indices()
    parts = []
    bounds = [0]
    origins = []
    for key in key_order:
        indices = [i for i, item in enumerate(keys) if item == key]
        for i in indices:
            start, stop = layout.group_bounds[i:i + 2]
            parts.append(source_rows[start:stop])
        bounds.append(bounds[-1] + sum(
            layout.group_bounds[i + 1] - layout.group_bounds[i]
            for i in indices))
        origins.append(layout.origins[indices[0]])
    rows = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    return replace(layout, rows=rows, group_bounds=tuple(bounds),
                   origins=tuple(origins), silhouette=None,
                   sort=layout.sort and layout.sort.reordered())


def apply_merge_relabel(layout, overrides):
    if len(overrides) != len(layout.base_names):
        raise ValueError('length new labels != length original labels')
    renamed = replace(layout, base_names=tuple(overrides))
    keys = [overrides[origin.source] for origin in renamed.origins]
    return merge_groups_by_key(renamed, keys)


def sort(matrix, layout, *, using='mean', method='no', cols=None,
         quantiles=1, threads):
    if quantiles < 1:
        raise ValueError('The number of quantiles should be bigger than or equal to 1.')
    if quantiles > 1 and method not in ('ascend', 'descend'):
        raise ValueError('Quantile grouping requires ascending or descending sorting.')
    if method == 'no':
        return layout
    current = layout.row_indices()
    if method == 'keep':
        keys = None
    elif using == 'region_length':
        keys = np.asarray([sum(end - start for start, end in
                               matrix.regions[int(row)][1]) for row in current])
    elif using == 'score':
        values, valid = [], []
        for row in current:
            try:
                score = float(matrix.regions[int(row)][5])
                finite = np.isfinite(score)
            except (TypeError, ValueError, IndexError):
                score, finite = np.nan, False
            values.append(score)
            valid.append(finite)
        keys = np.asarray(values, dtype=float)
        valid = np.asarray(valid, dtype=bool)
    elif using in ('mean', 'median', 'max', 'min', 'sum'):
        columns = _readonly_indices(cols)
        keys = _statistics.reduce_axis(
            matrix.values, 1, using, threads, rows=layout.rows,
            row_range=(0, layout.nrows) if layout.rows is None else None,
            cols=columns)
    else:
        raise ValueError(f'{using} is an unsupported sorting method')

    if quantiles > layout.nrows:
        raise ValueError('The number of quantiles is bigger than the number of regions. Please reduce the number of quantiles.')
    if quantiles > 1:
        bounds = (0, layout.nrows)
        origins = layout.origins[:1]
    else:
        bounds, origins = layout.group_bounds, layout.origins
    parts = []
    for start, stop in zip(bounds, bounds[1:]):
        if method == 'keep':
            order = np.arange(stop - start)
        elif using == 'score':
            local_valid = valid[start:stop]
            local_keys = keys[start:stop]
            valid_order = np.flatnonzero(local_valid)
            direction = -1 if method == 'descend' else 1
            valid_order = valid_order[np.argsort(
                direction * local_keys[valid_order], kind='stable')]
            order = np.concatenate([valid_order, np.flatnonzero(~local_valid)])
        else:
            order = keys[start:stop].argsort()
            if method == 'descend':
                order = order[::-1]
        parts.append(current[start:stop][order])
    rows = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    if quantiles > 1:
        bounds = tuple(int(q * layout.nrows / quantiles)
                       for q in range(quantiles + 1))
        origins = tuple(GroupOrigin(origins[0].source, q)
                        for q in range(1, quantiles + 1))
    return replace(layout, rows=rows, group_bounds=bounds, origins=origins,
                   sort=SortState(method, using), silhouette=None)


def cluster(matrix, layout, k, *, method='kmeans', cols=None, seed=0,
            threads, ward_distance_budget_bytes=2 << 30):
    from deeptoolsr.cluster import cluster as native_cluster

    return native_cluster(matrix, layout, k, method=method, cols=cols,
                          seed=seed, threads=threads,
                          ward_distance_budget_bytes=ward_distance_budget_bytes)


def silhouette(matrix, layout, threads):
    if len(layout.origins) < 2:
        scores = np.full(layout.nrows, np.nan)
    else:
        labels = np.repeat(np.arange(len(layout.origins)),
                           np.diff(layout.group_bounds)).tolist()
        scores = _statistics.silhouette_scores(
            matrix.values, labels, rows=layout.rows,
            row_range=(0, layout.nrows) if layout.rows is None else None,
            cols=layout.cluster_columns, zero_missing=True,
            num_threads=threads)
    return replace(layout, silhouette=scores)


def serialize_header(parameters):
    """Encode the single computeMatrix JSON header format."""
    header = {}
    sample_count = len(parameters['sample_labels'])
    for key, value in parameters.items():
        if type(value) is list and not value:
            value = None
        if key in SPECIAL_PARAMS and type(value) is not list:
            value = [value] * sample_count
        header[key] = value
    return json.dumps(header, separators=(',', ':'))


def _plot_parameters(matrix, layout, labels):
    parameters = dict(matrix.header.parameters)
    parameters['sample_labels'] = list(labels.samples)
    parameters['group_labels'] = list(labels.groups)
    parameters['sample_boundaries'] = list(matrix.header.sample_boundaries)
    parameters['group_boundaries'] = list(layout.group_bounds)
    # The header records the order of the rows written, not the source's.
    if layout.sort is None:
        parameters.pop('sort regions', None)
        parameters.pop('sort using', None)
    else:
        parameters['sort regions'] = layout.sort.method
        parameters['sort using'] = layout.sort.using
    return parameters


def _write_matrix(matrix, path, parameters, compressed, threads,
                  row_order=None, column_order=None):
    header = serialize_header(parameters)
    suffix = '.matrix.gz' if compressed else '.matrix.txt'
    with atomic_output_path(path, suffix=suffix) as temporary:
        _compute_matrix_io.write_matrix(
            str(temporary), header, matrix.regions, matrix.values,
            threads, 6, 0, row_order, column_order, bool(compressed))


def save(matrix, layout, labels, path, *, compressed, threads):
    """Write selected rows through the native writer, atomically."""
    _write_matrix(matrix, path, _plot_parameters(matrix, layout, labels),
                  compressed, threads, layout.rows)


def save_bed(matrix, layout, labels, handle):
    """Write BED12 plus the historical group and silhouette columns."""
    handle.write('#chrom\tstart\tend\tname\tscore\tstrand\tthickStart\tthickEnd'
                 '\titemRGB\tblockCount\tblockSizes\tblockStart\tdeepTools_group')
    if layout.silhouette is not None:
        handle.write('\tsilhouette')
    handle.write('\n')
    rows = layout.row_indices()
    for group, (start, stop) in enumerate(zip(
            layout.group_bounds, layout.group_bounds[1:])):
        for position in range(start, stop):
            region = matrix.regions[int(rows[position])]
            blocks = region[1]
            handle.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{1}\t{2}\t0'.format(
                region[0], blocks[0][0], blocks[-1][1], region[2],
                region[5], region[4]))
            handle.write('\t{0}\t{1}\t{2}\t{3}'.format(
                len(blocks),
                ','.join(str(int(end) - int(first)) for first, end in blocks),
                ','.join(str(int(first) - int(blocks[0][0]))
                         for first, _ in blocks), labels.groups[group]))
            if layout.silhouette is not None:
                handle.write(f'\t{layout.silhouette[position]}')
            handle.write('\n')
