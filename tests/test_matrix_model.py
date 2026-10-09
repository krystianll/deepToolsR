"""Frozen matrix and projected row operations against pinned CLI bytes."""
import bz2
from deeptoolsr import computeMatrixOperations as cmo
from deeptoolsr import computeMatrix
from tests.helpers.parity import original
from tests.helpers.bigwig import pyBigWig
from tests.helpers.parity import read_matrix

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from deeptoolsr import _statistics
from deeptoolsr.matrix import (
    Labels, Matrix, OwnedMatrix, RowLayout, apply_merge_relabel, cluster,
    filter_values, projection_kwargs, read_header,
    remove_empty_groups, save, save_bed, silhouette, sort,
)
from tests.helpers.generate_prepare_expected import (
    CASES, run_case,
)
from tests.helpers.prepare_fixtures import (
    assert_bed_equal, assert_diagnostics_equal,
)


FIXTURES = Path(__file__).parent / 'test_data' / 'prepare_expected'


def _input(name='input.gz'):
    matrix = Matrix.load(FIXTURES / name, threads=1)
    return matrix, RowLayout.identity(matrix)


def _filtered(matrix, layout):
    layout = filter_values(matrix, layout, 0, None, threads=1)
    layout, removed = remove_empty_groups(layout)
    return layout, removed


def _columns(matrix):
    bounds = matrix.header.sample_boundaries
    return np.asarray([*range(bounds[0], bounds[1]),
                       *range(bounds[2], bounds[3])], dtype=np.int64)


SORT_CASES = tuple(name for name in CASES if name.startswith(
    ('sort_ascend_', 'sort_descend_')))


@pytest.mark.parametrize('case', [
    *SORT_CASES, 'sort_disjoint_mean', 'sort_disjoint_median',
    'sort_no', 'sort_keep', 'quantile_custom', 'gzip_matrix',
    'kmeans', 'hclust_selected', 'kmeans_silhouette',
    'empty_override', 'show_counts', 'profile_merge_duplicate',
    'profile_together_duplicate', 'profile_cluster_relabel',
])
def test_row_operations_reproduce_cli_files(case, tmp_path):
    matrix, layout = _input('single.gz' if case == 'quantile_custom'
                            else 'input.gz')
    # Plots no longer re-filter by the header's min threshold (divergence
    # from deepTools 3.5.6); the 'Gone' cases drop Empty via --filterNans.
    if case in ('empty_override', 'profile_merge_duplicate',
                'profile_together_duplicate'):
        layout = filter_values(matrix, layout, None, None,
                               nan_mode='any_sample', threads=1)
    if case in ('kmeans', 'kmeans_silhouette', 'hclust_selected',
                'profile_cluster_relabel'):
        hierarchical = case in ('hclust_selected', 'profile_cluster_relabel')
        method = 'hierarchical' if hierarchical else 'kmeans'
        selected = case in ('hclust_selected', 'kmeans_silhouette')
        columns = _columns(matrix) if selected else None
        layout, warning = cluster(matrix, layout, 2, method=method,
                                  cols=columns, threads=1)
        assert warning is None
    if case == 'profile_merge_duplicate':
        layout, _ = remove_empty_groups(layout)
        layout = apply_merge_relabel(layout, ('Same', 'Same', 'Gone'))
    else:
        layout, _ = remove_empty_groups(layout)
    if case.startswith('sort_') and case not in ('sort_no', 'sort_keep'):
        parts = case.split('_')
        if case.startswith('sort_disjoint_'):
            using = parts[-1]
            method = 'descend' if using == 'median' else 'ascend'
            layout = sort(matrix, layout, using=using, method=method,
                          cols=_columns(matrix), threads=1)
        else:
            layout = sort(matrix, layout, using='_'.join(parts[2:-1]),
                          method=parts[1], quantiles=int(parts[-1][1:]),
                          threads=1)
    elif case == 'sort_keep':
        layout = sort(matrix, layout, method='keep', threads=1)
    elif case == 'quantile_custom':
        layout = sort(matrix, layout, method='ascend', using='mean',
                      quantiles=2, threads=1)
    elif case == 'gzip_matrix':
        layout = sort(matrix, layout, method='ascend', using='mean',
                      threads=1)
    elif case in ('kmeans_silhouette',):
        layout = sort(matrix, layout, method='ascend', using='mean',
                      threads=1)
        layout = silhouette(matrix, layout, threads=1)
    elif case in ('empty_override', 'show_counts',
                  'profile_together_duplicate', 'profile_merge_duplicate'):
        layout = sort(matrix, layout, method='ascend', using='mean',
                      threads=1)
    elif case == 'profile_cluster_relabel':
        pass
    elif case == 'sort_no':
        pass
    elif case in ('kmeans', 'hclust_selected'):
        pass

    suffix = '.gz' if case == 'gzip_matrix' else '.txt'
    expected_matrix = FIXTURES / (case + suffix)
    with open(expected_matrix, 'rb') as handle:
        first = (gzip.GzipFile(fileobj=handle).readline() if suffix == '.gz'
                 else handle.readline())
    header = json.loads(first[1:])
    labels = Labels(tuple(header['group_labels']),
                    tuple(header['group_labels']),
                    tuple(header['sample_labels']))
    output_matrix = tmp_path / (case + suffix)
    save(matrix, layout, labels, output_matrix,
         compressed=suffix == '.gz', threads=1)
    assert output_matrix.read_bytes() == expected_matrix.read_bytes()
    output_bed = tmp_path / (case + '.bed')
    with output_bed.open('w', newline='', encoding='utf-8') as handle:
        save_bed(matrix, layout, labels, handle)
    assert_bed_equal(output_bed, FIXTURES / (case + '.bed'),
                     silhouette=case == 'kmeans_silhouette')


@pytest.mark.parametrize('method', ['ascend', 'descend'])
@pytest.mark.parametrize('using', ['mean', 'median', 'sum', 'max', 'min'])
def test_sort_breaks_ties_like_default_argsort(method, using):
    # The CLI fixtures avoid tied keys because NumPy's default argsort orders
    # ties differently across platforms. Pin tie handling against argsort on
    # the running platform instead: keys in layout order, default kind.
    values = np.asarray([[1, 1], [2, 2], [1, 1], [3, 3], [2, 2], [1, 1],
                         [2, 2], [1, 1], [3, 3], [1, 1]], dtype=np.float32)
    header = read_header(FIXTURES / 'single.gz')
    parameters = {key: value[:1] if isinstance(value, list) and
                  len(value) == 3 else value
                  for key, value in header.parameters.items()}
    parameters.update(sample_labels=['one'],
                      sample_boundaries=[0, 2], group_labels=['All'],
                      group_boundaries=[0, len(values)])
    regions = [['chr1', [(i * 10, i * 10 + 5)], f'r{i}', i, '+', '0']
               for i in range(len(values))]
    matrix = OwnedMatrix.from_compute(parameters, values, regions).freeze()
    start = RowLayout.identity(matrix)
    shuffled = RowLayout(np.asarray([3, 0, 7, 1, 9, 4, 2, 8, 6, 5]),
                         start.group_bounds, start.base_names, start.origins)
    result = sort(matrix, shuffled, using=using, method=method, threads=1)
    keys = getattr(np, using)(values[shuffled.rows], axis=1)
    order = keys.argsort()
    if method == 'descend':
        order = order[::-1]
    assert result.rows.tolist() == shuffled.rows[order].tolist()


def test_silhouette_on_projected_rows_and_columns():
    matrix, layout = _input()
    layout, _ = _filtered(matrix, layout)
    columns = _columns(matrix)
    layout = sort(matrix, layout, method='descend', using='score',
                  threads=1)
    layout, warning = cluster(matrix, layout, 2, cols=columns, threads=1)
    assert warning is None
    layout = silhouette(matrix, layout, threads=1)
    copied = _statistics.copy_values(
        matrix.values, zero_missing=True, num_threads=1,
        rows=layout.rows, cols=columns)
    labels = np.repeat(np.arange(len(layout.origins)),
                       np.diff(layout.group_bounds)).tolist()
    expected = _statistics.silhouette_scores(
        copied, labels, num_threads=1)
    assert layout.silhouette.tobytes() == expected.tobytes()


def test_filter_on_reordered_rows_and_disjoint_samples():
    matrix, identity = _input()
    ordered = sort(matrix, identity, using='score', method='descend',
                   threads=1)
    columns = _columns(matrix)
    actual = filter_values(matrix, ordered, 0, None, cols=columns,
                           nan_mode='any_sample', threads=1)
    selected = matrix.values[ordered.rows][:, columns]
    keep, _ = _statistics.filter_matrix(
        selected, (0, 2, 4), (0, 1), 'perBin', 0, np.inf,
        _statistics.filter_nan_modes['any_sample'], 'removeRegion', 1)
    assert actual.rows.tobytes() == ordered.rows[keep].tobytes()


def test_independent_layouts_and_borrowed_blocks():
    matrix, identity = _input()
    first, _ = _filtered(matrix, identity)
    a1 = sort(matrix, first, using='median', method='ascend', threads=1)
    _ = sort(matrix, first, using='score', method='descend', threads=1)
    a2 = sort(matrix, first, using='median', method='ascend', threads=1)
    assert a1 == a2
    assert a1.digest() == a2.digest()
    assert a1.rows.tobytes() == a2.rows.tobytes()
    assert not matrix.values.flags.writeable
    assert identity.rows is None
    block = a1.block(matrix, 0, 0)
    assert block.values is matrix.values
    assert np.shares_memory(block.rows, a1.rows)
    assert projection_kwargs(block)['rows'] is block.rows
    assert projection_kwargs(identity.block(matrix, 0, 0))['row_range'] == (0, 5)


def test_header_and_owned_matrix():
    header = read_header(FIXTURES / 'input.gz')
    assert header.group_labels == ('North', 'South', 'Empty')
    owned = OwnedMatrix.load(FIXTURES / 'input.gz', threads=1)
    assert owned.values.flags.writeable
    frozen = owned.freeze()
    assert frozen.values is owned.values
    assert not frozen.values.flags.writeable
    assert frozen.source.st_size == (FIXTURES / 'input.gz').stat().st_size


@pytest.mark.parametrize('case', [
    'sort_disjoint_mean', 'gzip_matrix', 'kmeans_silhouette',
    'reject_label_count', 'reject_sort_sample',
])
def test_cli_fixture_generator_replays_status_and_bytes(case, tmp_path):
    expected = json.loads((FIXTURES / 'cases.json').read_text())[case]
    actual = run_case(case, tmp_path, FIXTURES)
    assert_diagnostics_equal(actual, expected,
                             silhouette=case == 'kmeans_silhouette')
    if not actual['status']:
        suffix = '.gz' if case == 'gzip_matrix' else '.txt'
        for ending in ('.bed', suffix):
            output = tmp_path / (case + ending)
            reference = FIXTURES / (case + ending)
            if ending == '.bed':
                assert_bed_equal(output, reference,
                                 silhouette=case == 'kmeans_silhouette')
            else:
                assert output.read_bytes() == reference.read_bytes()


# From general second audit regressions.

def signal(tmp_path):
    path = tmp_path / 'signal.bw'
    with pyBigWig.open(str(path), 'w') as bw:
        bw.addHeader([('chr1', 100)])
        bw.addEntries(['chr1', 'chr1'], [0, 50], ends=[50, 100], values=[2.0, 7.0])
    return path


def matrix_args(bw, regions, output, extra=()):
    return ['reference-point', '-S', str(bw), '-R', str(regions), '-o', str(output), '-b', '0', '-a', '10', '-bs', '10', *extra]


def assert_rows(path, names=('one', 'two'), values=(2.0, 7.0)):
    header, rows, observed = read_matrix(path)
    assert [row[3] for row in rows] == list(names)
    np.testing.assert_array_equal(observed, np.asarray(values).reshape(-1, 1))
    return header


def suffix_bed(tmp_path, kind):
    path = tmp_path / 'regions.bed'
    if kind == 'groups':
        path.write_text('chr1\t10\t20\tone\t0\t+\n#genes\nchr1\t60\t70\ttwo\t0\t+\n#genes_antisense\n')
        return (path, ['one', 'two'])
    path.write_text('chr1\t10\t20\tone\t0\t+\nchr1\t60\t70\ttwo_antisense\t0\t+\n#genes_antisense\n')
    return (path, ['one', 'two_antisense'])


@pytest.mark.parametrize('kind', ['groups', 'row'])
@pytest.mark.parametrize('quiet', [False, True])
def test_h01_keep_preserves_literal_antisense_names(tmp_path, kind, quiet):
    bw = signal(tmp_path)
    regions, names = suffix_bed(tmp_path, kind)
    out = tmp_path / 'out.gz'
    computeMatrix.main(matrix_args(bw, regions, out, ['--quiet'] if quiet else []))
    assert_rows(out, names)


@pytest.mark.parametrize('kind', ['groups', 'row'])
def test_h01_explicit_sort_preserves_literal_antisense_names(tmp_path, kind):
    bw = signal(tmp_path)
    regions, names = suffix_bed(tmp_path, kind)
    source, out = (tmp_path / 'source.gz', tmp_path / 'out.gz')
    computeMatrix.main(matrix_args(bw, regions, source, ['--sortRegions', 'no']))
    cmo.main(['sort', '-m', str(source), '-R', str(regions), '-o', str(out)])
    assert_rows(out, names)


@pytest.mark.parametrize('kind', ['groups', 'row'])
@pytest.mark.parametrize('implementation', ['plus_no_sort', 'original_keep'])
def test_literal_suffix_control(tmp_path, kind, implementation):
    bw = signal(tmp_path)
    regions, names = suffix_bed(tmp_path, kind)
    out = tmp_path / 'out.gz'
    if implementation == 'original_keep':
        original('computeMatrix', matrix_args(bw, regions, out))
    else:
        computeMatrix.main(matrix_args(bw, regions, out, ['--sortRegions', 'no']))
    assert_rows(out, names)


def compressed_bed(tmp_path, compression):
    text = b'chr1\t60\t70\ttwo\t0\t+\nchr1\t10\t20\tone\t0\t+\n'
    path = tmp_path / ('regions.bed' + ('.' + compression if compression else ''))
    path.write_bytes(gzip.compress(text) if compression == 'gz' else bz2.compress(text) if compression == 'bz2' else text)
    return path


@pytest.mark.parametrize('route', ['compute', 'source_sort', 'output_sort'])
def test_generated_groups_preserve_literal_suffixes_and_group_collisions(tmp_path, route):
    bw = signal(tmp_path)
    minus = tmp_path / 'minus.bw'
    with pyBigWig.open(str(minus), 'w') as handle:
        handle.addHeader([('chr1', 100)])
        handle.addEntries(['chr1'], [0], ends=[100], values=[11.0])
    bed = tmp_path / 'collisions.bed'
    bed.write_text('chr1\t60\t70\ttwo_antisense\t0\t+\nchr1\t10\t20\tone\t0\t+\n#genes\nchr1\t20\t30\tliteral_antisense\t0\t+\n#genes_antisense\n')
    source, out = (tmp_path / 'source.gz', tmp_path / 'out.gz')
    output_bed = tmp_path / 'output.bed'
    computeMatrix.main(matrix_args(bw, bed, source, ['--scoreFileNameMinus', str(minus), '--antisense', 'as_groups', '--sortRegions', 'no' if route == 'source_sort' else 'keep', '--outFileSortedRegions', str(output_bed)]))
    if route != 'compute':
        cmo.main(['sort', '-m', str(source), '-R', str(output_bed if route == 'output_sort' else bed), '-o', str(out)])
    else:
        out = source
    header = assert_rows(out, ['two_antisense', 'one', 'two_antisense_antisense', 'one_antisense', 'literal_antisense', 'literal_antisense_antisense'], [7, 2, 11, 11, 2, 11])
    assert header['group_labels'] == ['genes', 'genes_antisense_r1', 'genes_antisense', 'genes_antisense_antisense']
    assert header['group_boundaries'] == [0, 2, 4, 5, 6]
    assert header['antisense_group_sources'] == {'genes_antisense_r1': 'genes', 'genes_antisense_antisense': 'genes_antisense'}


@pytest.mark.parametrize('quiet', [False, True])
@pytest.mark.parametrize('filter_option', ['--skipZeros', '--minThreshold'])
def test_generated_group_survives_when_entire_source_group_is_filtered(tmp_path, quiet, filter_option):
    plus, minus = (tmp_path / 'plus.bw', tmp_path / 'minus.bw')
    for path, value in [(plus, 0.0), (minus, 7.0)]:
        with pyBigWig.open(str(path), 'w') as handle:
            handle.addHeader([('chr1', 100)])
            handle.addEntries(['chr1'], [0], ends=[100], values=[value])
    bed, out = (compressed_bed(tmp_path, None), tmp_path / 'out.gz')
    options = [filter_option] + (['1'] if filter_option == '--minThreshold' else [])
    if quiet:
        options.append('--quiet')
    computeMatrix.main(matrix_args(plus, bed, out, ['--scoreFileNameMinus', str(minus), '--antisense', 'as_groups', *options]))
    assert_rows(out, ['two_antisense', 'one_antisense'], [7, 7])


@pytest.mark.parametrize('missing', [False, True])
def test_partial_row_selection_preserves_nan_values(tmp_path, missing):
    from deeptoolsr.matrix import Matrix, OwnedMatrix
    values = np.array([[2.0, np.nan], [7.0, 4.0]], dtype=np.float32)
    if missing:
        values[1, 1] = np.nan
    regions = [['chr1', [(10, 20)], 'one', 0, '+', '0'], ['chr1', [(60, 70)], 'two', 0, '+', '0']]
    params = {'group_labels': ['genes'], 'group_boundaries': [0, 2], 'sample_labels': ['sample'], 'sample_boundaries': [0, 2], 'upstream': [0], 'downstream': [0], 'body': [2], 'unscaled 5 prime': [0], 'unscaled 3 prime': [0], 'ref point': [None], 'bin size': [1]}
    matrix = OwnedMatrix.from_compute(params, values, regions)
    bed = tmp_path / 'select.bed'
    bed.write_text('chr1\t60\t70\ttwo\t0\t+\n')
    order = cmo.sortMatrix(matrix, [str(bed)], 'transcript', 'transcript_id')
    assert order == [1]
    output = tmp_path / 'selected.gz'
    matrix.save(output, compressed=True, threads=1, row_order=order)
    selected = Matrix.load(output, 1)
    assert selected.regions[0][2] == 'two'
    assert selected.values[0, 0] == 7
    if missing:
        assert np.isnan(selected.values[0, 1])
    else:
        assert selected.values[0, 1] == 4
