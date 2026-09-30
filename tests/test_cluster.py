"""Native clustering contracts and optional SciPy oracle measurements."""
from deeptoolsr.matrix import MatrixHeader
from deeptoolsr.matrix import cluster
from deeptoolsr.matrix import silhouette
from deeptoolsr.matrix import OwnedMatrix
from tests.helpers.parity import original
from deeptoolsr import plotHeatmap
import matplotlib.pyplot as plt
from tests.helpers.parity import read_matrix
import math
from deeptoolsr.matrix import sort
from deeptoolsr import plotProfile

import gc
import os
from pathlib import Path
import subprocess
import sys
import tracemalloc
import weakref

import numpy as np
import pytest

from deeptoolsr import _cluster
from deeptoolsr.matrix import Matrix, RowLayout
from deeptoolsr import cluster as cluster_module


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / 'tests/test_data/prepare_expected/input.gz'


def _workspace(path, columns=None):
    values = Matrix.load(path, threads=1).values
    if columns is not None:
        values = values[:, columns]
    return np.nan_to_num(values, nan=0).astype(np.float32)


def _partition(labels):
    return {frozenset(np.flatnonzero(labels == label))
            for label in np.unique(labels)}


@pytest.mark.parametrize('selected', [None, [0, 1, 4, 5]])
def test_ward_fixture_matches_scipy(selected):
    hierarchy = pytest.importorskip('scipy.cluster.hierarchy')
    features = _workspace(PREPARE, selected)
    native = _cluster.ward_linkage(features, threads=1)
    reference = hierarchy.linkage(features, method='ward')
    assert np.array_equal(native[:, :2], reference[:, :2])
    assert np.allclose(native[:, 2], reference[:, 2], rtol=2e-7, atol=1e-10)
    for k in range(2, 7):
        labels = _cluster.cut_maxclust(native, k)
        expected = hierarchy.fcluster(reference, k, criterion='maxclust')
        assert np.array_equal(labels, expected)


def test_ward_separated_matches_scipy():
    hierarchy = pytest.importorskip('scipy.cluster.hierarchy')
    rng = np.random.default_rng(75)
    features = np.vstack([rng.normal(i * 20, 0.1, (20, 8))
                          for i in range(3)]).astype(np.float32)
    native = _cluster.ward_linkage(features, threads=1)
    reference = hierarchy.linkage(features, method='ward')
    assert np.array_equal(native[:, :2], reference[:, :2])
    assert np.allclose(native[:, 2], reference[:, 2], rtol=3e-6)
    assert _partition(_cluster.cut_maxclust(native, 3)) == _partition(
        hierarchy.fcluster(reference, 3, criterion='maxclust'))


def _benchmark_matrix(name, tmp_path):
    cached = ROOT / f'.cache/s4_bench/{name}.gz'
    if cached.exists():
        return cached
    regions, samples, bins = ((2000, 4, 200) if name == 'small'
                              else (5000, 8, 500))
    generated = tmp_path / f'{name}.gz'
    subprocess.run([sys.executable, str(ROOT / 'benchmarks/plots/make_matrix.py'),
                    str(regions), str(samples), str(bins), str(generated)],
                   check=True)
    return generated


def test_ward_small_benchmark_membership(tmp_path):
    hierarchy = pytest.importorskip('scipy.cluster.hierarchy')
    features = _workspace(_benchmark_matrix('small', tmp_path))
    native = _cluster.ward_linkage(features, threads=4)
    assert native.tobytes() == _cluster.ward_linkage(features, threads=1).tobytes()
    reference = hierarchy.linkage(features, method='ward')
    assert np.allclose(native[:, 2], reference[:, 2], rtol=2e-7)
    for k in range(2, 7):
        assert _partition(_cluster.cut_maxclust(native, k)) == _partition(
            hierarchy.fcluster(reference, k, criterion='maxclust'))


@pytest.mark.parametrize('path,selected', [
    (PREPARE, None), (PREPARE, [0, 1, 4, 5]), ('small', None),
])
def test_memory_saving_ward_matches_stored(path, selected, tmp_path):
    if path == 'small':
        path = _benchmark_matrix('small', tmp_path)
    features = _workspace(path, selected)
    stored = _cluster.ward_linkage(features, threads=1)
    memory_saving = _cluster.ward_linkage(
        features, threads=4, distance_budget_bytes=1)
    assert np.allclose(memory_saving[:, 2], stored[:, 2],
                       rtol=2e-7, atol=1e-9)
    assert features.tobytes() == _workspace(path, selected).tobytes()
    assert memory_saving.tobytes() == _cluster.ward_linkage(
        features, threads=1, distance_budget_bytes=1).tobytes()
    for k in range(2, 7):
        assert np.array_equal(_cluster.cut_maxclust(memory_saving, k),
                              _cluster.cut_maxclust(stored, k))


def test_ward_budget_message_only_on_fallback(tmp_path, capsys):
    from deeptoolsr import config, plotHeatmap

    settings = tmp_path / 'options.json'
    config.save_options({'ward_distance_budget_bytes': 1}, settings)
    common = ['-m', str(PREPARE), '--hclust', '2', '-p', '1',
              '--whatToShow', 'heatmap and colorbar']
    plotHeatmap.main([*common, '-o', str(tmp_path / 'fallback.png'),
                      '--config', str(settings)])
    fallback = capsys.readouterr().err
    assert fallback.count('memory-saving method') == 1
    config.save_options({'ward_distance_budget_bytes': 2 << 30}, settings)
    plotHeatmap.main([*common, '-o', str(tmp_path / 'stored.png'),
                      '--config', str(settings)])
    assert 'memory-saving method' not in capsys.readouterr().err


def test_ward_ties_follow_native_scan_rule():
    # Equal distances are scanned by ascending active index; equal merge
    # heights retain their NN-chain emission order under stable sorting.
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    linkage = _cluster.ward_linkage(features, threads=4)
    assert np.array_equal(linkage[:2, :2], [[0, 1], [2, 3]])
    assert _cluster.cut_maxclust(linkage, 3).tolist() == [1, 1, 2, 2]
    assert linkage.tobytes() == _cluster.ward_linkage(features, threads=1).tobytes()


def test_cut_maxclust_numbering_and_ties():
    hierarchy = pytest.importorskip('scipy.cluster.hierarchy')
    for values in ([0, 1, 10, 11], [0, 1, 10, 11, 20, 21],
                   [0, 1, 2, 3, 8, 9]):
        linkage = hierarchy.linkage(np.asarray(values)[:, None], method='ward')
        for k in range(1, len(values) + 1):
            assert np.array_equal(_cluster.cut_maxclust(linkage, k),
                                  hierarchy.fcluster(linkage, k, 'maxclust'))


def test_kmeans_separated_partition_and_thread_determinism():
    vq = pytest.importorskip('scipy.cluster.vq')
    rng = np.random.default_rng(127)
    features = np.vstack([rng.normal(i * 30, 0.2, (30, 6))
                          for i in range(3)]).astype(np.float32)
    labels, centroids, distortion = _cluster.kmeans(features, 3, seed=10,
                                                    threads=1)
    ref_centroids, _ = vq.kmeans(features, 3, seed=10)
    reference, _ = vq.vq(features, ref_centroids)
    assert _partition(labels) == _partition(reference)
    again = _cluster.kmeans(features, 3, seed=10, threads=4)
    assert labels.tobytes() == again[0].tobytes()
    assert centroids.tobytes() == again[1].tobytes()
    assert distortion == again[2]


def test_kmeans_mid_distortion_and_seeds(tmp_path):
    vq = pytest.importorskip('scipy.cluster.vq')
    features = _workspace(_benchmark_matrix('mid', tmp_path))
    native = _cluster.kmeans(features, 4, seed=0, threads=4)
    reference, _ = vq.kmeans(features, 4, seed=0)
    _, distances = vq.vq(features, reference)
    assert native[2] <= float(np.mean(distances)) * 1.01
    assert native[0].tobytes() == _cluster.kmeans(
        features, 4, seed=0, threads=1)[0].tobytes()
    other = _cluster.kmeans(features, 4, seed=1, threads=4)
    assert native[0].tobytes() != other[0].tobytes()


def test_plot_clustering_without_scipy(tmp_path):
    script = """
import sys
sys.modules['scipy'] = None
from deeptoolsr import plotHeatmap, plotProfile
for module in (plotHeatmap, plotProfile):
    for flag in ('--kmeans', '--hclust'):
        module.main(['-m', sys.argv[1], '-o', sys.argv[2], flag, '2', '-p', '1'])
"""
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / 'mpl'),
               DEEPTOOLSR_CONFIG_DIR=str(tmp_path / 'config'))
    completed = subprocess.run([sys.executable, '-c', script,
                                str(PREPARE), str(tmp_path / 'plot.png')],
                               cwd=ROOT, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_workspace_is_released(monkeypatch):
    matrix = Matrix.load(PREPARE, threads=1)
    original = cluster_module._statistics.copy_values
    references = []

    def capture(*args, **kwargs):
        workspace = original(*args, **kwargs)
        references.append(weakref.ref(workspace))
        return workspace

    monkeypatch.setattr(cluster_module._statistics, 'copy_values', capture)
    tracemalloc.start()
    for _ in range(2):
        cluster_module.cluster(matrix, RowLayout.identity(matrix), 2,
                               method='hierarchical', threads=1)
    gc.collect()
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(8):
        cluster_module.cluster(matrix, RowLayout.identity(matrix), 2,
                               method='kmeans', threads=1)
    gc.collect()
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    assert all(reference() is None for reference in references)
    assert after - before < 1_000_000


# From audit regressions.

@pytest.fixture()
def io_native_backends(monkeypatch):
    monkeypatch.delenv('DTP_BAM_BACKEND', raising=False)


@pytest.mark.usefixtures('io_native_backends')
def test_clustering_preserves_missing_values():
    data = np.array([[1, np.nan], [2, 3], [10, 11], [12, 13]], np.float32)
    regions = [['chr1', [(i, i + 1)], f'r{i}', 0, '+', '0'] for i in range(4)]
    matrix = Matrix(MatrixHeader.from_parameters({'group_boundaries': [0, 4], 'sample_boundaries': [0, 2], 'group_labels': ['g'], 'sample_labels': ['s']}), data, regions, None)
    layout, _ = cluster(matrix, RowLayout.identity(matrix), 2, method='hierarchical', threads=1)
    row = next((i for i, index in enumerate(layout.row_indices()) if matrix.regions[index][2] == 'r0'))
    assert np.isnan(matrix.values[layout.row_indices()[row], 1])


@pytest.mark.usefixtures('io_native_backends')
def test_two_member_clusters_have_nonzero_silhouette():
    matrix = Matrix(MatrixHeader.from_parameters({'group_boundaries': [0, 2, 4], 'sample_boundaries': [0, 1], 'group_labels': ['first', 'second'], 'sample_labels': ['sample']}), np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32), regions=[['chr1', [(index, index + 1)], str(index), 0, '+', '0'] for index in range(4)], source=None)
    layout = silhouette(matrix, RowLayout.identity(matrix), threads=1)
    np.testing.assert_allclose(layout.silhouette, [9.5 / 10.5, 8.5 / 9.5, 8.5 / 9.5, 9.5 / 10.5])


# From general audit regressions.

def make_matrix(path, values, bounds=None):
    data = np.asarray(values, dtype=np.float32)
    rows, bins = data.shape
    bounds = [0, rows] if bounds is None else bounds
    labels = [f'group{i}' for i in range(len(bounds) - 1)]
    regions = [['chr1', [(i * 10, i * 10 + 5)], f'r{i}', 0, '+', '0'] for i in range(rows)]
    parameters = {'upstream': [0], 'downstream': [0], 'body': [bins], 'unscaled 5 prime': [0], 'unscaled 3 prime': [0], 'ref point': [None], 'bin size': [1], 'sort regions': 'keep', 'sort using': 'mean', 'min threshold': None, 'max threshold': None, 'sample_labels': ['sample'], 'group_labels': labels, 'sample_boundaries': [0, bins], 'group_boundaries': bounds}
    OwnedMatrix.from_compute(parameters, data, regions).save(str(path), compressed=True, threads=1)


def render(tmp_path, values, extra=(), bounds=None, upstream=False):
    source, dest = (tmp_path / 'matrix.gz', tmp_path / 'heatmap.png')
    make_matrix(source, values, bounds)
    args = ['-m', str(source), '-o', str(dest), *extra]
    try:
        if upstream:
            original('plotHeatmap', args)
        else:
            plotHeatmap.main(args)
        assert dest.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    finally:
        plt.close('all')


@pytest.mark.parametrize('method', ['kmeans', 'hclust'])
def test_original_handles_empty_cluster_groups(tmp_path, method):
    render(tmp_path, np.tile([1.0, 2.0], (6, 1)), ['--' + method, '3'], upstream=True)


@pytest.mark.parametrize('method', ['kmeans', 'hclust'])
def test_nonempty_clusters_render(tmp_path, method):
    data = np.array([[1, 2], [1, 3], [2, 3], [10, 20], [10, 21], [11, 22]])
    render(tmp_path, data, ['--' + method, '2'])


@pytest.mark.parametrize('method', ['kmeans', 'hclust'])
def test_single_populated_cluster_silhouette_exports_nan(tmp_path, method, capsys):
    source, out, bed = (tmp_path / 'source.gz', tmp_path / 'out.gz', tmp_path / 'out.bed')
    make_matrix(source, np.tile([1.0, 2.0], (6, 1)))
    try:
        plotHeatmap.main(['-m', str(source), '-o', str(tmp_path / 'out.png'), '--' + method, '3', '--silhouette', '--outFileNameMatrix', str(out), '--outFileSortedRegions', str(bed)])
        header, rows, values = read_matrix(out)
        assert header['group_boundaries'] == [0, 6]
        assert header['group_labels'] == ['cluster_1']
        np.testing.assert_array_equal(values, np.tile([1, 2], (6, 1)))
        assert 'undefined' in capsys.readouterr().err
        assert all((line.split('\t')[-1] == 'nan' for line in bed.read_text().splitlines() if not line.startswith('#')))
    finally:
        plt.close('all')


def test_fewer_clusters_silhouette_uses_actual_group_count(tmp_path):
    source = tmp_path / 'source.gz'
    make_matrix(source, [[1, 2]] * 3 + [[10, 20]] * 3)
    matrix = Matrix.load(str(source), threads=1)
    layout, _ = cluster(matrix, RowLayout.identity(matrix), 3, method='hierarchical', threads=1)
    assert layout.group_bounds == (0, 3, 6)
    layout = silhouette(matrix, layout, threads=1)
    np.testing.assert_allclose(layout.silhouette, 1.0)


@pytest.mark.parametrize('method', ['kmeans', 'hierarchical'])
def test_one_region_one_cluster(tmp_path, method):
    source = tmp_path / 'source.gz'
    make_matrix(source, [[1, 2]])
    matrix = Matrix.load(str(source), threads=1)
    layout, _ = cluster(matrix, RowLayout.identity(matrix), 1, method=method, threads=1)
    assert layout.group_bounds == (0, 1)
    assert layout.base_names == ('cluster_1',)


# From numerical audit regressions.

def numerical_make_matrix(data, groups=None, labels=None):
    rows, cols = data.shape
    regions = [['chr1', [(i, i + 1)], f'r{i}', 0, '+'] for i in range(rows)]
    return Matrix(MatrixHeader.from_parameters({'group_boundaries': groups or [0, rows], 'sample_boundaries': list(range(cols + 1)), 'group_labels': labels or ['g'], 'sample_labels': [f's{i}' for i in range(cols)]}), data, regions, None)


def silhouette_reference(data, labels):
    values = data.tolist()
    expected = []
    for i, value in enumerate(values):
        same = [math.dist(value, other) for j, other in enumerate(values) if labels[i] == labels[j] and i != j]
        other_means = [math.fsum((math.dist(value, other) for j, other in enumerate(values) if labels[j] == group)) / labels.count(group) for group in set(labels) if group != labels[i]]
        if not same:
            expected.append(0.0)
            continue
        a, b = (math.fsum(same) / len(same), min(other_means))
        expected.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return expected


@pytest.mark.parametrize('selection', [[1], [2], None])
@pytest.mark.parametrize('fortran', [False, True])
def test_n03_silhouette_uses_clustering_samples(selection, fortran):
    data = np.array([[0, 0], [1, 100], [10, 0], [11, 100]], np.float32)
    data[0, 0] = np.nan
    if fortran:
        data = np.asfortranarray(data)
    matrix = numerical_make_matrix(data)
    columns = None if selection is None else [index - 1 for index in selection]
    layout, _ = cluster(matrix, RowLayout.identity(matrix), 2, method='hierarchical', cols=columns, threads=1)
    layout = sort(matrix, layout, using='mean', method='ascend', threads=1)
    layout = silhouette(matrix, layout, threads=1)
    values = matrix.values[layout.row_indices()]
    if selection is not None:
        values = values[:, [index - 1 for index in selection]]
    values = np.nan_to_num(values)
    labels = [group for group in range(2) for _ in range(layout.group_bounds[group + 1] - layout.group_bounds[group])]
    expected = silhouette_reference(values, labels)
    np.testing.assert_allclose(layout.silhouette, expected, rtol=1e-14, atol=0)
    missing_row = [matrix.regions[index][2] for index in layout.row_indices()].index('r0')
    assert np.isnan(matrix.values[layout.row_indices()[missing_row], 0])


def test_n03_multibin_sample_selection_and_reclustering_reset():
    data = np.array([[0, 0, 0, 0], [1, 1, 1, 100], [10, 10, 10, 0], [11, 11, 11, 100]], np.float32)
    matrix = numerical_make_matrix(data)
    parameters = dict(matrix.header.parameters)
    parameters['sample_boundaries'] = [0, 1, 3, 4]
    parameters['sample_labels'] = ['one', 'two bins', 'excluded']
    matrix = Matrix(MatrixHeader.from_parameters(parameters), data, matrix.regions, None)
    for selection, columns in (([2, 1], [1, 2, 0]), (None, [0, 1, 2, 3])):
        layout, _ = cluster(matrix, RowLayout.identity(matrix), 2, method='hierarchical', cols=columns, threads=1)
        layout = silhouette(matrix, layout, threads=1)
        labels = np.repeat(np.arange(2), np.diff(layout.group_bounds)).tolist()
        expected = silhouette_reference(matrix.values[layout.row_indices()][:, columns], labels)
        np.testing.assert_allclose(layout.silhouette, expected, rtol=1e-14, atol=0)


# Argument contract.

TEST_DATA = Path(__file__).parent / 'test_data'


MATRIX = TEST_DATA / 'computeMatrixOperations.mat.gz'


@pytest.mark.parametrize('module', [plotHeatmap, plotProfile])
def test_cluster_sample_index_is_checked_before_clustering(tmp_path, module):
    output = tmp_path / 'out.pdf'
    with pytest.raises(SystemExit, match='indices must be between'):
        module.main(['-m', str(MATRIX), '-o', str(output), '--kmeans', '2', '--clusterUsingSamples', '999'])
    assert not output.exists()
