"""Output determinism and multi-artifact publication contracts."""
from deeptoolsr.matrix import Matrix
from deeptoolsr.matrix import OwnedMatrix
from deeptoolsr import computeMatrixOperations as cmo
import numpy as np

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from deeptoolsr import plotHeatmap, plotProfile, plotMatrix


DATA = Path(__file__).parent / 'test_heatmapper'


def _child(code, *args, env=None):
    result = subprocess.run([sys.executable, '-c', code, *map(str, args)],
                            capture_output=True, text=True, timeout=90,
                            env=env)
    assert result.returncode == 0, result.stderr


VECTOR_CHILD = '''
import sys
from deeptoolsr import plotHeatmap
plotHeatmap.main(['-m', sys.argv[1], '-o', sys.argv[2]])
'''


@pytest.mark.parametrize('extension', ['pdf', 'svg'])
def test_vector_output_is_byte_deterministic(tmp_path, extension):
    env = dict(os.environ)
    env.pop('SOURCE_DATE_EPOCH', None)
    matrix = DATA / 'master.mat.gz'
    first, second = (tmp_path / ('first.' + extension),
                     tmp_path / ('second.' + extension))
    _child(VECTOR_CHILD, matrix, first, env=env)
    time.sleep(1.1)
    _child(VECTOR_CHILD, matrix, second, env=env)
    assert first.read_bytes() == second.read_bytes()


CLUSTER_CHILD = '''
import sys
from deeptoolsr import plotHeatmap, plotMatrix, prepare
seed = sys.argv[4]
if seed != 'default':
    original = prepare.cluster
    def with_seed(*args, **kwargs):
        kwargs['seed'] = int(seed)
        return original(*args, **kwargs)
    prepare.cluster = with_seed
plotMatrix.save_figure_atomic = lambda *_args, **_kwargs: None
plotHeatmap.main(['-m', sys.argv[1], '-o', sys.argv[2], '--kmeans', '4',
                  '--outFileSortedRegions', sys.argv[3]])
'''


def test_kmeans_cli_sorted_regions_are_seeded(tmp_path):
    matrix = DATA / 'large_matrix.mat.gz'
    outputs = []
    for index, seed in enumerate(('default', 'default', '0', '1', '10', '11')):
        plot = tmp_path / f'plot{index}.png'
        bed = tmp_path / f'regions{index}.bed'
        _child(CLUSTER_CHILD, matrix, plot, bed, seed)
        outputs.append(bed.read_bytes())
    assert outputs[0] == outputs[1]
    assert outputs[0] == outputs[2]
    assert len(set(outputs[2:])) > 1


def test_plot_failure_preserves_all_destinations(tmp_path, monkeypatch):
    matrix = DATA / 'master.mat.gz'
    plot = tmp_path / 'plot.png'
    saved_matrix = tmp_path / 'matrix.mat'
    bed = tmp_path / 'regions.bed'
    plot.write_bytes(b'old plot')
    saved_matrix.write_bytes(b'old matrix')

    def fail_after_staging(*_args, **_kwargs):
        # Every output is staged before the figure is built.
        staged = list(tmp_path.iterdir())
        assert any(path.name.endswith('.matrix.tmp') for path in staged)
        assert any(path.name.endswith('.regions.tmp') for path in staged)
        raise RuntimeError('injected after auxiliary staging')

    monkeypatch.setattr(plotMatrix, 'build_matrix_figure', fail_after_staging)
    with pytest.raises(RuntimeError, match='injected after auxiliary staging'):
        plotHeatmap.main([
            '-m', str(matrix), '-o', str(plot),
            '--outFileNameMatrix', str(saved_matrix),
            '--outFileSortedRegions', str(bed),
        ])
    assert plot.read_bytes() == b'old plot'
    assert saved_matrix.read_bytes() == b'old matrix'
    assert not bed.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        'matrix.mat', 'plot.png']


def test_profile_failure_preserves_all_destinations(tmp_path, monkeypatch):
    matrix = DATA / 'master.mat.gz'
    plot = tmp_path / 'plot.png'
    saved_matrix = tmp_path / 'matrix.mat'
    bed = tmp_path / 'regions.bed'
    plot.write_bytes(b'old plot')
    saved_matrix.write_bytes(b'old matrix')
    original_save_bed = plotMatrix.save_bed

    def fail_after_bed(matrix, layout, labels, handle):
        original_save_bed(matrix, layout, labels, handle)
        handle.flush()
        assert Path(handle.name).stat().st_size > 0
        assert any(path.name.endswith('.matrix.tmp') and path.stat().st_size
                   for path in tmp_path.iterdir())
        raise RuntimeError('injected after auxiliary staging')

    monkeypatch.setattr(plotMatrix, 'save_figure_atomic',
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plotMatrix, 'save_bed', fail_after_bed)
    with pytest.raises(RuntimeError, match='injected after auxiliary staging'):
        plotProfile.main([
            '-m', str(matrix), '-o', str(plot),
            '--outFileNameMatrix', str(saved_matrix),
            '--outFileSortedRegions', str(bed),
        ])
    assert plot.read_bytes() == b'old plot'
    assert saved_matrix.read_bytes() == b'old matrix'
    assert not bed.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        'matrix.mat', 'plot.png']


@pytest.mark.parametrize('tool', [plotHeatmap, plotProfile])
def test_invalid_plot_output_precedes_numeric_preparation(
        tmp_path, monkeypatch, capsys, tool):
    def unexpected_prepare(*_args, **_kwargs):
        raise AssertionError('numeric preparation ran before output check')

    monkeypatch.setattr(plotMatrix, 'prepare_matrix', unexpected_prepare)
    missing_parent = tmp_path / 'missing' / 'plot.png'
    with pytest.raises(SystemExit) as exit_info:
        tool.main(['-m', str(DATA / 'master.mat.gz'),
                   '-o', str(missing_parent)])
    assert exit_info.value.code == 2
    assert (f'argument --outFileName/-out/-o: {missing_parent} '
            "file can't be opened for writing" in capsys.readouterr().err)


# From general audit regressions.

def make_matrix(path, values, bounds=None):
    data = np.asarray(values, dtype=np.float32)
    rows, bins = data.shape
    bounds = [0, rows] if bounds is None else bounds
    labels = [f'group{i}' for i in range(len(bounds) - 1)]
    regions = [['chr1', [(i * 10, i * 10 + 5)], f'r{i}', 0, '+', '0'] for i in range(rows)]
    parameters = {'upstream': [0], 'downstream': [0], 'body': [bins], 'unscaled 5 prime': [0], 'unscaled 3 prime': [0], 'ref point': [None], 'bin size': [1], 'sort regions': 'keep', 'sort using': 'mean', 'min threshold': None, 'max threshold': None, 'sample_labels': ['sample'], 'group_labels': labels, 'sample_boundaries': [0, bins], 'group_boundaries': bounds}
    OwnedMatrix.from_compute(parameters, data, regions).save(str(path), compressed=True, threads=1)


OPERATIONS = {'subset': ['--samples', 'sample'], 'filterValues': ['--onFilterFail', 'maskSample'], 'transform': ['--scale', '2'], 'filterStrand': ['--strand', '+'], 'relabel': ['--sampleLabels', 'renamed']}


def limited_write(tmp_path, operation, suffix):
    if sys.platform == 'win32':
        pytest.skip('POSIX RLIMIT_FSIZE fault injection')
    source, dest = (tmp_path / 'source.gz', tmp_path / ('out' + suffix))
    make_matrix(source, [[1, 2], [3, 4]])
    previous = b'previous successful output'
    dest.write_bytes(previous)
    child = '\nimport resource, signal, sys\nfrom deeptoolsr import computeMatrixOperations\nsignal.signal(signal.SIGXFSZ, signal.SIG_IGN)\nresource.setrlimit(resource.RLIMIT_FSIZE,\n                  (80, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))\ncomputeMatrixOperations.main(sys.argv[1:])\n'
    result = subprocess.run([sys.executable, '-c', child, operation, '-m', str(source), '-o', str(dest), *OPERATIONS[operation]], capture_output=True, text=True, timeout=30)
    observed = dest.read_bytes()
    assert result.returncode != 0 and observed == previous, f'{operation}: status={result.returncode}, bytes={len(observed)}, old output preserved={observed == previous}; stderr={result.stderr!r}'
    assert sorted((p.name for p in tmp_path.iterdir())) == sorted([source.name, dest.name])


@pytest.mark.parametrize('operation', ['subset', 'filterValues', 'transform'])
@pytest.mark.parametrize('suffix', ['.gz', '.mat'])
def test_g01_late_write_failure_preserves_output(tmp_path, operation, suffix):
    limited_write(tmp_path, operation, suffix)


@pytest.mark.parametrize('operation,suffix', [('filterStrand', '.gz'), ('relabel', '.gz'), ('relabel', '.mat')])
def test_checked_writers_reject_the_same_late_failure(tmp_path, operation, suffix):
    limited_write(tmp_path, operation, suffix)


def test_g01_plain_strand_filter_late_write_failure(tmp_path):
    limited_write(tmp_path, 'filterStrand', '.mat')


@pytest.mark.parametrize('operation', list(OPERATIONS))
@pytest.mark.parametrize('suffix', ['.gz', '.mat'])
def test_normal_matrix_writes_round_trip(tmp_path, operation, suffix):
    source, dest = (tmp_path / 'source.gz', tmp_path / ('out' + suffix))
    make_matrix(source, [[1, 2], [3, 4]])
    cmo.main([operation, '-m', str(source), '-o', str(dest), *OPERATIONS[operation]])
    hm = Matrix.load(str(dest), threads=1)
    np.testing.assert_array_equal(hm.values, np.array([[1, 2], [3, 4]]) * (2 if operation == 'transform' else 1))
