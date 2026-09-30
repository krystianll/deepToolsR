from deeptoolsr import _bigwig
from deeptoolsr import computeMatrix
from tests.helpers.bigwig import pyBigWig
import struct
import gzip
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from deeptoolsr.matrix import Matrix
from deeptoolsr.matrix_validation import validate_matrix_header


native_io = pytest.importorskip('deeptoolsr._compute_matrix_io')
native_stream = pytest.importorskip('deeptoolsr._compute_matrix_stream')
native_transform = pytest.importorskip('deeptoolsr._transform')
native_compute = pytest.importorskip('deeptoolsr._compute_matrix_native')


def _header(rows=1, bins=2):
    return {
        'group_boundaries': [0, rows],
        'sample_boundaries': [0, bins],
        'group_labels': ['group'],
        'sample_labels': ['sample'],
        'bin size': [1],
        'upstream': [1],
        'body': [0],
        'downstream': [1],
        'unscaled 5 prime': [0],
        'unscaled 3 prime': [0],
        'ref point': ['TSS'],
    }


def _write(path, header, row='chr1\t0\t2\tr\t0\t+\t1\t2\n'):
    with gzip.open(path, 'wt') as handle:
        handle.write('@' + json.dumps(header) + '\n')
        handle.write(row)


@pytest.mark.parametrize('parameters', [
    {**_header(), 'group_boundaries': [1, 1]},
    {**_header(), 'group_boundaries': [0, 1, 0],
     'group_labels': ['a', 'b']},
    {**_header(), 'sample_boundaries': [0, 0]},
    {**_header(), 'sample_labels': []},
    {**_header(), 'bin size': [1, 2]},
])
def test_header_rejects_invalid_schema(parameters):
    with pytest.raises(ValueError):
        validate_matrix_header(parameters)


@pytest.mark.parametrize('processors', [1, 3])
@pytest.mark.parametrize('row', [
    'chr1\t0\t2\tr\t0\t+\t1\n',
    'chr1\t0,1\t2\tr\t0\t+\t1\t2\n',
    'chr1\t2\t1\tr\t0\t+\t1\t2\n',
    'chr1\t0\t2\tr\t0\t+\t1e100\t2\n',
])
def test_matrix_readers_reject_malformed_rows(tmp_path, monkeypatch,
                                              processors, row):
    path = tmp_path / 'bad.gz'
    _write(path, _header(), row)
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        Matrix.load(path, threads=processors)


def test_matrix_invariants_survive_optimized_python(tmp_path):
    code = (
        'import numpy as np; '
        'from deeptoolsr.matrix import OwnedMatrix; '
        'from tests.test_matrix_validation import _header; '
        'OwnedMatrix.from_compute(_header(), np.empty((1, 2)), [])'
    )
    result = subprocess.run(
        [sys.executable, '-O', '-c', code], cwd=os.getcwd(),
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'ValueError' in result.stderr


def test_native_io_validates_boundaries_before_allocation(tmp_path):
    path = tmp_path / 'one.gz'
    _write(path, _header())
    with pytest.raises(RuntimeError):
        native_io.read_matrix(str(path), 1, 2, [1, 1], 1)
    with pytest.raises(RuntimeError):
        native_io.read_matrix(str(path), 1, 2, [0, 1], 1, 0)


def test_native_transform_validates_all_ranges():
    matrix = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(RuntimeError):
        native_transform.scale_rows(matrix, [0, 5], [0], [4], 0, 1)
    with pytest.raises(RuntimeError):
        native_transform.scale_rows(matrix, [0, 4], [0], [5], 0, 1)
    with pytest.raises(RuntimeError):
        native_transform.binary_combine(
            matrix, matrix.copy(), [0, 4], [1.0], 9, 0, 1)
    with pytest.raises(RuntimeError):
        native_transform.combine([matrix, matrix.copy()], 9, 1)


def test_native_stream_validates_schema_before_opening_output(tmp_path):
    output = tmp_path / 'must-not-exist.gz'
    with pytest.raises(RuntimeError):
        native_stream.filter_values(
            'missing.gz', str(output), [0, 1], [0, 3, 2], [0],
            0, 0, False, 0.0, False, 0.0, 1)
    assert not output.exists()


def test_native_compute_validates_specs_before_opening_bigwig():
    with pytest.raises((ValueError, OverflowError)):
        native_compute.bin_bigwig_batch(
            ['missing.bw'], ['chr1'], [[([(0, 10)], 2)]],
            [[[0]]], [[[1.0]]], [False], [-1], [0], 'mean', False, 1)
    with pytest.raises((ValueError, OverflowError)):
        native_compute.bin_bigwig_batch(
            ['missing.bw'], ['chr1'], [[([(10, 0)], 2)]],
            [[[0]]], [[[1.0]]], [False], [0], [0], 'mean', False, 1)
    with pytest.raises((ValueError, OverflowError)):
        native_compute.bin_bigwig_batch(
            ['missing.bw'], ['chr1'], [[([(0, 10)], 2)]],
            [[[0]]], [[[float('nan')]]], [False], [0], [0],
            'mean', False, 1)


# From audit followup 2026 09 12.

@pytest.fixture()
def coverage_isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path / 'config'))


def write_track(path, values):
    with pyBigWig.open(str(path), 'w') as bw:
        bw.addHeader([('chr1', len(values))], maxZooms=0)
        starts = [i for i, value in enumerate(values) if np.isfinite(value)]
        bw.addEntries(['chr1'] * len(starts), starts, ends=[i + 1 for i in starts], values=[float(values[i]) for i in starts])


@pytest.fixture(params=['compressed_data', 'data_offset', 'child_index'])
def coverage_damaged_track(tmp_path, request):
    path = tmp_path / 'damaged.bw'
    write_track(path, np.arange(1000, dtype=float))
    data = bytearray(path.read_bytes())
    full_data_offset = struct.unpack_from('<Q', data, 16)[0]
    if request.param == 'compressed_data':
        assert data[full_data_offset + 8] == 120
        data[full_data_offset + 8] = 0
    else:
        root = struct.unpack_from('<Q', data, 24)[0] + 48
        assert data[root] == 1
        if request.param == 'child_index':
            data[root] = 0
            struct.pack_into('<H', data, root + 2, 1)
        struct.pack_into('<Q', data, root + 20, len(data) + 4096)
    path.write_bytes(data)
    assert _bigwig.bigwig_info(str(path), False)['chroms'] == {'chr1': 1000}
    return path


@pytest.mark.usefixtures('coverage_isolated_config')
@pytest.mark.parametrize('zero', [False, True])
def test_damaged_matrix_input_must_abort_and_preserve_output(tmp_path, coverage_damaged_track, zero):
    bed, out = (tmp_path / 'regions.bed', tmp_path / 'out.gz')
    bed.write_text('chr1\t100\t200\tr\t0\t+\n')
    out.write_bytes(b'previous successful result')
    failed = False
    try:
        computeMatrix.main(['reference-point', '-S', str(coverage_damaged_track), '-R', str(bed), '-o', str(out), '-b', '0', '-a', '100', '-bs', '10', '--quiet', *(['--missingDataAsZero'] if zero else [])])
    except (SystemExit, RuntimeError, ValueError) as error:
        assert 'Failed reading bigWig' in str(error)
        failed = True
    assert failed, 'corrupt data was silently committed as NaNs/zeros'
    assert out.read_bytes() == b'previous successful result'


# From general second audit regressions.

@pytest.mark.parametrize('sources', [None, [], {'bad': 1}, {'bad': []}])
def test_invalid_generated_group_metadata_is_rejected(sources):
    from deeptoolsr.region_provenance import antisense_sources
    with pytest.raises(ValueError, match='metadata'):
        antisense_sources({'antisense_group_sources': sources})
