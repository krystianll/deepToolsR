"""Successful sparse/delayed reads versus failed reads; native and CLI paths."""


import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
from tests.helpers.bigwig import pyBigWig
import pytest

from deeptoolsr import _bigwig, _compute_matrix_native, bigWigOperations


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("zero_missing", [False, True])
def test_empty_sparse_and_absent_chromosome_queries_are_successful(tmp_path, empty, zero_missing):
    path = tmp_path / "signal.bw"
    with pyBigWig.open(str(path), "w") as writer:
        writer.addHeader([("chr1", 1000)], maxZooms=0)
        if not empty:
            writer.addEntries(["chr1"], [100], ends=[200], values=[7.])
    reader = _compute_matrix_native.NativeBigWigReader([str(path)])
    for chrom, start in [("chr1", 300), ("absent", 100)]:
        actual = reader.bin_row(chrom, [[[(start, start + 20)], 2]],
                                [[0]], [[1.]], False, 0, 0, "mean", zero_missing)
        np.testing.assert_array_equal(actual, [0., 0.] if zero_missing else [np.nan, np.nan])
    # A previous empty query must not affect a later populated query.
    actual = reader.bin_row("chr1", [[[(100, 120)], 2]],
                            [[0]], [[1.]], False, 0, 0, "mean", zero_missing)
    expected = [7., 7.] if not empty else [0., 0.] if zero_missing else [np.nan, np.nan]
    np.testing.assert_array_equal(actual, expected)


def test_empty_bigwig_remains_valid_for_scale_merge_and_info(tmp_path):
    source, scaled, merged = (tmp_path / name for name in ("empty.bw", "scaled.bw", "merged.bw"))
    with pyBigWig.open(str(source), "w") as writer:
        writer.addHeader([("chr1", 1000)], maxZooms=0)
    bigWigOperations.main(["scale", "-b", str(source), "-o", str(scaled), "-s", "2", "--zoomLevels", "0"])
    bigWigOperations.main(["merge", "-b", str(source), "-o", str(merged), "--zoomLevels", "0"])
    assert _bigwig.bigwig_info(str(source))["nBasesCovered"] == 0
    assert _bigwig.bigwig_info(str(scaled))["nBasesCovered"] == 0
    with pyBigWig.open(str(merged)) as reader:
        np.testing.assert_array_equal(reader.values("chr1", 0, 1000), np.zeros(1000))


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("clang"),
                    reason="macOS stdio latency/error injection harness")
@pytest.mark.parametrize("mode", ["delay", "error", "short"])
def test_slow_reads_succeed_but_failed_reads_preserve_old_output(tmp_path, mode):
    track, bed, output = (tmp_path / name for name in ("input.bw", "regions.bed", "out.gz"))
    with pyBigWig.open(str(track), "w") as writer:
        writer.addHeader([("chr1", 1000)], maxZooms=0)
        writer.addEntries(["chr1"], [100], ends=[200], values=[7.])
    bed.write_text("chr1\t100\t200\tr\t0\t+\n")
    output.write_bytes(b"previous successful result")
    data = track.read_bytes()
    data_start = struct.unpack_from("<Q", data, 16)[0] + 8
    data_end = struct.unpack_from("<Q", data, 24)[0]
    source, library = tmp_path / "slow_io.c", tmp_path / "slow_io.dylib"
    source.write_text(r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdatomic.h>
static _Atomic int hits;
int dtp_injected_reads(void) { return atomic_load(&hits); }
static size_t audit_read(void *ptr, size_t size, size_t count, FILE *stream) {
    const char *target = getenv("DTP_TEST_IO_PATH");
    const char *mode = getenv("DTP_TEST_IO_MODE");
    char path[1024];
    if (target && mode && fcntl(fileno(stream), F_GETPATH, path) == 0 && !strcmp(path, target)) {
        long pos = ftell(stream);
        if (pos >= strtol(getenv("DTP_TEST_IO_BEGIN"), NULL, 10) &&
            pos < strtol(getenv("DTP_TEST_IO_END"), NULL, 10)) {
            atomic_fetch_add(&hits, 1);
            if (!strcmp(mode, "delay")) usleep(50000);
            else if (!strcmp(mode, "short")) {
                fread(ptr, 1, size * count / 2, stream);
                return 0;
            } else { errno = EIO; return 0; }
        }
    }
    return fread(ptr, size, count, stream);
}
__attribute__((used)) static struct { const void *replacement; const void *replacee; }
interpose __attribute__((section("__DATA,__interpose"))) = {
    (const void *)audit_read, (const void *)fread
};
''')
    subprocess.run(["clang", "-dynamiclib", str(source), "-o", str(library)], check=True)
    code = r'''
import ctypes, json, os, sys, time
from pathlib import Path
import numpy as np
from deeptoolsr import computeMatrix
from deeptoolsr.matrix import Matrix
track, bed, output, library, mode = json.loads(sys.argv[1])
handle = ctypes.CDLL(library)
started = time.monotonic()
failed = False
try:
    computeMatrix.main(['reference-point', '-S', track, '-R', bed, '-o', output,
                        '-b', '0', '-a', '100', '-bs', '10', '--quiet'])
except SystemExit as error:
    assert 'Failed reading bigWig' in str(error), str(error)
    failed = True
assert handle.dtp_injected_reads() > 0, 'fault/latency injection did not run'
if mode == 'delay':
    assert time.monotonic() - started >= .04
    assert not failed
    matrix = Matrix.load(output, threads=1)
    np.testing.assert_array_equal(matrix.values, np.full((1, 10), 7.))
else:
    assert failed
    assert Path(output).read_bytes() == b'previous successful result'
'''
    child = subprocess.run(
        [sys.executable, "-c", code, json.dumps([str(track), str(bed), str(output), str(library), mode])],
        env={**os.environ, "DYLD_INSERT_LIBRARIES": str(library),
             "DTP_TEST_IO_PATH": str(track.resolve()), "DTP_TEST_IO_MODE": mode,
             "DTP_TEST_IO_BEGIN": str(data_start), "DTP_TEST_IO_END": str(data_end)},
        capture_output=True, text=True, timeout=30,
    )
    assert child.returncode == 0, (child.returncode, child.stdout, child.stderr)


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
@pytest.mark.parametrize('command', ['scale', 'merge', 'info'])
def test_damaged_bigwig_operation_must_abort(tmp_path, coverage_damaged_track, command):
    output = tmp_path / 'out.bw'
    output.write_bytes(b'previous successful result')
    if command == 'info':
        args = ['info', str(coverage_damaged_track)]
    else:
        args = [command, '-b', str(coverage_damaged_track), '-o', str(output), '--zoomLevels', '0']
        if command == 'scale':
            args += ['--scaleFactor', '2']
    failed = False
    try:
        bigWigOperations.main(args)
    except (SystemExit, RuntimeError, ValueError) as error:
        assert 'Failed reading bigWig' in str(error)
        failed = True
    assert failed, 'corrupt data was silently omitted'
    assert output.read_bytes() == b'previous successful result'


# From read validation regressions.

@pytest.fixture()
def validation_isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path / 'config'))


@pytest.fixture
def validation_bigwig_tree_fixture(tmp_path):
    source = tmp_path / 'source.bw'
    with pyBigWig.open(str(source), 'w') as writer:
        writer.addHeader([('chr1', 1000), ('chr2', 1000)], maxZooms=0)
        writer.addEntries(['chr1'], [100], ends=[200], values=[7.0])
    return source


CHILD = "\nimport json, sys\ntry:\n    import resource\n    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\nexcept ImportError:\n    pass\nfrom deeptoolsr import _bigwig, _compute_matrix_native\ntry:\n    if sys.argv[2] == 'info':\n        result = _bigwig.bigwig_info(sys.argv[1])\n        assert result['nBasesCovered'] == 100 and result['mean'] == 7\n    else:\n        reader = _compute_matrix_native.NativeBigWigReader([sys.argv[1]])\n        result = reader.bin_row('chr1', [[[(100, 200)], 2]], [[0]], [[1.]],\n                                False, 0, 0, 'mean', False)\n        assert result.tolist() == [7., 7.]\nexcept (ValueError, RuntimeError, OverflowError) as error:\n    print(type(error).__name__, str(error))\n    sys.exit(7)\n"


def run_bigwig_child(path, consumer):
    return subprocess.run([sys.executable, '-c', CHILD, str(path), consumer], capture_output=True, text=True, timeout=8)


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('consumer', ['info', 'matrix'])
def test_valid_bigwig_tree_fixture(validation_bigwig_tree_fixture, consumer):
    child = run_bigwig_child(validation_bigwig_tree_fixture, consumer)
    assert child.returncode == 0, (child.returncode, child.stdout, child.stderr)


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('consumer', ['info', 'matrix'])
@pytest.mark.parametrize('damage', ['chrom_id_oob', 'duplicate_chrom_id', 'rtree_cycle', 'chrom_tree_cycle'])
def test_bad_bigwig_tree_must_raise_instead_of_crashing(validation_bigwig_tree_fixture, consumer, damage):
    data = bytearray(validation_bigwig_tree_fixture.read_bytes())
    chromosome_tree = struct.unpack_from('<Q', data, 8)[0]
    key_size = struct.unpack_from('<I', data, chromosome_tree + 8)[0]
    leaf = chromosome_tree + 32
    root = struct.unpack_from('<Q', data, 24)[0] + 48
    assert data[leaf] == data[root] == 1
    if damage == 'chrom_id_oob':
        struct.pack_into('<I', data, leaf + 4 + key_size, 1000000)
    elif damage == 'duplicate_chrom_id':
        struct.pack_into('<I', data, leaf + 4 + key_size + 8 + key_size, 0)
    elif damage == 'rtree_cycle':
        data[root] = 0
        struct.pack_into('<H', data, root + 2, 1)
        struct.pack_into('<Q', data, root + 20, root)
    else:
        data[leaf] = 0
        struct.pack_into('<H', data, leaf + 2, 1)
        struct.pack_into('<Q', data, leaf + 4 + key_size, leaf)
    validation_bigwig_tree_fixture.write_bytes(data)
    child = run_bigwig_child(validation_bigwig_tree_fixture, consumer)
    assert child.returncode == 7, (child.returncode, child.stdout, child.stderr)


def wrap_tree(data, kind, levels=1):
    """Repoint a small fixture through real, acyclic internal tree nodes."""
    if kind == 'chromosome':
        tree = struct.unpack_from('<Q', data, 8)[0]
        key_size = struct.unpack_from('<I', data, tree + 8)[0]
        root = tree + 32
        entry_size = key_size + 8
        child = len(data)
        leaves = [bytes(data[root + 4 + i * entry_size:root + 4 + (i + 1) * entry_size]) for i in range(2)]
        pointers = []
        for entry in leaves:
            pointers.append(len(data))
            data.extend(struct.pack('<BBH', 1, 0, 1) + entry)
        internal = bytearray(struct.pack('<BBH', 0, 0, 2))
        for entry, offset in zip(leaves, pointers):
            internal.extend(entry[:key_size] + struct.pack('<Q', offset))
        for _ in range(levels - 1):
            child = len(data)
            data.extend(internal)
            internal = bytearray(struct.pack('<BBH', 0, 0, 1) + leaves[0][:key_size] + struct.pack('<Q', child))
    else:
        root = struct.unpack_from('<Q', data, 24)[0] + 48
        count = struct.unpack_from('<H', data, root + 2)[0]
        leaf = bytes(data[root:root + 4 + count * 32])
        child = len(data)
        data.extend(leaf)
        internal = bytearray(struct.pack('<BBH', 0, 0, 1) + leaf[4:20] + struct.pack('<Q', child))
        for _ in range(levels - 1):
            child = len(data)
            data.extend(internal)
            internal = bytearray(struct.pack('<BBH', 0, 0, 1) + leaf[4:20] + struct.pack('<Q', child))
    data[root:root + len(internal)] = internal


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('kind', ['chromosome', 'interval'])
@pytest.mark.parametrize('levels', [1, 3, 65])
@pytest.mark.parametrize('consumer', ['info', 'matrix'])
def test_real_internal_trees_and_excessive_depth(validation_bigwig_tree_fixture, kind, levels, consumer):
    data = bytearray(validation_bigwig_tree_fixture.read_bytes())
    wrap_tree(data, kind, levels)
    validation_bigwig_tree_fixture.write_bytes(data)
    if levels < 64:
        with pyBigWig.open(str(validation_bigwig_tree_fixture)) as reader:
            assert reader.chroms() == {'chr1': 1000, 'chr2': 1000}
            np.testing.assert_array_equal(reader.values('chr1', 100, 200), np.full(100, 7.0))
    child = run_bigwig_child(validation_bigwig_tree_fixture, consumer)
    assert child.returncode == (7 if levels >= 64 else 0), (child.returncode, child.stdout, child.stderr)


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('damage', ['huge_key', 'huge_count', 'bad_value_size', 'chrom_child_count', 'interval_child_count', 'chrom_node_tag', 'interval_node_tag', 'truncated_zoom_header'])
@pytest.mark.parametrize('consumer', ['info', 'matrix'])
def test_malformed_tree_sizes_and_tags(validation_bigwig_tree_fixture, damage, consumer):
    data = bytearray(validation_bigwig_tree_fixture.read_bytes())
    ct = struct.unpack_from('<Q', data, 8)[0]
    root = struct.unpack_from('<Q', data, 24)[0] + 48
    if damage == 'huge_key':
        struct.pack_into('<I', data, ct + 8, 4294967295)
    elif damage == 'huge_count':
        struct.pack_into('<Q', data, ct + 16, 1 << 63)
    elif damage == 'bad_value_size':
        struct.pack_into('<I', data, ct + 12, 4)
    elif damage == 'chrom_child_count':
        struct.pack_into('<H', data, ct + 34, 65535)
    elif damage == 'interval_child_count':
        struct.pack_into('<H', data, root + 2, 65535)
    elif damage == 'chrom_node_tag':
        data[ct + 32] = 2
    elif damage == 'interval_node_tag':
        data[root] = 2
    else:
        struct.pack_into('<H', data, 6, 1)
        data = data[:68]
    validation_bigwig_tree_fixture.write_bytes(data)
    child = run_bigwig_child(validation_bigwig_tree_fixture, consumer)
    assert child.returncode == 7, (child.returncode, child.stdout, child.stderr)
