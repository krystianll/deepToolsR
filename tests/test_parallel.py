"""Outer scheduling keeps result order and caps nested native work."""
import os
import shutil
import subprocess
import sys

import pytest

from deeptoolsr.parallel import parallel_map


def test_parallel_map_passes_the_inner_budget_and_preserves_order():
    assert parallel_map(lambda item, inner: (item, inner), range(5), 3) == [
        (item, 1) for item in range(5)]
    assert parallel_map(lambda item, inner: (item, inner), [7], 4) == [(7, 4)]
    assert parallel_map(lambda item, inner: (item, inner), [7, 8], 1) == [
        (7, 1), (8, 1)]


def test_parallel_map_rejects_invalid_budget():
    with pytest.raises(ValueError, match='positive'):
        parallel_map(lambda item, inner: item, [1], 0)


# From audit regressions.

@pytest.fixture()
def io_native_backends(monkeypatch):
    monkeypatch.delenv('DTP_BAM_BACKEND', raising=False)


@pytest.mark.usefixtures('io_native_backends')
@pytest.mark.skipif(sys.platform != 'darwin' or not shutil.which('clang'), reason='macOS pthread fault-injection harness')
def test_partial_thread_pool_construction_raises_instead_of_aborting(tmp_path):
    if (os.cpu_count() or 1) < 2:
        pytest.skip('requires at least two available hardware threads')
    source, library = (tmp_path / 'fail_threads.c', tmp_path / 'fail_threads.dylib')
    source.write_text('\n#include <pthread.h>\n#include <stdlib.h>\n#include <errno.h>\nstatic int audit_create(pthread_t *t, const pthread_attr_t *a,\n                        void *(*fn)(void *), void *arg) {\n    static int count = 0;\n    if (getenv("DTP_AUDIT_FAIL_THREAD_CREATE") && ++count == 2) return EAGAIN;\n    return pthread_create(t, a, fn, arg);\n}\n__attribute__((used)) static struct { const void *replacement; const void *replacee; }\ninterpose __attribute__((section("__DATA,__interpose"))) = {\n    (const void *)audit_create, (const void *)pthread_create\n};\n')
    subprocess.run(['clang', '-dynamiclib', str(source), '-o', str(library)], check=True)
    child = subprocess.run([sys.executable, '-c', "\nimport os, resource\nimport numpy as np\nfrom deeptoolsr import _statistics\nresource.setrlimit(resource.RLIMIT_CORE, (0, 0))\ndata = np.ones((10, 10), np.float32)\nos.environ['DTP_AUDIT_FAIL_THREAD_CREATE'] = '1'\ntry:\n    _statistics.reduce_axis(data, 0, 'mean', 4)\nexcept Exception:\n    print('caught expected thread creation failure')\nelse:\n    raise AssertionError('fault injection did not run')\n"], env={**os.environ, 'DYLD_INSERT_LIBRARIES': str(library)}, capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, (child.returncode, child.stderr)
