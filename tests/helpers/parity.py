"""Upstream parity and matrix-file helpers."""

import gzip
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest


def original(module, arguments):
    if sys.platform == "win32" or importlib.util.find_spec("deeptools") is None:
        pytest.skip("requires original deepTools 3.5.6")
    # Upstream 3.5.6's filterValues calls the removed np.warnings alias.
    # Restore only that standard-library alias; leave its computations intact.
    code = (
        "import importlib,importlib.metadata,sys,warnings,numpy as np; "
        "np.warnings=warnings; "
        "assert importlib.metadata.version('deepTools')=='3.5.6'; "
        "module=importlib.import_module('deeptools.'+sys.argv[1]); "
        "module.main(sys.argv[2:])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, module] + arguments,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def read_matrix(path):
    with gzip.open(path, "rt") as handle:
        header = json.loads(handle.readline()[1:])
        fields = [line.rstrip().split("\t") for line in handle]
    return (
        header,
        [row[:6] for row in fields],
        np.array([[float(v) for v in row[6:]] for row in fields]),
    )
