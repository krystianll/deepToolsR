"""Cross-platform checks for binary bigWig output and zoom finalization.

Windows distinguishes text and binary update streams.  This module deliberately
avoids pyBigWig and pysam so it runs in the native-only Windows test lane.
"""

import os
import struct

import numpy as np
import pytest


_coverage = pytest.importorskip("deeptoolsr._coverage")
_bigwig = pytest.importorskip("deeptoolsr._bigwig")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
BAM = os.path.join(ROOT, "test1.bam")
BIGWIG_MAGIC = 0x888FFC26


def test_interval_bigwig_with_zoom_is_complete_and_readable(tmp_path):
    """Exercise the interval writer that used to crash in Windows text mode."""
    output = tmp_path / "coverage-with-zoom.bw"
    copied = tmp_path / "readback-copy.bw"

    _coverage.bam_coverage_bigwig(
        BAM, str(output), bin_size=50, threads=1, max_zooms=1)

    data = output.read_bytes()
    magic, version, levels = struct.unpack_from("<IHH", data)
    assert magic == BIGWIG_MAGIC
    assert version == 4
    assert levels == 1
    assert struct.unpack_from("<I", data, len(data) - 4)[0] == BIGWIG_MAGIC

    # Reading every interval and writing it again verifies the data blocks and
    # full-data index, without depending on optional pyBigWig Windows wheels.
    _bigwig.bigwig_scale(str(output), str(copied), 1.0, max_zooms=1)
    copied_data = copied.read_bytes()
    copied_magic, copied_version, copied_levels = struct.unpack_from(
        "<IHH", copied_data)
    assert copied_magic == BIGWIG_MAGIC
    assert copied_version == 4
    assert copied_levels == 1
    assert struct.unpack_from("<I", copied_data, len(copied_data) - 4)[0] == BIGWIG_MAGIC


def test_native_compute_matrix_does_not_require_pybigwig(tmp_path, monkeypatch):
    """The native command must obtain chromosome tables through libBigWig."""
    from deeptoolsr.computeMatrix import main
    from deeptoolsr.matrix import Matrix

    class ForbiddenPyBigWig:
        def __getattr__(self, name):
            raise AssertionError("native computeMatrix accessed pyBigWig")

    monkeypatch.setitem(__import__("sys").modules, "pyBigWig", ForbiddenPyBigWig())
    regions = tmp_path / "regions.bed"
    output = tmp_path / "matrix.gz"
    regions.write_text("3R\t150\t180\tplatform-check\t0\t+\n")

    main([
        "reference-point", "-S", os.path.join(ROOT, "testA.bw"),
        os.path.join(ROOT, "testB.bw"), "-R", str(regions),
        "-o", str(output), "-b", "50", "-a", "50", "--binSize", "5",
        "--sortRegions", "keep", "--missingDataAsZero", "--quiet",
    ])

    matrix = Matrix.load(str(output), threads=1)
    values = np.asarray(matrix.values, dtype=np.float32)
    mask = np.ma.getmaskarray(matrix.values)
    assert values.shape == (1, 40)
    assert not mask.any()
    np.testing.assert_array_equal(values[0, :20], [2.0] * 20)
    np.testing.assert_array_equal(values[0, 20:],
                                  [1.0] * 10 + [3.0] * 10)
