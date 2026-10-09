"""Zoom levels built from the written intervals (bigwig_writer.hpp).

Every case decodes the output's zoom section and compares it, level by level
and record by record, with an independent rebuild from the file's data
intervals (tests/helpers/zoom.py), including the level choice, block cuts and
index entries. libBigWig left the sum of the last record of each block and of
each level at 0; deepToolsR writes it.
"""

import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from deeptoolsr import bamCoverage, bigWigOperations
from tests.helpers.bigwig import pyBigWig
from tests.helpers.zoom import assert_zoom_matches_reference, data_section

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
BAM = os.path.join(ROOT, "test1.bam")
SPILL_RUNS = (8 << 20) // 16   # runs held in memory before spilling


def write_track(path, chroms, intervals):
    """Write [(chrom, starts, ends, values)] with pyBigWig (upstream libBigWig)."""
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader(chroms, maxZooms=10)
    for chrom, starts, ends, values in intervals:
        bw.addEntries([chrom] * len(starts), [int(s) for s in starts],
                      ends=[int(e) for e in ends], values=[float(v) for v in values])
    bw.close()


def scale(source, output, zooms=10, factor="1"):
    bigWigOperations.main(["scale", "-b", str(source), "-o", str(output), "-s", factor,
                           "--zoomLevels", str(zooms)])


@pytest.mark.parametrize("bin_size", [1, 50])
@pytest.mark.parametrize("zooms", [0, 1, 10])
def test_bamcoverage_zoom_levels_match_reference(tmp_path, bin_size, zooms):
    out = tmp_path / "cov.bw"
    bamCoverage.main(["-b", BAM, "-o", str(out), "-bs", str(bin_size),
                      "--zoomLevels", str(zooms)])
    levels = assert_zoom_matches_reference(out, zooms)
    assert (len(levels) > 0) == (zooms > 0)
    assert os.listdir(tmp_path) == ["cov.bw"]


def test_zoom_levels_do_not_change_the_data_section(tmp_path):
    with_zooms, without = tmp_path / "z10.bw", tmp_path / "z0.bw"
    for out, zooms in ((with_zooms, 10), (without, 0)):
        bamCoverage.main(["-b", BAM, "-o", str(out), "-bs", "1", "--zoomLevels", str(zooms)])
    assert data_section(with_zooms) == data_section(without)


def test_scale_writes_an_interval_too_wide_for_zoom_levels(tmp_path):
    # Mean interval width over 2^30 / 4: libBigWig can build no zoom level, and
    # the output is written with none instead of failing.
    source, out = tmp_path / "wide.bw", tmp_path / "out.bw"
    with pyBigWig.open(str(source), "w") as bw:
        bw.addHeader([("chr1", 400_000_000)], maxZooms=0)
        bw.addEntries(["chr1"], [0], ends=[400_000_000], values=[1.5])
    scale(source, out, factor="2")
    with pyBigWig.open(str(out)) as bw:
        assert bw.intervals("chr1") == ((0, 400_000_000, 3.0),)
        assert bw.header()["nLevels"] == 0


def test_interval_of_2_30_bases_gets_no_zoom_levels(tmp_path):
    # 4 x 2^30 wrapped to 0 in 32 bits and chose a 10-base zoom (minutes of work).
    source, out = tmp_path / "wide.bw", tmp_path / "out.bw"
    with pyBigWig.open(str(source), "w") as bw:
        bw.addHeader([("chr1", 2**30 + 10)], maxZooms=0)
        bw.addEntries(["chr1"], [0], ends=[2**30], values=[1.0])
    started = time.monotonic()
    scale(source, out)
    assert time.monotonic() - started < 10
    with pyBigWig.open(str(out)) as bw:
        assert bw.header()["nLevels"] == 0
    assert out.stat().st_size < 4096


@pytest.mark.parametrize("values,low,high", [((5, 3), 3, 5), ((-5, -3), -5, -3)])
def test_summary_records_minimum_and_maximum(tmp_path, values, low, high):
    source, out = tmp_path / "in.bw", tmp_path / "out.bw"
    write_track(source, [("chr1", 100)], [("chr1", [0, 10], [10, 20], values)])
    scale(source, out)
    with pyBigWig.open(str(out)) as bw:
        assert (bw.header()["minVal"], bw.header()["maxVal"]) == (low, high)


def test_summary_and_zoom_sums_are_exact(tmp_path):
    # 3 x 100000008 - 3 x 100000000: a float product made the sum 32.
    source, out = tmp_path / "in.bw", tmp_path / "out.bw"
    write_track(source, [("chr1", 1000)], [("chr1", [0, 3], [3, 6], [100000008, -100000000])])
    scale(source, out)
    with pyBigWig.open(str(out)) as bw:
        assert bw.header()["sumData"] == 24
        assert bw.stats("chr1", 0, 6)[0] == pytest.approx(4)
        assert bw.stats("chr1", 0, 6, exact=True)[0] == 4


@pytest.mark.skipif(not os.path.exists("/dev/full"), reason="needs /dev/full (Linux)")
def test_bigwig_write_errors_are_reported(tmp_path):
    from deeptoolsr import _bigwig
    source = tmp_path / "in.bw"
    write_track(source, [("chr1", 100000)], [("chr1", np.arange(5000) * 20,
                                              np.arange(5000) * 20 + 10, np.ones(5000))])
    with pytest.raises(RuntimeError):
        _bigwig.bigwig_scale(str(source), "/dev/full", 2.0, 0)


@pytest.mark.parametrize("case", ["sparse", "tiny_chromosome", "short_first_chromosome"])
def test_scale_zoom_levels_match_reference(tmp_path, case):
    rng = np.random.default_rng(7)
    if case == "sparse":
        chroms = [("chr1", 5_000_000), ("chr2", 3_000_000)]
        starts = np.sort(rng.choice(4_990, 300, replace=False)) * 1000
        intervals = [("chr1", starts, starts + rng.integers(1, 200, 300), rng.random(300)),
                     ("chr2", [10, 2_000_000], [20, 2_000_500], [3.0, 4.0])]
    elif case == "tiny_chromosome":
        chroms = [("chrM", 50)]
        intervals = [("chrM", [0, 5, 30], [5, 20, 50], [1.0, 2.0, 0.5])]
    else:  # level 1 (4 x 4 x mean width) is longer than the first chromosome
        chroms = [("chrS", 100), ("chrL", 2_000_000)]
        starts = np.arange(0, 2_000_000, 1000)
        intervals = [("chrS", [0, 40], [40, 100], [1.0, 2.0]),
                     ("chrL", starts, starts + 1000, rng.random(len(starts)))]
    source, out = tmp_path / "in.bw", tmp_path / "out.bw"
    write_track(source, chroms, intervals)
    for zooms in (0, 1, 10):
        scale(source, out, zooms)
        levels = assert_zoom_matches_reference(out, zooms)
        assert (len(levels) > 0) == (zooms > 0)


def test_merge_zoom_levels_match_reference(tmp_path):
    source, out = tmp_path / "in.bw", tmp_path / "out.bw"
    starts = np.arange(0, 400_000, 7)
    write_track(source, [("chr1", 400_000)],
                [("chr1", starts, starts + 7, (starts // 7) % 5)])
    bigWigOperations.main(["merge", "-b", str(source), "-o", str(out), "-bs", "1", "-p", "2"])
    levels = assert_zoom_matches_reference(out, 10)
    assert len(levels[0][1]) > 1   # several blocks: the block cut is exercised


@pytest.fixture(scope="module")
def dense_track(tmp_path_factory):
    """More runs than the in-memory buffer, spread over 30 chromosomes."""
    path = tmp_path_factory.mktemp("dense") / "dense.bw"
    per_chrom = SPILL_RUNS // 10
    starts = np.arange(per_chrom)
    values = (np.arange(per_chrom) % 3).astype(float)
    write_track(path, [(f"chr{i}", per_chrom) for i in range(30)],
                [(f"chr{i}", starts, starts + 1, values) for i in range(30)])
    return path


def test_spilled_runs_match_reference_and_leave_no_file(tmp_path, dense_track):
    out = tmp_path / "out.bw"
    scale(dense_track, out, zooms=2)
    levels = assert_zoom_matches_reference(out, 2)
    assert len(levels) == 2 and len(levels[0][1]) > 1
    assert os.listdir(tmp_path) == ["out.bw"]


def test_failed_write_removes_spill_file(tmp_path, dense_track):
    source, out = tmp_path / "in.bw", tmp_path / "out.bw"
    per_chrom = SPILL_RUNS // 10
    starts = np.arange(per_chrom)
    # The overflowing value comes after the in-memory buffer has spilled.
    write_track(source, [(f"chr{i}", per_chrom) for i in range(12)],
                [(f"chr{i}", starts, starts + 1, np.full(per_chrom, 1e38 if i == 11 else 1.0))
                 for i in range(12)])
    with pytest.raises(SystemExit):
        scale(source, out, factor="10")
    assert os.listdir(tmp_path) == ["in.bw"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGINT delivery")
def test_cancel_removes_spill_file(tmp_path, dense_track):
    out = tmp_path / "out.bw"
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys\nfrom deeptoolsr import bigWigOperations\n"
         "bigWigOperations.main(['scale', '-b', sys.argv[1], '-o', sys.argv[2], '-s', '2'])",
         str(dense_track), str(out)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 60
    while not any(n.endswith(".zoomruns") for n in os.listdir(tmp_path)):
        assert child.poll() is None and time.monotonic() < deadline, "spill file never appeared"
        time.sleep(0.002)
    child.send_signal(signal.SIGINT)
    child.communicate(timeout=60)
    assert child.returncode != 0
    assert os.listdir(tmp_path) == []


def test_constant_track_zoom_means_are_exact(tmp_path):
    """libBigWig's zoom means fall below 1.0 at block ends; ours stay 1.0."""
    chroms = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    starts = np.arange(0, 2_000_000, 10)
    source, out = tmp_path / "in.bw", tmp_path / "out.bw"
    write_track(source, chroms, [(c, starts, starts + 10, np.ones(len(starts))) for c, _ in chroms])
    scale(source, out)

    def zoom_means(path):
        bw = pyBigWig.open(str(path))
        means = [bw.stats(c, 0, n, type="mean", nBins=2000, exact=False) for c, n in chroms]
        bw.close()
        return np.array(means, dtype=float)

    assert np.any(np.abs(zoom_means(source) - 1.0) > 1e-6)   # upstream libBigWig
    np.testing.assert_allclose(zoom_means(out), 1.0, rtol=1e-6)


def test_zoom_means_equal_exact_means_on_real_data(tmp_path):
    out = tmp_path / "cov.bw"
    bamCoverage.main(["-b", BAM, "-o", str(out), "-bs", "1"])
    bw = pyBigWig.open(str(out))
    for chrom, length in bw.chroms().items():
        if not bw.intervals(chrom):
            continue
        zoomed = bw.stats(chrom, 0, length, type="mean", exact=False)[0]
        exact = bw.stats(chrom, 0, length, type="mean", exact=True)[0]
        assert zoomed == pytest.approx(exact, rel=1e-6)
    bw.close()
