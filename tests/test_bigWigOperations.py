"""Tests for bigWigOperationsR (info, scale)."""
from tests.helpers.bigwig import pyBigWig
import subprocess
import sys
from deeptoolsr import _transform
import math

from deeptoolsr import bigWigOperations
import gzip
import os

import numpy as np
import pytest

_bigwig = pytest.importorskip("deeptoolsr._bigwig")
_compute_matrix_native = pytest.importorskip(
    "deeptoolsr._compute_matrix_native")


def _make_bigwig(path):
    bw = pyBigWig.open(path, "w")
    bw.addHeader([("chr1", 1000)])
    bw.addEntries(["chr1", "chr1", "chr1"], [0, 100, 500], ends=[100, 200, 600],
                  values=[1.0, 2.0, 3.0])
    bw.close()


def _values(path, chrom, length):
    bw = pyBigWig.open(path)
    v = np.nan_to_num(np.array(bw.values(chrom, 0, length), dtype=np.float64))
    bw.close()
    return v


def test_scale_flips_and_scales(tmp_path):
    src = str(tmp_path / "in.bw")
    _make_bigwig(src)
    for factor in (-1.0, 2.5, 0.0):
        out = str(tmp_path / ("o%s.bw" % factor))
        _bigwig.bigwig_scale(src, out, factor)
        assert np.allclose(_values(out, "chr1", 1000), _values(src, "chr1", 1000) * factor)


def test_scale_preserves_resolution(tmp_path):
    src = str(tmp_path / "in.bw")
    _make_bigwig(src)
    out = str(tmp_path / "o.bw")
    _bigwig.bigwig_scale(src, out, 1.0)
    a = pyBigWig.open(src)
    b = pyBigWig.open(out)
    assert a.intervals("chr1") == b.intervals("chr1")
    a.close()
    b.close()


def test_scale_across_chunk_boundary(tmp_path):
    # An interval straddling the 8 Mbp read-chunk edge must be written exactly once.
    CHUNK = 8 << 20
    src = str(tmp_path / "big.bw")
    bw = pyBigWig.open(src, "w")
    bw.addHeader([("chr1", 16_000_000)])
    starts = [10, CHUNK - 8, 10_000_000]           # middle one straddles the boundary
    ends = [20, CHUNK + 8, 10_000_010]
    bw.addEntries(["chr1"] * 3, starts, ends=ends, values=[1.0, 5.0, 7.0])
    bw.close()
    out = str(tmp_path / "o.bw")
    _bigwig.bigwig_scale(src, out, 3.0)
    a = pyBigWig.open(src)
    b = pyBigWig.open(out)
    assert b.intervals("chr1") == tuple((s, e, v * 3.0) for s, e, v in a.intervals("chr1"))
    a.close()
    b.close()


def test_cli_info_chromsizes(tmp_path, capsys):
    src = str(tmp_path / "in.bw")
    _make_bigwig(src)
    cs = str(tmp_path / "chrom.sizes")
    bigWigOperations.main(["info", src, "--chromSizes", cs])
    assert open(cs).read().strip() == "chr1\t1000"


def test_cli_rejects_output_aliasing_input(tmp_path, capsys):
    src = str(tmp_path / "in.bw")
    _make_bigwig(src)
    with pytest.raises(SystemExit):
        bigWigOperations.main(["merge", "-b", src, "-o", src])
    assert "aliases input" in capsys.readouterr().err


def test_cli_scale_preserves_existing_output_on_writer_failure(
        tmp_path, monkeypatch):
    src = str(tmp_path / "in.bw")
    out = tmp_path / "out.bw"
    _make_bigwig(src)
    out.write_bytes(b"previous result")

    def fail_after_partial_write(_input, temporary, *_args, **_kwargs):
        with open(temporary, "wb") as handle:
            handle.write(b"partial replacement")
        raise RuntimeError("injected writer failure")

    monkeypatch.setattr(_bigwig, "bigwig_scale", fail_after_partial_write)
    with pytest.raises(SystemExit, match="injected writer failure"):
        bigWigOperations.main([
            "scale", "-b", src, "-o", str(out), "-s", "2"])
    assert out.read_bytes() == b"previous result"


@pytest.mark.parametrize("command", [
    ["scale", "-b", "in.bw", "-o", "out.bw", "-s", "nan"],
    ["scale", "-b", "in.bw", "-o", "out.bw", "-s", "inf"],
    ["merge", "-b", "in.bw", "-o", "out.bw", "--binSize", "5.5"],
    ["merge", "-b", "in.bw", "-o", "out.bw", "--binSize", "0"],
    ["merge", "-b", "in.bw", "-o", "out.bw", "--scale", "1e9999"],
    ["merge", "-b", "in.bw", "-o", "out.bw", "-p", "0"],
    ["merge", "-b", "in.bw", "-o", "out.bw", "--compressionLevel", "0"],
])
def test_cli_rejects_invalid_numeric_values_before_opening_files(command, tmp_path):
    command = [str(tmp_path / part) if part in ("in.bw", "out.bw") else part
               for part in command]
    with pytest.raises(SystemExit):
        bigWigOperations.main(command)
    assert not (tmp_path / "out.bw").exists()


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), float("-inf")])
def test_native_scale_rejects_nonfinite_factor_before_opening_output(tmp_path, factor):
    src = str(tmp_path / "in.bw")
    out = str(tmp_path / "out.bw")
    _make_bigwig(src)
    with pytest.raises(RuntimeError, match="factor must be finite"):
        _bigwig.bigwig_scale(src, out, factor)
    assert not os.path.exists(out)


def test_native_scale_rejects_float32_product_overflow(tmp_path):
    src = str(tmp_path / "in.bw")
    _make_bigwig(src)
    with pytest.raises(RuntimeError, match="float32 range"):
        _bigwig.bigwig_scale(src, str(tmp_path / "out.bw"), 1e300)


def test_scale_omits_nonfinite_input_values_with_one_warning(tmp_path, capfd):
    src = str(tmp_path / "nonfinite.bw")
    _mk(src, {"chr1": (30, [(0, 10, 2.0), (10, 20, float("inf")),
                            (20, 30, float("nan"))])})
    out = str(tmp_path / "scaled.bw")

    _bigwig.bigwig_scale(src, out, 3.0)

    assert pyBigWig.open(out).intervals("chr1") == ((0, 10, 6.0),)
    warning = capfd.readouterr().err
    assert warning.count("omitted non-finite values") == 1
    assert src in warning


def _mk(path, entries):
    """entries: {chrom: (length, [(start, end, value), ...])}."""
    bw = pyBigWig.open(path, "w")
    bw.addHeader([(c, l) for c, (l, _) in entries.items()])
    for c, (l, ivs) in entries.items():
        bw.addEntries([c] * len(ivs), [s for s, _, _ in ivs],
                      ends=[e for _, e, _ in ivs], values=[float(v) for _, _, v in ivs])
    bw.close()


def test_merge_sum_mean_scale(tmp_path):
    a = str(tmp_path / "a.bw")
    b = str(tmp_path / "b.bw")
    _mk(a, {"chr1": (1000, [(0, 100, 1.0), (100, 200, 2.0)])})
    _mk(b, {"chr1": (1000, [(0, 100, 3.0), (150, 250, 4.0)])})
    va, vb = _values(a, "chr1", 1000), _values(b, "chr1", 1000)
    o = str(tmp_path / "sum.bw")
    _bigwig.bigwig_merge([a, b], o, "bigwig", "sum", 1, 1.0, 1, 10, -1)
    assert np.allclose(_values(o, "chr1", 1000), va + vb)
    om = str(tmp_path / "mean.bw")
    _bigwig.bigwig_merge([a, b], om, "bigwig", "mean", 1, 1.0, 1, 10, -1)
    assert np.allclose(_values(om, "chr1", 1000), (va + vb) / 2)
    osc = str(tmp_path / "sc.bw")
    _bigwig.bigwig_merge([a, b], osc, "bigwig", "sum", 1, 3.0, 1, 10, -1)
    assert np.allclose(_values(osc, "chr1", 1000), 3 * (va + vb))


def test_merge_omits_nonfinite_input_values_and_names_each_file(tmp_path, capfd):
    a = str(tmp_path / "a.bw")
    b = str(tmp_path / "b.bw")
    _mk(a, {"chr1": (20, [(0, 10, 2.0), (10, 20, float("inf"))])})
    _mk(b, {"chr1": (20, [(0, 10, float("nan")), (10, 20, 4.0)])})
    out = str(tmp_path / "merged.bw")

    _bigwig.bigwig_merge([a, b], out, "bigwig", "sum", 10, 1.0, 2, 0, -1)

    assert np.allclose(_values(out, "chr1", 20),
                       np.r_[np.full(10, 2.0), np.full(10, 4.0)])
    warning = capfd.readouterr().err
    assert warning.count("omitted non-finite values") == 2
    assert a in warning and b in warning


def test_native_matrix_compute_omits_nonfinite_values_with_warning(tmp_path, capfd):
    src = str(tmp_path / "matrix-input.bw")
    _mk(src, {"chr1": (20, [(0, 10, 2.0), (10, 20, float("inf"))])})

    matrix = _compute_matrix_native.bin_bigwig_batch(
        [src], ["chr1"], [[([(0, 20)], 2)]], [[[0]]], [[[1.0]]],
        [False], [0], [0], "mean", False, 1)

    assert matrix[0, 0] == 2.0
    assert np.isnan(matrix[0, 1])
    warning = capfd.readouterr().err
    assert warning.count("omitted non-finite values") == 1
    assert src in warning


def test_merge_union_and_missing_is_zero(tmp_path):
    a = str(tmp_path / "a.bw")
    b = str(tmp_path / "b.bw")
    _mk(a, {"chr1": (1000, [(0, 100, 1.0)]), "chr2": (500, [(0, 50, 5.0)])})
    _mk(b, {"chr1": (1000, [(0, 100, 3.0)])})  # no chr2
    o = str(tmp_path / "u.bw")
    _bigwig.bigwig_merge([a, b], o, "bigwig", "sum", 1, 1.0, 1, 10, -1)
    bw = pyBigWig.open(o)
    assert set(bw.chroms()) == {"chr1", "chr2"}
    bw.close()
    assert np.allclose(_values(o, "chr2", 500), _values(a, "chr2", 500))  # a only


@pytest.mark.parametrize("name_length", [203, 1000])
@pytest.mark.parametrize("fmt,suffix,opener", [
    ("bedgraph", ".bedgraph", open),
    ("bedgraph.gz", ".bedgraph.gz", gzip.open),
])
def test_merge_bedgraph_supports_long_chromosome_names(
        tmp_path, name_length, fmt, suffix, opener):
    chrom = "c" * name_length
    src = str(tmp_path / "long.bw")
    _mk(src, {chrom: (100, [(0, 100, 1.25)])})
    out = str(tmp_path / ("out" + suffix))

    _bigwig.bigwig_merge([src], out, fmt, "sum", 1, 1.0, 1, 0, -1)

    with opener(out, "rt") as fh:
        assert fh.read() == f"{chrom}\t0\t100\t1.25\n"


def test_merge_length_mismatch_errors(tmp_path):
    a = str(tmp_path / "a.bw")
    c = str(tmp_path / "c.bw")
    _mk(a, {"chr1": (1000, [(0, 100, 1.0)])})
    _mk(c, {"chr1": (2000, [(0, 100, 1.0)])})
    with pytest.raises(RuntimeError, match="different lengths"):
        _bigwig.bigwig_merge([a, c], str(tmp_path / "x.bw"), "bigwig", "sum",
                             1, 1.0, 1, 10, -1)


def test_merge_preserves_chrom_order_chr10_accessible(tmp_path):
    # Input order chr1, chr10, chr2 (non-lexical) must survive so chr10 stays readable.
    ents = {"chr1": (100, [(0, 50, 1.0)]), "chr10": (100, [(0, 50, 2.0)]),
            "chr2": (100, [(0, 50, 3.0)])}
    a = str(tmp_path / "a.bw")
    b = str(tmp_path / "b.bw")
    _mk(a, ents)
    _mk(b, ents)
    o = str(tmp_path / "o.bw")
    _bigwig.bigwig_merge([a, b], o, "bigwig", "sum", 1, 1.0, 1, 10, -1)
    bw = pyBigWig.open(o)
    for chrom in ("chr1", "chr10", "chr2"):
        assert bw.stats(chrom, 0, 100, type="sum")[0] is not None  # accessible
    bw.close()


def test_merge_binsize_and_parallel(tmp_path):
    a = str(tmp_path / "a.bw")
    b = str(tmp_path / "b.bw")
    _mk(a, {"chr1": (10000, [(0, 5000, 2.0)])})
    _mk(b, {"chr1": (10000, [(0, 5000, 4.0)])})
    s = str(tmp_path / "s.bw")
    m = str(tmp_path / "m.bw")
    _bigwig.bigwig_merge([a, b], s, "bigwig", "sum", 50, 1.0, 1, 10, -1)
    _bigwig.bigwig_merge([a, b], m, "bigwig", "sum", 50, 1.0, 4, 10, -1)
    assert np.array_equal(_values(s, "chr1", 10000), _values(m, "chr1", 10000))


def test_cli_scale(tmp_path):
    src = str(tmp_path / "in.bw")
    out = str(tmp_path / "o.bw")
    _make_bigwig(src)
    bigWigOperations.main(["scale", "-b", src, "-o", out, "-s", "-1"])
    assert np.allclose(_values(out, "chr1", 1000), -_values(src, "chr1", 1000))


# From audit regressions.

@pytest.fixture()
def io_native_backends(monkeypatch):
    monkeypatch.delenv('DTP_BAM_BACKEND', raising=False)


@pytest.mark.usefixtures('io_native_backends')
@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX write-failure injection')
@pytest.mark.parametrize('kind', ['bedgraph', 'bigwig'])
def test_close_failure_preserves_existing_output(tmp_path, kind):
    source = tmp_path / 'input.bw'
    with pyBigWig.open(str(source), 'w') as bw:
        bw.addHeader([('chr1', 1000)])
        bw.addEntries(['chr1'] * 20, list(range(0, 1000, 50)), ends=list(range(50, 1001, 50)), values=[float(i % 2 + 1) for i in range(20)])
    output = tmp_path / ('output.bedgraph' if kind == 'bedgraph' else 'output.bw')
    previous = b'PREVIOUS VALID RESULT'
    output.write_bytes(previous)
    child = subprocess.run([sys.executable, '-c', "\nimport resource, signal, sys\nfrom deeptoolsr import bigWigOperations\nsignal.signal(signal.SIGXFSZ, signal.SIG_IGN)\nlimit = 64 if sys.argv[3] == 'bedgraph' else 200\nresource.setrlimit(resource.RLIMIT_FSIZE, (limit, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))\nif sys.argv[3] == 'bedgraph':\n    bigWigOperations.main(['merge', '-b', sys.argv[1], '-o', sys.argv[2],\n                          '-of', 'bedgraph', '-bs', '50'])\nelse:\n    bigWigOperations.main(['scale', '-b', sys.argv[1], '-o', sys.argv[2],\n                          '-s', '2', '--zoomLevels', '0'])\n", str(source), str(output), kind], capture_output=True, text=True, timeout=30)
    assert child.returncode != 0 and output.read_bytes() == previous, (child.returncode, child.stderr, output.read_bytes())


# From numerical audit regressions.

@pytest.mark.parametrize('numerator,denominator,pc', [(2e-20, 1e-20, 1.0), (2.0, 1.0, 1e+30)])
def test_n06_log2fc_preserves_small_representable_effect(numerator, denominator, pc):
    a = np.array([[numerator]], np.float32)
    b = np.array([[denominator]], np.float32)
    expected = math.log1p((float(a[0, 0]) - float(b[0, 0])) / (float(b[0, 0]) + pc)) / math.log(2)
    _transform.binary_combine(a, b, [0, 1], [pc], 2, 0, 1)
    assert math.isclose(float(a[0, 0]), expected, rel_tol=2 * np.finfo(np.float32).eps, abs_tol=0)
