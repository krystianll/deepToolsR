"""Output publication and input-integrity safety (case aliases, modes,
all-or-nothing replacement, temporary names, truncated inputs, native paths)."""

import gzip
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from deeptoolsr import (bamCoverage, bigWigOperations, computeMatrix,
                        computeMatrixOperations, path_validation)
from deeptoolsr.native_path import native_path
from deeptoolsr.path_validation import (OutputSet, atomic_output_path,
                                        validate_input_output_paths)
from tests.helpers.bigwig import pyBigWig

DATA = Path(__file__).parent / "test_data"
ROOT = Path(__file__).parents[1]


def _umask():
    mask = os.umask(0)
    os.umask(mask)
    return mask


def test_outputs_differing_in_case_or_normalisation_are_aliases(tmp_path):
    with pytest.raises(ValueError, match="same file"):
        validate_input_output_paths([], [tmp_path / "out.plus.bw", tmp_path / "out.PLUS.bw"])
    nfc, nfd = "caf\u00e9.gz", "cafe\u0301.gz"
    with pytest.raises(ValueError, match="same file"):
        validate_input_output_paths([], [tmp_path / nfc, tmp_path / nfd])


def test_split_suffixes_differing_in_case_write_nothing(tmp_path, capsys):
    with pytest.raises(SystemExit):
        bamCoverage.main(["-b", str(DATA / "test_paired.bam"), "-o", str(tmp_path / "out.bedgraph"),
                          "--outFileFormat", "bedgraph", "--filterRNAstrand", "split",
                          "--strandedness", "forward", "--suffix", ".plus", ".PLUS"])
    assert "same file" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_compute_matrix_outputs_differing_in_case_are_rejected(tmp_path):
    with pytest.raises(SystemExit, match="same file"):
        computeMatrix.main(["reference-point", "-S", str(DATA / "testA.bw"),
                            "-R", str(DATA / "computeMatrixOperations.bed"),
                            "-o", str(tmp_path / "res.gz"),
                            "--outFileNameMatrix", str(tmp_path / "RES.GZ")])
    assert list(tmp_path.iterdir()) == []


def test_file_named_dash_is_a_path_not_a_stream(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Path("-").write_bytes((DATA / "computeMatrixOperations.mat.gz").read_bytes())
    with pytest.raises(SystemExit):
        computeMatrixOperations.main(["relabel", "-m", "-", "-o", "-", "--groupLabels", "a"])
    assert "aliases input" in capsys.readouterr().err


def test_chrom_sizes_dash_still_writes_stdout(capsys):
    bigWigOperations.main(["info", str(DATA / "testA.bw"), "--chromSizes", "-"])
    assert re.match(r"\S+\t\d+\n", capsys.readouterr().out)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_published_outputs_get_the_default_file_mode(tmp_path):
    destination = tmp_path / "out.txt"
    with atomic_output_path(destination) as temporary:
        Path(temporary).write_text("x")
    assert destination.stat().st_mode & 0o777 == 0o666 & ~_umask()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX umask")
def test_new_outputs_follow_the_umask_current_at_write_time(tmp_path):
    # A fresh interpreter controls the umask the package is imported under:
    # importing leaves the umask alone, and a later umask, tighter or looser,
    # decides the mode of each new output.
    code = ("import os, sys\n"
            "os.umask(0o022)\n"
            "from deeptoolsr.config import save_options\n"
            "print(oct(os.umask(0o077)))\n"
            "for mask, name in ((0o077, 'tight.txt'), (0o002, 'loose.txt')):\n"
            "    os.umask(mask)\n"
            "    path = save_options({}, os.path.join(sys.argv[1], name))\n"
            "    print(oct(os.stat(path).st_mode & 0o777))\n")
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    umask_after_import, tight, loose = result.stdout.split()
    assert umask_after_import == oct(0o022)
    assert (tight, loose) == (oct(0o600), oct(0o664))


def test_failed_replacement_restores_every_previous_output(tmp_path, monkeypatch):
    first, second = tmp_path / "out.plus.bedgraph", tmp_path / "out.minus.bedgraph"
    first.write_text("old plus")
    second.write_text("old minus")
    real_replace = os.replace

    def failing_replace(source, destination):
        if os.fspath(destination) == str(second):
            raise PermissionError("destination is open elsewhere")
        real_replace(source, destination)

    monkeypatch.setattr(path_validation.os, "replace", failing_replace)
    with pytest.raises(PermissionError):
        with OutputSet() as outputs:
            Path(outputs.stage(first)).write_text("new plus")
            Path(outputs.stage(second)).write_text("new minus")
    assert first.read_text() == "old plus"
    assert second.read_text() == "old minus"
    assert sorted(path.name for path in tmp_path.iterdir()) == [second.name, first.name]


def test_interrupt_during_publication_restores_previous_outputs(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("a.txt", "b.txt", "c.txt")]
    for path in paths:
        path.write_text("old " + path.name)
    real_replace = os.replace

    def interrupted(source, destination):
        if os.fspath(destination) == str(paths[2]):
            raise KeyboardInterrupt
        real_replace(source, destination)

    monkeypatch.setattr(path_validation.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        with OutputSet() as outputs:
            for path in paths:
                Path(outputs.stage(path)).write_text("new")
    assert [path.read_text() for path in paths] == ["old " + path.name for path in paths]
    assert len(list(tmp_path.iterdir())) == 3


def _windows_long_paths_off():
    from deeptoolsr.native_path import long_paths_enabled
    return sys.platform == "win32" and not long_paths_enabled()


def _refused_as_too_long(capsys, argv, main=None):
    with pytest.raises(SystemExit) as stop:
        (main or bamCoverage.main)(argv)
    assert "Windows long paths are off" in (
        str(stop.value.code) + capsys.readouterr().err)


def test_long_output_basenames_fit_their_temporaries(tmp_path, capsys):
    name = "a" * 240 + ".bw"
    argv = ["-b", str(DATA / "test1.bam"), "-o", str(tmp_path / name),
            "--binSize", "10"]
    if _windows_long_paths_off() and len(os.path.abspath(tmp_path / name)) > 259:
        # Windows with long paths off: refused up front, nothing written.
        _refused_as_too_long(capsys, argv)
        assert list(tmp_path.iterdir()) == []
        return
    bamCoverage.main(argv)
    assert [path.name for path in tmp_path.iterdir()] == [name]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows MAX_PATH")
@pytest.mark.parametrize("side", ["input", "output"])
def test_long_paths_off_is_a_clear_error_before_any_work(
        tmp_path, monkeypatch, capsys, side):
    from deeptoolsr import native_path as native
    monkeypatch.setattr(native, "long_paths_enabled", lambda: False)
    deep = tmp_path.joinpath("d" * 100, "e" * 100, "f" * 100)
    _extended(deep).mkdir(parents=True)
    bam = DATA / "test1.bam"
    if side == "input":
        for name in ("test1.bam", "test1.bam.bai"):
            _extended(deep / name).write_bytes((DATA / name).read_bytes())
        bam, output = deep / "test1.bam", tmp_path / "out.bw"
    else:
        output = deep / "out.bw"
    with pytest.raises(SystemExit) as stop:
        bamCoverage.main(["-b", str(bam), "-o", str(output), "--binSize", "10"])
    message = str(stop.value.code) + capsys.readouterr().err
    assert "Windows long paths are off" in message
    assert not _extended(output).exists()


def _extended(path):
    # The test's own file operations reach a path longer than MAX_PATH on
    # Windows through the extended-length form; deepToolsR gets plain paths.
    if sys.platform == "win32":
        return Path("\\\\?\\" + os.path.abspath(path).replace("/", "\\"))
    return Path(path)


def test_paths_longer_than_260_characters_work_for_every_file_kind(tmp_path, capsys):
    import bz2
    from deeptoolsr import deeptoolsr_list_tools, plotHeatmap, plotProfile
    deep = tmp_path.joinpath("d" * 100, "e" * 100, "f" * 100)
    _extended(deep).mkdir(parents=True)
    for name in ("test1.bam", "test1.bam.bai", "test1.cram", "test1.cram.crai",
                 "test1.fa", "test1.fa.fai"):
        _extended(deep / name).write_bytes((DATA / name).read_bytes())
    bed = b"3R\t100\t200\tr1\t0\t+\n3R\t400\t600\tr2\t0\t-\n"
    gtf = (DATA / "test.gtf").read_bytes()
    _extended(deep / "regions.bed").write_bytes(bed)
    _extended(deep / "regions.bed.gz").write_bytes(gzip.compress(bed))
    _extended(deep / "regions.gtf").write_bytes(gtf)
    _extended(deep / "regions.gtf.bz2").write_bytes(bz2.compress(gtf))
    _extended(deep / "blacklist.bed").write_bytes(b"3R\t1400\t1450\n")

    def at(name):
        return str(deep / name)

    assert len(at("regions.bed")) > 260
    if _windows_long_paths_off():
        # Windows with long paths off: refused up front, nothing written.
        _refused_as_too_long(capsys, ["-b", at("test1.bam"), "-o", at("bam.bw")])
        assert not _extended(deep / "bam.bw").exists()
        return
    deeptoolsr_list_tools.main(["options", "--config", at("options.txt")])
    bamCoverage.main(["-b", at("test1.bam"), "-o", at("bam.bw"), "--binSize", "10",
                      "--blackListFileName", at("blacklist.bed")])
    bamCoverage.main(["-b", at("test1.cram"), "--referenceFasta", at("test1.fa"),
                      "-o", at("cram.bedgraph"), "--outFileFormat", "bedgraph"])
    bigWigOperations.main(["scale", "--bigwig", at("bam.bw"),
                           "-o", at("scaled.bw"), "--scaleFactor", "2"])
    bigWigOperations.main(["merge", "--bigwig", at("bam.bw"), at("scaled.bw"),
                           "-o", at("merged.bedgraph"), "-of", "bedgraph"])
    bigWigOperations.main(["info", at("scaled.bw"), "--chromSizes", at("chrom.sizes")])
    computeMatrix.main([
        "reference-point", "-S", at("bam.bw"), at("scaled.bw"),
        "-R", at("regions.bed"), at("regions.bed.gz"), at("regions.gtf"),
        at("regions.gtf.bz2"), "-bl", at("blacklist.bed"), "-a", "100", "-b", "100",
        "--config", at("options.txt"), "-o", at("matrix.gz"),
        "--outFileNameMatrix", at("matrix.tab"),
        "--outFileSortedRegions", at("sorted.bed")])
    computeMatrixOperations.main(["subset", "-m", at("matrix.gz"),
                                  "-o", at("subset.mat.gz"), "--samples", "bam"])
    for plot, figure, extra in (
            (plotHeatmap, "heatmap.png", ["--outFileNameMatrix", at("heatmap.tab"),
                                          "--outFileSortedRegions", at("heatmap.bed")]),
            (plotHeatmap, "heatmap.svg", []),
            (plotProfile, "profile.pdf", ["--outFileNameData", at("profile.tab")])):
        plot.main(["-m", at("subset.mat.gz"), "-o", at(figure),
                   "--config", at("options.txt"), *extra])
    for name in ("options.txt", "bam.bw", "cram.bedgraph", "scaled.bw",
                 "merged.bedgraph", "chrom.sizes", "matrix.gz", "matrix.tab",
                 "sorted.bed", "subset.mat.gz", "heatmap.png", "heatmap.tab",
                 "heatmap.bed", "heatmap.svg", "profile.pdf", "profile.tab"):
        assert _extended(deep / name).stat().st_size > 0, name


def _truncated_bam(tmp_path):
    data = (DATA / "test1.bam").read_bytes()
    block = struct.unpack("<H", data[16:18])[0] + 1  # first BGZF block: the header
    path = tmp_path / "cut.bam"
    path.write_bytes(data[:block])
    (tmp_path / "cut.bam.bai").write_bytes((DATA / "test1.bam.bai").read_bytes())
    return path


def _truncated_cram(tmp_path):
    path = tmp_path / "cut.cram"
    path.write_bytes((DATA / "test1.cram").read_bytes()[:236])  # header container only
    (tmp_path / "cut.cram.crai").write_bytes((DATA / "test1.cram.crai").read_bytes())
    return path


@pytest.mark.parametrize("make", [_truncated_bam, _truncated_cram])
def test_truncated_alignments_are_an_error(tmp_path, make):
    source = make(tmp_path)
    with pytest.raises(SystemExit, match="file may be truncated"):
        bamCoverage.main(["-b", str(source), "-o", str(tmp_path / "out.bedgraph"),
                          "--outFileFormat", "bedgraph"])
    assert not (tmp_path / "out.bedgraph").exists()


def test_failed_region_query_is_an_error_not_zeros(tmp_path):
    cram = tmp_path / "reads.cram"
    cram.write_bytes((DATA / "test1.cram").read_bytes())
    entries = gzip.decompress((DATA / "test1.cram.crai").read_bytes()).decode().split("\t")
    entries[3] = "-5"  # container offset htslib cannot seek to
    (tmp_path / "reads.cram.crai").write_bytes(gzip.compress("\t".join(entries).encode()))
    with pytest.raises(SystemExit, match="does not match the file.*re-index it"):
        bamCoverage.main(["-b", str(cram), "-o", str(tmp_path / "out.bedgraph"),
                          "--outFileFormat", "bedgraph"])


def _patched_bigwig(tmp_path, start, end):
    """A bigWig whose second interval is [start, end) (written as [20, 25))."""
    import zlib
    path = tmp_path / "patched.bw"
    with pyBigWig.open(str(path), "w") as writer:
        writer.addHeader([("chr1", 100)], maxZooms=0)
        writer.addEntries(["chr1"] * 3, [0, 20, 30], ends=[10, 25, 40], values=[1.0, 5.0, 2.0])
    data = bytearray(path.read_bytes())
    full, index = struct.unpack("<QQ", data[16:32])
    block = bytes(data[full + 8:index])
    raw = bytearray(zlib.decompress(block))
    raw[36:44] = struct.pack("<II", start, end)
    for level in range(9, -1, -1):
        packed = zlib.compress(bytes(raw), level)
        if len(packed) <= len(block):
            break
    data[full + 8:index] = packed + b"\0" * (len(block) - len(packed))
    path.write_bytes(bytes(data))
    return path


@pytest.mark.parametrize("start,end", [(20, 15), (20, 20)])
def test_empty_or_reversed_bigwig_intervals_are_rejected(tmp_path, start, end):
    source = _patched_bigwig(tmp_path, start, end)
    with pytest.raises(SystemExit, match="Failed reading bigWig"):
        bigWigOperations.main(["merge", "-b", str(source), "-o", str(tmp_path / "m.bedgraph"),
                               "--outFileFormat", "bedgraph"])


def test_failed_bigwig_write_open_is_an_error_not_a_crash(tmp_path):
    # In a subprocess: a regression would crash the interpreter.
    code = ("from deeptoolsr import _bigwig\n"
            "try:\n"
            "    _bigwig.bigwig_scale({!r}, {!r}, 2.0)\n"
            "except RuntimeError as error:\n"
            "    print('error:', error)\n").format(
        str(DATA / "testA.bw"), str(tmp_path / "missing" / "out.bw"))
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "could not open bigWig" in result.stdout


def test_regions_reject_truncated_gzip_with_zlib_132(tmp_path):
    """zlib >= 1.3.2 reports a truncated stream only on the read that hit it."""
    keg = Path("/opt/homebrew/opt/zlib/lib")
    if sys.platform != "darwin" or not (keg / "libz.1.dylib").exists():
        pytest.skip("needs the Homebrew zlib keg")
    raw = b"".join(b"chr1\t%d\t%d\n" % (i, i + 5) for i in range(0, 50000, 10))
    stored = gzip.compress(raw, compresslevel=0)
    cut = 10 + 5 + raw.index(b"\n", 30000) + 1  # inside the first stored block, at a line end
    path = tmp_path / "cut.bed.gz"
    path.write_bytes(stored[:cut])
    code = ("from deeptoolsr.regions import load_regions\n"
            "try:\n"
            "    load_regions({!r})\n"
            "except RuntimeError as error:\n"
            "    print(error)\n").format(str(path))
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                            text=True, env=dict(os.environ, DYLD_LIBRARY_PATH=str(keg)))
    assert result.returncode == 0, result.stderr
    assert "truncated gzip file" in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ANSI code page")
def test_native_path_uses_the_ansi_code_page():
    assert native_path("C:\\data\\reads.bam") == b"C:\\data\\reads.bam"
    with pytest.raises(ValueError, match="ANSI code page"):
        native_path("C:\\data\\\U0001f600.bam")


def test_native_path_is_identity_on_posix():
    if sys.platform == "win32":
        pytest.skip("POSIX only")
    assert native_path(Path("x/\u00e9.bam")) == "x/\u00e9.bam"


# Native entry points that open files, and where compute.py reads chromosome
# sizes through an injectable reader.
_NATIVE_OPENERS = (
    "estimate_fragment_length", "bam_index_stats", "infer_strandedness",
    "resolve_bam_index_path", "bam_coverage_bigwig", "bigwig_info", "bigwig_scale",
    "bigwig_merge", "read_matrix", "write_matrix", "read_file", "bin_bigwig_batch",
    "filter_strand", "filter_values", "filter_values_mask", "subset_columns",
    "bigwig_chrom_sizes")


def test_every_native_file_open_receives_a_native_path():
    call = re.compile(r"\.({})\(\s*(\S+)".format("|".join(_NATIVE_OPENERS)))
    seen = set()
    for module in sorted((ROOT / "deeptoolsr").rglob("*.py")):
        for match in call.finditer(module.read_text()):
            seen.add(match.group(1))
            assert match.group(2).lstrip("[").startswith("native_path("), (
                module.name, match.group(0))
    assert seen == set(_NATIVE_OPENERS)
