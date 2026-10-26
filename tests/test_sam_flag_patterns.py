"""--samFlagInclude patterns and --samFlagExclude masks: parser and behaviour."""
import argparse
import shutil
import subprocess

import numpy as np
import pytest

from deeptoolsr import bamCoverage
from deeptoolsr.bamCoverage import sam_flag_mask, sam_flag_pattern
from tests.helpers.coverage import record, write_bam, read_values

pysam = pytest.importorskip("pysam")

PAIRED, PROPER, REVERSE, MREVERSE = 0x1, 0x2, 0x10, 0x20
READ1, READ2, SECONDARY, DUP = 0x40, 0x80, 0x100, 0x400


@pytest.mark.parametrize("value,expected", [
    ("3", (3, 0)),
    ("PAIRED,PROPER_PAIR", (3, 0)),
    ("1,2", (3, 0)),
    ("0x3", (3, 0)),
    ("0X1f", (31, 0)),
    ("paired,Proper_Pair", (3, 0)),
    ("READ1", (64, 0)),
    ("^READ2", (0, 128)),
    ("^DUP", (0, 1024)),
    ("READ1,REVERSE", (80, 0)),
    ("READ2,^REVERSE", (128, 16)),
    ("^0x90", (0, 144)),
    ("0", (0, 0)),
    ("65535", (65535, 0)),
    ("UNMAP,MUNMAP,MREVERSE,SECONDARY,QCFAIL,SUPPLEMENTARY",
     (4 | 8 | 32 | 256 | 512 | 2048, 0)),
])
def test_include_pattern_parser(value, expected):
    assert sam_flag_pattern(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("1024", 1024), ("0x400", 1024), ("DUP,SECONDARY", 1280),
    ("dup,256", 1280), ("0", 0),
])
def test_exclude_mask_parser(value, expected):
    assert sam_flag_mask(value) == expected


@pytest.mark.parametrize("parse,value,message", [
    (sam_flag_pattern, "READ3", "unknown flag"),
    (sam_flag_pattern, "65536", "out of range"),
    (sam_flag_pattern, "0x10000", "out of range"),
    (sam_flag_pattern, "1,,2", "empty flag term"),
    (sam_flag_pattern, "", "empty flag term"),
    (sam_flag_pattern, "^", "empty flag term"),
    (sam_flag_pattern, "-1", "unknown flag"),
    (sam_flag_pattern, "1.5", "unknown flag"),
    (sam_flag_pattern, "REVERSE,^16", "both set and unset"),
    (sam_flag_pattern, "3,^PAIRED", "both set and unset"),
    (sam_flag_mask, "^DUP", "not allowed"),
    (sam_flag_mask, "BOGUS", "unknown flag"),
    (sam_flag_mask, "70000", "out of range"),
    (sam_flag_mask, "DUP,", "empty flag term"),
])
def test_parser_errors(parse, value, message):
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        parse(value)


def test_cli_errors_before_reading_input(tmp_path, capsys):
    out = tmp_path / "out.bw"
    with pytest.raises(SystemExit):
        bamCoverage.main(["-b", str(tmp_path / "missing.bam"), "-o", str(out),
                          "--samFlagInclude", "READ1", "READ1,^READ1"])
    assert "both set and unset" in capsys.readouterr().err
    assert not out.exists()


def test_cli_include_takes_several_values():
    args = bamCoverage.parse_arguments().parse_args([
        "-b", "x.bam", "-o", "y.bw",
        "--samFlagInclude", "READ1,REVERSE", "READ2,^REVERSE",
        "--samFlagExclude", "DUP,SECONDARY"])
    assert args.samFlagInclude == [(80, 0), (128, 16)]
    assert args.samFlagExclude == 1280


# Synthetic BAM: one proper pair per READ1 orientation x mate orientation, so
# every READ1/READ2 x REVERSE x MREVERSE combination occurs, plus a duplicate
# pair and a secondary alignment. Every alignment covers its own 10 bases.
def _synthetic_reads(fr_only=False):
    reads, start = [], 10
    for r1_rev in (False, True):
        for r2_rev in (False, True):
            if fr_only and r1_rev == r2_rev:
                continue
            base = PAIRED | PROPER
            f1 = base | READ1 | (REVERSE if r1_rev else 0) | (MREVERSE if r2_rev else 0)
            f2 = base | READ2 | (REVERSE if r2_rev else 0) | (MREVERSE if r1_rev else 0)
            name = "p%d" % start
            reads.append(record(name, start, start + 20, f1, 30))
            reads.append(record(name, start + 20, start, f2, -30))
            start += 50
    name = "dup"
    reads.append(record(name, start, start + 20, 99 | DUP, 30))
    reads.append(record(name, start + 20, start, 147 | DUP, -30))
    start += 50
    reads.append(record("sec", start, start + 20, 99 | SECONDARY, 30))
    return reads


@pytest.fixture(scope="module")
def synthetic_bam(tmp_path_factory):
    path = tmp_path_factory.mktemp("flags") / "flags.bam"
    write_bam(path, _synthetic_reads(), (400,))
    return path


def _oracle(keep):
    expected = np.zeros(400)
    for read in _synthetic_reads():
        if keep(read.flag):
            expected[read.reference_start:read.reference_start + 10] += 1
    return expected


def _run(bam, out, flag_args, extra=()):
    bamCoverage.main(["-b", str(bam), "-o", str(out), "--binSize", "1",
                      "--strandedness", "none", "--filterMode", "deeptools",
                      *extra, *flag_args])
    return out


# Each oracle is written from the flag meaning, not from set/unset masks.
BEHAVIOUR = [
    (["--samFlagInclude", "3"], lambda f: bool(f & PAIRED) and bool(f & PROPER)),
    (["--samFlagInclude", "READ1"], lambda f: bool(f & READ1)),
    (["--samFlagInclude", "READ1", "REVERSE"],
     lambda f: bool(f & READ1) or bool(f & REVERSE)),
    (["--samFlagInclude", "^READ2", "^REVERSE"],
     lambda f: not (f & READ2) or not (f & REVERSE)),
    (["--samFlagInclude", "^DUP"], lambda f: not f & DUP),
    (["--samFlagInclude", "READ1,REVERSE", "READ2,^REVERSE"],
     lambda f: (bool(f & READ1) and bool(f & REVERSE))
     or (bool(f & READ2) and not f & REVERSE)),
    (["--samFlagInclude", "MREVERSE,^REVERSE"],
     lambda f: bool(f & MREVERSE) and not f & REVERSE),
    (["--samFlagInclude", "SECONDARY"], lambda f: bool(f & SECONDARY)),
    (["--samFlagExclude", "DUP,SECONDARY"], lambda f: not f & (DUP | SECONDARY)),
    (["--samFlagExclude", "0x10"], lambda f: not f & REVERSE),
    (["--samFlagInclude", "READ2", "--samFlagExclude", "DUP"],
     lambda f: bool(f & READ2) and not f & DUP),
]


@pytest.mark.parametrize("unit", ["read", "fragment"])
@pytest.mark.parametrize("threads", ["1", "4"])
@pytest.mark.parametrize("flag_args,keep", BEHAVIOUR)
def test_patterns_keep_expected_reads(tmp_path, synthetic_bam, unit, threads,
                                      flag_args, keep):
    values = read_values(_run(synthetic_bam, tmp_path / "out.bw", flag_args,
                              ["--countUnit", unit, "-p", threads]))
    np.testing.assert_array_equal(values, _oracle(keep))


def test_deeptools_forward_recipe_in_one_run(tmp_path):
    # Proper pairs in real libraries face each other (flags 99/147, 83/163);
    # the same-orientation pairs of synthetic_bam (115/179, 67/131) would
    # also match the one-run patterns but not the flag-163/83 runs.
    bam = tmp_path / "fr.bam"
    write_bam(bam, _synthetic_reads(fr_only=True), (400,))
    common = ["--countUnit", "read", "--collapse", "5prime"]

    def run(name, *include):
        return read_values(_run(bam, tmp_path / name,
                                ["--samFlagInclude", *include], common))

    one = run("one.bw", "READ1,REVERSE", "READ2,^REVERSE")
    np.testing.assert_array_equal(one, run("a.bw", "163") + run("b.bw", "83"))
    assert one.sum() > 0


@pytest.mark.skipif(shutil.which("samtools") is None, reason="needs samtools")
@pytest.mark.parametrize("samtools_args,flag_args", [
    (["-f", "3"], ["--samFlagInclude", "3"]),
    (["-F", "1024"], ["--samFlagExclude", "1024"]),
    (["-F", "1024"], ["--samFlagInclude", "^DUP"]),
    (["--rf", "80"], ["--samFlagInclude", "READ1", "REVERSE"]),
    (["-G", "144"], ["--samFlagInclude", "^READ2", "^REVERSE"]),
])
def test_equivalent_to_samtools_filtering(tmp_path, synthetic_bam,
                                          samtools_args, flag_args):
    filtered = tmp_path / "filtered.bam"
    subprocess.run(["samtools", "view", "-b", "-o", str(filtered),
                    *samtools_args, str(synthetic_bam)], check=True)
    subprocess.run(["samtools", "index", str(filtered)], check=True)
    common = ["--countUnit", "read"]
    expected = _run(filtered, tmp_path / "a.bw", [], common)
    actual = _run(synthetic_bam, tmp_path / "b.bw", flag_args, common)
    assert actual.read_bytes() == expected.read_bytes()
