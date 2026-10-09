"""Each BAM/CRAM pair in tests/test_data gives byte-identical coverage.

The deepTools CRAMs name their reference as `./<name>.fa` (@SQ UR), so the
runs start in tests/test_data. encode_chip_slice is real data: ENCODE
ENCFF772IUE (single-end ChIP-seq), chr1:10-11 Mb, with a reference-free CRAM.
pointseq_slice is real paired, spliced POINT-seq data (100 kb of chr22, read
names and tags stripped), also with a reference-free CRAM.
"""

import collections
import os
import subprocess
import sys

import pytest

from deeptoolsr import _coverage, bamCoverage

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
PAIRS = ["test1", "test2", "testA", "testB", "test_filtering", "test_filtering2",
         "test_paired2", "encode_chip_slice", "pointseq_slice"]
OPTIONS = {
    "plain": [],
    "cpm": ["--normalizeUsing", "CPM"],
    "extend": ["--extendReads", "150"],
    "collapse": ["--collapse", "5prime"],
    "split": ["--strandedness", "reverse", "--filterRNAstrand", "split"],
    "blacklist": ["--blackListFileName", "{blacklist}"],
}


def outputs(tmp_path, name, kind, threads, options):
    out = tmp_path / f"{kind}-{threads}"
    out.mkdir()
    bamCoverage.main(["-b", f"{name}.{kind}", "-o", str(out / "cov.bw"), "-bs", "10",
                      "-p", str(threads), *options])
    return {path.name: path.read_bytes() for path in out.iterdir()}


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("name", PAIRS)
def test_cram_coverage_is_byte_identical_to_bam(tmp_path, monkeypatch, name, option):
    monkeypatch.chdir(ROOT)
    first = _coverage.bam_index_stats(f"{name}.bam")["targets"][0]
    blacklist = tmp_path / "blacklist.bed"
    blacklist.write_text(f"{first['name']}\t{first['length'] // 4}\t{first['length'] // 2}\n")
    options = [value.format(blacklist=blacklist) for value in OPTIONS[option]]
    expected = outputs(tmp_path, name, "bam", 1, options)
    assert expected
    for kind, threads in (("bam", 4), ("cram", 1), ("cram", 4)):
        assert outputs(tmp_path, name, kind, threads, options) == expected, (kind, threads)


def run_coverage(cwd, bam, out, threads, options, env=None):
    # A subprocess, so a native crash in a worker thread fails this test
    # instead of killing pytest.
    code = "import sys; from deeptoolsr import bamCoverage; bamCoverage.main(sys.argv[1:])"
    subprocess.run([sys.executable, "-c", code, "-b", bam, "-o", str(out), "-bs", "10",
                    "-p", str(threads), *options], cwd=cwd, check=True, capture_output=True, env=env)
    return out.read_bytes()


@pytest.mark.parametrize("option", ["plain", "extend"])
@pytest.mark.parametrize("name", ["encode_chip_slice", "test_paired2"])
def test_cram31_coverage_is_byte_identical_to_bam(tmp_path, name, option):
    """CRAM 3.1 codecs need deep stacks: worker threads must not overflow."""
    pysam = pytest.importorskip("pysam")
    cram = tmp_path / f"{name}.cram"
    with pysam.AlignmentFile(os.path.join(ROOT, f"{name}.bam")) as src, \
            pysam.AlignmentFile(str(cram), "wc", template=src,
                                format_options=[b"version=3.1", b"no_ref=1"]) as dst:
        for read in src:
            dst.write(read)
    assert cram.read_bytes()[4:6] == b"\x03\x01"
    pysam.index(str(cram))
    options = OPTIONS[option]
    expected = run_coverage(ROOT, f"{name}.bam", tmp_path / "bam.bg", 1,
                            ["--outFileFormat", "bedgraph", *options])
    assert expected
    for threads in (1, 4):
        assert run_coverage(tmp_path, cram.name, tmp_path / f"cram{threads}.bg", threads,
                            ["--outFileFormat", "bedgraph", *options]) == expected, threads


@pytest.mark.parametrize("options", [
    ["--extendReads"],
    ["--extendReads", "--skipIntrons"],
    ["--collapse", "3prime"],
    ["--strandedness", "reverse", "--filterRNAstrand", "split", "--normalizeUsing", "CPM"],
], ids=["extend", "skip-introns", "collapse", "split-cpm"])
def test_spliced_pairs_cram_is_byte_identical_to_bam(tmp_path, monkeypatch, options):
    monkeypatch.chdir(ROOT)
    expected = outputs(tmp_path, "pointseq_slice", "bam", 1, options)
    assert expected
    for kind, threads in (("bam", 4), ("cram", 1), ("cram", 4)):
        assert outputs(tmp_path, "pointseq_slice", kind, threads, options) == expected, \
            (kind, threads)


def per_base(path):
    values = {}
    for line in path.read_text().splitlines():
        _, start, end, value = line.split()
        if float(value):  # zero runs span the whole 50 Mb chromosome
            values.update(dict.fromkeys(range(int(start), int(end)), float(value)))
    return values


def introns(read):
    gaps, pos = set(), read.reference_start
    for op, length in read.cigartuples:
        if op == 3:
            gaps.update(range(pos, pos + length))
        if op in (0, 2, 3, 7, 8):
            pos += length
    return gaps


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("kind", ["bam", "cram"])
def test_skip_introns_empties_exactly_the_reads_introns(tmp_path, kind, threads):
    """--skipIntrons removes, per fragment, exactly its mates' N gaps.

    A proper pair (forward mate first) is one fragment; a dovetailed pair is
    extended as two reads, each its own unit.
    """
    pysam = pytest.importorskip("pysam")
    tracks = {}
    for name, extra in (("filled", []), ("skipped", ["--skipIntrons"])):
        run_coverage(ROOT, f"pointseq_slice.{kind}", tmp_path / name, threads,
                     ["--outFileFormat", "bedgraph", "-bs", "1", "--extendReads", *extra])
        tracks[name] = per_base(tmp_path / name)
    pairs = collections.defaultdict(list)
    with pysam.AlignmentFile(os.path.join(ROOT, "pointseq_slice.bam")) as reads:
        for read in reads:
            pairs[read.query_name].append(read)
    expected = collections.Counter()
    for mates in pairs.values():
        forward = [read for read in mates if not read.is_reverse]
        reverse = [read for read in mates if read.is_reverse]
        fragment = (len(forward) == len(reverse) == 1 and
                    forward[0].reference_start <= reverse[0].reference_start)
        for unit in [mates] if fragment else [[read] for read in mates]:
            expected.update(set().union(*map(introns, unit)))
    assert sum(all(map(introns, mates)) for mates in pairs.values()) > 500
    removed = {pos: value - tracks["skipped"].get(pos, 0.0)
               for pos, value in tracks["filled"].items()
               if value != tracks["skipped"].get(pos, 0.0)}
    assert removed == {pos: float(count) for pos, count in expected.items()}


def test_cram_reference_fasta(tmp_path, capsys):
    """Coverage decodes no bases, so a CRAM whose @SQ UR is relative to another
    directory needs no reference; the option still gives the same output."""
    bam, cram, fasta = (os.path.join(ROOT, f"test1.{ext}") for ext in ("bam", "cram", "fa"))
    for threads in (1, 4):
        for extra in ([], ["--extendReads"], ["--normalizeUsing", "CPM", "--exactScaling"]):
            expected = run_coverage(tmp_path, bam, tmp_path / f"bam{threads}.bw", threads, extra)
            for reference in ([], ["--referenceFasta", fasta]):
                assert run_coverage(tmp_path, cram, tmp_path / f"cram{threads}.bw", threads,
                                    [*reference, *extra]) == expected, (threads, extra, reference)

    bamCoverage.main(["-b", bam, "-o", str(tmp_path / "warn.bw"), "--referenceFasta", fasta])
    assert "--referenceFasta is unused for BAM/SAM input" in capsys.readouterr().err
    unindexed = tmp_path / "plain.fa"
    unindexed.write_bytes(open(fasta, "rb").read())
    with pytest.raises(SystemExit):
        bamCoverage.main(["-b", cram, "-o", str(tmp_path / "x.bw"),
                          "--referenceFasta", str(unindexed)])
    assert "plain.fa.fai not found" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        bamCoverage.main(["-b", cram, "-o", fasta, "--referenceFasta", fasta])


def test_multi_reference_slice_cram_needs_reference_fasta(tmp_path):
    """htslib loads the reference for a slice that mixes chromosomes, even
    without decoding bases; the hidden FASTA must then come from the option."""
    pysam = pytest.importorskip("pysam")
    reference = tmp_path / "ref" / "genome.fa"
    reference.parent.mkdir()
    contigs = {f"c{i}": "ACGTTGCAAT" * 200 for i in range(1, 4)}
    reference.write_text("".join(f">{name}\n{bases}\n" for name, bases in contigs.items()))
    pysam.faidx(str(reference))
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": name, "LN": len(bases)} for name, bases in contigs.items()]}
    bam, cram = str(tmp_path / "mixed.bam"), str(tmp_path / "mixed.cram")
    with pysam.AlignmentFile(bam, "wb", header=header) as out:
        for tid, bases in enumerate(contigs.values()):
            for start in range(100, 1600, 300):
                read = pysam.AlignedSegment(out.header)
                read.query_name, read.reference_id, read.reference_start = f"r{tid}-{start}", tid, start
                read.mapping_quality, read.cigarstring = 60, "50M"
                read.query_sequence = bases[start:start + 50]
                out.write(read)
    pysam.index(bam)
    with pysam.AlignmentFile(bam) as src, pysam.AlignmentFile(
            cram, "wc", template=src, reference_filename=str(reference),
            format_options=[b"multi_seq_per_slice=1"]) as out:
        for read in src:
            out.write(read)
    pysam.index(cram)
    hidden = tmp_path / "hidden"
    reference.parent.rename(hidden)
    empty = tmp_path / "empty"
    empty.mkdir()
    env = {**os.environ, "REF_PATH": str(empty), "REF_CACHE": str(empty)}
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_coverage(empty, cram, tmp_path / "missing.bw", 1, [], env)
    assert ("error reading BAM/CRAM data (truncated or corrupt input, or a CRAM whose "
            "reference is needed but not found: pass --referenceFasta FILE)"
            ) in error.value.stderr.decode()
    assert run_coverage(empty, cram, tmp_path / "cram.bw", 1,
                        ["--referenceFasta", str(hidden / "genome.fa")], env) == \
        run_coverage(empty, bam, tmp_path / "bam.bw", 1, [], env)


@pytest.mark.parametrize("threads", [1, 4])
def test_reference_free_cram_needs_no_reference(tmp_path, threads):
    cram = os.path.join(ROOT, "pointseq_slice.cram")
    bam = os.path.join(ROOT, "pointseq_slice.bam")
    assert run_coverage(tmp_path, cram, tmp_path / "cram.bw", threads, ["-e"]) == \
        run_coverage(tmp_path, bam, tmp_path / "bam.bw", threads, ["-e"])
