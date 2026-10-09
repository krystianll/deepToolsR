"""BAM, CRAM and bgzipped SAM give the same coverage; damaged indexes are errors."""

import gzip
import io
import subprocess
import sys

import pytest

from deeptoolsr import _bigwig, _coverage, bamCoverage, regions

pysam = pytest.importorskip("pysam")

CHROM_LENGTH = 3_000_000


def _write_sorted(path, header, reads):
    unsorted = path.with_suffix(".unsorted.bam")
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as out:
        for name, flag, pos, mate_pos, tlen in reads:
            read = pysam.AlignedSegment(out.header)
            read.query_name, read.flag, read.reference_id = name, flag, 0
            read.reference_start, read.mapping_quality = pos, 30
            read.cigarstring, read.query_sequence = "50M", "ACGT" * 12 + "AC"
            if flag & 1:
                read.next_reference_id, read.next_reference_start = 0, mate_pos
                read.template_length = tlen
            out.write(read)
    pysam.sort("-o", str(path), str(unsorted))
    pysam.index(str(path))


@pytest.fixture(scope="module")
def alignments(tmp_path_factory):
    """The same reads as BAM, CRAM and bgzipped SAM (+ .csi), spread over
    several coverage windows, with pairs whose mates sit across window edges."""
    base = tmp_path_factory.mktemp("alignments")
    reads = [(f"s{i}", 16 if i % 3 else 0, i * 7_919, -1, 0) for i in range(350)]
    for i in range(120):
        start = (i * 24_989) % (CHROM_LENGTH - 2_000)
        mate = start + 150 + (i % 5) * 40
        size = mate + 50 - start
        reads += [(f"p{i}", 99, start, mate, size), (f"p{i}", 147, mate, start, -size)]
    bam = base / "reads.bam"
    _write_sorted(bam, {"HD": {"VN": "1.6", "SO": "coordinate"},
                        "SQ": [{"SN": "chr1", "LN": CHROM_LENGTH}]}, reads)
    # AlignmentFile rather than pysam.view: pysam's samtools view crashes on Windows.
    cram, sam = base / "reads.cram", base / "reads.sam"
    for path, mode, options in ((cram, "wc", [b"no_ref=1"]), (sam, "w", [])):
        with pysam.AlignmentFile(str(bam)) as src, pysam.AlignmentFile(
                str(path), mode, template=src, format_options=options) as dst:
            for read in src:
                dst.write(read)
    pysam.index(str(cram))
    sam_gz = base / "reads.sam.gz"
    pysam.tabix_compress(str(sam), str(sam_gz))
    pysam.index("-c", str(sam_gz))
    blacklist = base / "black.bed"
    blacklist.write_text("chr1\t1048000\t1049000\nchr1\t2097000\t2098500\n")
    return {"bam": bam, "cram": cram, "sam.gz": sam_gz, "blacklist": blacklist}


def _bedgraph(alignments, kind, tmp_path, options):
    out = tmp_path / f"{kind}.bedgraph"
    options = [str(alignments["blacklist"]) if o == "BLACKLIST" else o for o in options]
    bamCoverage.main(["-b", str(alignments[kind]), "-o", str(out),
                      "--outFileFormat", "bedgraph", "-bs", "100"] + options)
    return out.read_text()


@pytest.mark.parametrize("threads", ["1", "4"])
@pytest.mark.parametrize("options", [
    [],
    ["--normalizeUsing", "CPM"],
    ["--normalizeUsing", "RPKM"],
    ["--extendReads"],
    ["--extendReads", "--blackListFileName", "BLACKLIST", "--filterByOverlap"],
], ids=["plain", "cpm", "rpkm", "extend", "filter-by-overlap"])
def test_cram_and_sam_gz_coverage_equals_bam(alignments, tmp_path, threads, options):
    # Each worker and mate lookup must load its own index: a CRAM index set
    # the region on the handle that loaded it (wrong ranges), and a SAM.gz
    # worker without its header crashed.
    expected = _bedgraph(alignments, "bam", tmp_path, options + ["-p", threads])
    assert expected.count("\n") > 100
    for kind in ("cram", "sam.gz"):
        assert _bedgraph(alignments, kind, tmp_path, options + ["-p", threads]) == expected, kind


def test_cram_read_counts_match_bam_index_statistics(alignments):
    # CRAI stores no per-reference counts; they are counted from the CRAM.
    bam = _coverage.bam_index_stats(str(alignments["bam"]))
    assert bam["total_mapped"] == 590
    assert _coverage.bam_index_stats(str(alignments["cram"])) == bam


@pytest.fixture
def tiny_bam(tmp_path):
    """One 1500 bp reference with one index bin, as in the damaged-index repros."""
    bam = tmp_path / "tiny.bam"
    _write_sorted(bam, {"HD": {"VN": "1.6", "SO": "coordinate"},
                        "SQ": [{"SN": "3R", "LN": 1500}]},
                  [(f"r{i}", 0, i * 70, -1, 0) for i in range(20)])
    return bam


@pytest.mark.parametrize("offset, value", [
    (4, 0x00),     # no references listed (crashed in htslib)
    (12, 0x26),    # the data bin moved off the chromosome (empty track)
    (14, 0xFF),    # bin id out of range (empty track)
    (15, 0xFF),    # bin id above INT_MAX (hung inside htslib)
], ids=["n-ref", "moved-bin", "bad-bin", "negative-bin"])
def test_damaged_bai_is_an_error(tiny_bam, tmp_path, offset, value):
    index = tiny_bam.parent / (tiny_bam.name + ".bai")
    data = bytearray(index.read_bytes())
    data[offset] = value
    index.write_bytes(bytes(data))
    run = subprocess.run(
        [sys.executable, "-c", "from deeptoolsr.bamCoverage import main; main()",
         "-b", str(tiny_bam), "-o", str(tmp_path / "out.bedgraph"),
         "--outFileFormat", "bedgraph"],
        capture_output=True, text=True, timeout=120)
    assert run.returncode == 1, run.stderr
    assert "does not match the file" in run.stderr and "re-index it" in run.stderr
    assert not (tmp_path / "out.bedgraph").exists()


def test_damaged_csi_bin_is_an_error(tiny_bam, tmp_path):
    # A CSI bin id above INT_MAX hung inside htslib, as for a .bai.
    (tiny_bam.parent / (tiny_bam.name + ".bai")).unlink()
    pysam.index("-c", str(tiny_bam))
    index = tiny_bam.parent / (tiny_bam.name + ".csi")
    data = bytearray(gzip.decompress(index.read_bytes()))
    data[24:28] = (0xFFFFFFF0).to_bytes(4, "little")  # the first bin id
    index.write_bytes(gzip.compress(bytes(data)))
    run = subprocess.run(
        [sys.executable, "-c", "from deeptoolsr.bamCoverage import main; main()",
         "-b", str(tiny_bam), "-o", str(tmp_path / "out.bw")],
        capture_output=True, text=True, timeout=120)
    assert run.returncode == 1, run.stderr
    assert "bin id 4294967280 is impossible" in run.stderr and "re-index it" in run.stderr


def test_empty_crai_is_an_error(alignments, tmp_path):
    cram = tmp_path / "reads.cram"
    cram.write_bytes(alignments["cram"].read_bytes())
    (tmp_path / "reads.cram.crai").write_bytes(gzip.compress(b""))
    with pytest.raises(SystemExit, match="does not match the file.*re-index it"):
        bamCoverage.main(["-b", str(cram), "-o", str(tmp_path / "out.bw")])


def test_large_nearly_empty_chromosome_writes_a_bigwig(tmp_path):
    # Too wide for any zoom level: the bigWig is written without zoom levels.
    bam = tmp_path / "wide.bam"
    _write_sorted(bam, {"HD": {"VN": "1.6", "SO": "coordinate"},
                        "SQ": [{"SN": "chr1", "LN": 2_000_000_000}]}, [("r", 0, 1000, -1, 0)])
    out = tmp_path / "wide.bw"
    bamCoverage.main(["-b", str(bam), "-o", str(out), "-bs", "10000"])
    info = _bigwig.bigwig_info(str(out))
    assert info["chroms"] == {"chr1": 2_000_000_000} and info["nLevels"] == 0
    assert info["maxVal"] == pytest.approx(50 / 10000) and info["nBasesCovered"] == 2_000_000_000


def test_messages_and_labels_quote_paths_and_lines_as_valid_utf8(tmp_path):
    # A native path that is not valid UTF-8 (a Windows ANSI path) never breaks
    # a message or a label, and an overlong bad line is quoted shortly.
    def region_set():
        return regions._regions.RegionSet("transcript", "exon", "transcript_id", False, None)

    # Windows decodes native paths from the ANSI code page; elsewhere they
    # are UTF-8, so bytes that are not show as U+FFFD.
    e, y = ((b"\xe9".decode("mbcs"), b"\xff".decode("mbcs")) if sys.platform == "win32"
            else ("\ufffd", "\ufffd"))
    name = b"/data/r\xe9gions.bed"
    lines = b"chr1\t1\t100\tn\t0\t+\nchr1\t5\t" + b"x" * 5000 + b"\n"
    with pytest.raises(RuntimeError) as error:
        region_set().read_stream(io.BytesIO(lines).read, name)
    message = str(error.value)
    assert message.startswith("BED line with 3 columns in /data/r" + e + "gions.bed, "
                              "whose first region line has 6 (6 or more needed): chr1")
    assert len(message) < 400 and message.endswith("\u2026")
    with pytest.raises(RuntimeError, match="cannot open .*missing" + y + ".bed"):
        region_set().read_file(str(tmp_path).encode() + b"/missing\xff.bed")
    labelled = region_set()
    labelled.read_stream(io.BytesIO(b"chr1\t1\t100\n").read, name)
    labelled.finish()
    assert labelled.labels == ["r" + e + "gions.bed"]
