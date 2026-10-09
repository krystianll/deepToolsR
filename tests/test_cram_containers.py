"""CRAM coverage decodes each container once and stays byte-identical to BAM.

Every worker shares the decoded CRAM containers, and a query visits exactly the
records htslib's region iterator would return. The real slices are re-encoded
with 100 reads per container, so window and container boundaries cross often:
small windows sit inside one container, and default windows span many.
"""

import gzip
import os

import pytest

from deeptoolsr import _coverage, bamCoverage

pysam = pytest.importorskip("pysam")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
SLICES = ["encode_chip_slice", "pointseq_slice"]
OPTIONS = {
    "plain": [],
    "bs1": ["-bs", "1"],
    "extend": ["--extendReads", "150"],
    "collapse": ["--collapse", "5prime"],
    "split": ["--strandedness", "reverse", "--filterRNAstrand", "split"],
    "blacklist": ["--blackListFileName", "{regions}"],
    "whitelist-overlap": ["--whiteListFileName", "{regions}", "--filterByOverlap",
                          "--extendReads", "150"],
    "cpm": ["--normalizeUsing", "CPM"],
    "bpm": ["--normalizeUsing", "BPM"],
}


def small_containers(tmp_path, name):
    """The slice as a reference-free CRAM with 100 reads per container."""
    cram = tmp_path / f"{name}.small.cram"
    if not cram.exists():
        with pysam.AlignmentFile(os.path.join(ROOT, f"{name}.bam")) as src, \
                pysam.AlignmentFile(str(cram), "wc", template=src,
                                    format_options=[b"seqs_per_slice=100", b"no_ref=1"]) as dst:
            for read in src:
                dst.write(read)
        pysam.index(str(cram))
    return cram


def containers(cram):
    with gzip.open(f"{cram}.crai", "rt") as handle:
        return len({line.split("\t")[3] for line in handle})


def regions(tmp_path, name):
    target = _coverage.bam_index_stats(os.path.join(ROOT, f"{name}.bam"))["targets"][0]
    length = target["length"]
    bed = tmp_path / "regions.bed"
    rows = [(length * k // 20, length * k // 20 + length // 40)
            for k in range(0, 20, 3)]
    bed.write_text("".join(f"{target['name']}\t{a}\t{b}\n" for a, b in rows))
    return bed


def cli(tmp_path, source, label, threads, options):
    out = tmp_path / f"{label}-{threads}"
    out.mkdir()
    has_bs = "-bs" in options
    bamCoverage.main(["-b", str(source), "-o", str(out / "cov.bw"), "-p", str(threads),
                      *([] if has_bs else ["-bs", "10"]), *options])
    return {path.name: path.read_bytes() for path in out.iterdir()}


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("name", SLICES)
def test_cram_with_many_containers_is_byte_identical_to_bam(tmp_path, monkeypatch, name, option):
    monkeypatch.chdir(ROOT)
    bed = regions(tmp_path, name)
    options = [value.format(regions=bed) for value in OPTIONS[option]]
    expected = cli(tmp_path, f"{name}.bam", "bam", 1, options)
    assert expected
    small = small_containers(tmp_path, name)
    for label, source, threads in (("cram", f"{name}.cram", 4), ("small", small, 1),
                                   ("small", small, 4)):
        assert cli(tmp_path, source, label, threads, options) == expected, (label, threads)


def api(source, out, threads, window_size, knob, **options):
    _coverage.bam_coverage_bigwig(str(source), str(out), out_format="bedgraph", bin_size=10,
                                  threads=threads, window_size=window_size, max_zooms=0,
                                  mate_budget=knob, **options)
    return out.read_bytes()


API_OPTIONS = {
    "plain": {},
    "extend": dict(extend_reads=150),
    "collapse": dict(collapse="5prime"),
    "whitelist-overlap": dict(extend_reads=150, filter_by_overlap=True),
}


@pytest.mark.parametrize("option", API_OPTIONS)
@pytest.mark.parametrize("window_size", [7, 0])
def test_each_container_is_decoded_once(tmp_path, option, window_size):
    """7 bins of 10 bp: every container spans many windows. 0: the default
    windows, each spanning many containers. Either way one decode per
    container (plus the end-of-file container and an unindexed
    trailing one), and a cache too small to keep
    any idle container still gives the same bytes."""
    name = "pointseq_slice"
    small = small_containers(tmp_path, name)
    options = dict(API_OPTIONS[option])
    if options.get("filter_by_overlap"):
        bed = regions(tmp_path, name)
        target = bed.read_text().split("\t")[0]
        options["whitelist"] = {target: [tuple(map(int, row.split("\t")[1:3]))
                                         for row in bed.read_text().splitlines()]}
    expected = api(os.path.join(ROOT, f"{name}.bam"), tmp_path / "bam.bg", 1, window_size,
                   None, **options)
    assert expected
    for threads in (1, 4):
        knob = {}
        assert api(small, tmp_path / f"cram{threads}.bg", threads, window_size, knob,
                   **options) == expected, threads
        if threads == 1:
            assert 0 < knob["cram_container_decodes"] <= containers(small) + 2, knob
        tiny = {"cram_cache_bytes": 1}
        assert api(small, tmp_path / f"tiny{threads}.bg", threads, window_size, tiny,
                   **options) == expected, threads
