"""bamCoverageR --countUnit read|fragment against an independent oracle.

One synthetic BAM holds every pair class: overlapping mates (p), separate
mates (q), a pair whose forward mate fails --minMappingQuality (m), a
dovetailed pair (d) and an unpaired read (s). The oracle lists, for each
count unit and read model, the intervals each counted unit covers and its
weight; every metric is then derived from that list.
"""

import importlib.metadata
import json
import subprocess
import sys

import numpy as np
import pytest

from tests.helpers.coverage import record, write_bam, read_values
from deeptoolsr import _coverage, bamCoverage, config

pysam = pytest.importorskip("pysam")

LENGTH, BIN, EXTEND, MAPQ = 1000, 10, 60, 10
M50 = ((0, 50),)


def reads():
    return [
        record("p", 100, 130, 99, 80, cigar=M50),
        record("p", 130, 100, 147, -80, cigar=M50),
        record("q", 300, 450, 99, 200, cigar=M50),
        record("q", 450, 300, 147, -200, cigar=M50),
        record("m", 600, 650, 99, 100, mapq=0, cigar=M50),
        record("m", 650, 600, 147, -100, cigar=M50),
        # Dovetailed: the reverse mate starts left of the forward mate.
        record("d", 800, 820, 83, -70, cigar=M50),
        record("d", 820, 800, 163, 70, cigar=M50),
        record("s", 900, -1, 0, 0, mtid=-1, cigar=((0, 40),)),
    ]


def units(unit, model):
    """[(intervals, weight)] of every counted read or fragment."""
    if model == "plain":
        if unit == "read":
            spans = [[(100, 150)], [(130, 180)], [(300, 350)], [(450, 500)],
                     [(650, 700)], [(800, 850)], [(820, 870)], [(900, 940)]]
        else:
            spans = [[(100, 180)], [(300, 350), (450, 500)], [(650, 700)],
                     [(800, 870)], [(900, 940)]]
        return [(span, 1) for span in spans]
    if model == "extend":
        pair = 2 if unit == "read" else 1
        rows = [([(100, 180)], pair), ([(300, 500)], pair), ([(600, 700)], 1),
                ([(900, 900 + EXTEND)], 1)]
        if unit == "read":  # dovetailed mates are extended as single reads
            rows += [([(850 - EXTEND, 850)], 1), ([(820, 820 + EXTEND)], 1)]
        else:
            rows += [([(800, 870)], 1)]
        return rows
    end = {"5prime": 0, "3prime": 1, "center": 2}[model]
    if unit == "read":  # each read's own 5', 3' end or centre
        sites = {0: [100, 179, 300, 499, 699, 849, 820, 900],
                 1: [149, 130, 349, 450, 650, 800, 869, 939],
                 2: [125, 155, 325, 475, 675, 825, 845, 920]}[end]
    else:
        sites = {0: [100, 300, 600, 800, 900],
                 1: [179, 499, 699, 869, 939],
                 2: [140, 400, 650, 835, 920]}[end]
    return [([(site, site + 1)], 1) for site in sites]


def expected(unit, model, metric, denominator="library"):
    depth = np.zeros(LENGTH)
    counts = np.zeros(LENGTH // BIN)
    for span, weight in units(unit, model):
        touched = set()
        for start, end in span:
            depth[start:end] += weight
            touched |= set(range(start // BIN, (end - 1) // BIN + 1))
        counts[sorted(touched)] += weight
    if metric == "coverage-sum":
        values = depth.reshape(-1, BIN).sum(axis=1)
    elif metric == "coverage-mean":
        values = depth.reshape(-1, BIN).mean(axis=1)
    else:
        library = {("read", "library"): 9, ("read", "filtered"): 8,
                   ("fragment", "library"): 5, ("fragment", "filtered"): 5}
        scale = {"read-count": 1, "cpm": 1e6 / library[unit, denominator],
                 "rpkm": 1e9 / library[unit, denominator] / BIN,
                 "bpm": 1e6 / counts.sum()}[metric]
        values = counts * scale
    return np.repeat(values, BIN)


AGG = {"coverage-mean": ("mean", "none"), "coverage-sum": ("sum", "none"),
       "read-count": ("count", "none"), "cpm": ("count", "cpm"),
       "rpkm": ("count", "rpkm"), "bpm": ("count", "bpm")}
MODELS = ["plain", "extend", "5prime", "3prime", "center"]


def run(source, output, unit, model, metric, **extra):
    aggregation, normalization = AGG[metric]
    _coverage.bam_coverage_bigwig(
        str(source), str(output), bin_size=BIN, aggregation=aggregation,
        normalization=normalization, min_mapping_quality=MAPQ,
        extend_reads=EXTEND if model == "extend" else 0,
        collapse=model if model not in ("plain", "extend") else "none",
        count_unit=unit, max_zooms=0, **extra)


@pytest.fixture(scope="module")
def bam(tmp_path_factory):
    path = tmp_path_factory.mktemp("count_unit") / "pairs.bam"
    write_bam(path, reads(), (LENGTH,))
    return path


@pytest.mark.parametrize("metric", list(AGG))
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("unit", ["read", "fragment"])
def test_every_metric_and_read_model_matches_the_oracle(tmp_path, bam, unit, model, metric):
    output = tmp_path / "out.bw"
    run(bam, output, unit, model, metric)
    np.testing.assert_allclose(read_values(output), expected(unit, model, metric), rtol=1e-6)


@pytest.mark.parametrize("metric", ["cpm", "rpkm"])
@pytest.mark.parametrize("unit", ["read", "fragment"])
def test_filtered_denominator_counts_retained_units(tmp_path, bam, unit, metric):
    output = tmp_path / "out.bw"
    run(bam, output, unit, "plain", metric, normalization_denominator="filtered")
    np.testing.assert_allclose(read_values(output),
                               expected(unit, "plain", metric, "filtered"), rtol=1e-6)


def test_fragment_library_denominator_honours_ignore_for_normalization(tmp_path):
    path, output = tmp_path / "two.bam", tmp_path / "out.bw"
    write_bam(path, reads() + [record("x", 10, 40, 99, 80, tid=1, mtid=1, cigar=M50),
                               record("x", 40, 10, 147, -80, tid=1, mtid=1, cigar=M50)],
              (LENGTH, LENGTH))
    for ignore, fragments in ((None, 6), (["chr2"], 5)):
        run(path, output, "fragment", "plain", "cpm", ignore_for_normalization=ignore)
        assert read_values(output)[100] == pytest.approx(1e6 / fragments)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("unit", ["read", "fragment"])
def test_threads_windows_and_cram_are_identical(tmp_path, bam, unit, model):
    cram = tmp_path / "pairs.cram"
    with pysam.AlignmentFile(str(bam)) as src, pysam.AlignmentFile(
            str(cram), "wc", template=src, format_options=[b"no_ref=1"]) as dst:
        for read in src:
            dst.write(read)
    pysam.index(str(cram))
    reference = expected(unit, model, "read-count")
    for source, threads, window in ((bam, 1, 0), (bam, 4, 3), (cram, 1, 0), (cram, 4, 3)):
        output = tmp_path / f"{source.suffix}-{threads}.bw"
        run(source, output, unit, model, "read-count", threads=threads, window_size=window)
        np.testing.assert_array_equal(read_values(output), reference)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("unit", ["read", "fragment"])
def test_split_strands_partition_the_unsplit_counts(tmp_path, bam, unit, model):
    forward, reverse, whole = (tmp_path / f"{name}.bw" for name in ("f", "r", "all"))
    run(bam, forward, unit, model, "read-count", out_path_reverse=str(reverse),
        strandedness="reverse", filter_rna_strand="split")
    run(bam, whole, unit, model, "read-count", strandedness="reverse")
    np.testing.assert_array_equal(read_values(forward) + read_values(reverse),
                                  read_values(whole))


@pytest.mark.parametrize("model", MODELS)
def test_single_end_data_is_identical_in_both_units(tmp_path, model):
    path = tmp_path / "se.bam"
    write_bam(path, [record("a", 100, -1, 0, 0, mtid=-1, cigar=M50),
                     record("b", 130, -1, 16, 0, mtid=-1, cigar=M50),
                     record("c", 400, -1, 16, 0, mtid=-1, mapq=0, cigar=M50)], (LENGTH,))
    for metric in ("coverage-mean", "read-count", "cpm", "bpm"):
        values = []
        for unit in ("read", "fragment"):
            output = tmp_path / f"{unit}.bw"
            run(path, output, unit, model, metric, strandedness="forward")
            values.append(read_values(output))
        np.testing.assert_array_equal(*values)


def test_dovetailed_pair_with_filtered_mate_is_a_single_read(tmp_path):
    path, output = tmp_path / "dove.bam", tmp_path / "out.bw"
    write_bam(path, [record("d", 800, 820, 83, -70, mapq=0, cigar=M50),
                     record("d", 820, 800, 163, 70, cigar=M50)], (LENGTH,))
    run(path, output, "fragment", "extend", "read-count")
    values = read_values(output)
    assert values[815] == 0 and values[820] == 1 and values[879] == 1


def test_circular_origin_pair_stays_two_reads_in_fragment_unit(tmp_path):
    path, output = tmp_path / "origin.bam", tmp_path / "out.bw"
    write_bam(path, [record("o", 0, 950, 83, -50, cigar=M50),
                     record("o", 950, 0, 163, 50, cigar=M50)], (LENGTH,))
    run(path, output, "fragment", "plain", "read-count")
    values = read_values(output)
    assert values[0] == 1 and values[960] == 1 and values[500] == 0


def test_count_unit_is_validated(tmp_path, bam):
    with pytest.raises(RuntimeError, match="count_unit"):
        run(bam, tmp_path / "out.bw", "pair", "plain", "read-count")


def bedgraph(tmp_path, bam, *options):
    output = tmp_path / "out.bedgraph"
    bamCoverage.main(["-b", str(bam), "-o", str(output), "-of", "bedgraph", "-bs", "10",
                      "--minMappingQuality", str(MAPQ), *options])
    return output.read_text()


def test_options_file_default_and_cli_override(tmp_path, bam, monkeypatch, capsys):
    read = bedgraph(tmp_path, bam)
    fragment = bedgraph(tmp_path, bam, "--countUnit", "fragment")
    assert read != fragment
    from deeptoolsr import deeptoolsr_list_tools
    deeptoolsr_list_tools.process_args(["options", "--countUnit", "fragment"])
    assert "bamCoverageR count unit: fragment" in capsys.readouterr().out
    assert json.loads(config.options_path().read_text())["count_unit"] == "fragment"
    assert bedgraph(tmp_path, bam) == fragment
    assert bedgraph(tmp_path, bam, "--countUnit", "read") == read


def test_invalid_stored_count_unit_warns_and_counts_reads(tmp_path, bam, capsys):
    read = bedgraph(tmp_path, bam)
    config.options_path().parent.mkdir(parents=True, exist_ok=True)
    config.options_path().write_text(json.dumps({"count_unit": "pairs"}))
    capsys.readouterr()
    assert bedgraph(tmp_path, bam) == read
    assert "ignoring invalid count_unit" in capsys.readouterr().err


def test_upstream_read_unit_parity(tmp_path):
    """Read unit matches deepTools 3.5.6 for plain and extended pairs."""
    if sys.platform == "win32":
        pytest.skip("Original deepTools is unsupported on Windows")
    pytest.importorskip("deeptools")
    if importlib.metadata.version("deepTools") != "3.5.6":
        pytest.skip("requires deepTools 3.5.6")
    # A separate interpreter keeps deepTools' process-global state out of ours.
    upstream = [sys.executable, "-c",
                "import sys; from deeptools.bamCoverage import main; main(sys.argv[1:])"]
    path = tmp_path / "pq.bam"
    write_bam(path, reads()[:4] + reads()[-1:], (LENGTH,))

    def rows(path):
        return [(c, int(a), int(b), float(v)) for c, a, b, v in
                (line.split() for line in path.read_text().splitlines()) if float(v) != 0]

    for options in ([], ["-e", str(EXTEND)]):
        for theirs, ours in (("None", "read-count"), ("CPM", "CPM"), ("RPKM", "RPKM")):
            args = ["-b", str(path), "-of", "bedgraph", "-bs", "10", *options]
            subprocess.run([*upstream, *args, "--normalizeUsing", theirs,
                            "-o", str(tmp_path / "theirs.bg")], check=True, capture_output=True)
            bamCoverage.main([*args, "--countUnit", "read", "--normalizeUsing", ours,
                              "-o", str(tmp_path / "ours.bg")])
            got, want = rows(tmp_path / "ours.bg"), rows(tmp_path / "theirs.bg")
            assert [r[:3] for r in got] == [r[:3] for r in want]
            np.testing.assert_allclose([r[3] for r in got], [r[3] for r in want], rtol=1e-5)


# Read-unit counterparts of the fragment-unit scenarios in
# test_coverage_pair_context.py, test_read_extension.py and test_bamCoverage.py.

from tests.test_coverage_pair_context import (  # noqa: E402
    _MATE_OPTIONS, _budget_run, _deep_pileup_bam, _mate_matrix_bam)


def _pair_values(tmp_path, reads, **options):
    bam, out = tmp_path / "pair.bam", tmp_path / "pair.bw"
    write_bam(bam, reads, (200,))
    _coverage.bam_coverage_bigwig(str(bam), str(out), bin_size=1, max_zooms=0,
                                  count_unit="read", **options)
    return read_values(out)


def _spans(*spans):
    values = np.zeros(200)
    for begin, end, value in spans:
        values[begin:end] += value
    return values


@pytest.mark.parametrize("aggregation", ["sum", "mean", "count"])
@pytest.mark.parametrize("filtered", [None, "forward", "reverse"])
def test_read_unit_extended_pair_depth_is_two(tmp_path, aggregation, filtered):
    mapqs = {None: (60, 60), "forward": (0, 60), "reverse": (60, 0)}[filtered]
    reads = [record("p", 10, 60, 99, 60, mapq=mapqs[0]),
             record("p", 60, 10, 147, -60, mapq=mapqs[1])]
    values = _pair_values(tmp_path, reads, aggregation=aggregation, extend_reads=20,
                          min_mapping_quality=10)
    np.testing.assert_array_equal(values, _spans((10, 70, 2 if filtered is None else 1)))


@pytest.mark.parametrize("aggregation", ["sum", "count"])
@pytest.mark.parametrize("filtered", [None, "forward", "reverse"])
def test_read_unit_skip_introns_depth(tmp_path, aggregation, filtered):
    # Fragment [10, 80): forward intron [15, 35), reverse intron [65, 75).
    M, N = 0, 3
    mapqs = {None: (60, 60), "forward": (0, 60), "reverse": (60, 0)}[filtered]
    reads = [record("p", 10, 60, 99, 70, mapq=mapqs[0], cigar=((M, 5), (N, 20), (M, 5))),
             record("p", 60, 10, 147, -70, mapq=mapqs[1], cigar=((M, 5), (N, 10), (M, 5)))]
    values = _pair_values(tmp_path, reads, aggregation=aggregation, extend_reads=20,
                          skip_introns=True, min_mapping_quality=10)
    forward = _spans((10, 15, 1), (35, 80, 1))
    reverse = _spans((10, 65, 1), (75, 80, 1))
    # Both mates retained: each covers the fragment minus every intron in it.
    both = _spans((10, 15, 2), (35, 65, 2), (75, 80, 2))
    expected = {None: both, "forward": reverse, "reverse": forward}[filtered]
    np.testing.assert_array_equal(values, expected)


# Read 1 (flag 99) forward at [10, 20); read 2 (flag 147) reverse at [50, 60).
# Expected bases per read: (read 1, read 2).
_COLLAPSE_PER_READ = {
    ("5prime", "none"): (10, 59), ("3prime", "none"): (19, 50), ("center", "none"): (15, 55),
    # Forward library: the RNA is on +; read 2 opposes it and uses its own 3' end.
    ("5prime", "forward"): (10, 50), ("3prime", "forward"): (19, 59),
    ("center", "forward"): (15, 55),
    # Reverse library: the RNA is on -; read 1 opposes it and uses its own 3' end.
    ("5prime", "reverse"): (19, 59), ("3prime", "reverse"): (10, 50),
    ("center", "reverse"): (15, 55),
}


@pytest.mark.parametrize("collapse,strandedness", sorted(_COLLAPSE_PER_READ))
def test_read_unit_collapse_places_each_read_strandedness_aware(tmp_path, collapse,
                                                                strandedness):
    reads = [record("p", 10, 50, 99, 50), record("p", 50, 10, 147, -50)]
    values = _pair_values(tmp_path, reads, aggregation="count", collapse=collapse,
                          strandedness=strandedness)
    first, second = _COLLAPSE_PER_READ[collapse, strandedness]
    np.testing.assert_array_equal(values, _spans((first, first + 1, 1), (second, second + 1, 1)))


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("option", sorted(_MATE_OPTIONS))
def test_read_unit_tiny_mate_budgets_are_byte_identical(tmp_path, option, threads):
    bam = tmp_path / "matrix.bam"
    _mate_matrix_bam(bam)
    options = dict(_MATE_OPTIONS[option], threads=threads, count_unit="read")
    expected = _budget_run(bam, tmp_path / "default", options)
    assert expected[0]
    for entries in (1, 4):
        budget = {"entries": entries}
        assert _budget_run(bam, tmp_path / f"tiny{entries}", options, budget) == expected
        assert budget["peak_batch_entries"] <= entries
        assert budget["peak_cache_entries"] <= entries


def test_read_unit_deep_pileup_collapse_is_linear_and_memory_bounded(tmp_path):
    import time
    bam, out = tmp_path / "deep.bam", tmp_path / "deep.bw"
    _deep_pileup_bam(bam, 80000)
    budget = {}
    started = time.monotonic()
    _coverage.bam_coverage_bigwig(str(bam), str(out), bin_size=1, collapse="5prime",
                                  threads=4, max_zooms=0, mate_budget=budget,
                                  count_unit="read")
    assert time.monotonic() - started < 30
    expected = np.zeros(5000)
    expected[1000] = expected[1109] = 80000  # each read's own 5' end
    np.testing.assert_array_equal(read_values(out), expected)
    assert budget["peak_cache_entries"] <= 32768
    assert budget["peak_cache_bytes"] <= 8 << 20
    assert budget["peak_batch_bytes"] <= 4 << 20
