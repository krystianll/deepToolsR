"""Independent pair/filter/window oracles; no original deepTools dependency."""
from tests.helpers.parity import original
import math
from tests.helpers.bigwig import pyBigWig

import numpy as np
import pytest

from tests.helpers.coverage import record, write_bam, read_values
from deeptoolsr import _coverage, bamCoverage

pysam = pytest.importorskip("pysam")


@pytest.mark.parametrize("start", [150, 250])
@pytest.mark.parametrize("overlap_filter", [False, True])
@pytest.mark.parametrize("redundant_regions", [False, True])
@pytest.mark.parametrize("window,threads,halo", [(0, 1, 0), (7, 2, 10000)])
def test_multiple_whitelist_regions_never_multiply_read_counts(
    tmp_path, start, overlap_filter, redundant_regions, window, threads, halo
):
    bam, output = tmp_path / "read.bam", tmp_path / "out.bw"
    write_bam(
        bam, [record("read", start, -1, 0, 0, mtid=-1,
                     cigar=((0, 350 - start),))], (500,),
    )
    regions = [(100, 200), (300, 400)]
    if redundant_regions:
        regions += [(100, 200), (150, 200), (300, 350)]
    _coverage.bam_coverage_bigwig(
        str(bam), str(output), bin_size=1, aggregation="count",
        whitelist={"chr1": regions}, filter_by_overlap=overlap_filter,
        filter_by_overlap_halo=halo, window_size=window, threads=threads,
        max_zooms=0,
    )
    expected = np.zeros(500)
    expected[start:350] = 1
    if not overlap_filter:
        # Ordinary whitelist mode masks output; overlap mode retains the read.
        expected[:100] = expected[200:300] = expected[400:] = 0
    np.testing.assert_array_equal(read_values(output), expected)


def test_two_whitelisted_exons_in_one_bin_count_as_one_read(tmp_path):
    bam, output = tmp_path / "spliced.bam", tmp_path / "out.bw"
    write_bam(
        bam, [record("read", 150, -1, 0, 0, mtid=-1,
                     cigar=((0, 50), (3, 100), (0, 50)))], (500,),
    )
    _coverage.bam_coverage_bigwig(
        str(bam), str(output), bin_size=500, aggregation="count",
        whitelist={"chr1": [(100, 200), (300, 400)]},
        filter_by_overlap=True, max_zooms=0,
    )
    np.testing.assert_array_equal(read_values(output), np.ones(500))


SHAPES = {
    "extend": (10, 60),
    "5prime": (10, 15),
    "3prime": (55, 60),
    "center": (33, 38),
}
FILTERS = [
    ("both", [], (60, 60), (99, 147), 2),
    ("include-left", ["--samFlagInclude", "64"], (60, 60), (99, 147), 1),
    ("include-right", ["--samFlagInclude", "128"], (60, 60), (99, 147), 1),
    ("exclude-left", ["--samFlagExclude", "64"], (60, 60), (99, 147), 1),
    ("exclude-right", ["--samFlagExclude", "128"], (60, 60), (99, 147), 1),
    ("mapq-left", ["--minMappingQuality", "30"], (60, 0), (99, 147), 1),
    ("mapq-right", ["--minMappingQuality", "30"], (0, 60), (99, 147), 1),
    ("duplicate-left", ["--ignoreDuplicates"], (60, 60), (99 | 1024, 147), 1),
    ("duplicate-right", ["--ignoreDuplicates"], (60, 60), (99, 147 | 1024), 1),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("metric", ["coverage-mean", "coverage-sum", "read-count"])
@pytest.mark.parametrize(
    "name,extra,mapqs,flags,kept", FILTERS, ids=[row[0] for row in FILTERS]
)
def test_pair_shapes_respect_each_filter(
    tmp_path, shape, metric, name, extra, mapqs, flags, kept
):
    bam, output = tmp_path / "pair.bam", tmp_path / "out.bw"
    write_bam(
        bam,
        [
            record("pair", 10, 50, flags[0], 50, mapq=mapqs[0]),
            record("pair", 50, 10, flags[1], -50, mapq=mapqs[1]),
        ],
    )
    shape_args = (
        ["--extendReads", "50"]
        if shape == "extend"
        else ["--collapse", shape, "--collapsedLength", "5"]
    )
    bamCoverage.main(
        [
            "-b",
            str(bam),
            "-o",
            str(output),
            "--binSize",
            "1",
            "--normalizeUsing",
            metric,
            "--zoomLevels",
            "0",
        ]
        + shape_args
        + extra
    )
    expected = np.zeros(200)
    begin, end = SHAPES[shape]
    expected[begin:end] = kept if metric == "read-count" else 1
    np.testing.assert_array_equal(read_values(output), expected)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("aggregation", ["mean", "sum", "count"])
def test_missing_mate_contributes_only_the_observed_alignment(
    tmp_path, shape, side, aggregation
):
    bam, output = tmp_path / "orphan.bam", tmp_path / "out.bw"
    read = (
        record("pair", 10, 50, 99, 50)
        if side == "left"
        else record("pair", 50, 10, 147, -50)
    )
    write_bam(bam, [read])
    kwargs = (
        {"extend_reads": 50}
        if shape == "extend"
        else {"collapse": shape, "collapsed_length": 5}
    )
    _coverage.bam_coverage_bigwig(
        str(bam),
        str(output),
        bin_size=1,
        aggregation=aggregation,
        window_size=7,
        threads=2,
        max_zooms=0,
        **kwargs,
    )
    expected = np.zeros(200)
    begin, end = SHAPES[shape]
    expected[begin:end] = 1
    np.testing.assert_array_equal(read_values(output), expected)


@pytest.mark.parametrize("shape", ["extend", "5prime", "3prime", "center"])
@pytest.mark.parametrize("aggregation", ["sum", "count"])
@pytest.mark.parametrize("threads,window", [(1, 0), (2, 11)])
def test_long_pair_entire_placement_is_window_independent(
    tmp_path, shape, aggregation, threads, window
):
    bam, output = tmp_path / "long.bam", tmp_path / "out.bw"
    start, end, length = 130, 2080090, 2100000
    write_bam(
        bam,
        [
            record("pair", start, end - 10, 99, end - start),
            record("pair", end - 10, start, 147, start - end),
        ],
        (length,),
    )
    kwargs = (
        {"extend_reads": 100}
        if shape == "extend"
        else {"collapse": shape, "collapsed_length": 20}
    )
    _coverage.bam_coverage_bigwig(
        str(bam),
        str(output),
        bin_size=100,
        aggregation=aggregation,
        threads=threads,
        window_size=window,
        max_zooms=0,
        **kwargs,
    )
    begin, stop = {
        "extend": (start, end),
        "5prime": (start, start + 20),
        "3prime": (end - 20, end),
        "center": ((start + end) // 2 - 10, (start + end) // 2 + 10),
    }[shape]
    expected_bins = np.zeros(length // 100)
    for i in range(begin // 100, (stop - 1) // 100 + 1):
        expected_bins[i] = (
            2
            if aggregation == "count"
            else min(stop, (i + 1) * 100) - max(begin, i * 100)
        )
    np.testing.assert_array_equal(read_values(output), np.repeat(expected_bins, 100))


@pytest.mark.parametrize("threads", [1, 2])
def test_reverse_single_end_extension_reaches_previous_windows(tmp_path, threads):
    bam, output = tmp_path / "reverse.bam", tmp_path / "out.bw"
    write_bam(bam, [record("single", 1200000, -1, 16, 0)], (1500000,))
    _coverage.bam_coverage_bigwig(
        str(bam),
        str(output),
        bin_size=10,
        aggregation="count",
        extend_reads=400000,
        threads=threads,
        max_zooms=0,
    )
    expected = np.zeros(1500000)
    expected[800010:1200010] = 1
    np.testing.assert_array_equal(read_values(output), expected)


@pytest.mark.parametrize(
    "cigar,length",
    [
        (((0, 10),), 10),
        (((0, 5), (3, 100), (0, 5)), 10),
        (((0, 5), (2, 5), (0, 5)), 15),
        (((0, 5), (1, 5), (0, 5)), 10),
        (((4, 5), (7, 5), (8, 5)), 10),
    ],
)
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("which", ["min", "max"])
def test_zero_tlen_cigar_length_filter_boundaries(
    tmp_path, cigar, length, offset, which
):
    bam, output = tmp_path / "single.bam", tmp_path / "out.bw"
    read = record("single", 10, -1, 0, 0, cigar=cigar)
    write_bam(bam, [read])
    threshold = length + offset
    _coverage.bam_coverage_bigwig(
        str(bam),
        str(output),
        bin_size=1,
        aggregation="count",
        max_zooms=0,
        **{which + "_fragment_length": threshold},
    )
    expected = np.zeros(200)
    retained = length >= threshold if which == "min" else length <= threshold
    if retained:
        for begin, end in read.get_blocks():
            expected[begin:end] = 1
    np.testing.assert_array_equal(read_values(output), expected)


@pytest.mark.parametrize("region_kind", ["whitelist", "blacklist"])
def test_overlap_finds_mate_on_another_chromosome(tmp_path, region_kind):
    bam, output = tmp_path / "interchrom.bam", tmp_path / "out.bw"
    write_bam(
        bam,
        [
            record("pair", 10, 90, 97, 0, tid=0, mtid=1),
            record("pair", 90, 10, 145, 0, tid=1, mtid=0),
        ],
        (200, 200),
    )
    _coverage.bam_coverage_bigwig(
        str(bam),
        str(output),
        bin_size=10,
        aggregation="count",
        filter_by_overlap=True,
        filter_by_overlap_halo=0,
        max_zooms=0,
        threads=2,
        **{region_kind: {"chr1": [(10, 20)]}},
    )
    for chrom, start in [("chr1", 10), ("chr2", 90)]:
        expected = np.zeros(200)
        if region_kind == "whitelist":
            expected[start: start + 10] = 1
        np.testing.assert_array_equal(read_values(output, chrom), expected)


def test_overlap_does_not_mix_read_groups_or_failed_mate_filters(tmp_path):
    bam, output = tmp_path / "groups.bam", tmp_path / "out.bw"
    write_bam(
        bam,
        [
            record("same", 10, 90, 99, 90, group="A"),
            record("same", 10, 90, 99, 90, group="B", mapq=0),
            record("same", 90, 10, 147, -90, group="A"),
            record("same", 90, 10, 147, -90, group="B"),
        ],
    )
    _coverage.bam_coverage_bigwig(
        str(bam),
        str(output),
        bin_size=10,
        aggregation="count",
        min_mapping_quality=30,
        filter_by_overlap=True,
        whitelist={"chr1": [(10, 20)]},
        window_size=1,
        threads=2,
        normalization="cpm",
        normalization_denominator="filtered",
        max_zooms=0,
    )
    expected = np.zeros(200)
    expected[10:20] = expected[90:100] = 500000
    np.testing.assert_array_equal(read_values(output), expected)


@pytest.mark.parametrize("extra_flag", [256, 2048])
@pytest.mark.parametrize("model", ["aligned", "extended", "depth"])
def test_filtered_alignment_class_cannot_overwrite_primary_mate(
    tmp_path, extra_flag, model
):
    bam, output = tmp_path / "classes.bam", tmp_path / "out.bw"
    write_bam(
        bam,
        [
            record("pair", 10, 90, 99, 90),
            record("pair", 10, 90, 99 | extra_flag, 90),
            record("pair", 90, 10, 147, -90),
        ],
    )
    _coverage.bam_coverage_bigwig(
        str(bam),
        str(output),
        bin_size=1,
        aggregation="sum" if model == "depth" else "count",
        extend_reads=0 if model == "aligned" else 90,
        filter_by_overlap=True,
        whitelist={"chr1": [(10, 20)]},
        max_zooms=0,
    )
    expected = np.zeros(200)
    if model == "aligned":
        expected[10:20] = expected[90:100] = 1
    else:
        expected[10:100] = 1 if model == "depth" else 2
    np.testing.assert_array_equal(read_values(output), expected)


@pytest.mark.parametrize("threads", [1, 2])
def test_overlap_cache_eviction_preserves_exact_mate_decisions(tmp_path, threads):
    bam, output = tmp_path / "many.bam", tmp_path / "out.bw"
    reads = []
    for i in range(2100):
        left, right = 10 * i, 2000000 + 10 * i
        reads += [
            record(str(i), left, right, 99, right - left + 10),
            record(str(i), right, left, 147, left - right - 10),
        ]
    write_bam(bam, reads, (2100000,))
    _coverage.bam_coverage_bigwig(
        str(bam),
        str(output),
        bin_size=10,
        aggregation="count",
        filter_by_overlap=True,
        whitelist={"chr1": [(0, 21000)]},
        threads=threads,
        filter_by_overlap_halo=0,
        max_zooms=0,
    )
    expected = np.zeros(2100000)
    expected[:21000] = expected[2000000:2021000] = 1
    np.testing.assert_array_equal(read_values(output), expected)


def test_duplicate_qnames_use_reciprocal_mate_coordinates(tmp_path):
    bam, output = tmp_path / "duplicate-qname.bam", tmp_path / "out.bw"
    write_bam(
        bam,
        [
            record("same", 10, 50, 99, 50),
            record("same", 50, 10, 147, -50),
            record("same", 10, 90, 99, 90),
            record("same", 90, 10, 147, -90),
        ],
    )
    _coverage.bam_coverage_bigwig(
        str(bam), str(output), bin_size=1, aggregation="sum",
        extend_reads=100, max_zooms=0,
    )
    expected = np.zeros(200)
    expected[10:60] += 1
    expected[10:100] += 1
    np.testing.assert_array_equal(read_values(output), expected)


def test_duplicate_qnames_do_not_borrow_overlap_decisions(tmp_path):
    bam, output = tmp_path / "duplicate-overlap.bam", tmp_path / "out.bw"
    write_bam(
        bam,
        [
            record("same", 10, 50, 99, 50),
            record("same", 10, 90, 99, 90,
                   cigar=((0, 3), (3, 7), (0, 7))),
            record("same", 50, 10, 147, -50),
            record("same", 90, 10, 147, -90),
        ],
    )
    _coverage.bam_coverage_bigwig(
        str(bam), str(output), bin_size=1, aggregation="count",
        filter_by_overlap=True, whitelist={"chr1": [(15, 18)]},
        max_zooms=0,
    )
    expected = np.zeros(200)
    expected[10:20] = expected[50:60] = 1
    np.testing.assert_array_equal(read_values(output), expected)


def test_short_extension_never_shrinks_singleton_cigar_blocks(tmp_path):
    bam, output = tmp_path / "short-extension.bam", tmp_path / "out.bw"
    reads = [
        record("forward", 10, -1, 0, 0),
        record("reverse", 30, -1, 16, 0),
        record("spliced", 50, -1, 0, 0,
               cigar=((0, 5), (3, 10), (0, 5))),
        record("discordant", 80, 120, 65, 0),
    ]
    write_bam(bam, reads)
    _coverage.bam_coverage_bigwig(
        str(bam), str(output), bin_size=1, aggregation="count",
        extend_reads=3, filter_mode="deeptools", max_zooms=0,
    )
    expected = np.zeros(200)
    for read in reads:
        for start, end in read.get_blocks():
            expected[start:end] += 1
    np.testing.assert_array_equal(read_values(output), expected)


def test_false_proper_pair_flag_uses_singleton_model(tmp_path):
    bam, output = tmp_path / "false-proper.bam", tmp_path / "out.bw"
    write_bam(
        bam,
        [record("bad", 10, 50, 67, 50), record("bad", 50, 10, 131, -50)],
    )
    _coverage.bam_coverage_bigwig(
        str(bam), str(output), bin_size=1, aggregation="count",
        extend_reads=20, filter_mode="deeptools", max_zooms=0,
    )
    expected = np.zeros(200)
    expected[10:30] += 1
    expected[50:70] += 1
    np.testing.assert_array_equal(read_values(output), expected)


def test_auto_fragment_length_samples_beyond_coordinate_prefix(tmp_path):
    bam = tmp_path / "lengths.bam"
    reads = []
    for tid, (count, span) in enumerate(((60, 50), (200, 200))):
        for i in range(count):
            left = 10 + i * (span + 10)
            right = left + span - 10
            reads.extend([
                record(f"p{tid}-{i}", left, right, 99, span, tid=tid, mtid=tid),
                record(f"p{tid}-{i}", right, left, 147, -span, tid=tid, mtid=tid),
            ])
    write_bam(bam, reads, (10000, 50000))
    # A first-100-record implementation sees only 50 bp chr1 fragments.  The
    # bounded whole-file sampler correctly reflects the much larger chr2 group.
    assert _coverage.estimate_fragment_length(str(bam), sample=100) == 200


def test_strandedness_deduplicates_probe_queries_and_pair_mates(tmp_path):
    single = tmp_path / "single.bam"
    write_bam(single, [record("single", 20, -1, 0, 0)])
    duplicated = _coverage.infer_strandedness(
        str(single), [("chr1", 15, 35, 1), ("chr1", 10, 40, 1)])
    conflicting = _coverage.infer_strandedness(
        str(single), [("chr1", 15, 35, 1), ("chr1", 10, 40, -1)])
    assert duplicated["forward"] == 1 and duplicated["total"] == 1
    assert conflicting["total"] == 0

    paired = tmp_path / "paired.bam"
    write_bam(
        paired,
        [record("pair", 20, 40, 99, 30), record("pair", 40, 20, 147, -30)],
    )
    result = _coverage.infer_strandedness(
        str(paired), [("chr1", 0, 100, 1)], require_proper_pair=True)
    assert result["forward"] == 1 and result["total"] == 1


def test_strandedness_strict_pair_mode_ignores_discordant_evidence(tmp_path):
    bam = tmp_path / "mixed.bam"
    write_bam(
        bam,
        [
            record("good", 20, 40, 99, 30),
            record("good", 40, 20, 147, -30),
            record("bad", 80, 100, 81, 30),
            record("bad", 100, 80, 129, -30),
        ],
    )
    result = _coverage.infer_strandedness(
        str(bam), [("chr1", 0, 150, 1)], require_proper_pair=True)
    assert result["forward"] == 1 and result["reverse"] == 0


def test_strandedness_sample_bounds_dense_and_filtered_probes(tmp_path):
    bam = tmp_path / 'dense-probes.bam'
    reads = [record(f'read-{i}', 20 + i % 100, -1, 0, 0, mapq=10)
             for i in range(5000)]
    write_bam(bam, reads)
    probes = [('chr1', 0, 180, 1)] * 30
    sampled = _coverage.infer_strandedness(str(bam), probes, sample_size=40)
    assert sampled['total'] == 40
    assert sampled['sampled_probes'] == 1
    assert sampled['alignments_examined'] == 40
    assert sampled['forward_fraction'] == 1
    filtered = _coverage.infer_strandedness(
        str(bam), probes, min_mapping_quality=20, sample_size=40)
    assert filtered['total'] == 0
    assert filtered['alignments_examined'] <= 40 * 50


def test_strandedness_sample_visits_separated_probes_and_is_reproducible(tmp_path):
    bam = tmp_path / 'distributed-probes.bam'
    reads = [record(f'forward-{i}', 20 + i % 30, -1, 0, 0)
             for i in range(1000)]
    reads += [record(f'reverse-{i}', 20 + i % 30, -1, 16, 0, tid=1)
              for i in range(1000)]
    write_bam(bam, reads, (200, 200))
    probes = [('chr1', 0, 100, 1), ('chr2', 0, 100, 1)]
    result = _coverage.infer_strandedness(str(bam), probes, sample_size=100)
    assert result['forward'] == result['reverse'] == 50
    assert result == _coverage.infer_strandedness(str(bam), probes[::-1], sample_size=100)


@pytest.mark.parametrize('reverse_order', [False, True])
def test_overlap_prefetch_verifies_useful_and_zero_identity_collision(tmp_path, reverse_order):
    bam, out = tmp_path / 'ambiguous.bam', tmp_path / 'ambiguous.bw'
    reads = [record('same', 10, 40, 99, 40),
             record('same', 10, 40, 99, 40, cigar=((3, 5), (0, 5))),
             record('same', 40, 10, 147, -40)]
    if reverse_order:
        reads.reverse()
    write_bam(bam, reads)
    _coverage.bam_coverage_bigwig(str(bam), str(out), bin_size=1,
                                  aggregation='count', filter_by_overlap=True,
                                  whitelist={'chr1': [(10, 11)]}, max_zooms=0)
    expected = np.zeros(200)
    expected[10:20] = 1
    np.testing.assert_array_equal(read_values(out), expected)


@pytest.mark.parametrize('halo', [-1, 0, 10000])
@pytest.mark.parametrize('region_kind', ['whitelist', 'blacklist'])
@pytest.mark.parametrize('touch_exon', [False, True])
def test_overlap_spliced_distant_mate_is_exact_for_every_halo(
    tmp_path, halo, region_kind, touch_exon
):
    bam, out = tmp_path / 'spliced-distant.bam', tmp_path / 'out.bw'
    # Only the separate, distant mate touches the feature. Its intron must
    # never count as a touch, even though indexed queries retrieve the read.
    write_bam(bam, [record('pair', 10, 120000, 99, 140010),
                    record('pair', 120000, 10, 147, -140010,
                           cigar=((0, 10), (3, 20000), (0, 10)))], (150000,))
    region = (140010, 140020) if touch_exon else (120100, 120110)
    _coverage.bam_coverage_bigwig(
        str(bam), str(out), bin_size=1, aggregation='count',
        filter_by_overlap=True, filter_by_overlap_halo=halo,
        window_size=2000, threads=2, max_zooms=0,
        **{region_kind: {'chr1': [region]}},
    )
    expected = np.zeros(150000)
    if touch_exon == (region_kind == 'whitelist'):
        expected[10:20] = expected[120000:120010] = expected[140010:140020] = 1
    np.testing.assert_array_equal(read_values(out), expected)


@pytest.mark.parametrize('sample_size', [0, 1000001])
def test_strandedness_rejects_unbounded_sample_size(tmp_path, sample_size):
    with pytest.raises(ValueError, match='sample_size'):
        _coverage.infer_strandedness(str(tmp_path / 'unused.bam'), [], sample_size=sample_size)


def test_overlap_prefetch_capacity_preserves_exact_fallback(tmp_path):
    bam, out = tmp_path / 'dense-mates.bam', tmp_path / 'dense-mates.bw'
    # More useful identities than the bounded cache can hold. Only one omitted
    # identity needs a mate query, keeping this regression small and fast.
    reads = [record(f'filler-{i}', 10, 90, 99, 90) for i in range(33000)]
    reads += [record('target', 10, 50, 99, 50), record('target', 50, 10, 147, -50)]
    write_bam(bam, reads)
    _coverage.bam_coverage_bigwig(str(bam), str(out), bin_size=1,
                                  aggregation='count', filter_by_overlap=True,
                                  whitelist={'chr1': [(10, 11)]}, max_zooms=0)
    expected = np.zeros(200)
    expected[10:20] = 33001
    expected[50:60] = 1
    np.testing.assert_array_equal(read_values(out), expected)


@pytest.mark.parametrize("region_kind", ["whitelist", "blacklist"])
def test_nonprimary_touch_propagates_to_complete_template(tmp_path, region_kind):
    bam, output = tmp_path / "supplementary.bam", tmp_path / "out.bw"
    write_bam(
        bam,
        [
            record("pair", 10, 50, 99, 50),
            record("pair", 50, 10, 147, -50),
            record("pair", 100, 50, (99 & ~2) | 2048, 0),
        ],
    )
    _coverage.bam_coverage_bigwig(
        str(bam), str(output), bin_size=1, aggregation="count",
        filter_mode="deeptools", filter_by_overlap=True, max_zooms=0,
        **{region_kind: {"chr1": [(100, 110)]}},
    )
    expected = np.zeros(200)
    if region_kind == "whitelist":
        expected[10:20] = expected[50:60] = expected[100:110] = 1
    np.testing.assert_array_equal(read_values(output), expected)


@pytest.mark.parametrize("metric", ["CPM", "RPKM"])
@pytest.mark.parametrize("split", [False, True])
def test_empty_filtered_overlap_denominator_preserves_outputs(tmp_path, metric, split):
    bam, output, bed = tmp_path / "pair.bam", tmp_path / "out.bw", tmp_path / "keep.bed"
    write_bam(bam, [record("pair", 10, 50, 99, 50), record("pair", 50, 10, 147, -50)])
    # The fragment spans the BED interval, but neither aligned mate touches it.
    bed.write_text("chr1\t30\t40\n")
    outputs = (
        [tmp_path / "out.plus.bw", tmp_path / "out.minus.bw"] if split else [output]
    )
    for path in outputs:
        path.write_bytes(b"previous successful output")
    args = [
        "-b",
        str(bam),
        "-o",
        str(output),
        "--normalizeUsing",
        metric,
        "--normalizationDenominator",
        "filtered",
        "--whiteListFileName",
        str(bed),
        "--filterByOverlap",
    ]
    if split:
        args += ["--strandedness", "forward", "--filterRNAstrand", "split"]
    with pytest.raises(SystemExit, match="denominator is zero"):
        bamCoverage.main(args)
    for path in outputs:
        assert path.read_bytes() == b"previous successful output"


# From audit followup 2026 09 12.

@pytest.fixture()
def coverage_isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path / 'config'))


@pytest.mark.usefixtures('coverage_isolated_config')
@pytest.mark.parametrize('include_exon', [False, True])
@pytest.mark.parametrize('paired', [False, True])
@pytest.mark.parametrize('gap_op', [2, 3])
def test_skipped_intron_is_not_strandedness_evidence(tmp_path, include_exon, paired, gap_op):
    bam = tmp_path / 'spliced.bam'
    reads = [record('r', 10, 140 if paired else -1, 99 if paired else 0, 140 if paired else 0, mtid=0 if paired else -1, cigar=((0, 10), (gap_op, 100), (0, 10)))]
    if paired:
        reads.append(record('r', 140, 10, 147, -140))
    write_bam(bam, reads)
    probes = [('chr1', 50, 60, -1)]
    if include_exon:
        probes += [('chr1', 10, 20, 1)]
        if paired:
            probes += [('chr1', 140, 150, 1)]
    result = _coverage.infer_strandedness(str(bam), probes, 0, True)
    assert result['total'] == int(include_exon)
    assert result['forward'] == int(include_exon)
    assert result['reverse'] == 0


@pytest.fixture
def coverage_mixed_bam(tmp_path):
    path = tmp_path / 'mixed.bam'
    reads = []
    cigars = [((0, 23),), ((0, 10), (3, 35), (0, 10)), ((4, 3), (0, 8), (1, 2), (0, 9)), ((0, 8), (2, 4), (0, 13))]
    for tid in range(2):
        for i in range(24):
            flag = (16 if i % 2 else 0) | (1024 if i % 7 == 0 else 0)
            reads.append(record(f'r{tid}-{i}', i * 9, -1, flag, 0, mtid=-1, tid=tid, mapq=20 if i % 3 == 0 else 60, cigar=cigars[i % len(cigars)]))
    write_bam(path, reads, (300, 400))
    return (path, reads)


coverage_FILTERS = [[], ['--minMappingQuality', '30'], ['--samFlagInclude', '16'], ['--samFlagExclude', '1024'], ['--minFragmentLength', '22'], ['--maxFragmentLength', '22']]


def eligible(read, args):
    if not args:
        return True
    option, value = (args[0], int(args[1]))
    length = sum((n for op, n in read.cigartuples if op in (0, 2, 7, 8)))
    return {'--minMappingQuality': read.mapping_quality >= value, '--samFlagInclude': read.flag & value == value, '--samFlagExclude': not read.flag & value, '--minFragmentLength': length >= value, '--maxFragmentLength': length <= value}[option]


@pytest.mark.usefixtures('coverage_isolated_config')
@pytest.mark.parametrize('filter_args', coverage_FILTERS)
@pytest.mark.parametrize('metric', ['read-count', 'CPM', 'RPKM'])
@pytest.mark.parametrize('bin_size', [1, 10])
def test_mixed_cigar_filtered_normalization_matches_oracle_and_upstream(tmp_path, coverage_mixed_bam, filter_args, metric, bin_size):
    bam, reads = coverage_mixed_bam
    kept = [read for read in reads if eligible(read, filter_args)]
    expected = [np.zeros(300), np.zeros(400)]
    factor = {'read-count': 1, 'CPM': 1000000.0 / len(kept), 'RPKM': 1000000000.0 / (len(kept) * bin_size)}[metric]
    for read in kept:
        touched = set()
        position = read.reference_start
        for op, length in read.cigartuples:
            if op in (0, 7, 8):
                touched.update(range(position // bin_size, (position + length - 1) // bin_size + 1))
            if op in (0, 2, 3, 7, 8):
                position += length
        for b in touched:
            expected[read.reference_id][b * bin_size:(b + 1) * bin_size] += factor
    plus, upstream = (tmp_path / 'plus.bw', tmp_path / 'original.bw')
    common = ['-b', str(bam), '-bs', str(bin_size), '-p', '1', *filter_args]
    bamCoverage.main(common + ['-o', str(plus), '--normalizeUsing', metric, '--normalizationDenominator', 'filtered', '--zoomLevels', '0'])
    original('bamCoverage', common + ['-o', str(upstream), '--normalizeUsing', 'None' if metric == 'read-count' else metric, '--exactScaling'])
    for tid, oracle in enumerate(expected):
        actual, reference = (read_values(plus, f'chr{tid + 1}'), read_values(upstream, f'chr{tid + 1}'))
        np.testing.assert_allclose(actual, oracle, rtol=1.3e-07, atol=0)
        np.testing.assert_allclose(actual, reference, rtol=5.2e-06, atol=1e-06)


@pytest.mark.usefixtures('coverage_isolated_config')
@pytest.mark.parametrize('metric', ['coverage-mean', 'coverage-sum', 'read-count', 'CPM', 'RPKM', 'BPM'])
@pytest.mark.parametrize('touch_mode', ['mask', 'ignore', 'sense', 'antisense'])
@pytest.mark.parametrize('extend', [False, True])
def test_paired_split_overlap_metrics_against_independent_bin_oracle(tmp_path, metric, touch_mode, extend):
    length, bin_size = (997, 13)
    bam, plus, minus = (tmp_path / name for name in ('pairs.bam', 'plus.bw', 'minus.bw'))
    white = [(40, 300, 1), (340, 650, -1), (700, 990, 0)]
    black = [(135, 145, 0), (515, 555, 1), (835, 845, -1)]
    pairs = []
    for i, start in enumerate(range(10, 900, 30)):
        flags = (99, 147) if i % 2 == 0 else (163, 83)
        pair = [record(f'pair{i}', start, start + 60, flags[0], 80, mapq=20 if i % 3 == 0 else 60, cigar=((0, 10), (3, 30), (0, 10))), record(f'pair{i}', start + 60, start, flags[1] | (1024 if i % 7 == 0 else 0), -80, cigar=((0, 20),))]
        pairs.append((1 if i % 2 == 0 else -1, start, pair))
    write_bam(bam, [read for _, _, pair in pairs for read in pair], (length,))

    def touches(reads, features, strand):
        for read in reads:
            for begin, end in read.get_blocks():
                for first, last, orientation in features:
                    strand_ok = orientation == 0 or touch_mode == 'ignore' or (orientation == strand if touch_mode == 'sense' else orientation != strand)
                    if strand_ok and begin < last and (first < end):
                        return True
        return False
    widths = np.array([min(bin_size, length - b) for b in range(0, length, bin_size)])
    expected = np.zeros((2, len(widths)))
    read_count = metric not in ('coverage-mean', 'coverage-sum')
    denominator = 0
    for strand, start, pair in pairs:
        retained = [r for r in pair if r.mapping_quality >= 30 and (not r.flag & 1024)]
        if touch_mode != 'mask' and (not touches(retained, white, strand) or touches(retained, black, strand)):
            continue
        denominator += len(retained)
        intervals = [[(start, start + 80)]] * (len(retained) if read_count else bool(retained)) if extend else [r.get_blocks() for r in retained]
        stream = 0 if strand == 1 else 1
        for blocks in intervals:
            for b, width in enumerate(widths):
                begin, end = (b * bin_size, b * bin_size + width)
                overlaps = [max(0, min(end, last) - max(begin, first)) for first, last in blocks]
                expected[stream, b] += int(any(overlaps)) if read_count else sum(overlaps)
    if touch_mode == 'mask':
        for b, width in enumerate(widths):
            begin, end = (b * bin_size, b * bin_size + width)
            has_white = any((first < end and begin < last for first, last, _ in white))
            has_black = any((first < end and begin < last for first, last, _ in black))
            if not has_white or has_black:
                expected[:, b] = 0
    if metric == 'coverage-mean':
        expected /= widths
    elif metric == 'CPM':
        expected *= 1000000.0 / denominator
    elif metric == 'RPKM':
        expected *= 1000000000.0 / denominator / widths
    elif metric == 'BPM':
        expected *= 1000000.0 / expected.sum()
    expected[0] *= 3
    expected[1] *= -6
    normalization = metric.lower() if metric in ('CPM', 'RPKM', 'BPM') else 'none'
    aggregation = 'count' if read_count else 'mean' if metric == 'coverage-mean' else 'sum'
    _coverage.bam_coverage_bigwig(str(bam), str(plus), str(minus), bin_size=bin_size, aggregation=aggregation, min_mapping_quality=30, ignore_duplicates=True, strandedness='forward', filter_rna_strand='split', whitelist={'chr1': white}, blacklist={'chr1': black}, filter_by_overlap=touch_mode != 'mask', feature_touch_strand='ignore' if touch_mode == 'mask' else touch_mode, normalization=normalization, normalization_denominator='filtered', scale_factor=3, scale_factor_minus=-2, extend_reads=100 if extend else 0, threads=3, window_size=7, filter_by_overlap_halo=0, max_zooms=0)
    for output, oracle in zip((plus, minus), expected):
        np.testing.assert_allclose(read_values(output), np.repeat(oracle, widths), rtol=1.4e-07, atol=0)


# From equivalence regressions.

@pytest.mark.parametrize('metric', ['count', 'mean', 'sum', 'cpm', 'rpkm'])
@pytest.mark.parametrize('strand', ['forward', 'reverse'])
@pytest.mark.parametrize('touch', ['ignore', 'sense', 'antisense'])
@pytest.mark.parametrize('window', [1, 29])
def test_overlap_union_cigar_strand_filters_and_normalizations(tmp_path, metric, strand, touch, window):
    bam, output = (tmp_path / 'reads.bam', tmp_path / 'plus.bw')
    reads = []
    for i in range(18):
        start = 8 + i * 18
        flags = (99, 147) if i % 2 == 0 else (163, 83)
        for side in [0, 1]:
            reads.append(record(f'r{i}', start + side * 28, start + (1 - side) * 28, flags[side] | (1024 if (i + side) % 11 == 0 else 0), 38 if side == 0 else -38, mapq=0 if (i + side) % 7 == 0 else 60, cigar=((0, 3), (3 if i % 2 else 2, 4), (0, 3))))
    write_bam(bam, reads, lengths=(401,))
    whitelist = [(a, a + 6, s) for a, s in [(12, 1), (47, -1), (84, 0), (119, 1), (163, -1), (208, 0), (271, 1), (307, -1)]]
    blacklist = [(55, 59, 0), (192, 196, 1), (311, 314, -1)]
    eligible = [r for r in reads if r.mapping_quality >= 30 and (not r.flag & 1024)]

    def transcript(r):
        sign = -1 if r.is_reverse else 1
        if r.is_read2:
            sign *= -1
        return sign if strand == 'forward' else -sign

    def touching(r, features):
        return any((a < end and b > start and (touch == 'ignore' or s == 0 or s == transcript(r) * (1 if touch == 'sense' else -1)) for a, b in r.get_blocks() for start, end, s in features))
    by_name = {}
    for r in eligible:
        by_name.setdefault(r.query_name, []).append(r)
    kept = [r for pair in by_name.values() if any((touching(x, whitelist) for x in pair)) and (not any((touching(x, blacklist) for x in pair))) for r in pair]
    assert kept
    normalized = metric in ['cpm', 'rpkm']
    _coverage.bam_coverage_bigwig(str(bam), str(output), bin_size=7, aggregation='count' if normalized else metric, min_mapping_quality=30, ignore_duplicates=True, strandedness=strand, filter_by_overlap=True, feature_touch_strand=touch, whitelist={'chr1': whitelist}, blacklist={'chr1': blacklist}, window_size=window * 7, threads=2, normalization=metric if normalized else 'none', normalization_denominator='filtered')
    bins = np.zeros(math.ceil(401 / 7))
    for r in kept:
        if metric in ['count', 'cpm', 'rpkm']:
            touched = {i for a, b in r.get_blocks() for i in range(a // 7, (b - 1) // 7 + 1)}
            bins[list(touched)] += 1
        else:
            for a, b in r.get_blocks():
                for pos in range(a, b):
                    bins[pos // 7] += 1
    widths = np.minimum(7, 401 - 7 * np.arange(len(bins)))
    if metric == 'mean':
        bins /= widths
    if metric == 'cpm':
        bins *= 1000000.0 / len(kept)
    if metric == 'rpkm':
        bins *= 1000000000.0 / (len(kept) * widths)
    np.testing.assert_allclose(read_values(output), np.repeat(bins, widths), rtol=2e-07, atol=1e-07)


@pytest.mark.parametrize('region_kind', ['whitelist', 'blacklist'])
@pytest.mark.parametrize('normalization', ['none', 'cpm'])
def test_e07_supplementary_read_uses_primary_mates_overlap(tmp_path, region_kind, normalization):
    bam, output = (tmp_path / 'reads.bam', tmp_path / 'out.bw')
    write_bam(bam, [record('pair', 10, 50, 99, 50), record('pair', 50, 10, 147, -50), record('pair', 100, 50, 99 & ~2 | 2048, 0), record('control', 160, -1, 0, 0)])
    _coverage.bam_coverage_bigwig(str(bam), str(output), bin_size=1, aggregation='count', filter_mode='deeptools', filter_by_overlap=True, normalization=normalization, normalization_denominator='filtered', **{region_kind: {'chr1': [(50, 60)]}})
    expected = np.zeros(200)
    if region_kind == 'whitelist':
        expected[10:20] = expected[50:60] = expected[100:110] = 1
        kept = 3
    else:
        expected[160:170] = 1
        kept = 1
    if normalization == 'cpm':
        expected *= 1000000.0 / kept
    np.testing.assert_allclose(read_values(output), expected, rtol=2e-07)


# From parity audit regressions.

def make_bam(path, records, lengths=(200,)):
    """Records: tid, start, flag, mapq, TLEN, mate position, CIGAR tuples."""
    with pysam.AlignmentFile(str(path), 'wb', header={'HD': {'SO': 'coordinate'}, 'SQ': [{'SN': f'chr{i + 1}', 'LN': length} for i, length in enumerate(lengths)]}) as handle:
        for i, (tid, start, flag, mapq, tlen, mate, cigar) in enumerate(records):
            read = pysam.AlignedSegment()
            read.query_name = 'pair' if flag & 1 else f'r{i}'
            read.query_sequence = 'A' * sum((n for op, n in cigar if op in (0, 1, 4, 7, 8)))
            read.reference_id, read.reference_start = (tid, start)
            read.flag, read.mapping_quality, read.cigartuples = (flag, mapq, cigar)
            if flag & 1:
                read.template_length = tlen
                read.next_reference_id, read.next_reference_start = (tid, mate)
            handle.write(read)
    pysam.index(str(path))


def values(path, chrom='chr1'):
    with pyBigWig.open(str(path)) as bw:
        return np.asarray(bw.values(chrom, 0, bw.chroms(chrom)))


@pytest.mark.parametrize('filter_args', [[], ['--samFlagInclude', '64'], ['--samFlagInclude', '128'], ['--minMappingQuality', '30']])
@pytest.mark.parametrize('metric', ['read-count', 'CPM', 'RPKM'])
@pytest.mark.parametrize('implementation', ['plus', 'original'])
def test_p01_extended_pairs_honor_each_mates_filters(tmp_path, filter_args, metric, implementation):
    bam, upstream, plus = [tmp_path / name for name in ('pair.bam', 'original.bw', 'plus.bw')]
    make_bam(bam, [(0, 10, 99, 60, 50, 50, [(0, 10)]), (0, 50, 147, 0, -50, 10, [(0, 10)])])
    args = ['-b', str(bam), '--binSize', '10', '--extendReads', '50', '-p', '1'] + filter_args
    if implementation == 'original':
        original('bamCoverage', args + ['-o', str(upstream), '--normalizeUsing', 'None' if metric == 'read-count' else metric])
        result = upstream
    else:
        bamCoverage.main(args + ['-o', str(plus), '--normalizeUsing', metric, '--normalizationDenominator', 'filtered', '--filterMode', 'deeptools'])
        result = plus
    kept = 1 if filter_args else 2
    scale = {'read-count': 1.0, 'CPM': 1000000.0 / kept, 'RPKM': 1000000000.0 / (kept * 10)}[metric]
    expected = np.zeros(200)
    expected[10:60] = kept * scale
    np.testing.assert_allclose(values(result), expected, rtol=5.2e-06 if implementation == 'original' else 2e-07, atol=1e-06)


@pytest.mark.parametrize('threshold,kept', [('--minFragmentLength', 1), ('--maxFragmentLength', 0)])
@pytest.mark.parametrize('implementation', ['plus', 'original'])
def test_p02_single_end_length_filters_use_observed_cigar_length(tmp_path, threshold, kept, implementation):
    bam, upstream, plus = [tmp_path / name for name in ('single.bam', 'original.bw', 'plus.bw')]
    make_bam(bam, [(0, 10, 0, 60, 0, 0, [(0, 10)]), (0, 70, 0, 60, 0, 0, [(0, 20)])])
    args = ['-b', str(bam), '--binSize', '10', threshold, '15', '-p', '1']
    if implementation == 'original':
        original('bamCoverage', args + ['-o', str(upstream)])
        result = upstream
    else:
        bamCoverage.main(args + ['-o', str(plus), '--normalizeUsing', 'read-count', '--filterMode', 'deeptools'])
        result = plus
    expected = np.zeros(200)
    expected[70:90] = 1 if kept else 0
    expected[10:20] = 0 if kept else 1
    np.testing.assert_array_equal(values(result), expected)


@pytest.mark.parametrize('region_filter', ['--whiteListFileName', '--blackListFileName'])
@pytest.mark.parametrize('denominator', ['library', 'filtered'])
@pytest.mark.parametrize('metric', ['CPM', 'RPKM'])
def test_p03_filtered_denominator_counts_region_filtered_reads(tmp_path, region_filter, denominator, metric):
    bam, bed, output = (tmp_path / 'single.bam', tmp_path / 'filter.bed', tmp_path / 'plus.bw')
    make_bam(bam, [(0, 10, 0, 60, 0, 0, [(0, 10)]), (0, 70, 0, 60, 0, 0, [(0, 10)])])
    bed.write_text('chr1\t0\t30\n' if region_filter == '--whiteListFileName' else 'chr1\t60\t90\n')
    args = ['-b', str(bam), '-o', str(output), '--binSize', '10', '-p', '2', '--normalizeUsing', metric, '--normalizationDenominator', denominator, region_filter, str(bed), '--filterByOverlap']
    bamCoverage.main(args)
    count = 2 if denominator == 'library' else 1
    expected = np.zeros(200)
    expected[10:20] = {'CPM': 1000000.0 / count, 'RPKM': 1000000000.0 / (count * 10)}[metric]
    np.testing.assert_allclose(values(output), expected, rtol=2e-07, atol=1e-06)


@pytest.mark.parametrize('threads', [1, 2])
def test_p04_long_pair_placement_survives_production_window_boundary(tmp_path, threads):
    bam = tmp_path / 'long.bam'
    make_bam(bam, [(0, 1030000, 99, 60, 50000, 1079990, [(0, 10)]), (0, 1079990, 147, 60, -50000, 1030000, [(0, 10)])], lengths=(1200000,))
    observed = []
    for maximum in (None, 100000):
        output = tmp_path / f'maximum-{maximum}.bw'
        args = ['-b', str(bam), '-o', str(output), '--binSize', '10', '-p', str(threads), '--extendReads', '100', '--normalizeUsing', 'read-count']
        if maximum is not None:
            args += ['--maxFragmentLength', str(maximum)]
        bamCoverage.main(args)
        with pyBigWig.open(str(output)) as bw:
            observed.append(bw.values('chr1', 1060000, 1060010))
    np.testing.assert_array_equal(observed, np.full((2, 10), 2.0))


@pytest.mark.parametrize('metric', ['coverage-mean', 'coverage-sum', 'read-count', 'CPM', 'RPKM', 'BPM'])
@pytest.mark.parametrize('scenario', ['plain', 'mapq', 'filtered', 'ignore', 'exact'])
def test_all_normalizations_against_independent_bin_oracle(tmp_path, metric, scenario):
    bam, output = (tmp_path / 'library.bam', tmp_path / 'plus.bw')
    records = [(0, 0, 0, 60, 0, 0, [(0, 10)]), (0, 20, 0, 10, 0, 0, [(0, 10)]), (0, 50, 256, 60, 0, 0, [(0, 10)]), (0, 98, 0, 60, 0, 0, [(0, 5)]), (1, 30, 0, 60, 0, 0, [(0, 10)])]
    lengths = [103, 100]
    make_bam(bam, records, lengths)
    args = ['-b', str(bam), '-o', str(output), '--binSize', '10', '-p', '2', '--normalizeUsing', metric, '--scaleFactor', '2']
    if scenario in ('mapq', 'filtered'):
        args += ['--minMappingQuality', '30']
    if scenario == 'filtered':
        args += ['--normalizationDenominator', 'filtered']
    if scenario == 'ignore':
        args += ['--ignoreForNormalization', 'chr2']
    if scenario == 'exact':
        args += ['--exactScaling']
    bamCoverage.main(args)
    counts, depths = ([np.zeros(math.ceil(n / 10)) for n in lengths], [np.zeros(n) for n in lengths])
    eligible = []
    for i, (tid, start, flag, mapq, _, _, cigar) in enumerate(records):
        keep = not flag & 256 and (scenario not in ('mapq', 'filtered') or mapq >= 30)
        if keep:
            eligible.append(i)
            end = start + cigar[0][1]
            depths[tid][start:end] += 1
            counts[tid][start // 10:(end - 1) // 10 + 1] += 1
    included = [i for i, record in enumerate(records) if scenario != 'ignore' or record[0] != 1]
    if scenario == 'filtered':
        included = [i for i in included if i in eligible]
    if scenario == 'exact':
        included = [i for i in included if not records[i][2] & 256]
    denominator = len(included)
    bpm_total = sum((math.fsum(row) for tid, row in enumerate(counts) if scenario != 'ignore' or tid != 1))
    for tid, length in enumerate(lengths):
        expected = np.zeros(length)
        for bin_index, count in enumerate(counts[tid]):
            start, end = (10 * bin_index, min(length, 10 * (bin_index + 1)))
            width = end - start
            value = {'coverage-mean': math.fsum(depths[tid][start:end]) / width, 'coverage-sum': math.fsum(depths[tid][start:end]), 'read-count': count, 'CPM': count * 1000000.0 / denominator, 'RPKM': count * 1000000000.0 / (denominator * width), 'BPM': count * 1000000.0 / bpm_total}[metric]
            expected[start:end] = 2 * value
        np.testing.assert_allclose(values(output, f'chr{tid + 1}'), expected, rtol=2e-07, atol=1e-06)


def _mate_matrix_bam(path):
    """Pairs near, far, on another chromosome, stacked, and non-primary."""
    rng = np.random.default_rng(7)
    reads = []
    for i in range(500):
        kind, name = i % 6, f't{i}'
        group = 'AB'[i % 2] if i % 7 == 0 else None
        a = int(rng.integers(0, 9500))
        b = a + int(rng.integers(0, 400))
        left, right = (99, 147) if rng.random() < 0.5 else (163, 83)
        if kind == 1:
            b = a + int(rng.integers(20000, 25000))
        elif kind == 4:
            name, a, b = f'stack{i % 9}', 30000, 30000 + int(rng.integers(0, 60))
        if kind == 2:
            reads += [record(name, a, b, 97, 0, mtid=1, group=group),
                      record(name, b, a, 145, 0, tid=1, mtid=0, group=group)]
            continue
        if kind == 5:
            reads.append(record(name, a, 0, 73, 0, group=group))
            continue
        tlen = b + 10 - a
        reads += [record(name, a, b, left, tlen, group=group),
                  record(name, b, a, right, -tlen, group=group)]
        if kind == 3:
            elsewhere = int(rng.integers(0, 39000))
            reads += [record(name, elsewhere, b, left | 256, 0, group=group),
                      record(name, elsewhere + 7, b, left | 2048, 0, group=group)]
    write_bam(path, reads, (40000, 12000))


_REGIONS = {'chr1': [(100, 400), (2000, 2600), (20500, 22000), (30000, 30005)],
            'chr2': [(0, 3000)]}
_MATE_OPTIONS = {
    'collapse-5prime': dict(collapse='5prime'),
    'collapse-3prime': dict(collapse='3prime'),
    'collapse-center': dict(collapse='center'),
    'extend': dict(extend_reads=50),
    'split': dict(collapse='5prime', strandedness='forward', filter_rna_strand='split'),
    'whitelist': dict(aggregation='count', filter_by_overlap=True, whitelist=_REGIONS),
    'blacklist-collapse': dict(collapse='5prime', filter_by_overlap=True, blacklist=_REGIONS),
    'filtered-denominator': dict(aggregation='count', normalization='cpm',
                                 normalization_denominator='filtered',
                                 filter_by_overlap=True, whitelist=_REGIONS),
    'nonprimary': dict(filter_mode='deeptools', extend_reads=30,
                       filter_by_overlap=True, whitelist=_REGIONS),
}


def _budget_run(bam, out, options, budget=None):
    paths = [out.with_suffix('.fwd.bedgraph'), out.with_suffix('.rev.bedgraph')]
    _coverage.bam_coverage_bigwig(
        str(bam), str(paths[0]), out_path_reverse=str(paths[1]), out_format='bedgraph',
        bin_size=10, window_size=500, mate_budget=budget, **options)
    return [p.read_bytes() for p in paths if p.exists()]


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('option', sorted(_MATE_OPTIONS))
def test_tiny_mate_budgets_are_byte_identical(tmp_path, option, threads):
    bam = tmp_path / 'matrix.bam'
    _mate_matrix_bam(bam)
    options = dict(_MATE_OPTIONS[option], threads=threads)
    expected = _budget_run(bam, tmp_path / 'default', options)
    assert expected[0]
    for entries in (1, 4):
        budget = {'entries': entries}
        assert _budget_run(bam, tmp_path / f'tiny{entries}', options, budget) == expected
        assert 0 < budget['peak_batch_entries'] <= entries
        assert budget['peak_cache_entries'] <= entries


def _deep_pileup_bam(path, pairs):
    reads = []
    for i in range(pairs):
        reads += [record(f'p{i}', 1000, 1100, 99, 110), record(f'p{i}', 1100, 1000, 147, -110)]
    write_bam(path, reads, (5000,))


def test_deep_pileup_collapse_is_linear_and_memory_bounded(tmp_path):
    """More pairs than the mate cache holds once stalled per-read queries."""
    import time
    bam, out = tmp_path / 'deep.bam', tmp_path / 'deep.bw'
    _deep_pileup_bam(bam, 80000)
    budget = {}
    started = time.monotonic()
    _coverage.bam_coverage_bigwig(str(bam), str(out), bin_size=1, collapse='5prime',
                                  threads=4, max_zooms=0, mate_budget=budget)
    assert time.monotonic() - started < 30
    expected = np.zeros(5000)
    expected[1000] = 80000
    np.testing.assert_array_equal(read_values(out), expected)
    assert budget['peak_cache_entries'] <= 32768
    assert budget['peak_cache_bytes'] <= 8 << 20
    assert 0 < budget['peak_batch_entries'] <= 16384
    assert budget['peak_batch_bytes'] <= 4 << 20


def test_deep_pileup_batch_flushes_repeatedly_within_one_window(tmp_path):
    bam, out = tmp_path / 'deep.bam', tmp_path / 'deep.bw'
    _deep_pileup_bam(bam, 6000)
    budget = {'entries': 256, 'bytes': 64 << 10}
    _coverage.bam_coverage_bigwig(str(bam), str(out), bin_size=1, extend_reads=50,
                                  threads=4, max_zooms=0, mate_budget=budget)
    expected = np.zeros(5000)
    expected[1000:1110] = 6000
    np.testing.assert_array_equal(read_values(out), expected)
    # Over 5000 deferred fragments through a 256-entry batch: many flushes.
    assert budget['peak_batch_entries'] <= 256
    assert budget['peak_batch_bytes'] <= 64 << 10
    assert budget['peak_cache_entries'] <= 256


@pytest.mark.parametrize('entries', [None, 64])
def test_mates_in_a_deep_pileup_on_another_chromosome(tmp_path, entries):
    bam, out = tmp_path / 'interchrom.bam', tmp_path / 'interchrom.bw'
    reads = []
    for i in range(6000):
        reads += [record(f'p{i}', 10 * i, 1000, 97, 0, mtid=1),
                  record(f'p{i}', 1000, 10 * i, 145, 0, tid=1, mtid=0)]
    write_bam(bam, reads, (60000, 5000))
    budget = {} if entries is None else {'entries': entries}
    _coverage.bam_coverage_bigwig(str(bam), str(out), bin_size=10, aggregation='count',
                                  filter_by_overlap=True, whitelist={'chr2': [(1000, 1001)]},
                                  threads=4, max_zooms=0, mate_budget=budget)
    np.testing.assert_array_equal(read_values(out, 'chr1'), np.ones(60000))
    expected = np.zeros(5000)
    expected[1000:1010] = 6000
    np.testing.assert_array_equal(read_values(out, 'chr2'), expected)
    assert 0 < budget['peak_batch_entries'] <= (entries or 16384)


def _clipped_pair_reads():
    """STAR-like pairs: TLEN spans soft clips, so its sign can point either way.

    Each entry: name, forward (start, cigar, flag), reverse (start, cigar, flag),
    forward TLEN, and the MAPQ of (forward, reverse).
    """
    M, S = 0, 4
    return [
        # chr21 example: the reverse mate carries the positive, clip-inflated TLEN.
        ('star', (30, ((S, 1), (M, 44), (S, 1)), 99), (33, ((S, 6), (M, 41), (S, 1)), 147), -48, (60, 60)),
        ('same_pos', (100, ((M, 10),), 99), (100, ((M, 10),), 147), 10, (60, 60)),
        ('same_neg', (100, ((M, 10),), 99), (100, ((M, 12),), 147), -12, (60, 60)),
        ('normal', (150, ((M, 10),), 163), (180, ((M, 10),), 83), 40, (60, 60)),
        ('dovetail', (210, ((M, 20),), 99), (210, ((M, 15),), 147), 20, (60, 60)),
        ('outward', (232, ((M, 10),), 99), (228, ((M, 10),), 147), -14, (60, 60)),
        ('fwd_only', (240, ((M, 10),), 99), (260, ((S, 1), (M, 10), (S, 1)), 147), 32, (60, 0)),
        ('rev_only', (275, ((S, 2), (M, 10),), 163), (280, ((M, 10), (S, 3)), 83), 18, (0, 60)),
        ('long', (5, ((M, 10),), 99), (285, ((M, 10),), 147), 290, (60, 60)),
    ]


def _write_clipped_pairs(path):
    reads = []
    for name, (fpos, fcig, fflag), (rpos, rcig, rflag), tlen, (fq, rq) in _clipped_pair_reads():
        for pos, cig, flag, mpos, t, q in ((fpos, fcig, fflag, rpos, tlen, fq),
                                           (rpos, rcig, rflag, fpos, -tlen, rq)):
            reads.append(record(name, pos, mpos, flag, t, mapq=q, cigar=cig))
    write_bam(path, reads, (300,))


def _aligned_end(pos, cigar):
    return pos + sum(n for op, n in cigar if op in (0, 2, 3, 7, 8))


def _expected_clipped(collapse, strandedness, rna_filter, extend):
    """Independent oracle: a proper pair is [forward start, reverse end), once."""
    out = [np.zeros(300), np.zeros(300)]
    for name, (fpos, fcig, fflag), (rpos, rcig, rflag), tlen, (fq, rq) in _clipped_pair_reads():
        mates = [(fpos, fcig, fflag, fq >= 10), (rpos, rcig, rflag, rq >= 10)]
        if name == 'outward':  # not inward-facing: each mate is its own alignment
            pieces = [(pos, _aligned_end(pos, cig), flag, bool(flag & 16))
                      for pos, cig, flag, ok in mates if ok]
        elif mates[1][3]:  # reverse mate kept: it holds both bounds
            pieces = [(fpos, _aligned_end(rpos, rcig), rflag, False)]
        elif mates[0][3]:  # forward mate alone extends by |TLEN|
            pieces = [(fpos, fpos + abs(tlen), fflag, False)]
        else:
            pieces = []
        for start, end, flag, reverse in pieces:
            read_sign = -1 if flag & 16 else 1
            ts = -read_sign if flag & 128 else read_sign
            if strandedness == 'reverse':
                ts = -ts
            if rna_filter == 'forward' and ts != 1:
                continue
            stream = 1 if rna_filter == 'split' and ts != 1 else 0
            sign = ts if strandedness != 'none' else (-1 if reverse else 1)
            if extend:
                out[stream][start:end] += 1
            elif collapse == 'center':
                out[stream][(start + end) // 2] += 1
            else:
                left = (sign > 0) == (collapse == '5prime')
                out[stream][start if left else end - 1] += 1
    return out


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('window_size', [None, 16])
@pytest.mark.parametrize('mode', [
    ('5prime', 'none', 'none'), ('3prime', 'none', 'none'), ('center', 'none', 'none'),
    ('5prime', 'forward', 'forward'), ('3prime', 'reverse', 'none'),
    ('5prime', 'forward', 'split'), ('center', 'reverse', 'split'),
    ('extend', 'none', 'none'), ('extend', 'forward', 'split')])
def test_proper_fragment_spans_forward_start_to_reverse_end(tmp_path, mode, window_size, threads):
    collapse, strandedness, rna_filter = mode
    bam = tmp_path / 'clipped.bam'
    _write_clipped_pairs(bam)
    out, rev = tmp_path / 'out.bw', tmp_path / 'rev.bw'
    options = dict(extend_reads=5) if collapse == 'extend' else dict(collapse=collapse)
    if window_size:
        options['window_size'] = window_size
    _coverage.bam_coverage_bigwig(
        str(bam), str(out), out_path_reverse=str(rev), bin_size=1, min_mapping_quality=10,
        strandedness=strandedness, filter_rna_strand=rna_filter, threads=threads,
        max_zooms=0, **options)
    expected = _expected_clipped(collapse, strandedness, rna_filter, collapse == 'extend')
    for stream, path in enumerate([out, rev] if rna_filter == 'split' else [out]):
        np.testing.assert_array_equal(np.nan_to_num(read_values(path)), expected[stream])


_NONPRIMARY_REGIONS = {'chr1': [(k, k + 300) for k in range(0, 40000, 1000)]}


def _nonprimary_bam(path, templates=400):
    """Secondary/supplementary alignments before and after their primaries or
    paired with another secondary; ineligible (MAPQ 0) and ambiguous (two
    primaries for one end) identities."""
    rng = np.random.default_rng(11)
    reads = []
    for i in range(templates):
        name, group = f'n{i}', ('AB'[i % 2] if i % 5 == 0 else None)
        a = int(rng.integers(0, 39000))
        b = a + int(rng.integers(0, 400))
        left, right = (99, 147) if i % 2 else (163, 83)
        tlen = b + 10 - a
        reads += [record(name, a, b, left, tlen, group=group, mapq=0 if i % 11 == 0 else 60),
                  record(name, b, a, right, -tlen, group=group)]
        for j in range(1 + i % 3):
            reads.append(record(name, int(rng.integers(0, 39000)), b,
                                left | (256 if j % 2 == 0 else 2048), 0, group=group))
        reads.append(record(name, int(rng.integers(0, 39000)), int(rng.integers(0, 39000)),
                            left | 256, 0, group=group))  # mate is another secondary
        if i % 13 == 0:
            reads.append(record(name, a + 5, b, left, tlen, group=group))
    write_bam(path, reads, (40000,))


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('region_kind', ['whitelist', 'blacklist'])
def test_nonprimary_index_budget_spills_identically(tmp_path, region_kind, threads):
    bam = tmp_path / 'nonprimary.bam'
    _nonprimary_bam(bam)
    options = {'filter_mode': 'deeptools', 'filter_by_overlap': True, 'min_mapping_quality': 10,
               'extend_reads': 30, 'threads': threads, region_kind: _NONPRIMARY_REGIONS}
    default = {}
    expected = _budget_run(bam, tmp_path / 'default', options, default)
    assert expected[0] and default['nonprimary_passes'] == 2
    assert not default['nonprimary_spilled']
    for nonprimary_bytes, spilled in ((1 << 20, False), (4096, True), (512, True)):
        budget = {'nonprimary_bytes': nonprimary_bytes}
        assert _budget_run(bam, tmp_path / f'tiny{nonprimary_bytes}', options, budget) == expected
        assert budget['nonprimary_passes'] == 2
        assert budget['nonprimary_spilled'] == spilled
        assert budget['peak_nonprimary_bytes'] <= nonprimary_bytes + 256  # a few records of slack
    assert sorted(p.name for p in tmp_path.iterdir() if 'nonprimary' in p.name) == ['nonprimary.bam', 'nonprimary.bam.bai']


def _multimapper_bam(path, templates):
    """Primaries in [0, 10000); secondaries in the whitelisted [20000, 30000).
    Even templates' secondaries share their primary end's identity; odd ones
    pair with each other, as STAR writes them, so they resolve to nothing."""
    rng = np.random.default_rng(5)
    reads, kept = [], []
    for i in range(templates):
        name = f'm{i}'
        a = int(rng.integers(0, 9500))
        b = a + int(rng.integers(0, 400))
        pair = [record(name, a, b, 99, b + 10 - a), record(name, b, a, 147, a - b - 10)]
        secondaries = []
        for _ in range(3):
            where = int(rng.integers(20000, 29990))
            mate = b if i % 2 == 0 else int(rng.integers(20000, 29990))
            secondaries.append(record(name, where, mate, 99 | 256, 0))
        reads += pair + secondaries
        kept += secondaries + (pair if i % 2 == 0 else [])
    write_bam(path, reads, (40000,))
    expected = np.zeros(40000)
    for read in kept:
        expected[read.reference_start:read.reference_start + 10] += 1
    return expected


@pytest.mark.parametrize('templates', [4000, 20000])
def test_multimapper_nonprimary_index_stays_within_budget_in_linear_time(tmp_path, templates):
    import time
    bam = tmp_path / 'multi.bam'
    expected = _multimapper_bam(bam, templates)
    timings = {}
    for nonprimary_bytes in (None, 64 << 10, 4096):
        budget = {} if nonprimary_bytes is None else {'nonprimary_bytes': nonprimary_bytes}
        out = tmp_path / f'multi{nonprimary_bytes}.bw'
        started = time.monotonic()
        _coverage.bam_coverage_bigwig(
            str(bam), str(out), bin_size=1, aggregation='count', filter_mode='deeptools',
            filter_by_overlap=True, whitelist={'chr1': [(20000, 30000)]}, threads=4,
            max_zooms=0, mate_budget=budget)
        timings[nonprimary_bytes] = time.monotonic() - started
        np.testing.assert_array_equal(read_values(out), expected)
        if nonprimary_bytes:
            assert budget['peak_nonprimary_bytes'] <= nonprimary_bytes + 256
            assert budget['nonprimary_passes'] == 2 and budget['nonprimary_spilled']
    # Exceeding the budget costs disk merges, never extra BAM passes.
    assert timings[64 << 10] < 2 and timings[4096] < 2, timings


def test_nonprimary_spill_files_are_removed_after_an_error(tmp_path):
    bam, out = tmp_path / 'nonprimary.bam', tmp_path / 'out.bw'
    _nonprimary_bam(bam)
    blocker = tmp_path / 'out.bw.nonprimary.index'  # the spilled index cannot be written
    blocker.mkdir()
    (blocker / 'keep').touch()
    with pytest.raises(RuntimeError, match='spill file'):
        _coverage.bam_coverage_bigwig(
            str(bam), str(out), bin_size=10, filter_mode='deeptools', filter_by_overlap=True,
            whitelist=_NONPRIMARY_REGIONS, mate_budget={'nonprimary_bytes': 4096})
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        'nonprimary.bam', 'nonprimary.bam.bai', 'out.bw.nonprimary.index']


def test_nonprimary_bloom_false_positives_do_not_change_results(tmp_path):
    """At 1 KB the 256-byte Bloom filter is saturated by 20,000 identities, so
    nearly every alignment reaches the join; only real matches inherit."""
    bam = tmp_path / 'multi.bam'
    expected = _multimapper_bam(bam, 20000)
    for nonprimary_bytes in (None, 1024):
        budget = {} if nonprimary_bytes is None else {'nonprimary_bytes': nonprimary_bytes}
        out = tmp_path / f'bloom{nonprimary_bytes}.bw'
        _coverage.bam_coverage_bigwig(
            str(bam), str(out), bin_size=1, aggregation='count', filter_mode='deeptools',
            filter_by_overlap=True, whitelist={'chr1': [(0, 5000), (20000, 30000)]}, threads=2,
            max_zooms=0, mate_budget=budget)
        values = read_values(out)
        if nonprimary_bytes is None:
            reference = values
        else:
            np.testing.assert_array_equal(values, reference)
            assert budget['nonprimary_passes'] == 2 and budget['nonprimary_spilled']
    assert values.sum() > expected.sum()  # primaries inside (0, 5000) count on their own
