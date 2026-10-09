"""Read extension: every read is a solid fragment from its 5' end, never
shorter than its own span; --skipIntrons leaves each alignment's own introns
empty."""

import numpy as np
import pytest

from deeptoolsr import _coverage, bamCoverage
from tests.helpers.coverage import read_values, record, write_bam

M, N, S = 0, 3, 4
LENGTH = 300
SPLICED = ((M, 5), (N, 100), (M, 5))


def expected(*intervals, value=1.0):
    depth = np.zeros(LENGTH)
    for start, end in intervals:
        depth[start:end] += value
    return depth


def coverage(tmp_path, reads, aggregation="sum", **options):
    bam = tmp_path / "in.bam"
    write_bam(bam, reads, lengths=(LENGTH,))
    values = []
    for threads in (1, 4):
        out = tmp_path / f"out{threads}.bw"
        _coverage.bam_coverage_bigwig(
            str(bam), str(out), bin_size=1, aggregation=aggregation,
            threads=threads, window_size=16, **options)
        values.append(read_values(out))
    assert np.array_equal(values[0], values[1])
    return values[0]


SINGLE_CASES = [
    # (start, cigar, reverse, extension, skip, expected intervals)
    (10, ((M, 10),), False, 20, False, [(10, 30)]),
    (10, ((M, 10),), False, 5, False, [(10, 20)]),
    (50, ((M, 10),), True, 20, False, [(40, 60)]),
    (50, ((M, 10),), True, 5, False, [(50, 60)]),
    (10, ((S, 3), (M, 10)), False, 12, False, [(10, 22)]),
    (10, SPLICED, False, 20, False, [(10, 120)]),
    (10, SPLICED, False, 20, True, [(10, 15), (115, 120)]),
    (10, SPLICED, False, 200, False, [(10, 210)]),
    (10, SPLICED, False, 200, True, [(10, 15), (115, 210)]),
    (150, SPLICED, True, 20, False, [(150, 260)]),
    (150, SPLICED, True, 20, True, [(150, 155), (255, 260)]),
    (150, SPLICED, True, 200, False, [(60, 260)]),
    (150, SPLICED, True, 200, True, [(60, 155), (255, 260)]),
]


@pytest.mark.parametrize("start,cigar,reverse,extension,skip,intervals", SINGLE_CASES)
def test_single_read_extension(tmp_path, start, cigar, reverse, extension, skip, intervals):
    read = record("r", start, -1, 16 if reverse else 0, 0, mtid=-1, cigar=cigar)
    values = coverage(tmp_path, [read], extend_reads=extension, skip_introns=skip)
    assert np.array_equal(values, expected(*intervals))


def spliced_pair(forward_mapq=60, reverse_mapq=60):
    # Fragment [10, 80): forward 10-40 with intron [15, 35); reverse 60-80 with
    # intron [65, 75); mates leave the gap [40, 60) unobserved.
    return [
        record("p", 10, 60, 99, 70, mapq=forward_mapq, cigar=((M, 5), (N, 20), (M, 5))),
        record("p", 60, 10, 147, -70, mapq=reverse_mapq, cigar=((M, 5), (N, 10), (M, 5))),
    ]


@pytest.mark.parametrize("aggregation,weight", [("sum", 1), ("count", 2)])
@pytest.mark.parametrize("skip,intervals", [
    (False, [(10, 80)]),
    (True, [(10, 15), (35, 65), (75, 80)]),
])
def test_proper_pair_with_spliced_mates(tmp_path, aggregation, weight, skip, intervals):
    values = coverage(tmp_path, spliced_pair(), aggregation=aggregation,
                      extend_reads=20, skip_introns=skip)
    assert np.array_equal(values, expected(*intervals, value=weight))


@pytest.mark.parametrize("forward_mapq,reverse_mapq,intervals", [
    (60, 0, [(10, 15), (35, 80)]),   # forward mate alone: |TLEN|
    (0, 60, [(10, 65), (75, 80)]),   # reverse mate alone
])
@pytest.mark.parametrize("aggregation", ["sum", "count"])
def test_lone_mate_skips_only_its_own_introns(tmp_path, forward_mapq, reverse_mapq,
                                              intervals, aggregation):
    values = coverage(tmp_path, spliced_pair(forward_mapq, reverse_mapq),
                      aggregation=aggregation, min_mapping_quality=10,
                      extend_reads=20, skip_introns=True)
    assert np.array_equal(values, expected(*intervals))


def clipped_pair():
    # |TLEN| 48 includes soft clips; the aligned fragment is [30, 74).
    return [
        record("c", 30, 33, 99, 48, cigar=((S, 1), (M, 44), (S, 1))),
        record("c", 33, 30, 147, -48, cigar=((S, 6), (M, 41), (S, 1))),
    ]


@pytest.mark.parametrize("options,intervals", [
    (dict(extend_reads=20), [(30, 74)]),
    (dict(collapse="3prime"), [(73, 74)]),
    (dict(collapse="center"), [(52, 53)]),
])
@pytest.mark.parametrize("aggregation,weight", [("sum", 1), ("count", 2)])
def test_both_metrics_use_the_aligned_fragment(tmp_path, options, intervals,
                                               aggregation, weight):
    values = coverage(tmp_path, clipped_pair(), aggregation=aggregation, **options)
    assert np.array_equal(values, expected(*intervals, value=weight))


def test_cpm_counts_each_mate_of_the_aligned_fragment(tmp_path):
    reads = clipped_pair() + [record("s", 200, -1, 0, 0, mtid=-1)]
    values = coverage(tmp_path, reads, aggregation="count", normalization="cpm",
                      extend_reads=20, skip_introns=True)
    assert np.allclose(values, expected((30, 74), value=2e6 / 3)
                       + expected((200, 220), value=1e6 / 3), rtol=1e-6)


def test_skip_introns_requires_extension(tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        bamCoverage.main(["-b", "in.bam", "-o", str(tmp_path / "o.bw"), "--skipIntrons"])
    assert raised.value.code == 2
    assert "--skipIntrons requires --extendReads" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="skip_introns requires extend_reads"):
        _coverage.bam_coverage_bigwig("in.bam", str(tmp_path / "o.bw"), skip_introns=True)
