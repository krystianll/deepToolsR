"""The sole BAM backend checked against records, including BAI and CSI inputs."""

from pathlib import Path

import numpy as np
from tests.helpers.bigwig import pyBigWig
import pytest

from deeptoolsr import _coverage

pysam = pytest.importorskip("pysam")


DATA = Path(__file__).parent / "test_data"
BAMS = [
    "test1.bam",
    "test2.bam",
    "testA.bam",
    "testB.bam",
    "test_paired.bam",
    "test_paired2.bam",
    "test_filtering.bam",
    "test_filtering2.bam",
    "test1_csi.bam",
    "test_paired_csi.bam",
]


@pytest.mark.parametrize("name", BAMS)
def test_index_counts_match_actual_alignment_flags(name):
    path = DATA / name
    with pysam.AlignmentFile(str(path), "rb") as bam:
        mapped, unmapped = (
            np.zeros(bam.nreferences, dtype=int),
            np.zeros(bam.nreferences, dtype=int),
        )
        unplaced = 0
        for read in bam:
            if read.reference_id < 0:
                unplaced += 1
            elif read.is_unmapped:
                unmapped[read.reference_id] += 1
            else:
                mapped[read.reference_id] += 1
        expected_targets = list(zip(bam.references, bam.lengths))
    stats = _coverage.bam_index_stats(str(path))
    assert [
        (row["name"], row["length"]) for row in stats["targets"]
    ] == expected_targets
    assert [row["mapped"] for row in stats["targets"]] == mapped.tolist()
    assert [row["unmapped"] for row in stats["targets"]] == unmapped.tolist()
    assert stats["total_mapped"] == sum(mapped)
    assert stats["total_unmapped"] == sum(unmapped)
    assert stats["n_no_coor"] == unplaced
    if "_csi" in name:
        assert _coverage.resolve_bam_index_path(str(path)).endswith(".csi")


@pytest.mark.parametrize(
    "name",
    [
        "test1.bam",
        "testA.bam",
        "test_filtering.bam",
        "test1_csi.bam",
        "test_paired_csi.bam",
    ],
)
@pytest.mark.parametrize("bin_size", [1, 13, 50])
def test_read_count_bins_match_cigar_oracle(tmp_path, name, bin_size):
    path, output = DATA / name, tmp_path / "out.bw"
    _coverage.bam_coverage_bigwig(
        str(path),
        str(output),
        bin_size=bin_size,
        aggregation="count",
        threads=2,
        max_zooms=0,
    )
    with pysam.AlignmentFile(str(path), "rb") as bam, pyBigWig.open(str(output)) as bw:
        expected = {
            chrom: np.zeros((length + bin_size - 1) // bin_size)
            for chrom, length in zip(bam.references, bam.lengths)
        }
        for read in bam:
            if read.flag & (4 | 256 | 2048):
                continue
            bins = {
                i
                for start, end in read.get_blocks()
                for i in range(start // bin_size, (end - 1) // bin_size + 1)
            }
            for i in bins:
                expected[read.reference_name][i] += 1
        for chrom, length in zip(bam.references, bam.lengths):
            actual = np.asarray(bw.values(chrom, 0, length))[::bin_size]
            np.testing.assert_array_equal(actual, expected[chrom])
