"""Synthetic coverage input helpers."""

import numpy as np
import pytest
from tests.helpers.bigwig import pyBigWig

pysam = pytest.importorskip("pysam")


def record(
    name,
    start,
    mate,
    flag,
    tlen,
    *,
    tid=0,
    mtid=0,
    mapq=60,
    cigar=((0, 10),),
    group=None,
):
    read = pysam.AlignedSegment()
    read.query_name = name
    read.query_sequence = "A" * sum(
        length for op, length in cigar if op in (0, 1, 4, 7, 8)
    )
    read.reference_id, read.reference_start = tid, start
    read.next_reference_id, read.next_reference_start = mtid, mate
    read.flag, read.template_length, read.mapping_quality = flag, tlen, mapq
    read.cigartuples = cigar
    if group is not None:
        read.set_tag("RG", group)
    return read


def write_bam(path, reads, lengths=(200,)):
    header = {
        "HD": {"SO": "coordinate"},
        "SQ": [{"SN": f"chr{i + 1}", "LN": length} for i, length in enumerate(lengths)],
        "RG": [{"ID": "A"}, {"ID": "B"}],
    }
    with pysam.AlignmentFile(str(path), "wb", header=header) as handle:
        for read in sorted(reads, key=lambda r: (r.reference_id, r.reference_start)):
            handle.write(read)
    pysam.index(str(path))


def read_values(path, chrom="chr1"):
    with pyBigWig.open(str(path)) as bw:
        return np.asarray(bw.values(chrom, 0, bw.chroms(chrom)))
