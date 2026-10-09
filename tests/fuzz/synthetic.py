"""Deterministic synthetic inputs for the fuzz harnesses.

A paired-end spliced BAM (with its CRAM, BGZF SAM and indices), two
multi-chromosome bigWigs large enough for a multi-level R-tree and zoom
levels, region files (BED6, BED12, GTF), three matrices and an options file.
pysam and pyBigWig are imported lazily so that collecting the harness does
not need them.
"""

import os
import random
import subprocess
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "test_data"
ALN_CHROMS = (("chrA", 40000), ("chrB", 25000))
# 80 scaffolds give more data blocks than one R-tree node holds (two levels).
BW_CHROMS = (("chr1", 1_500_000), ("chr2", 800_000), ("chr3", 300_000)) + tuple(
    (f"scaf{i}", 30000) for i in range(80))


def _segment(pysam, rng, name, flag, ref, pos, cigar, mate, tlen):
    seg = pysam.AlignedSegment()
    seg.query_name, seg.flag, seg.reference_id = name, flag, ref
    seg.reference_start = pos
    seg.mapping_quality = 60 if rng.random() > 0.1 else 0
    seg.next_reference_id, seg.next_reference_start = mate
    seg.template_length = tlen
    length = 50
    if cigar:
        seg.cigarstring = cigar
        length = seg.infer_query_length()
    seg.query_sequence = "".join(rng.choice("ACGT") for _ in range(length))
    seg.query_qualities = pysam.qualitystring_to_array("I" * length)
    return seg


def write_pe_bam(path, pairs=1500, seed=7):
    """Proper spliced pairs with soft clips, plus singletons and discordant pairs."""
    import pysam

    rng = random.Random(seed)
    header = {"HD": {"VN": "1.6", "SO": "unsorted"},
              "SQ": [{"SN": n, "LN": ln} for n, ln in ALN_CHROMS]}
    unsorted = f"{path}.unsorted.bam"
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for i in range(pairs):
            ref = rng.randrange(2)
            p1 = rng.randrange(100, ALN_CHROMS[ref][1] - 3000)
            c1 = rng.choice(["50M", "20M300N30M", "5S45M", "25M150N20M5S"])
            c2 = rng.choice(["50M", "30M500N20M", "3S47M"])
            p2 = p1 + rng.randrange(50, 600)
            kind, rev, name = rng.random(), rng.random() < 0.5, f"r{i}"
            if kind < 0.05:  # singleton with an unmapped mate placed beside it
                out.write(_segment(pysam, rng, name, 73 | (16 * rev), ref, p1, c1,
                                   (ref, p1), 0))
                out.write(_segment(pysam, rng, name, 133 | (32 * rev), ref, p1, None,
                                   (ref, p1), 0))
            elif kind < 0.1:  # discordant: mates on different chromosomes
                r2 = 1 - ref
                p2 = rng.randrange(100, ALN_CHROMS[r2][1] - 1000)
                out.write(_segment(pysam, rng, name, 97, ref, p1, c1, (r2, p2), 0))
                out.write(_segment(pysam, rng, name, 145, r2, p2, c2, (ref, p1), 0))
            else:
                f1, f2 = (163, 83) if rev else (99, 147)
                tlen = p2 + 600 - p1
                out.write(_segment(pysam, rng, name, f1, ref, p1, c1, (ref, p2), tlen))
                out.write(_segment(pysam, rng, name, f2, ref, p2, c2, (ref, p1), -tlen))
    pysam.sort("-o", str(path), unsorted)
    os.remove(unsorted)
    pysam.index(str(path))


def write_alignment_family(base):
    """pe.bam(+bai), reference-free pe.cram(+crai), BGZF pe.sam.gz(+csi)."""
    import pysam

    write_pe_bam(base / "pe.bam")
    pysam.view("-C", "--output-fmt-option", "no_ref=1", "-o", str(base / "pe.cram"),
               str(base / "pe.bam"), catch_stdout=False)
    pysam.index(str(base / "pe.cram"))
    pysam.view("-h", "-o", str(base / "pe.sam"), str(base / "pe.bam"),
               catch_stdout=False)
    pysam.tabix_compress(str(base / "pe.sam"), str(base / "pe.sam.gz"), force=True)
    os.remove(base / "pe.sam")
    pysam.index("-c", str(base / "pe.sam.gz"))


def write_bigwig(path, seed, step=40):
    """Variable-step bigWig over three chromosomes with zoom levels."""
    import pyBigWig

    rng = random.Random(seed)
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader(list(BW_CHROMS), maxZooms=10)
    for chrom, length in BW_CHROMS:
        starts = list(range(0, length - step, step))
        starts = [s for s in starts if rng.random() < 0.6]
        bw.addEntries([chrom] * len(starts), starts,
                      ends=[s + step - rng.randrange(step // 2) for s in starts],
                      values=[round(rng.uniform(0, 50), 2) for _ in starts])
    bw.close()


def _regions(rng, n, chroms=BW_CHROMS):
    rows = []
    for i in range(n):
        chrom, length = chroms[rng.randrange(len(chroms))]
        start = rng.randrange(1000, length - 10000)
        rows.append(f"{chrom}\t{start}\t{start + rng.randrange(500, 5000)}\t"
                    f"g{i}\t0\t{'+-'[rng.randrange(2)]}\n")
    return "".join(sorted(rows))


def _bed12(rng, n):
    rows = []
    for i in range(n):
        chrom, length = BW_CHROMS[rng.randrange(len(BW_CHROMS))]
        start = rng.randrange(1000, length - 10000)
        sizes, starts, pos = [], [], 0
        for _ in range(rng.randrange(1, 4)):
            size = rng.randrange(100, 600)
            sizes.append(size)
            starts.append(pos)
            pos += size + rng.randrange(200, 1500)
        end = start + starts[-1] + sizes[-1]
        rows.append("\t".join(map(str, (
            chrom, start, end, f"t{i}", 0, "+-"[i % 2], start, end, 0, len(sizes),
            ",".join(map(str, sizes)) + ",", ",".join(map(str, starts)) + ","))) + "\n")
    return "".join(sorted(rows))


def _gtf(bed12):
    rows = []
    for line in bed12.splitlines():
        f = line.split("\t")
        start = int(f[1])
        attrs = f'gene_id "{f[3]}g"; transcript_id "{f[3]}";'
        rows.append("\t".join((f[0], "x", "transcript", str(start + 1), f[2], ".",
                               f[5], ".", attrs)) + "\n")
        for size, off in zip(f[10].rstrip(",").split(","), f[11].rstrip(",").split(",")):
            s = start + int(off)
            rows.append("\t".join((f[0], "x", "exon", str(s + 1), str(s + int(size)),
                                   ".", f[5], ".", attrs)) + "\n")
    return "".join(rows)


def _run(base, *cmd):
    subprocess.run(cmd, cwd=base, check=True, capture_output=True)


def write_all(base, seed=11):
    """Write every synthetic input into ``base``; return the file names."""
    rng = random.Random(seed)
    base.mkdir(parents=True, exist_ok=True)
    write_alignment_family(base)
    if (DATA / "pointseq_slice.bam").exists():
        for suffix in ("", ".bai"):
            (base / f"ps.bam{suffix}").write_bytes(
                (DATA / f"pointseq_slice.bam{suffix}").read_bytes())
    text = {
        "bl2.bed": "chrA\t5000\t9000\nchrB\t1000\t3000\n",
        "wl.bed": "chrA\t2000\t30000\nchrB\t0\t15000\n",
        "s.bed": "chrA\t0\t20000\tg1\t0\t+\nchrB\t0\t12000\tg2\t0\t-\n",
        "mr1.bed": _regions(rng, 120),
        "mr2.bed": _regions(rng, 80),
        "mr3.bed": _regions(rng, 60),
        "r12.bed": _bed12(rng, 60),
    }
    text["mg.gtf"] = _gtf(text["r12.bed"])
    for name, body in text.items():
        (base / name).write_text(body)
    write_bigwig(base / "mb.bw", seed=1)
    write_bigwig(base / "mb2.bw", seed=2)
    geometry = ("-m", "1000", "-a", "500", "-b", "500", "-bs", "50")
    _run(base, "computeMatrixR", "scale-regions", "-S", "mb.bw", "mb2.bw",
         "-R", "mr1.bed", "mr2.bed", "-o", "mA.mat.gz", *geometry)
    _run(base, "computeMatrixR", "scale-regions", "-S", "mb.bw", "mb2.bw",
         "-R", "mr3.bed", "-o", "mB.mat.gz", *geometry)
    _run(base, "computeMatrixR", "scale-regions", "-S", "mb2.bw",
         "-R", "mr1.bed", "mr2.bed", "-o", "mC.mat.gz", *geometry)
    _run(base, "deeptoolsr", "options", "--config", "cfg.txt", "--fontMultiplier", "1.5")
    return sorted(p.name for p in base.iterdir())
