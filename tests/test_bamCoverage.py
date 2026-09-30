"""Tests for the native bamCoverageR backend (deeptoolsr._coverage).

Ground truth is computed independently with pysam (per-base depth from aligned
blocks), so these do not depend on stock bamCoverage being installed. Skipped
when the native extension is unavailable.
"""
from deeptoolsr import bamCoverage
import struct
from deeptoolsr import _coverage as numerical__coverage
from tests.helpers.bigwig import pyBigWig
from pathlib import Path

import os

import numpy as np
import pytest

pysam = pytest.importorskip("pysam")
_coverage = pytest.importorskip("deeptoolsr._coverage")

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/test_data/"
PAIRED = ROOT + "test_paired.bam"
SINGLE = ROOT + "test1.bam"
CIGAR = ROOT + "testA.bam"  # 3R + chr_cigar (has an N-spliced, soft-clipped read)

BAM_FUNMAP = 0x4
BAM_FSECONDARY = 0x100
BAM_FSUPPLEMENTARY = 0x800


def test_split_suffixes_must_be_nonempty_and_unique(tmp_path, capsys):
    base = str(tmp_path / "split.bw")
    with pytest.raises(SystemExit):
        from deeptoolsr import bamCoverage
        bamCoverage.main(["-b", PAIRED, "-o", base,
                          "--filterRNAstrand", "split",
                          "--strandedness", "forward",
                          "--suffix", "", ".minus"])
    assert "must not be empty" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        bamCoverage.main(["-b", PAIRED, "-o", base,
                          "--filterRNAstrand", "split",
                          "--strandedness", "forward",
                          "--suffix", ".same", ".same"])
    assert "same file" in capsys.readouterr().err


def test_overlap_halo_can_be_zero_with_max_fragment(tmp_path):
    from deeptoolsr import bamCoverage
    regions = tmp_path / 'keep.bed'
    regions.write_text('chr2\t0\t10000000\n')
    output = tmp_path / 'out.bw'
    bamCoverage.main([
        "-b", PAIRED, "-o", str(output),
        "--whiteListFileName", str(regions), "--filterByOverlap",
        "--maxFragmentLength", "500", "--filterByOverlapHalo", "0"])
    assert np.any(_values(str(output), 'chr2', _chrom_len(PAIRED, 'chr2')) > 0)


def test_feature_touch_strand_requires_overlap_filter(tmp_path):
    from deeptoolsr import bamCoverage
    with pytest.raises(SystemExit, match="requires --filterByOverlap"):
        bamCoverage.main([
            "-b", PAIRED, "-o", str(tmp_path / "out.bw"),
            "--strandedness", "forward", "--featureTouchStrand", "sense"])


@pytest.mark.parametrize("option,value", [
    ("--binSize", "5.5"),
    ("--binSize", "0"),
    ("--minMappingQuality", "256"),
    ("--samFlagInclude", "65536"),
    ("--minFragmentLength", "-1"),
    ("--scaleFactor", "nan"),
    ("--scaleFactorMinus", "inf"),
    ("--zoomLevels", "-1"),
    ("--compressionLevel", "0"),
    ("--numberOfProcessors", "0"),
])
def test_cli_rejects_invalid_numeric_values_before_opening_output(
        tmp_path, option, value):
    from deeptoolsr import bamCoverage
    out = tmp_path / "out.bw"
    with pytest.raises(SystemExit):
        bamCoverage.main(["-b", PAIRED, "-o", str(out), option, value])
    assert not out.exists()


def test_cli_rejects_inverted_fragment_length_range(tmp_path, capsys):
    from deeptoolsr import bamCoverage
    out = tmp_path / "out.bw"
    with pytest.raises(SystemExit):
        bamCoverage.main([
            "-b", PAIRED, "-o", str(out),
            "--minFragmentLength", "501", "--maxFragmentLength", "500"])
    assert "minFragmentLength must be less than or equal" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("removed", [
    ["--normalizeUsing", "RPGC"],
    ["--effectiveGenomeSize", "100"],
])
def test_removed_rpgc_options_are_rejected(tmp_path, removed):
    from deeptoolsr import bamCoverage
    output = tmp_path / "out.bw"
    with pytest.raises(SystemExit):
        bamCoverage.main(["-b", PAIRED, "-o", str(output)] + removed)
    assert not output.exists()


def test_native_coverage_rejects_nonfinite_scale_before_opening_output(tmp_path):
    out = tmp_path / "out.bw"
    with pytest.raises(RuntimeError, match="scale factors must be finite"):
        _coverage.bam_coverage_bigwig(
            PAIRED, str(out), scale_factor=float("nan"))
    assert not out.exists()


def test_cli_preserves_existing_output_on_native_failure(tmp_path, monkeypatch):
    from deeptoolsr import bamCoverage
    out = tmp_path / "out.bw"
    out.write_bytes(b"previous coverage")

    def fail_after_partial_write(_bam, temporary, *_args, **_kwargs):
        with open(temporary, "wb") as handle:
            handle.write(b"partial coverage")
        raise RuntimeError("injected coverage failure")

    monkeypatch.setattr(_coverage, "bam_coverage_bigwig",
                        fail_after_partial_write)
    with pytest.raises(SystemExit, match="injected coverage failure"):
        bamCoverage.main(["-b", PAIRED, "-o", str(out)])
    assert out.read_bytes() == b"previous coverage"


def test_native_resolves_the_index_path_used_by_htslib(tmp_path):
    bam = tmp_path / "reads.bam"
    bam.write_bytes(b"bam placeholder")
    stem_bai = tmp_path / "reads.bai"
    stem_bai.write_bytes(b"stem bai")
    assert _coverage.resolve_bam_index_path(str(bam)) == str(stem_bai)

    appended_bai = tmp_path / "reads.bam.bai"
    appended_bai.write_bytes(b"appended bai")
    assert _coverage.resolve_bam_index_path(str(bam)) == str(appended_bai)

    appended_csi = tmp_path / "reads.bam.csi"
    appended_csi.write_bytes(b"appended csi")
    assert _coverage.resolve_bam_index_path(str(bam)) == str(appended_csi)


@pytest.mark.parametrize("bin_size,expected", [
    (1, 262_144),
    (4, 262_144),
    (50, 20_971),
    (500, 2_097),
    (2_000_000, 1),
])
def test_default_coverage_window_has_bin_and_basepair_caps(bin_size, expected):
    bins = _coverage.coverage_window_bins(bin_size)
    assert bins == expected
    assert bins <= 262_144
    # A single very large bin is the only permitted production span >1 MiB.
    assert bins == 1 or bins * bin_size <= 1_048_576


def test_explicit_test_window_cannot_bypass_safety_caps():
    assert _coverage.coverage_window_bins(1, 1_000_000) == 262_144
    assert _coverage.coverage_window_bins(500, 1_000_000) == 2_097
    assert _coverage.coverage_window_bins(50, 7) == 7


def test_cli_rejects_output_aliasing_loaded_bam_index(tmp_path, capsys):
    from deeptoolsr import bamCoverage
    bam = tmp_path / "reads.bam"
    index = tmp_path / "reads.bam.bai"
    with open(PAIRED, "rb") as handle:
        bam.write_bytes(handle.read())
    with open(PAIRED + ".bai", "rb") as handle:
        index.write_bytes(handle.read())
    original = index.read_bytes()

    with pytest.raises(SystemExit):
        bamCoverage.main(["-b", str(bam), "-o", str(index)])

    assert "aliases input" in capsys.readouterr().err
    assert index.read_bytes() == original


# --- helpers ---------------------------------------------------------------

def _chrom_len(bam, chrom):
    with pysam.AlignmentFile(bam, "rb") as f:
        return f.get_reference_length(chrom)


def _values(path, chrom, length):
    bw = pyBigWig.open(path)
    v = np.nan_to_num(np.array(bw.values(chrom, 0, length), dtype=np.float64))
    bw.close()
    return v


def _perbin(path, chrom, length, binsize):
    bw = pyBigWig.open(path)
    ivs = bw.intervals(chrom)
    bw.close()
    nb = (length + binsize - 1) // binsize
    a = np.zeros(nb)
    for s, e, val in (ivs or []):
        a[s // binsize:(e + binsize - 1) // binsize] = val
    return a


def _primary_reads(bam, chrom):
    """Yield primary, mapped reads on a chromosome (matches the default filter_mode)."""
    with pysam.AlignmentFile(bam, "rb") as f:
        for r in f.fetch(chrom):
            if r.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY):
                continue
            yield r


def _gt_perbase_depth(bam, chrom):
    """True per-base depth from aligned blocks (get_blocks splits on N/D)."""
    L = _chrom_len(bam, chrom)
    d = np.zeros(L, dtype=np.int64)
    for r in _primary_reads(bam, chrom):
        for bs, be in r.get_blocks():
            d[bs:be] += 1
    return d


def _gt_fragment_depth(bam, chrom, extend):
    """Single-count fragment depth: proper pair -> mate span once; else extend read."""
    L = _chrom_len(bam, chrom)
    d = np.zeros(L, dtype=np.int64)
    seen = set()
    for r in _primary_reads(bam, chrom):
        proper = (r.flag & 0x2) and not (r.flag & 0x8) and \
            r.next_reference_id == r.reference_id and r.template_length != 0
        if proper:
            s = min(r.reference_start, r.next_reference_start)
            e = s + abs(r.template_length)
            fragment = (r.query_name, s, e)
            if fragment in seen:
                continue
            seen.add(fragment)
        elif r.is_reverse:
            e, s = r.reference_end, r.reference_end - extend
        else:
            s, e = r.reference_start, r.reference_start + extend
        s, e = max(0, s), min(L, e)
        if e > s:
            d[s:e] += 1
    return d


def _gt_fragment_read_weight_depth(bam, chrom, extend):
    """Place every observed alignment; absent mates cannot contribute weight."""
    length = _chrom_len(bam, chrom)
    depth = np.zeros(length, dtype=np.int64)
    for read in _primary_reads(bam, chrom):
        proper = (read.flag & 0x2) and not (read.flag & 0x8) and \
            read.next_reference_id == read.reference_id and \
            read.template_length != 0
        if proper:
            start = min(read.reference_start, read.next_reference_start)
            end = start + abs(read.template_length)
        elif read.is_reverse:
            end, start = read.reference_end, read.reference_end - extend
        else:
            start, end = read.reference_start, read.reference_start + extend
        start, end = max(0, start), min(length, end)
        if end > start:
            depth[start:end] += 1
    return depth


# --- per-base / aggregation correctness ------------------------------------

def test_binsize1_mean_is_perbase_depth(tmp_path):
    out = str(tmp_path / "d.bw")
    _coverage.bam_coverage_bigwig(PAIRED, out, bin_size=1, aggregation="mean",
                                  filter_mode="deeptools")
    L = _chrom_len(PAIRED, "chr2")
    assert np.array_equal(_values(out, "chr2", L), _gt_perbase_depth(PAIRED, "chr2").astype(float))


def test_mean_equals_binnedaverage(tmp_path):
    out = str(tmp_path / "m.bw")
    bs = 50
    _coverage.bam_coverage_bigwig(PAIRED, out, bin_size=bs, aggregation="mean",
                                  filter_mode="deeptools")
    L = _chrom_len(PAIRED, "chr2")
    depth = _gt_perbase_depth(PAIRED, "chr2").astype(float)
    nb = (L + bs - 1) // bs
    pad = nb * bs - L
    if pad:
        depth = np.concatenate([depth, np.zeros(pad)])
    expected = depth.reshape(nb, bs).sum(axis=1)
    width = np.full(nb, bs)
    if pad:
        width[-1] = bs - pad
    expected = expected / width
    assert np.allclose(expected, _perbin(out, "chr2", L, bs), atol=1e-5)


def test_sum_equals_mean_times_binsize(tmp_path):
    bs = 50
    ms = str(tmp_path / "s.bw")
    mm = str(tmp_path / "m.bw")
    _coverage.bam_coverage_bigwig(PAIRED, ms, bin_size=bs, aggregation="sum",
                                  filter_mode="deeptools")
    _coverage.bam_coverage_bigwig(PAIRED, mm, bin_size=bs, aggregation="mean",
                                  filter_mode="deeptools")
    L = _chrom_len(PAIRED, "chr2")
    # full bins only (last bin may be partial; chr2 len is a multiple of 50 here)
    assert np.allclose(_perbin(ms, "chr2", L, bs), _perbin(mm, "chr2", L, bs) * bs, atol=1e-3)


def test_cigar_n_split_and_softclip(tmp_path):
    # chr_cigar contains 10S20M10N10M10S: 30 aligned bases, intron + clips uncovered.
    out = str(tmp_path / "c.bw")
    _coverage.bam_coverage_bigwig(CIGAR, out, bin_size=1, aggregation="mean",
                                  filter_mode="deeptools")
    L = _chrom_len(CIGAR, "chr_cigar")
    assert np.array_equal(_values(out, "chr_cigar", L),
                          _gt_perbase_depth(CIGAR, "chr_cigar").astype(float))


# --- read models -----------------------------------------------------------

def test_extend_counts_fragment_once(tmp_path):
    out = str(tmp_path / "e.bw")
    ext = 300
    _coverage.bam_coverage_bigwig(PAIRED, out, bin_size=1, aggregation="sum",
                                  filter_mode="deeptools", extend_reads=ext)
    L = _chrom_len(PAIRED, "chr2")
    assert np.allclose(_values(out, "chr2", L),
                       _gt_fragment_depth(PAIRED, "chr2", ext).astype(float), atol=1e-4)


def test_extended_read_count_uses_alignment_weight(tmp_path):
    out = str(tmp_path / "weighted.bw")
    extension = 300
    _coverage.bam_coverage_bigwig(
        PAIRED, out, bin_size=1, aggregation="count",
        filter_mode="deeptools", extend_reads=extension)
    length = _chrom_len(PAIRED, "chr2")
    assert np.array_equal(
        _values(out, "chr2", length),
        _gt_fragment_read_weight_depth(
            PAIRED, "chr2", extension).astype(float))


@pytest.mark.parametrize("mode", ["5prime", "3prime", "center"])
@pytest.mark.parametrize("length", [1, 20, 500])
def test_collapse_matches_ground_truth(tmp_path, mode, length):
    out = str(tmp_path / f"col_{mode}_{length}.bw")
    _coverage.bam_coverage_bigwig(PAIRED, out, bin_size=1, aggregation="sum",
                                  filter_mode="deeptools", collapse=mode,
                                  collapsed_length=length)
    L = _chrom_len(PAIRED, "chr2")
    d = np.zeros(L, dtype=np.int64)
    seen = set()
    for r in _primary_reads(PAIRED, "chr2"):
        proper = (r.flag & 0x2) and not (r.flag & 0x8) and \
            r.next_reference_id == r.reference_id and r.template_length != 0
        if proper:
            fs = min(r.reference_start, r.next_reference_start)
            fe = fs + abs(r.template_length)
            fragment = (r.query_name, fs, fe)
            if fragment in seen:
                continue
            seen.add(fragment)
        else:
            fs, fe = r.reference_start, r.reference_end
        fwd = not (r.mate_is_reverse if proper and r.template_length < 0 else r.is_reverse)
        if mode == "center":
            c = (fs + fe) // 2
            a, b = c - length // 2, c - length // 2 + length
        else:
            anchor_left = fwd if mode == "5prime" else (not fwd)
            a, b = (fs, fs + length) if anchor_left else (fe - length, fe)
        a, b = max(a, fs), min(b, fe)
        a, b = max(0, a), min(L, b)
        if b > a:
            d[a:b] += 1
    assert np.array_equal(_values(out, "chr2", L), d.astype(float))


# --- strand ----------------------------------------------------------------

def test_split_partitions_unstranded(tmp_path):
    plus = str(tmp_path / "p.bw")
    minus = str(tmp_path / "m.bw")
    uns = str(tmp_path / "u.bw")
    common = dict(bin_size=50, aggregation="count", filter_mode="deeptools",
                  strandedness="reverse")
    _coverage.bam_coverage_bigwig(PAIRED, plus, minus, filter_rna_strand="split", **common)
    _coverage.bam_coverage_bigwig(PAIRED, uns, filter_rna_strand="none", **common)
    L = _chrom_len(PAIRED, "chr2")
    p = _perbin(plus, "chr2", L, 50)
    m = _perbin(minus, "chr2", L, 50)
    u = _perbin(uns, "chr2", L, 50)
    assert np.array_equal(p + m, u)  # complete partition, nothing lost or doubled


def test_split_equals_separate_strand_runs(tmp_path):
    plus = str(tmp_path / "p.bw")
    minus = str(tmp_path / "m.bw")
    fwd = str(tmp_path / "f.bw")
    rev = str(tmp_path / "r.bw")
    common = dict(bin_size=50, aggregation="count", filter_mode="deeptools",
                  strandedness="reverse")
    _coverage.bam_coverage_bigwig(PAIRED, plus, minus, filter_rna_strand="split", **common)
    _coverage.bam_coverage_bigwig(PAIRED, fwd, filter_rna_strand="forward", **common)
    _coverage.bam_coverage_bigwig(PAIRED, rev, filter_rna_strand="reverse", **common)
    L = _chrom_len(PAIRED, "chr2")
    assert np.array_equal(_perbin(plus, "chr2", L, 50), _perbin(fwd, "chr2", L, 50))
    assert np.array_equal(_perbin(minus, "chr2", L, 50), _perbin(rev, "chr2", L, 50))


# --- normalization (the correctness fix) -----------------------------------

def _implied_D(tmp_path, denom, frs):
    raw = str(tmp_path / f"raw_{denom}_{frs}.bw")
    cpm = str(tmp_path / f"cpm_{denom}_{frs}.bw")
    common = dict(bin_size=50, aggregation="count", filter_mode="deeptools",
                  strandedness="reverse", filter_rna_strand=frs)
    _coverage.bam_coverage_bigwig(PAIRED, raw, **common)
    _coverage.bam_coverage_bigwig(PAIRED, cpm, normalization="cpm",
                                  normalization_denominator=denom, **common)
    L = _chrom_len(PAIRED, "chr2")
    r = _perbin(raw, "chr2", L, 50)
    c = _perbin(cpm, "chr2", L, 50)
    nz = r > 0
    return float(np.median(r[nz] * 1e6 / c[nz]))


def test_library_denominator_is_filter_independent(tmp_path):
    # The fix: library size is the same for forward and reverse strand filtering.
    df = _implied_D(tmp_path, "library", "forward")
    dr = _implied_D(tmp_path, "library", "reverse")
    assert np.isclose(df, dr)


def test_filtered_denominator_depends_on_strand(tmp_path):
    # The old behaviour (opt-in): filtered denominator differs by strand.
    ff = _implied_D(tmp_path, "filtered", "forward")
    fr = _implied_D(tmp_path, "filtered", "reverse")
    assert not np.isclose(ff, fr)
    # the two strand subsets partition the library
    assert np.isclose(ff + fr, _implied_D(tmp_path, "library", "none"))


def test_cpm_scale(tmp_path):
    raw = str(tmp_path / "raw.bw")
    cpm = str(tmp_path / "cpm.bw")
    st = _coverage.bam_index_stats(PAIRED)
    D = st["total_mapped"]
    _coverage.bam_coverage_bigwig(PAIRED, raw, bin_size=50, aggregation="count",
                                  filter_mode="deeptools")
    _coverage.bam_coverage_bigwig(PAIRED, cpm, bin_size=50, aggregation="count",
                                  filter_mode="deeptools", normalization="cpm")
    L = _chrom_len(PAIRED, "chr2")
    r = _perbin(raw, "chr2", L, 50)
    c = _perbin(cpm, "chr2", L, 50)
    nz = r > 0
    assert np.allclose(c[nz], r[nz] * 1e6 / D, rtol=1e-5)


def test_rpkm_uses_each_bins_actual_width(tmp_path):
    raw = str(tmp_path / "raw-rpkm.bw")
    rpkm = str(tmp_path / "rpkm.bw")
    bin_size = 137
    denominator = _coverage.bam_index_stats(SINGLE)["total_mapped"]
    common = dict(bin_size=bin_size, aggregation="count",
                  filter_mode="deeptools")
    _coverage.bam_coverage_bigwig(SINGLE, raw, **common)
    _coverage.bam_coverage_bigwig(
        SINGLE, rpkm, normalization="rpkm", **common)
    length = _chrom_len(SINGLE, "3R")
    counts = _perbin(raw, "3R", length, bin_size)
    observed = _perbin(rpkm, "3R", length, bin_size)
    widths = np.full(counts.size, bin_size, dtype=float)
    widths[-1] = length - bin_size * (counts.size - 1)
    np.testing.assert_allclose(
        observed, counts * 1e9 / (denominator * widths), rtol=2e-5)


def test_native_rejects_normalization_with_coverage_aggregation(tmp_path):
    with pytest.raises(RuntimeError, match="read-count aggregation"):
        _coverage.bam_coverage_bigwig(
            PAIRED, str(tmp_path / "invalid.bw"), aggregation="mean",
            normalization="cpm")


def test_native_rejects_removed_rpgc_normalization(tmp_path):
    with pytest.raises(RuntimeError, match="normalization must be"):
        _coverage.bam_coverage_bigwig(
            PAIRED, str(tmp_path / "invalid.bw"), aggregation="count",
            normalization="rpgc")


# --- region masks ----------------------------------------------------------

def test_whitelist_and_blacklist_masks(tmp_path):
    base = str(tmp_path / "b.bw")
    masked = str(tmp_path / "m.bw")
    _coverage.bam_coverage_bigwig(PAIRED, base, bin_size=50, aggregation="count",
                                  filter_mode="deeptools")
    _coverage.bam_coverage_bigwig(
        PAIRED, masked, bin_size=50, aggregation="count", filter_mode="deeptools",
        whitelist={"2": [(5000000, 5000600)]},
        blacklist={"2": [(5000300, 5000400)]})
    L = _chrom_len(PAIRED, "chr2")
    b = _perbin(base, "chr2", L, 50)
    m = _perbin(masked, "chr2", L, 50)
    w0, w1 = 5000000 // 50, (5000600 - 1) // 50
    k0, k1 = 5000300 // 50, (5000400 - 1) // 50
    assert np.all(m[:w0] == 0) and np.all(m[w1 + 1:] == 0)      # outside whitelist zeroed
    assert np.all(m[k0:k1 + 1] == 0)                            # blacklisted subregion zeroed
    assert np.array_equal(m[w0:k0], b[w0:k0])                   # kept region unchanged


@pytest.mark.parametrize("overlap", [False, True])
def test_empty_whitelist_retains_nothing(tmp_path, overlap):
    out = str(tmp_path / ("empty_touch.bw" if overlap else "empty_mask.bw"))
    _coverage.bam_coverage_bigwig(
        PAIRED, out, bin_size=50, aggregation="sum",
        filter_mode="deeptools", whitelist={}, filter_by_overlap=overlap)
    length = _chrom_len(PAIRED, "chr2")
    assert np.all(_perbin(out, "chr2", length, 50) == 0)


def test_whitelist_on_different_chromosome_retains_nothing(tmp_path):
    out = str(tmp_path / "mismatch.bw")
    _coverage.bam_coverage_bigwig(
        PAIRED, out, bin_size=50, aggregation="sum",
        filter_mode="deeptools", whitelist={"not_chr2": [(0, 100)]},
        filter_by_overlap=True)
    length = _chrom_len(PAIRED, "chr2")
    assert np.all(_perbin(out, "chr2", length, 50) == 0)


# --- parallel == serial (determinism) --------------------------------------

CONFIGS = [
    dict(bin_size=1, aggregation="mean"),
    dict(bin_size=50, aggregation="count"),
    dict(bin_size=7, aggregation="sum"),
    dict(bin_size=50, aggregation="mean", extend_reads=300),
    dict(bin_size=1, aggregation="sum", collapse="5prime", collapsed_length=20),
    dict(bin_size=50, aggregation="count", min_mapping_quality=10, sam_flag_exclude=16),
    dict(bin_size=1, aggregation="count", normalization="cpm"),
]


@pytest.mark.parametrize("bam,chrom", [(PAIRED, "chr2"), (SINGLE, "3R"), (CIGAR, "3R")])
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("threads,window", [(4, 0), (8, 7), (3, 1)])
def test_parallel_matches_serial(tmp_path, bam, chrom, cfg, threads, window):
    ser = str(tmp_path / "s.bw")
    par = str(tmp_path / "p.bw")
    _coverage.bam_coverage_bigwig(bam, ser, filter_mode="deeptools", threads=1,
                                  window_size=0, **cfg)
    _coverage.bam_coverage_bigwig(bam, par, filter_mode="deeptools", threads=threads,
                                  window_size=window, **cfg)
    L = _chrom_len(bam, chrom)
    assert np.array_equal(_values(ser, chrom, L), _values(par, chrom, L))


# --- P8: fragment-touch overlap filter -------------------------------------

def _touch_qnames(bam, chrom, region):
    tq = set()
    for r in _primary_reads(bam, chrom):
        for bs, be in r.get_blocks():
            if bs < region[1] and be > region[0]:
                tq.add(r.query_name)
                break
    return tq


def _perbase_for_qnames(bam, chrom, qnames):
    L = _chrom_len(bam, chrom)
    d = np.zeros(L, dtype=np.int64)
    for r in _primary_reads(bam, chrom):
        if r.query_name not in qnames:
            continue
        for bs, be in r.get_blocks():
            d[bs:be] += 1
    return d


def test_overlap_whitelist_keep_touch(tmp_path):
    region = (5000100, 5000300)
    out = str(tmp_path / "w.bw")
    _coverage.bam_coverage_bigwig(PAIRED, out, bin_size=1, aggregation="sum",
                                  filter_mode="deeptools", filter_by_overlap=True,
                                  whitelist={"chr2": [region]})
    L = _chrom_len(PAIRED, "chr2")
    keep = _touch_qnames(PAIRED, "chr2", region)
    assert np.array_equal(_values(out, "chr2", L),
                          _perbase_for_qnames(PAIRED, "chr2", keep).astype(float))


def test_overlap_blacklist_drop_touch(tmp_path):
    region = (5000100, 5000300)
    out = str(tmp_path / "b.bw")
    _coverage.bam_coverage_bigwig(PAIRED, out, bin_size=1, aggregation="sum",
                                  filter_mode="deeptools", filter_by_overlap=True,
                                  blacklist={"chr2": [region]})
    L = _chrom_len(PAIRED, "chr2")
    allq = {r.query_name for r in _primary_reads(PAIRED, "chr2")}
    keep = allq - _touch_qnames(PAIRED, "chr2", region)
    assert np.array_equal(_values(out, "chr2", L),
                          _perbase_for_qnames(PAIRED, "chr2", keep).astype(float))


def test_overlap_is_pair_aware(tmp_path):
    # A kept fragment's non-touching mate still contributes coverage outside the region.
    region = (5000100, 5000300)
    out = str(tmp_path / "p.bw")
    _coverage.bam_coverage_bigwig(PAIRED, out, bin_size=1, aggregation="sum",
                                  filter_mode="deeptools", filter_by_overlap=True,
                                  whitelist={"chr2": [region]})
    L = _chrom_len(PAIRED, "chr2")
    v = _values(out, "chr2", L)
    assert (v[:region[0]] > 0).any() or (v[region[1]:] > 0).any()


def test_overlap_configurable_halo_keeps_long_pair_across_windows(tmp_path):
    bam = str(tmp_path / 'long_pair.bam')
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'},
              'SQ': [{'SN': 'chr1', 'LN': 1000}]}
    records = []
    for pos, mate_pos, flag, tlen in (
            (90, 250, 99, 210), (250, 90, 147, -210)):
        read = pysam.AlignedSegment()
        read.query_name = 'long-pair'
        read.query_sequence = 'A' * 50
        read.query_qualities = pysam.qualitystring_to_array('I' * 50)
        read.flag = flag
        read.reference_id = 0
        read.reference_start = pos
        read.mapping_quality = 60
        read.cigar = [(0, 50)]
        read.next_reference_id = 0
        read.next_reference_start = mate_pos
        read.template_length = tlen
        records.append(read)
    with pysam.AlignmentFile(bam, 'wb', header=header) as handle:
        for read in records:
            handle.write(read)
    pysam.index(bam)

    short = str(tmp_path / 'short.bw')
    enough = str(tmp_path / 'enough.bw')
    common = dict(bin_size=1, aggregation='count', window_size=100,
                  filter_by_overlap=True,
                  whitelist={'chr1': [(90, 100)]}, max_zooms=0)
    _coverage.bam_coverage_bigwig(
        bam, short, filter_by_overlap_halo=50, **common)
    _coverage.bam_coverage_bigwig(
        bam, enough, filter_by_overlap_halo=300, **common)

    assert _values(short, 'chr1', 1000)[250] == 1
    assert _values(enough, 'chr1', 1000)[250] == 1
    np.testing.assert_array_equal(_values(short, 'chr1', 1000),
                                  _values(enough, 'chr1', 1000))


def test_overlap_is_splice_aware(tmp_path):
    # chr_cigar read is 10S20M10N10M10S: whitelisting the N gap must NOT keep it.
    r = next(iter(pysam.AlignmentFile(CIGAR, "rb").fetch("chr_cigar")))
    blocks = r.get_blocks()
    gap = (blocks[0][1], blocks[1][0])
    L = _chrom_len(CIGAR, "chr_cigar")
    out = str(tmp_path / "s.bw")
    _coverage.bam_coverage_bigwig(CIGAR, out, bin_size=1, aggregation="sum",
                                  filter_mode="deeptools", filter_by_overlap=True,
                                  whitelist={"chr_cigar": [gap]})
    assert np.all(_values(out, "chr_cigar", L) == 0)          # spliced over -> dropped
    out2 = str(tmp_path / "s2.bw")
    _coverage.bam_coverage_bigwig(CIGAR, out2, bin_size=1, aggregation="sum",
                                  filter_mode="deeptools", filter_by_overlap=True,
                                  whitelist={"chr_cigar": [(blocks[0][0], blocks[0][0] + 5)]})
    assert (_values(out2, "chr_cigar", L) > 0).any()           # aligned block -> kept


@pytest.mark.parametrize("threads,window", [(4, 0), (8, 7), (3, 1)])
def test_overlap_parallel_matches_serial(tmp_path, threads, window):
    region = (5000100, 5000400)
    kw = dict(bin_size=1, aggregation="mean", filter_mode="deeptools",
              filter_by_overlap=True, whitelist={"chr2": [region]})
    ser = str(tmp_path / "s.bw")
    par = str(tmp_path / "p.bw")
    _coverage.bam_coverage_bigwig(PAIRED, ser, threads=1, window_size=0, **kw)
    _coverage.bam_coverage_bigwig(PAIRED, par, threads=threads, window_size=window, **kw)
    L = _chrom_len(PAIRED, "chr2")
    assert np.array_equal(_values(ser, "chr2", L), _values(par, "chr2", L))


# --- BPM normalization -----------------------------------------------------

def test_bpm_sums_to_million(tmp_path):
    raw = str(tmp_path / "raw.bw")
    bpm = str(tmp_path / "bpm.bw")
    _coverage.bam_coverage_bigwig(PAIRED, raw, bin_size=50, aggregation="count",
                                  filter_mode="deeptools")
    _coverage.bam_coverage_bigwig(PAIRED, bpm, bin_size=50, aggregation="count",
                                  filter_mode="deeptools", normalization="bpm")
    L = _chrom_len(PAIRED, "chr2")
    r = _perbin(raw, "chr2", L, 50)
    b = _perbin(bpm, "chr2", L, 50)
    assert np.allclose(b, r * 1e6 / r.sum(), rtol=1e-4)
    assert abs(b.sum() - 1e6) < 1.0


def test_bpm_parallel_matches_serial(tmp_path):
    s = str(tmp_path / "s.bw")
    m = str(tmp_path / "m.bw")
    kw = dict(bin_size=1, aggregation="count", filter_mode="deeptools", normalization="bpm")
    _coverage.bam_coverage_bigwig(PAIRED, s, threads=1, **kw)
    _coverage.bam_coverage_bigwig(PAIRED, m, threads=4, window_size=7, **kw)
    L = _chrom_len(PAIRED, "chr2")
    assert np.array_equal(_values(s, "chr2", L), _values(m, "chr2", L))


@pytest.mark.parametrize("normalization", ["none", "cpm", "bpm"])
def test_scale_factor_minus_is_additional(tmp_path, normalization):
    common = dict(bin_size=50, aggregation="count", filter_mode="deeptools",
                  strandedness="reverse", filter_rna_strand="split",
                  normalization=normalization)
    p0 = str(tmp_path / "p0.bw")
    m0 = str(tmp_path / "m0.bw")
    _coverage.bam_coverage_bigwig(PAIRED, p0, m0, **common)
    p1 = str(tmp_path / "p1.bw")
    m1 = str(tmp_path / "m1.bw")
    _coverage.bam_coverage_bigwig(PAIRED, p1, m1, scale_factor=2.0,
                                  scale_factor_minus=-3.0, **common)
    L = _chrom_len(PAIRED, "chr2")
    assert np.allclose(_perbin(p1, "chr2", L, 50),
                       2 * _perbin(p0, "chr2", L, 50))
    assert np.allclose(_perbin(m1, "chr2", L, 50),
                       -6 * _perbin(m0, "chr2", L, 50))


def test_bpm_split_pooled_denominator(tmp_path):
    plus = str(tmp_path / "p.bw")
    minus = str(tmp_path / "m.bw")
    _coverage.bam_coverage_bigwig(
        PAIRED, plus, minus, bin_size=50, aggregation="count", filter_mode="deeptools",
        strandedness="reverse", filter_rna_strand="split", normalization="bpm")
    L = _chrom_len(PAIRED, "chr2")
    total = _perbin(plus, "chr2", L, 50).sum() + _perbin(minus, "chr2", L, 50).sum()
    assert abs(total - 1e6) < 1.0  # pooled denominator -> the two tracks share it


# --- P8b: strand-oriented touch (--featureTouchStrand) ---------------------

def _tf_reverse(r):  # transcript strand under reverse (dUTP) library
    read_sign = -1 if r.is_reverse else 1
    is_r2 = (r.flag & 0x1) and (r.flag & 0x80)
    return read_sign if is_r2 else -read_sign


def _touchstrand_gt(region, feat_strand, mode):
    keep = set()
    for r in _primary_reads(PAIRED, "chr2"):
        if not any(bs < region[1] and be > region[0] for bs, be in r.get_blocks()):
            continue
        fs = _tf_reverse(r)
        ok = {"ignore": True, "sense": fs == feat_strand,
              "antisense": fs == -feat_strand}[mode]
        if ok:
            keep.add(r.query_name)
    return _perbase_for_qnames(PAIRED, "chr2", keep)


@pytest.mark.parametrize("feat", [1, -1])
@pytest.mark.parametrize("mode", ["ignore", "sense", "antisense"])
def test_feature_touch_strand(tmp_path, feat, mode):
    region = (5000100, 5000300)
    out = str(tmp_path / "t.bw")
    _coverage.bam_coverage_bigwig(
        PAIRED, out, bin_size=1, aggregation="sum", filter_mode="deeptools",
        whitelist={"chr2": [(region[0], region[1], feat)]}, filter_by_overlap=True,
        strandedness="reverse", feature_touch_strand=mode)
    L = _chrom_len(PAIRED, "chr2")
    assert np.array_equal(_values(out, "chr2", L),
                          _touchstrand_gt(region, feat, mode).astype(float))


# --- P9: strandedness inference --------------------------------------------

def _tf_forward(r):
    read_sign = -1 if r.is_reverse else 1
    is_r2 = (r.flag & 0x1) and (r.flag & 0x80)
    return -read_sign if is_r2 else read_sign


@pytest.mark.parametrize("gene", [1, -1])
def test_infer_strandedness_matches_ground_truth(gene):
    reg = (5000000, 5010000)
    fwd = rev = 0
    for r in _primary_reads(PAIRED, "chr2"):
        if r.reference_start >= reg[1] or r.reference_end <= reg[0]:
            continue
        if _tf_forward(r) == gene:
            fwd += 1
        else:
            rev += 1
    res = _coverage.infer_strandedness(PAIRED, [("2", reg[0], reg[1], gene)], 0, False)
    assert res["forward"] == fwd and res["reverse"] == rev
    assert res["total"] == fwd + rev
    # flipping the gene strand flips forward/reverse
    flip = _coverage.infer_strandedness(PAIRED, [("chr2", reg[0], reg[1], -gene)], 0, False)
    assert flip["forward"] == res["reverse"] and flip["reverse"] == res["forward"]


def test_infer_strandedness_empty():
    res = _coverage.infer_strandedness(PAIRED, [("chr2", 0, 10, 1)], 0, False)
    assert res["total"] == 0 and res["forward_fraction"] == 0.0


@pytest.mark.parametrize("probe_chrom", ["chr2", "2"])
def test_cli_strandedness_file(tmp_path, probe_chrom):
    from deeptoolsr import bamCoverage
    bed = tmp_path / "probes.bed"
    bed.write_text(f"{probe_chrom}\t5000000\t5010000\tg\t0\t-\n")
    out = str(tmp_path / "o.bw")
    # inference returns "none" on this mixed synthetic data -> plain run succeeds
    bamCoverage.main(["-b", PAIRED, "-o", out, "-bs", "50",
                      "--normalizeUsing", "read-count",
                      "--filterMode", "deeptools", "--strandedness", "file=" + str(bed)])
    assert os.path.exists(out)


def test_strandedness_colon_alias_is_rejected(tmp_path):
    from deeptoolsr import bamCoverage
    with pytest.raises(SystemExit, match='file=BED'):
        bamCoverage.main([
            '-b', PAIRED, '-o', str(tmp_path / 'old.bw'),
            '--strandedness', 'file:' + str(tmp_path / 'probes.bed')])


def test_bam_bed_chromosome_aliases_are_exact_first(tmp_path):
    from deeptoolsr import bamCoverage
    bed = tmp_path / "regions.bed"
    bed.write_text("1\t0\t5\nchr1\t5\t10\nMT\t0\t5\n")
    parsed = bamCoverage._parse_blacklist(
        bed, {"1", "chr1", "chrM"})
    assert parsed == {
        "1": [(0, 5, 0)],
        "chr1": [(5, 10, 0)],
        "chrM": [(0, 5, 0)],
    }


def test_unresolved_bed_chromosomes_warn_once_and_remain_empty(
        tmp_path, capsys):
    from deeptoolsr import bamCoverage
    bed = tmp_path / 'regions.bed'
    bed.write_text(
        'missing	0	10\nmissing	20	30\n'
        'also_missing	0	10\tgene\t0\t+\n')

    assert bamCoverage._parse_blacklist(bed, {'chr1'}) == {}
    warning = capsys.readouterr().err
    assert 'omitted unresolved chromosome names' in warning
    assert warning.count('missing') == 2  # one within also_missing, one distinct name

    assert bamCoverage._parse_probe_bed(bed, {'chr1'}) == []
    warning = capsys.readouterr().err
    assert 'omitted unresolved chromosome name' in warning
    assert 'also_missing' in warning


def test_unresolved_ignore_for_normalization_warns(tmp_path, capsys):
    from deeptoolsr import bamCoverage
    output = tmp_path / 'out.bw'
    bamCoverage.main([
        '-b', PAIRED, '-o', str(output), '--binSize', '10000000',
        '--normalizeUsing', 'CPM', '--ignoreForNormalization', 'typoChrom'])
    assert output.exists()
    warning = capsys.readouterr().err
    assert '--ignoreForNormalization' in warning
    assert 'typoChrom' in warning


def test_probe_regions_merge_duplicates_and_remove_opposite_overlap(tmp_path):
    from deeptoolsr import bamCoverage
    bed = tmp_path / "probes.bed"
    bed.write_text(
        "1\t10\t30\ta\t0\t+\n"
        "chr1\t20\t40\tb\t0\t+\n"
        "chr1\t25\t35\tc\t0\t-\n"
    )
    assert bamCoverage._parse_probe_bed(bed, {"chr1"}) == [
        ("chr1", 10, 25, 1),
        ("chr1", 35, 40, 1),
    ]


# --- bedGraph output -------------------------------------------------------

def _bg_perbase(text, chrom, L):
    v = np.zeros(L)
    for line in text.strip().split("\n"):
        c, s, e, val = line.split("\t")
        if c == chrom:
            v[int(s):int(e)] = float(val)
    return v


def test_bedgraph_matches_bigwig(tmp_path):
    L = _chrom_len(PAIRED, "chr2")
    bw = str(tmp_path / "o.bw")
    bg = str(tmp_path / "o.bedgraph")
    _coverage.bam_coverage_bigwig(PAIRED, bw, "", "bigwig", bin_size=1,
                                  aggregation="sum", filter_mode="deeptools")
    _coverage.bam_coverage_bigwig(PAIRED, bg, "", "bedgraph", bin_size=1,
                                  aggregation="sum", filter_mode="deeptools")
    with open(bg) as fh:
        assert np.array_equal(_bg_perbase(fh.read(), "chr2", L), _values(bw, "chr2", L))


def test_bedgraph_gz_matches_bigwig(tmp_path):
    import gzip
    L = _chrom_len(PAIRED, "chr2")
    bw = str(tmp_path / "o.bw")
    gz = str(tmp_path / "o.bedgraph.gz")
    _coverage.bam_coverage_bigwig(PAIRED, bw, "", "bigwig", bin_size=1,
                                  aggregation="sum", filter_mode="deeptools")
    _coverage.bam_coverage_bigwig(PAIRED, gz, "", "bedgraph.gz", bin_size=1,
                                  aggregation="sum", filter_mode="deeptools")
    with gzip.open(gz, "rt") as fh:
        assert np.array_equal(_bg_perbase(fh.read(), "chr2", L), _values(bw, "chr2", L))


def test_bedgraph_parallel_matches_serial(tmp_path):
    s = str(tmp_path / "s.bedgraph")
    m = str(tmp_path / "m.bedgraph")
    kw = dict(bin_size=1, aggregation="count", filter_mode="deeptools")
    _coverage.bam_coverage_bigwig(PAIRED, s, "", "bedgraph", threads=1, **kw)
    _coverage.bam_coverage_bigwig(PAIRED, m, "", "bedgraph", threads=4, window_size=7, **kw)
    assert open(s).read() == open(m).read()


def test_cli_bedgraph_and_gz(tmp_path):
    from deeptoolsr import bamCoverage
    bg = str(tmp_path / "o.bedgraph")
    bamCoverage.main(["-b", PAIRED, "-o", bg, "-of", "bedgraph", "-bs", "50",
                      "--filterMode", "deeptools"])
    assert os.path.exists(bg) and os.path.getsize(bg) > 0
    gz = str(tmp_path / "o.bedgraph.gz")
    bamCoverage.main(["-b", PAIRED, "-o", gz, "-of", "bedgraph", "-bs", "50",
                      "--filterMode", "deeptools"])
    import gzip
    assert gzip.open(gz, "rt").read() == open(bg).read()


# --- CLI -------------------------------------------------------------------

def test_cli_exposes_one_semantic_metric_selector():
    from deeptoolsr import bamCoverage
    parser = bamCoverage.parse_arguments()
    args = parser.parse_args(["-b", PAIRED, "-o", "out.bw"])
    assert args.normalizeUsing == "coverage-mean"
    assert not hasattr(args, "binAggregation")
    with pytest.raises(SystemExit):
        parser.parse_args([
            "-b", PAIRED, "-o", "out.bw", "--binAggregation", "count"])


def test_cli_runs_and_splits(tmp_path):
    from deeptoolsr import bamCoverage
    out = str(tmp_path / "cli.bw")
    bamCoverage.main(["-b", PAIRED, "-o", out, "-bs", "50",
                      "--normalizeUsing", "read-count",
                      "--filterMode", "deeptools"])
    assert os.path.exists(out)
    sp = str(tmp_path / "s.bw")
    bamCoverage.main(["-b", PAIRED, "-o", sp, "--filterRNAstrand", "split",
                      "--strandedness", "reverse", "--suffix", ".plus", ".minus"])
    assert os.path.exists(str(tmp_path / "s.plus.bw"))
    assert os.path.exists(str(tmp_path / "s.minus.bw"))


def test_cli_split_without_strandedness_errors(tmp_path):
    from deeptoolsr import bamCoverage
    with pytest.raises(SystemExit):
        bamCoverage.main(["-b", PAIRED, "-o", str(tmp_path / "x.bw"),
                          "--filterRNAstrand", "split"])


# From audit regressions.

@pytest.fixture()
def io_native_backends(monkeypatch):
    monkeypatch.delenv('DTP_BAM_BACKEND', raising=False)


@pytest.mark.usefixtures('io_native_backends')
def test_corrupt_bam_is_not_committed_as_success(tmp_path):
    source = tmp_path / 'corrupt.bam'
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': 'chr1', 'LN': 100000}]}
    with pysam.AlignmentFile(str(source), 'wb', header=header) as bam:
        for i in range(2000):
            record = pysam.AlignedSegment()
            record.query_name = f'r{i}'
            record.query_sequence = 'A' * 50
            record.flag = 0
            record.reference_id = 0
            record.reference_start = i * 20
            record.mapping_quality = 60
            record.cigar = [(0, 50)]
            bam.write(record)
    pysam.index(str(source))
    data = bytearray(source.read_bytes())
    offsets, pos = ([], 0)
    while pos < len(data):
        offsets.append(pos)
        pos += struct.unpack_from('<H', data, pos + 16)[0] + 1
    data[offsets[2] + 20] ^= 255
    source.write_bytes(data)
    output = tmp_path / 'output.bedgraph'
    output.write_bytes(b'PREVIOUS VALID RESULT')
    with pytest.raises(SystemExit):
        bamCoverage.main(['-b', str(source), '-o', str(output), '--outFileFormat', 'bedgraph', '--binSize', '100'])
    assert output.read_bytes() == b'PREVIOUS VALID RESULT'


# From numerical audit regressions.

def write_bam(path, paired=False):
    chromosomes = ['chr1'] if paired else ['chr1', 'chr2']
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': chrom, 'LN': 100} for chrom in chromosomes]}
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        for i in range(2):
            read = pysam.AlignedSegment()
            read.query_name = 'pair' if paired else f'read{i}'
            read.query_sequence = 'A' * 10
            read.flag = [99, 147][i] if paired else 0
            read.reference_id = 0 if paired else i
            read.reference_start = [10, 50][i] if paired else 10
            read.mapping_quality = 60
            read.cigarstring = '10M'
            if paired:
                read.next_reference_id = 0
                read.next_reference_start = [50, 10][i]
                read.template_length = [50, -50][i]
            out.write(read)
    pysam.index(str(path))


def bw_values(path, chrom='chr1'):
    with pyBigWig.open(str(path)) as bw:
        return np.array(bw.values(chrom, 0, bw.chroms(chrom)))


@pytest.mark.parametrize('normalization', ['BPM', 'CPM'])
@pytest.mark.parametrize('threads', [1, 2])
def test_n02_ignore_chromosome_changes_normalization_denominator(tmp_path, normalization, threads):
    bam = tmp_path / 'two.bam'
    write_bam(bam)
    observed = []
    for ignore in (False, True):
        output = tmp_path / f'{normalization}{ignore}.bw'
        args = ['-b', str(bam), '-o', str(output), '--binSize', '10', '-p', str(threads), '--normalizeUsing', normalization]
        if ignore:
            args += ['--ignoreForNormalization', '2']
        bamCoverage.main(args)
        observed.append(bw_values(output)[10])
        assert bw_values(output, 'chr2')[10] == observed[-1]
    assert observed == [500000, 1000000]


def test_native_fragment_estimation_uses_tlen(tmp_path):
    backend = numerical__coverage
    bam = tmp_path / 'paired.bam'
    write_bam(bam, paired=True)
    assert backend.estimate_fragment_length(str(bam)) == 50
    with pytest.raises(RuntimeError, match='sample must be > 0'):
        backend.estimate_fragment_length(str(bam), sample=0)


@pytest.mark.parametrize('threads', [1, 2])
def test_n02_bpm_exclusions_share_denominator_across_split_tracks(tmp_path, threads):
    backend = numerical__coverage
    bam, plus, minus = (tmp_path / 'split.bam', tmp_path / 'plus.bw', tmp_path / 'minus.bw')
    with pysam.AlignmentFile(str(bam), 'wb', header={'HD': {'SO': 'coordinate'}, 'SQ': [{'SN': chrom, 'LN': 100} for chrom in ('chr1', 'chr2')]}) as out:
        for chrom in range(2):
            for reverse in (False, True):
                read = pysam.AlignedSegment()
                read.query_name = f'read{chrom}{reverse}'
                read.query_sequence = 'A' * 10
                read.reference_id = chrom
                read.reference_start = 30 if reverse else 10
                read.cigarstring = '10M'
                read.flag = 16 if reverse else 0
                out.write(read)
    pysam.index(str(bam))
    backend.bam_coverage_bigwig(str(bam), str(plus), str(minus), bin_size=10, aggregation='count', normalization='bpm', strandedness='forward', filter_rna_strand='split', ignore_for_normalization=['2'], threads=threads, window_size=2, scale_factor=3, scale_factor_minus=-2)
    for chrom in ('chr1', 'chr2'):
        expected = np.zeros(100)
        expected[10:20] = 3 * 1000000.0 / 2
        np.testing.assert_array_equal(bw_values(plus, chrom), expected)
        expected[:] = 0
        expected[30:40] = -6 * 1000000.0 / 2
        np.testing.assert_array_equal(bw_values(minus, chrom), expected)


@pytest.mark.parametrize('split', [False, True])
def test_n02_bpm_empty_denominator_preserves_existing_outputs(tmp_path, split):
    bam = tmp_path / 'two.bam'
    write_bam(bam)
    base = tmp_path / 'output.bw'
    paths = [tmp_path / 'output.plus.bw', tmp_path / 'output.minus.bw'] if split else [base]
    for path in paths:
        path.write_bytes(b'previous valid coverage')
    args = ['-b', str(bam), '-o', str(base), '--normalizeUsing', 'BPM', '--ignoreForNormalization', 'chr1', 'chr2', '-p', '2']
    if split:
        args += ['--strandedness', 'forward', '--filterRNAstrand', 'split']
    before = set(tmp_path.iterdir())
    with pytest.raises(SystemExit, match='BPM denominator is zero'):
        bamCoverage.main(args)
    for path in paths:
        assert path.read_bytes() == b'previous valid coverage'
    assert set(tmp_path.iterdir()) == before


# Argument contract.

TEST_DATA = Path(__file__).parent / 'test_data'


CONTRACT_PAIRED = TEST_DATA / 'test_paired.bam'


def test_bam_collapse_extension_conflict_is_rejected_before_output(tmp_path, capsys):
    from deeptoolsr import bamCoverage
    output = tmp_path / 'out.bw'
    with pytest.raises(SystemExit):
        bamCoverage.main(['-b', str(CONTRACT_PAIRED), '-o', str(output), '--extendReads', '200', '--collapse', 'center'])
    assert 'cannot be used together' in capsys.readouterr().err
    assert not output.exists()


def test_direct_native_bam_collapse_extension_conflict(tmp_path):
    native = pytest.importorskip('deeptoolsr._coverage')
    output = tmp_path / 'out.bw'
    with pytest.raises(RuntimeError, match='cannot be used together'):
        native.bam_coverage_bigwig(str(CONTRACT_PAIRED), str(output), extend_reads=200, collapse='center')
    assert not output.exists()


@pytest.mark.parametrize('extra,message', [(['--collapsedLength', '5'], 'unused without --collapse'), (['--scaleFactorMinus', '-1'], 'unused unless --filterRNAstrand split'), (['--suffix', '.a', '.b'], 'unused unless --filterRNAstrand split'), (['--normalizationDenominator', 'filtered'], 'unused unless --normalizeUsing is CPM or RPKM'), (['--normalizeUsing', 'BPM', '--exactScaling'], 'unused unless --normalizeUsing is CPM or RPKM'), (['--normalizeUsing', 'CPM', '--normalizationDenominator', 'filtered', '--exactScaling'], 'redundant with --normalizationDenominator filtered'), (['--ignoreForNormalization', 'chrM'], 'unused for unnormalized coverage metrics')])
def test_explicit_inert_bam_options_warn(tmp_path, monkeypatch, capsys, extra, message):
    from deeptoolsr import _coverage, bamCoverage

    def empty_output(_bam, output, *_args, **_kwargs):
        Path(output).write_bytes(b'')
    monkeypatch.setattr(_coverage, 'bam_coverage_bigwig', empty_output)
    bamCoverage.main(['-b', str(CONTRACT_PAIRED), '-o', str(tmp_path / 'out.bw')] + extra)
    assert message in capsys.readouterr().err
