"""bamCoverageR black/whitelists, read models and their edge cases on real reads.

encode_chip_slice (single-end ChIP-seq, chr1) and pointseq_slice (paired,
spliced POINT-seq, chr22) cover small read spans of full-length chromosomes.
Partial filters are checked per base against coverage computed here from the
reads with pysam.
"""

import collections
import os

import pytest

from deeptoolsr import _coverage, bamCoverage

pysam = pytest.importorskip("pysam")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
# Each slice's chromosome and its read span [start, end).
SPANS = {"encode_chip_slice": ("chr1", 10_000_220, 10_999_494),
         "pointseq_slice": ("chr22", 36_250_691, 36_349_977)}
RUNS = [("bam", 1), ("bam", 4), ("cram", 1), ("cram", 4)]
MODES = {"mask": [], "overlap": ["--filterByOverlap"]}


def coverage(tmp_path, capsys, name, kind="bam", threads=1, *options, bs=50):
    """Runs bamCoverageR to bedGraph; returns (output text, stderr WARNING lines)."""
    out = tmp_path / f"{name}-{kind}-{threads}-{len(list(tmp_path.iterdir()))}.bg"
    capsys.readouterr()
    bamCoverage.main(["-b", os.path.join(ROOT, f"{name}.{kind}"), "-o", str(out),
                      "-of", "bedgraph", "-bs", str(bs), "-p", str(threads), *options])
    warnings = [line for line in capsys.readouterr().err.splitlines()
                if line.startswith("WARNING:")]
    return out.read_text(), warnings


def bed(tmp_path, label, *rows):
    path = tmp_path / f"{label}.bed"
    path.write_text("".join("\t".join(map(str, row)) + "\n" for row in rows))
    return str(path)


def nonzero(text):
    return [line for line in text.splitlines() if float(line.split()[3])]


def per_base(text):
    values = {}
    for line in nonzero(text):
        _, start, end, value = line.split()
        values.update(dict.fromkeys(range(int(start), int(end)), float(value)))
    return values


def reads_by_pair(name):
    pairs = collections.defaultdict(list)
    with pysam.AlignmentFile(os.path.join(ROOT, f"{name}.bam")) as reads:
        for read in reads:
            pairs[read.query_name].append(read)
    return list(pairs.values())


def blocks(read):
    """Aligned blocks [start, end): M/=/X; D and N gap them."""
    out, pos = [], read.reference_start
    for op, length in read.cigartuples:
        if op in (0, 7, 8):
            out.append((pos, pos + length))
        if op in (0, 2, 3, 7, 8):
            pos += length
    return out


def bases(read):
    return {pos for start, end in blocks(read) for pos in range(start, end)}


def touches(read, regions):
    return any(start < r_end and r_start < end
               for start, end in blocks(read) for r_start, r_end in regions)


def depth(units):
    """Per-base depth: each unit is a set of covered positions."""
    counts = collections.Counter()
    for unit in units:
        counts.update(unit)
    return {pos: float(count) for pos, count in counts.items()}


def fragment(mates, dovetail=True):
    """[start, end) of a proper pair, counted as one fragment.

    Inward-facing: forward start to reverse end. Dovetailed (the reverse mate
    starts left of the forward mate and overlaps it): leftmost to rightmost
    aligned base, or None with dovetail=False (--countUnit read places its
    reads as single reads). None for mates that face outward without
    overlapping.
    """
    forward = [read for read in mates if not read.is_reverse]
    reverse = [read for read in mates if read.is_reverse]
    assert len(forward) == len(reverse) == 1 and forward[0].is_proper_pair
    forward, reverse = forward[0], reverse[0]
    if forward.reference_start <= reverse.reference_start:
        return forward.reference_start, reverse.reference_end
    if dovetail and reverse.reference_end > forward.reference_start:
        return reverse.reference_start, max(forward.reference_end, reverse.reference_end)
    return None


def transcript_strand(read):
    """--strandedness reverse: read2 is sense."""
    sign = -1 if read.is_reverse else 1
    return sign if read.is_read2 else -sign


def header_rows(name):
    return [(t["name"], 0, t["length"])
            for t in _coverage.bam_index_stats(os.path.join(ROOT, f"{name}.bam"))["targets"]]


# --- A: lists that remove nothing --------------------------------------------

def no_op_lists(tmp_path, name):
    chrom, start, _ = SPANS[name]
    return {
        "empty-blacklist": ["--blackListFileName", bed(tmp_path, "empty")],
        "blacklist-away": ["--blackListFileName", bed(tmp_path, "away", (chrom, 0, start - 1000))],
        "blacklist-unknown-chrom": ["--blackListFileName",
                                    bed(tmp_path, "unknown", ("chrUnknown", 0, 10**9))],
        "whole-chrom-whitelist": ["--whiteListFileName",
                                  bed(tmp_path, "whole", *header_rows(name))],
    }


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", SPANS)
def test_lists_that_remove_nothing_give_the_plain_output(tmp_path, capsys, name, mode):
    plain, _ = coverage(tmp_path, capsys, name)
    assert nonzero(plain)
    for label, options in no_op_lists(tmp_path, name).items():
        for kind, threads in RUNS:
            text, _ = coverage(tmp_path, capsys, name, kind, threads, *options, *MODES[mode])
            assert text == plain, (label, kind, threads)


# --- B: lists that remove everything warn once --------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", SPANS)
def test_lists_that_remove_everything_warn_once(tmp_path, capsys, name, mode):
    chrom = SPANS[name][0]
    cases = {
        "whole-chrom-blacklist": ["--blackListFileName",
                                  bed(tmp_path, "black", *header_rows(name))],
        "whitelist-without-reads": ["--whiteListFileName", bed(tmp_path, "white", (chrom, 100, 200))],
    }
    for label, options in cases.items():
        for kind, threads in RUNS:
            text, warnings = coverage(tmp_path, capsys, name, kind, threads, *options, *MODES[mode])
            assert text and not nonzero(text), (label, kind, threads)
            assert len(warnings) == 1, (label, kind, threads, warnings)
            assert options[0] in warnings[0] and "removed all coverage" in warnings[0]


@pytest.mark.parametrize("name", SPANS)
def test_both_lists_are_named_in_the_warning(tmp_path, capsys, name):
    chrom = SPANS[name][0]
    _, warnings = coverage(
        tmp_path, capsys, name, "bam", 1,
        "--blackListFileName", bed(tmp_path, "black", *header_rows(name)),
        "--whiteListFileName", bed(tmp_path, "white", (chrom, 100, 200)))
    assert len(warnings) == 1
    assert "--blackListFileName" in warnings[0] and "--whiteListFileName" in warnings[0]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", SPANS)
def test_no_warning_unless_the_lists_removed_the_coverage(tmp_path, capsys, name, mode):
    chrom, start, end = SPANS[name]
    middle = (start + end) // 2
    partial = bed(tmp_path, "partial", (chrom, middle, middle + 5000))
    for options in (["--blackListFileName", partial], ["--whiteListFileName", partial]):
        text, warnings = coverage(tmp_path, capsys, name, "bam", 4, *options, *MODES[mode])
        assert nonzero(text) and not warnings, options
    text, warnings = coverage(tmp_path, capsys, name)
    assert nonzero(text) and not warnings
    # Other filters leave nothing for the lists to remove.
    text, warnings = coverage(tmp_path, capsys, name, "bam", 1, "--minFragmentLength", "100000000",
                              "--whiteListFileName", partial, *MODES[mode])
    assert not nonzero(text) and not warnings


# --- C: whitelists without intervals ------------------------------------------

@pytest.mark.parametrize("name", SPANS)
def test_whitelist_without_intervals_is_an_error(tmp_path, capsys, name):
    for whitelist in (bed(tmp_path, "empty"), bed(tmp_path, "unknown", ("chrUnknown", 0, 1000))):
        for mode in MODES.values():
            with pytest.raises(SystemExit, match="has no intervals"):
                coverage(tmp_path, capsys, name, "bam", 1, "--whiteListFileName", whitelist, *mode)


# --- D: zero normalization denominators ---------------------------------------

@pytest.mark.parametrize("name", SPANS)
def test_zero_denominator_names_what_removed_the_reads(tmp_path, capsys, name):
    chrom = SPANS[name][0]
    white = ["--whiteListFileName", bed(tmp_path, "white", (chrom, 100, 200))]
    for method in ("CPM", "RPKM"):
        with pytest.raises(SystemExit, match=r"normalization denominator is zero \(nothing left "
                                             r"after filtering: --whiteListFileName, read filters\)"):
            coverage(tmp_path, capsys, name, "bam", 1, *white, "--filterByOverlap",
                     "--normalizeUsing", method, "--normalizationDenominator", "filtered")
        # The library denominator is unaffected: an all-zero track and a warning.
        text, warnings = coverage(tmp_path, capsys, name, "cram", 4, *white, "--filterByOverlap",
                                  "--normalizeUsing", method)
        assert text and not nonzero(text) and len(warnings) == 1
        with pytest.raises(SystemExit, match=r"normalization denominator is zero \(nothing left "
                                             r"after filtering: --ignoreForNormalization\)"):
            coverage(tmp_path, capsys, name, "bam", 1, "--normalizeUsing", method,
                     "--ignoreForNormalization", chrom)
    for mode in MODES.values():
        with pytest.raises(SystemExit, match=r"BPM denominator is zero \(nothing left after "
                                             r"filtering: --whiteListFileName, read filters\)"):
            coverage(tmp_path, capsys, name, "bam", 1, *white, *mode, "--normalizeUsing", "BPM")


# --- E: partial filters against coverage computed from the reads ---------------

def inner_regions(name):
    chrom, start, end = SPANS[name]
    third = (end - start) // 3
    return chrom, [(start + third, start + third + 2000), (start + 2 * third, start + 2 * third + 7)]


@pytest.mark.parametrize("kind,threads", RUNS)
@pytest.mark.parametrize("name", SPANS)
def test_mask_zeroes_exactly_the_listed_bases(tmp_path, capsys, name, kind, threads):
    chrom, regions = inner_regions(name)
    listed = {pos for start, end in regions for pos in range(start, end)}
    path = bed(tmp_path, "regions", *[(chrom, *region) for region in regions])
    plain = depth(bases(read) for mates in reads_by_pair(name) for read in mates)
    assert listed & set(plain) and set(plain) - listed
    text, _ = coverage(tmp_path, capsys, name, kind, threads, bs=1)
    assert per_base(text) == plain
    text, _ = coverage(tmp_path, capsys, name, kind, threads, "--blackListFileName", path, bs=1)
    assert per_base(text) == {pos: value for pos, value in plain.items() if pos not in listed}
    text, _ = coverage(tmp_path, capsys, name, kind, threads, "--whiteListFileName", path, bs=1)
    assert per_base(text) == {pos: value for pos, value in plain.items() if pos in listed}


@pytest.mark.parametrize("kind,threads", RUNS)
@pytest.mark.parametrize("name", SPANS)
def test_filter_by_overlap_keeps_or_drops_whole_fragments(tmp_path, capsys, name, kind, threads):
    chrom, regions = inner_regions(name)
    path = bed(tmp_path, "regions", *[(chrom, *region) for region in regions])
    pairs = reads_by_pair(name)
    touching = [any(touches(read, regions) for read in mates) for mates in pairs]
    assert 0 < sum(touching) < len(pairs)
    for option, keep in (("--whiteListFileName", True), ("--blackListFileName", False)):
        expected = depth(bases(read) for mates, touched in zip(pairs, touching)
                         if touched == keep for read in mates)
        text, _ = coverage(tmp_path, capsys, name, kind, threads, option, path,
                           "--filterByOverlap", bs=1)
        assert per_base(text) == expected, option


@pytest.mark.parametrize("touch", ["sense", "antisense"])
@pytest.mark.parametrize("kind,threads", RUNS)
def test_intron_whitelist_keeps_fragments_touching_introns_by_strand(
        tmp_path, capsys, kind, threads, touch):
    """Nascent-RNA coverage: fragments touching an intron of the given strand."""
    pairs = reads_by_pair("pointseq_slice")
    introns = {}
    for mates in pairs:
        for read in mates:
            pos = read.reference_start
            for op, length in read.cigartuples:
                if op == 3:
                    introns.setdefault((pos, pos + length), transcript_strand(read))
                if op in (0, 2, 3, 7, 8):
                    pos += length
    path = bed(tmp_path, "introns", *[("chr22", start, end, ".", 0, "+" if strand > 0 else "-")
                                      for (start, end), strand in sorted(introns.items())])
    want = 1 if touch == "sense" else -1
    kept = []
    for mates in pairs:
        strand = transcript_strand(mates[0]) * want
        regions = [region for region, sign in introns.items() if sign == strand]
        if any(touches(read, regions) for read in mates):
            kept.append(mates)
    assert 0 < len(kept) < len(pairs)
    expected = depth(bases(read) for mates in kept for read in mates)
    text, _ = coverage(tmp_path, capsys, "pointseq_slice", kind, threads,
                       "--whiteListFileName", path, "--filterByOverlap",
                       "--strandedness", "reverse", "--featureTouchStrand", touch, bs=1)
    assert per_base(text) == expected


def collapsed(start, end, reverse, how):
    if how == "center":
        middle = (start + end) // 2
        return {middle}
    return {start if (how == "5prime") != reverse else end - 1}


@pytest.mark.parametrize("how", ["5prime", "3prime", "center"])
@pytest.mark.parametrize("kind,threads", RUNS)
@pytest.mark.parametrize("name", SPANS)
def test_collapse_places_one_base_per_fragment(tmp_path, capsys, name, kind, threads, how):
    units = []
    for mates in reads_by_pair(name):
        if len(mates) == 2 and fragment(mates):  # its forward mate's orientation
            units.append(collapsed(*fragment(mates), False, how))
        else:
            units.extend(collapsed(read.reference_start, read.reference_end,
                                   read.is_reverse, how) for read in mates)
    text, _ = coverage(tmp_path, capsys, name, kind, threads, "--collapse", how,
                       "--countUnit", "fragment", bs=1)
    assert per_base(text) == depth(units)


@pytest.mark.parametrize("kind,threads", RUNS)
@pytest.mark.parametrize("name", SPANS)
def test_extend_reads_fills_fragments(tmp_path, capsys, name, kind, threads):
    length = 200
    units = []
    for mates in reads_by_pair(name):
        if len(mates) == 2 and fragment(mates):
            units.append(set(range(*fragment(mates))))
            continue
        for read in mates:  # from the 5' end, never shorter than the read
            span = max(length, read.reference_end - read.reference_start)
            start, end = ((read.reference_end - span, read.reference_end) if read.is_reverse
                          else (read.reference_start, read.reference_start + span))
            units.append(set(range(max(start, 0), end)))
    text, _ = coverage(tmp_path, capsys, name, kind, threads, "--extendReads", str(length),
                       "--countUnit", "fragment", bs=1)
    assert per_base(text) == depth(units)


@pytest.mark.parametrize("how", ["5prime", "3prime", "center"])
@pytest.mark.parametrize("kind,threads", RUNS)
@pytest.mark.parametrize("name", SPANS)
def test_read_mode_collapse_places_each_dovetailed_mate(tmp_path, capsys, name, kind, threads, how):
    """--countUnit read: every alignment, a dovetailed pair's mates included,
    collapses within its own aligned span."""
    units = [collapsed(read.reference_start, read.reference_end, read.is_reverse, how)
             for mates in reads_by_pair(name) for read in mates]
    text, _ = coverage(tmp_path, capsys, name, kind, threads, "--collapse", how,
                       "--countUnit", "read", bs=1)
    assert per_base(text) == depth(units)


@pytest.mark.parametrize("kind,threads", RUNS)
@pytest.mark.parametrize("name", SPANS)
def test_read_mode_extends_each_dovetailed_mate(tmp_path, capsys, name, kind, threads):
    """--countUnit read: an inward pair fills its fragment twice; a dovetailed
    pair's mates extend on their own."""
    length = 200
    units = []
    for mates in reads_by_pair(name):
        if len(mates) == 2 and fragment(mates, dovetail=False):
            units += [set(range(*fragment(mates, dovetail=False)))] * 2
            continue
        for read in mates:
            span = max(length, read.reference_end - read.reference_start)
            start, end = ((read.reference_end - span, read.reference_end) if read.is_reverse
                          else (read.reference_start, read.reference_start + span))
            units.append(set(range(max(start, 0), end)))
    text, _ = coverage(tmp_path, capsys, name, kind, threads, "--extendReads", str(length),
                       "--countUnit", "read", bs=1)
    assert per_base(text) == depth(units)
