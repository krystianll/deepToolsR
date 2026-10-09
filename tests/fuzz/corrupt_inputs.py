"""Corrupt-input harness: run the real CLI tools on damaged copies of small inputs.

Each case copies the inputs a command needs into its own directory, replaces
one of them with a corrupted variant and runs the command in a subprocess.
The run is classified as

- ``ok-error``: non-zero exit with a message and no Python traceback;
- ``success-same`` / ``success-differs``: exit 0, output equal to or different
  from the run on valid inputs;
- ``crash``: killed by a signal (negative return code or >= 128);
- ``hang``: timed out;
- ``traceback``: an unhandled Python exception reached stderr;
- ``leftover``: a temporary file beside the output, or a partial output after
  a failed run (reported in addition to the exit classification).

Run it directly (``python -m tests.fuzz.corrupt_inputs OUTDIR``) or through
``pytest tests/fuzz --fuzz``. Variants are deterministic (fixed seed).
"""

import bz2
import collections
import concurrent.futures
import gzip
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from tests.fuzz import synthetic

DATA = Path(__file__).resolve().parents[1] / "test_data"
SEED = 20261003
TIMEOUT = 20
LEFTOVER = re.compile(r"(\.deeptoolsr\.|\.zoomruns|spill|\.bak$|\.tmp$|~$)")
REGIONS = "3R\t100\t400\tr1\t0\t+\n3R\t500\t900\tr2\t0\t-\n3R\t1000\t1300\tr3\t0\t+\n"


# ---------------------------------------------------------------- variants

def _flip(data, offset, value):
    out = bytearray(data)
    out[offset] = value if value is not None else out[offset] ^ 0xFF
    return bytes(out)


def _put(data, offset, fmt, value):
    out = bytearray(data)
    if offset + struct.calcsize(fmt) <= len(out):
        struct.pack_into(fmt, out, offset, value)
    return bytes(out)


def binary_variants(data, rng, *, step, header=64, samples=24, extra=()):
    """Truncations, byte flips and caller-specific field inflations."""
    out = [("empty", b"")]
    cuts = set(range(step, len(data), step)) | {len(data) - k for k in range(1, 17)}
    out += [(f"trunc@{c}", data[:c]) for c in sorted(c for c in cuts if c > 0)]
    offsets = set(range(min(header, len(data))))
    offsets |= {rng.randrange(len(data)) for _ in range(samples)}
    for off in sorted(offsets):
        value = (None, 0x00, 0xFF)[off % 3]
        name = {None: "xor", 0: "zero", 0xFF: "ff"}[value]
        out.append((f"{name}@{off}", _flip(data, off, value)))
    out += list(extra)
    return out


def bgzf_blocks(data):
    offsets, pos = [], 0
    while pos + 18 <= len(data) and data[pos:pos + 4] == b"\x1f\x8b\x08\x04":
        offsets.append(pos)
        pos += struct.unpack_from("<H", data, pos + 16)[0] + 1
    return offsets


def bgzf_extra(data):
    out = []
    for i, off in enumerate(bgzf_blocks(data)):
        out.append((f"block-cut@{off}", data[:off]))
        if off:
            out.append((f"block-cut-mid@{off}", data[:off + 9]))
        if i < 4:
            for size in (0x0000, 0x0013, 0xFFFF):
                out.append((f"bsize={size:#x}@{off}", _put(data, off + 16, "<H", size)))
            out.append((f"isize-ff@{off}",
                        _put(data, off + struct.unpack_from("<H", data, off + 16)[0] - 3,
                             "<I", 0xFFFFFFFF)))
    return out


def bam_extra(data):
    """Inflate the BAM header text length and reference count."""
    out = bgzf_extra(data)
    raw = gzip.decompress(data)
    for name, off, value in (("l_text", 4, 0x7FFFFFFF), ("l_text", 4, -1),
                             ("n_ref", 8 + struct.unpack_from("<i", raw, 4)[0],
                              0x7FFFFFFF)):
        out.append((f"{name}={value}", gzip.compress(_put(raw, off, "<i", value))))
    return out


BIGWIG_FIELDS = (("zoomLevels", 6, "<H"), ("chromTreeOffset", 8, "<Q"),
                 ("fullDataOffset", 16, "<Q"), ("fullIndexOffset", 24, "<Q"),
                 ("totalSummaryOffset", 44, "<Q"), ("uncompressBufSize", 52, "<I"))


def bigwig_extra(data, stride=1):
    """Header field inflation plus dense damage over the full-resolution part.

    ``stride`` thins the dense damage for large files.
    """
    end = struct.unpack_from("<Q", data, 24)[0] + 128  # past the data R-tree
    out = [(f"trunc@{c}", data[:c]) for c in range(100, end, 16 * stride)]
    out += [(f"xor@{o}", _flip(data, o, None)) for o in range(96, end, 5 * stride)]
    for name, off, fmt in BIGWIG_FIELDS:
        big = 0xFFFF if fmt == "<H" else (0xFFFFFFFF if fmt == "<I" else len(data) * 4)
        for value in (0, big, min(len(data) - 1, big)):
            out.append((f"{name}={value}", _put(data, off, fmt, value)))
    data_off = struct.unpack_from("<Q", data, 16)[0]
    out.append(("dataCount=huge", _put(data, data_off, "<Q", 0xFFFFFFFFFFFF)))
    zooms = struct.unpack_from("<H", data, 6)[0]
    for z in range(zooms):
        base = 64 + 24 * z
        out.append((f"zoom{z}.dataOffset=huge", _put(data, base + 8, "<Q", len(data) * 8)))
        out.append((f"zoom{z}.indexOffset=0", _put(data, base + 16, "<Q", 0)))
    for match in list(re.finditer(rb"\x78[\x01\x5e\x9c\xda]", data[data_off:]))[::stride]:
        off = data_off + match.start()
        out.append((f"zblock-cut@{off}", data[:off]))
        out.append((f"zblock-garble@{off}", _flip(data, off + 3, None)))
    return out


def text_variants(text, rng):
    lines = text.splitlines(keepends=True)
    first = lines[0].rstrip("\n").split("\t")
    variants = [
        ("binary-line", text + "\x89PNG\r\n\x1a\n\x00\xff\xfe\n"),
        ("binary-first", "\x00\x01\x02\xff\n" + text),
        ("overlong-line", text + "3R\t1\t2\t" + "x" * 2_000_000 + "\n"),
        ("nul-mid", text[:len(text) // 2] + "\x00" + text[len(text) // 2:]),
        ("missing-cols", "".join("\t".join(ln.split("\t")[:2]) + "\n" for ln in lines)),
        ("one-col", "3R\n" + text),
        ("huge-start", text + "\t".join(
            ["3R", "99999999999999999999", "99999999999999999999"] + first[3:]) + "\n"),
        ("neg-coords", text + "\t".join(["3R", "-5", "-1"] + first[3:]) + "\n"),
        ("end-before-start", text + "\t".join(["3R", "900", "100"] + first[3:]) + "\n"),
        ("float-coords", text + "\t".join(["3R", "1.5e3", "2e3"] + first[3:]) + "\n"),
        ("int64-max", text + "\t".join(
            ["3R", "9223372036854775807", "9223372036854775808"] + first[3:]) + "\n"),
        ("no-trailing-newline", text.rstrip("\n")),
        ("crlf", text.replace("\n", "\r\n")),
        ("unknown-chrom", text + "\t".join(["chrZZ", "1", "100"] + first[3:]) + "\n"),
        ("only-comments", "# x\ntrack name=x\n"),
        ("utf8-bad", text + "3R\t1\t50\t\xe9\xff\n"),
        ("empty", ""),
        ("blank-lines", "\n\n" + text + "\n\n"),
        ("many-cols", "".join(ln.rstrip("\n") + "\tx" * 40 + "\n" for ln in lines)),
    ]
    for i in range(6):
        cut = rng.randrange(1, len(text))
        variants.append((f"cut-mid{i}", text[:cut]))
    return variants


def matrix_text_variants(text):
    header, body = text.split("\n", 1)
    meta = json.loads(header[1:])
    rows = body.splitlines(keepends=True)

    def with_meta(**kw):
        m = dict(meta, **kw)
        return "@" + json.dumps(m) + "\n" + body

    out = [
        ("hdr-not-json", "@{not json\n" + body),
        ("hdr-missing", body),
        ("hdr-empty-json", "@{}\n" + body),
        ("group_boundaries-huge", with_meta(group_boundaries=[0, 10**12])),
        ("group_boundaries-neg", with_meta(group_boundaries=[0, -3])),
        ("group_boundaries-unsorted", with_meta(group_boundaries=[3, 0])),
        ("sample_boundaries-huge", with_meta(sample_boundaries=[0, 10**12])),
        ("sample_boundaries-short", with_meta(sample_boundaries=[0])),
        ("sample_labels-extra", with_meta(sample_labels=meta["sample_labels"] * 3)),
        ("bin size-zero", with_meta(**{"bin size": [0] * len(meta["bin size"])})),
        ("body-huge", with_meta(body=[10**15] * len(meta["body"]))),
        ("upstream-neg", with_meta(upstream=[-100] * len(meta["upstream"]))),
        ("group_labels-none", with_meta(group_labels=None)),
        ("hdr-overlong", "@" + json.dumps(dict(meta, pad="x" * 3_000_000)) + "\n" + body),
        ("row-missing-cols", "\n".join([header, rows[0].split("\t", 8)[0]] + rows[1:])),
        ("row-extra-cols", header + "\n" + rows[0].rstrip("\n") + "\t1\t2\n" + "".join(rows[1:])),
        ("row-nonnumeric", header + "\n" + rows[0].rstrip("\n").rsplit("\t", 1)[0]
         + "\tabc\n" + "".join(rows[1:])),
        ("row-huge-number", header + "\n" + rows[0].rstrip("\n").rsplit("\t", 1)[0]
         + "\t1e400\n" + "".join(rows[1:])),
        ("row-binary", header + "\n" + body + "\x00\xff\x89garbage\n"),
        ("row-nul", header + "\n" + body.replace("\t", "\t\x00", 1)),
        ("rows-dropped", header + "\n" + rows[0]),
        ("rows-none", header + "\n"),
        ("row-overlong", header + "\n" + body + "3R\t1\t2\tx\t0\t+" + "\t1" * 500_000 + "\n"),
    ]
    return out


# ---------------------------------------------------------------- cases

def _gz(data):
    return gzip.compress(data, mtime=0)


def build_inputs(base):
    """Write the valid inputs into ``base`` and return {name: bytes}."""
    base.mkdir(parents=True, exist_ok=True)
    files = {
        "in.bam": (DATA / "test1.bam").read_bytes(),
        "in.bam.bai": (DATA / "test1.bam.bai").read_bytes(),
        "in.cram": (DATA / "test1.cram").read_bytes(),
        "in.cram.crai": (DATA / "test1.cram.crai").read_bytes(),
        "test1.fa": (DATA / "test1.fa").read_bytes(),
        "test1.fa.fai": (DATA / "test1.fa.fai").read_bytes(),
        "in.bw": (DATA / "test1.bw.bw").read_bytes(),
        "r.bed": REGIONS.encode(),
        "in.gtf": (DATA / "test.gtf").read_bytes(),
        "bl.bed": (DATA / "test_filtering.blacklist.bed").read_bytes(),
    }
    for name, data in files.items():
        (base / name).write_bytes(data)
    subprocess.run(["bamCoverageR", "-b", "in.bam", "-o", "good.bw"], cwd=base,
                   check=True, capture_output=True)
    subprocess.run(["computeMatrixR", "scale-regions", "-S", "in.bw", "good.bw",
                    "-R", "r.bed", "-o", "in.mat.gz", "-a", "100", "-b", "100",
                    "-m", "200", "-bs", "10"], cwd=base, check=True, capture_output=True)
    files["good.bw"] = (base / "good.bw").read_bytes()
    files["in.mat.gz"] = (base / "in.mat.gz").read_bytes()
    for name in synthetic.write_all(base):
        files[name] = (base / name).read_bytes()
    return files


BAM_CMDS = (("bamCoverageR", "-b", "in.bam", "-o", "out.bw"),
            ("bamCoverageR", "-b", "in.bam", "-o", "out.bg", "-of", "bedgraph"))
CRAM_CMDS = tuple(tuple("in.cram" if a == "in.bam" else a for a in c) for c in BAM_CMDS)
BW_CMDS = (("computeMatrixR", "reference-point", "-S", "in.bw", "-R", "r.bed",
            "-o", "out.mat.gz", "-a", "100", "-b", "100"),
           ("bigWigOperationsR", "scale", "-b", "in.bw", "-o", "out.bw", "-s", "2"),
           ("bigWigOperationsR", "merge", "-b", "in.bw", "good.bw", "-o", "out.bw"),
           ("bigWigOperationsR", "info", "in.bw"))
MAT_CMDS = (("plotHeatmapR", "-m", "in.mat.gz", "-o", "out.png"),
            ("plotProfileR", "-m", "in.mat.gz", "-o", "out.png"),
            ("computeMatrixOperationsR", "info", "-m", "in.mat.gz"),
            ("computeMatrixOperationsR", "filterValues", "-m", "in.mat.gz",
             "-o", "out.mat.gz", "--min", "0"),
            ("computeMatrixOperationsR", "dataRange", "-m", "in.mat.gz"),
            ("deeptoolsr", "describe", "plotHeatmapR", "-m", "in.mat.gz", "-o", "out.png"))


def _regions_cmd(name, blacklist=False):
    if blacklist:
        return ("computeMatrixR", "scale-regions", "-S", "good.bw", "-R", "r.bed",
                "-bl", name, "-o", "out.mat.gz", "-m", "200", "-bs", "10")
    return ("computeMatrixR", "scale-regions", "-S", "good.bw", "-R", name,
            "-o", "out.mat.gz", "-m", "200", "-bs", "10")


def cases(files, rng, valid):
    """Yield (kind, target_name, variant_name, bytes, commands).

    Records the valid bytes of every target in ``valid``.
    """
    valid.update(files)
    yield from (("bam", "in.bam", n, d, BAM_CMDS) for n, d in
                binary_variants(files["in.bam"], rng, step=1024,
                                extra=bam_extra(files["in.bam"])))
    yield from (("bai", "in.bam.bai", n, d, BAM_CMDS) for n, d in
                binary_variants(files["in.bam.bai"], rng, step=8, header=32, samples=8))
    yield from (("cram", "in.cram", n, d, CRAM_CMDS) for n, d in
                binary_variants(files["in.cram"], rng, step=256, header=48))
    yield from (("crai", "in.cram.crai", n, d, CRAM_CMDS) for n, d in
                binary_variants(files["in.cram.crai"], rng, step=8, header=20, samples=6)
                + [(f"text:{n}", _gz(t.encode("latin-1"))) for n, t in
                   text_variants(gzip.decompress(files["in.cram.crai"]).decode(), rng)])
    yield from (("bigwig", "in.bw", n, d, BW_CMDS) for n, d in
                binary_variants(files["in.bw"], rng, step=1024, header=96,
                                extra=bigwig_extra(files["in.bw"])))
    sources = (("bed", REGIONS, False), ("gtf", files["in.gtf"].decode(), False),
               ("blacklist", files["bl.bed"].decode(), True))
    for kind, valid_text, blacklist in sources:
        base = {"gtf": "x.gtf", "bed": "x.bed", "blacklist": "blk.bed"}[kind]
        for suffix, encode in (("", lambda b: b), (".gz", _gz), (".bz2", bz2.compress)):
            target = base + suffix
            valid[target] = encode(valid_text.encode())
            for vname, text in text_variants(valid_text, rng):
                yield (kind + suffix, target, vname, encode(text.encode("latin-1")),
                       (_regions_cmd(target, blacklist),))
            if suffix:
                packed = valid[target]
                for vname, data in binary_variants(packed, rng, step=4, header=12, samples=4):
                    yield (kind + suffix, target, "archive:" + vname, data,
                           (_regions_cmd(target, blacklist),))
    mat_text = gzip.decompress(files["in.mat.gz"]).decode()
    yield from (("matrix", "in.mat.gz", n, d, MAT_CMDS) for n, d in
                binary_variants(files["in.mat.gz"], rng, step=64, header=24, samples=16))
    yield from (("matrix", "in.mat.gz", "text:" + n, _gz(t.encode("latin-1")), MAT_CMDS)
                for n, t in matrix_text_variants(mat_text))
    yield ("matrix", "in.mat.gz", "plain-uncompressed", mat_text.encode(), MAT_CMDS)
    yield from more_cases(files, rng, valid)


# ------------------------------------------- synthetic inputs (paired, multi-chrom)

ALN_OPTS = ((), ("--extendReads",), ("--extendReads", "--skipIntrons"),
            ("--filterRNAstrand", "split", "--strandedness", "reverse"),
            ("--normalizeUsing", "CPM"), ("--collapse", "5prime"),
            ("--filterByOverlap", "-bl", "bl2.bed"))


def aln_cmds(name, i):
    """Plain plus one rotating option set, each at -p 1 and -p 4."""
    return tuple(("bamCoverageR", "-b", name, "-o", "out.bw", "-p", p, *opts)
                 for opts in (ALN_OPTS[0], ALN_OPTS[1 + i % (len(ALN_OPTS) - 1)])
                 for p in ("1", "4"))


def _sampled(data, rng, cuts, **kw):
    return binary_variants(data, rng, step=max(1, len(data) // cuts), **kw)


MB_CMDS = (("computeMatrixR", "reference-point", "-S", "mb.bw", "-R", "mr1.bed",
            "-o", "out.mat.gz", "-a", "500", "-b", "500", "-bs", "50"),
           ("computeMatrixR", "scale-regions", "-S", "mb2.bw", "mb.bw", "-R", "mr1.bed",
            "-o", "out.mat.gz", "-m", "1000", "-bs", "50"),
           ("bigWigOperationsR", "scale", "-b", "mb.bw", "-o", "out.bw", "-s", "2"),
           ("bigWigOperationsR", "merge", "-b", "mb.bw", "mb2.bw", "-o", "out.bw"),
           ("bigWigOperationsR", "merge", "-b", "mb2.bw", "mb.bw", "-o", "out.bw"),
           ("bigWigOperationsR", "info", "--chromSizes", "out.sizes", "mb.bw"))
CMO = ("computeMatrixOperationsR",)
MATRIX_CMDS = {
    "mA.mat.gz": ((*CMO, "info", "-m", "mA.mat.gz"),
                  (*CMO, "info", "--json", "-m", "mA.mat.gz"),
                  (*CMO, "relabel", "-m", "mA.mat.gz", "-o", "out.mat.gz",
                   "--sampleLabels", "a", "b"),
                  (*CMO, "subset", "-m", "mA.mat.gz", "-o", "out.mat.gz",
                   "--samples", "mb2"),
                  (*CMO, "filterStrand", "-m", "mA.mat.gz", "-o", "out.mat.gz",
                   "--strand", "+"),
                  (*CMO, "filterValues", "-m", "mA.mat.gz", "-o", "out.mat.gz",
                   "--min", "1"),
                  (*CMO, "rbind", "-m", "mA.mat.gz", "mB.mat.gz", "-o", "out.mat.gz"),
                  (*CMO, "cbind", "-m", "mA.mat.gz", "mC.mat.gz", "-o", "out.mat.gz"),
                  (*CMO, "cbind", "--blind", "-m", "mC.mat.gz", "mA.mat.gz",
                   "-o", "out.mat.gz"),
                  (*CMO, "sort", "-m", "mA.mat.gz", "-R", "mr2.bed", "mr1.bed",
                   "-o", "out.mat.gz"),
                  (*CMO, "dataRange", "-m", "mA.mat.gz")),
    "mB.mat.gz": ((*CMO, "rbind", "-m", "mA.mat.gz", "mB.mat.gz", "-o", "out.mat.gz"),
                  (*CMO, "rbind", "-m", "mB.mat.gz", "mA.mat.gz", "-o", "out.mat.gz")),
    "mC.mat.gz": ((*CMO, "cbind", "-m", "mA.mat.gz", "mC.mat.gz", "-o", "out.mat.gz"),
                  (*CMO, "cbind", "--blind", "-m", "mA.mat.gz", "mC.mat.gz",
                   "-o", "out.mat.gz")),
}
CM = ("computeMatrixR", "scale-regions", "-S", "mb.bw", "-o", "out.mat.gz",
      "-m", "1000", "-bs", "100")
ANNOTATION_CMDS = {
    "bl2.bed": (("bamCoverageR", "-b", "pe.bam", "-o", "out.bw", "-bl", "bl2.bed"),
                ("bamCoverageR", "-b", "pe.bam", "-o", "out.bw", "-bl", "bl2.bed",
                 "--filterByOverlap")),
    "wl.bed": (("bamCoverageR", "-b", "pe.bam", "-o", "out.bw", "-wl", "wl.bed"),
               ("bamCoverageR", "-b", "pe.bam", "-o", "out.bw", "-wl", "wl.bed",
                "--filterByOverlap", "-p", "4")),
    "s.bed": (("bamCoverageR", "-b", "pe.bam", "-o", "out.bw",
               "--strandedness", "file=s.bed"),),
    "r12.bed": ((*CM, "-R", "r12.bed"), (*CM, "-R", "r12.bed", "--metagene"),
                (*CM, "-R", "mr1.bed", "r12.bed")),
    "mg.gtf": ((*CM, "-R", "mg.gtf", "--metagene"), (*CM, "-R", "mr1.bed", "mg.gtf")),
    "mr2.bed": ((*CM, "-R", "mr1.bed", "mr2.bed"),
                (*CMO, "sort", "-m", "mA.mat.gz", "-R", "mr1.bed", "mr2.bed",
                 "-o", "out.mat.gz")),
}
CONFIG_CMDS = (("deeptoolsr", "options", "--config", "cfg.txt"),
               ("deeptoolsr", "options", "--config", "cfg.txt", "--fontMultiplier", "2"),
               ("plotHeatmapR", "--config", "cfg.txt", "-m", "in.mat.gz", "-o", "out.png"),
               ("computeMatrixR", "reference-point", "--config", "cfg.txt", "-S", "in.bw",
                "-R", "r.bed", "-o", "out.mat.gz"))


def json_variants(text, rng):
    """Truncations plus wrong-typed values for each top-level key of a JSON object."""
    data = json.loads(text)
    out = [(f"cut@{c}", text[:c]) for c in sorted(rng.sample(range(1, len(text)), 8))]
    out += [("list-root", "[1, 2]\n"), ("null-root", "null\n"), ("nan", text.replace(
        "1.5", "NaN", 1)), ("dup-key", text.replace("{", '{"font_multiplier": "x", ', 1))]
    for key in sorted(data):
        for label, value in (("str", "x"), ("null", None), ("list", [1]),
                             ("huge", 1e308), ("neg", -1), ("obj", {"a": {}})):
            out.append((f"{key}={label}", json.dumps(dict(data, **{key: value}))))
    return out


def more_cases(files, rng, valid):
    aln = [("pe.bam", _sampled(files["pe.bam"], rng, 20, header=32, samples=8,
                               extra=bam_extra(files["pe.bam"]))),
           ("pe.bam.bai", binary_variants(files["pe.bam.bai"], rng, step=8, header=32,
                                          samples=8)),
           ("pe.cram", _sampled(files["pe.cram"], rng, 20, header=48, samples=8)),
           ("pe.cram.crai", binary_variants(files["pe.cram.crai"], rng, step=8,
                                            header=20, samples=6)),
           ("pe.sam.gz", _sampled(files["pe.sam.gz"], rng, 20, header=32, samples=8,
                                  extra=bgzf_extra(files["pe.sam.gz"]))),
           ("pe.sam.gz.csi", binary_variants(files["pe.sam.gz.csi"], rng, step=8,
                                             header=32, samples=8))]
    if "ps.bam" in files:
        aln.append(("ps.bam", _sampled(files["ps.bam"], rng, 20, header=32, samples=8,
                                       extra=bgzf_extra(files["ps.bam"]))))
    for target, variants in aln:
        reads = target.rsplit(".", 1)[0] if target.endswith(("bai", "crai", "csi")) \
            else target
        for i, (vname, data) in enumerate(variants):
            yield (target, target, vname, data, aln_cmds(reads, i))
    mb = files["mb.bw"]
    yield from (("multi-bigwig", "mb.bw", n, d, MB_CMDS) for n, d in
                _sampled(mb, rng, 40, header=96, samples=16,
                         extra=bigwig_extra(mb, stride=max(1, len(mb) // 1500))))
    for target, cmds in ANNOTATION_CMDS.items():
        yield from ((target, target, n, t.encode("latin-1"), cmds)
                    for n, t in text_variants(files[target].decode(), rng))
    for target, cmds in MATRIX_CMDS.items():
        data = files[target]
        variants = _sampled(data, rng, 12, header=24, samples=8)
        variants += [("text:" + n, _gz(t.encode("latin-1"))) for n, t in
                     matrix_text_variants(gzip.decompress(data).decode())]
        yield from ((target, target, n, d, cmds) for n, d in variants)
    yield from (("config", "cfg.txt", n, t.encode("latin-1"), CONFIG_CMDS)
                for n, t in json_variants(files["cfg.txt"].decode(), rng)
                + text_variants(files["cfg.txt"].decode(), rng)[:6])


# ---------------------------------------------------------------- running

def _output(rundir, cmd, stdout):
    """Stdout, or every out.* file (split outputs too), gunzipped when possible."""
    if not any(a.startswith("out.") for a in cmd):
        return stdout
    out = []
    for path in sorted(rundir.glob("out.*")):
        data = path.read_bytes()
        if data[:2] == b"\x1f\x8b":
            try:
                data = gzip.decompress(data)
            except (OSError, EOFError):
                pass
        out.append(path.name.encode() + b"\0" + data)
    return b"\0".join(out) or None


def run_case(rundir, base, inputs, target, data, cmd):
    rundir.mkdir(parents=True)
    for name in inputs:
        shutil.copyfile(base / name, rundir / name)
    (rundir / target).write_bytes(data)
    before = set(os.listdir(rundir))
    env = dict(os.environ, PYTHONFAULTHANDLER="1")
    try:
        proc = subprocess.run(cmd, cwd=rundir, capture_output=True, timeout=TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"class": "hang", "rc": None, "stderr": "", "leftovers": []}
    stderr = proc.stderr.decode("utf-8", "replace")
    rc = proc.returncode
    if rc < 0 or rc >= 128:
        cls = "crash"
    elif "Traceback (most recent call last)" in stderr:
        cls = "traceback"
    elif rc:
        cls = "ok-error"
    else:
        cls = "success"
    new = sorted(set(os.listdir(rundir)) - before)
    outputs = {n for n in new if n.startswith("out.")}
    leftovers = [n for n in new if n not in outputs or LEFTOVER.search(n)]
    if rc:
        leftovers += [n for n in new if n in outputs]
    return {"class": cls, "rc": rc, "stderr": stderr[-4000:], "leftovers": leftovers,
            "output": _output(rundir, cmd, proc.stdout) if cls == "success" else None}


COMPANIONS = {"in.bam": ("in.bam.bai",), "pe.bam": ("pe.bam.bai",),
              "ps.bam": ("ps.bam.bai",), "pe.cram": ("pe.cram.crai",),
              "pe.sam.gz": ("pe.sam.gz.csi",),
              "in.cram": ("in.cram.crai", "test1.fa", "test1.fa.fai")}


def _inputs_for(cmd, target, known):
    """Every known input file the command names, plus its index companions."""
    names = {a.split("=", 1)[-1] for a in cmd} & set(known)
    names |= {c for n in list(names) for c in COMPANIONS.get(n, ())}
    names.discard(target)
    return sorted(names)


# ------------------------------------------------------------ serve protocol

PROBE = b'{"id": "probe", "info": ["-m", "in.mat.gz"]}\n'
CLOSE = b'{"close": true}\n'
BAD_REQUESTS = (
    ("truncated-json", b'{"id": 1, "info": ["-m", "in.mat'),
    ("not-json", b"hello there"),
    ("huge-garbage", b"x" * 20_000_000),
    ("huge-path", b'{"id": "h", "info": ["-m", "' + b"x" * 20_000_000 + b'"]}'),
    ("info-str", b'{"id": 1, "info": "in.mat.gz"}'),
    ("id-bool", b'{"id": true, "info": ["-m", "in.mat.gz"]}'),
    ("id-float", b'{"id": 1.5, "info": ["-m", "in.mat.gz"]}'),
    ("id-list", b'{"id": [1], "describe": ["plotHeatmapR"]}'),
    ("command-ints", b'{"id": 2, "command": [1, 2]}'),
    ("command-unknown-tool", b'{"id": 2, "command": ["rm", "-rf", "/"]}'),
    ("progress-str", b'{"id": 3, "command": ["plotHeatmapR"], "progress": "yes"}'),
    ("close-str", b'{"id": 4, "close": "yes"}'),
    ("cancel-obj", b'{"cancel": {}}'),
    ("two-ops", b'{"id": 5, "info": ["-m", "a"], "describe": ["plotHeatmapR"]}'),
    ("list-root", b"[1, 2]"),
    ("null-root", b"null"),
    ("number-overflow", b'{"id": 1e999, "info": ["-m", "in.mat.gz"]}'),
    ("deep-nesting", b"[" * 100_000 + b"]" * 100_000),
    ("non-utf8-string", b'{"id": 6, "info": ["-m", "\xff\xfe.mat.gz"]}'),
    ("non-utf8-raw", b"\xff\xfe\xfd\x80"),
    ("nul-bytes", b'{"id": 7,\x00 "info": ["-m", "in.mat.gz"]}'),
    ("lone-surrogate-path", b'{"id": 9, "info": ["-m", "\\ud800.mat.gz"]}'),
    ("bad-plot-args", b'{"id": 8, "command": ["plotHeatmapR", "-m", "in.mat.gz", '
                      b'"--colorMap", "nope", "-o", "out.png"]}'),
)
SERVE_CASES = tuple((n, line + b"\n", 1, True) for n, line in BAD_REQUESTS) + (
    ("all-bad-at-once", b"\n".join(r for _, r in BAD_REQUESTS) + b"\n",
     len(BAD_REQUESTS), True),
    # Per the contract, EOF and close answer queued requests "cancelled".
    ("eof-mid-request", b'{"id": 1, "info": ', 1, False),
    ("eof-after-request", PROBE, 1, False),
    ("eof-request-without-newline", PROBE.rstrip(b"\n"), 1, False),
    ("request-after-close", CLOSE + PROBE, 0, False),
    ("empty-stdin", b"", 0, False),
)


def _talk(proc, payload, probe):
    """Send ``payload``; with ``probe``, wait for the probe reply, then close."""
    out = []
    try:
        proc.stdin.write(payload + (PROBE if probe else b""))
        proc.stdin.flush()
        while probe:
            line = proc.stdout.readline()
            if not line:
                break
            out.append(line)
            if b'"probe"' in line:
                proc.stdin.write(CLOSE)
                break
        proc.stdin.close()
    except BrokenPipeError:
        pass
    out += proc.stdout.read().splitlines()
    return out


def run_serve(rundir, base, payload, expected_errors, probe):
    """One serve session; expect ``expected_errors`` error replies and a live worker."""
    rundir.mkdir(parents=True)
    shutil.copyfile(base / "in.mat.gz", rundir / "in.mat.gz")
    proc = subprocess.Popen(("deeptoolsr", "serve"), cwd=rundir, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        lines = pool.submit(_talk, proc, payload, probe)
        err = pool.submit(proc.stderr.read)
        try:
            lines = lines.result(timeout=TIMEOUT * 3)
            proc.wait(timeout=TIMEOUT)
        except (concurrent.futures.TimeoutError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait()
            return {"class": "hang", "rc": None, "stderr": "", "leftovers": []}
        stderr = err.result().decode("utf-8", "replace")
    replies, problems = [], []
    for line in lines:
        try:
            reply = json.loads(line)
        except ValueError:
            problems.append(f"non-JSON stdout line {line[:80]!r}")
            continue
        if isinstance(reply, dict) and ("ok" in reply or "error" in reply):
            replies.append(reply)
    errors = [r for r in replies if r.get("ok") is False]
    probes = [r for r in replies if r.get("id") == "probe" and r.get("ok")]
    if len(errors) != expected_errors:
        problems.append(f"{len(errors)} error replies, expected {expected_errors}: "
                        + json.dumps(errors)[:300])
    if probe != bool(probes):
        problems.append(f"probe answered ok: {bool(probes)}, expected {probe}")
    rc = proc.returncode
    if rc < 0 or rc >= 128:
        cls = "crash"
    elif "Traceback (most recent call last)" in stderr:
        cls = "traceback"
    elif problems or rc:
        cls = "bad-reply"
    else:
        cls = "ok-error"
    return {"class": cls, "rc": rc, "stderr": "\n".join(problems) + "\n" + stderr[-3000:],
            "leftovers": sorted(set(os.listdir(rundir)) - {"in.mat.gz"})}


def signature(stderr):
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    last = lines[-1] if lines else "<empty stderr>"
    last = re.sub(r"/\S+", "<path>", last)
    return re.sub(r"\d+", "N", last)[:160]


def run_all(workdir, jobs=6, limit=None):
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    base = workdir / "base"
    files = build_inputs(base)
    rng = random.Random(SEED)
    todo = []
    valid = {}
    for kind, target, vname, data, cmds in cases(files, rng, valid):
        for cmd in cmds:
            todo.append((kind, target, vname, data, cmd))
    todo = todo[:limit] if limit else todo
    baselines = {}
    for _, target, _, _, cmd in todo:
        if cmd not in baselines:
            res = run_case(workdir / "valid" / str(len(baselines)), base,
                           _inputs_for(cmd, target, files), target, valid[target], cmd)
            assert res["class"] == "success", (cmd, res["stderr"])
            baselines[cmd] = res["output"]
    results = []

    def one(i):
        kind, target, vname, data, cmd = todo[i]
        rundir = workdir / "runs" / f"{i:05d}"
        res = run_case(rundir, base, _inputs_for(cmd, target, files), target, data, cmd)
        if res["class"] == "success":
            res["class"] = ("success-same" if res["output"] == baselines[cmd]
                            else "success-differs")
        res.pop("output", None)
        if res["class"] in ("ok-error", "success-same", "success-differs") \
                and not res["leftovers"]:
            shutil.rmtree(rundir)
        res.update(kind=kind, target=target, variant=vname, cmd=" ".join(cmd),
                   rundir=str(rundir), sig=signature(res["stderr"]))
        return res

    def serve(case):
        name, payload, expected, probe = case
        rundir = workdir / "serve" / name
        res = run_serve(rundir, base, payload, expected, probe)
        if res["class"] == "ok-error" and not res["leftovers"]:
            shutil.rmtree(rundir)
        res.update(kind="serve", target="stdin", variant=name, cmd="deeptoolsr serve",
                   rundir=str(rundir), sig=signature(res["stderr"]))
        return res

    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(one, range(len(todo))))
        if not limit:
            results += list(pool.map(serve, SERVE_CASES))
    for res in results:
        if res["class"] in ("ok-error", "success-same", "success-differs"):
            res["stderr"] = res["stderr"][-400:]
    (workdir / "results.json").write_text(json.dumps(results, indent=1))
    return results


def summarize(results):
    counts = collections.Counter((r["kind"], r["cmd"].split(" ")[0], r["class"]) for r in results)
    groups = collections.defaultdict(list)
    for r in results:
        groups[(r["class"], r["cmd"], r["kind"], r["sig"])].append(r)
        if r["leftovers"]:
            groups[("leftover", r["cmd"], r["kind"], str(r["leftovers"]))].append(r)
    return counts, groups


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    workdir = Path(argv[0] if argv else "fuzz-work")
    results = run_all(workdir, jobs=int(os.environ.get("FUZZ_JOBS", "6")))
    counts, groups = summarize(results)
    for key in sorted(counts):
        print(*key, counts[key], sep="\t")
    print()
    for (cls, cmd, kind, sig), rs in sorted(groups.items()):
        if cls in ("ok-error", "success-same"):
            continue
        print(f"[{cls}] {kind} :: {cmd} :: {sig} :: n={len(rs)} e.g. {rs[0]['variant']}"
              f" {rs[0]['rundir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
