"""Independent bigWig zoom-level decoder and reference builder for tests.

``read_zoom_levels`` decodes a file's zoom section (headers, R-tree index,
compressed blocks). ``reference_zoom_levels`` rebuilds the expected levels from
the file's data intervals, following libBigWig 0.4.8's level choice and record
rules with every record's sum and sum of squares filled in.
"""

import struct
import zlib

import numpy as np

UINT32_MAX = 0xFFFFFFFF
BLOCK_RECORDS = 32768 // 32 - 1   # libBigWig fills a zoom buffer to one short of bufSize/32
RECORD = struct.Struct("<IIIIffff")


def _leaves(data, node):
    is_leaf, _, count = struct.unpack_from("<BBH", data, node)
    out = []
    for i in range(count):
        if is_leaf:
            out.append(struct.unpack_from("<IIIIQQ", data, node + 4 + 32 * i))
        else:
            child = struct.unpack_from("<IIIIQ", data, node + 4 + 24 * i)[4]
            out.extend(_leaves(data, child))
    return out


def read_zoom_levels(path):
    """[(reduction, blocks)], blocks = [(index entry, [record tuples])]."""
    data = open(path, "rb").read()
    n_levels = struct.unpack_from("<H", data, 6)[0]
    levels = []
    for i in range(n_levels):
        reduction, _, data_offset, index_offset = struct.unpack_from("<IIQQ", data, 0x40 + 24 * i)
        assert struct.unpack_from("<I", data, index_offset)[0] == 0x2468ACE0
        blocks = []
        for entry in _leaves(data, index_offset + 48):
            offset, size = entry[4], entry[5]
            raw = zlib.decompress(data[offset:offset + size])
            records = [RECORD.unpack_from(raw, j) for j in range(0, len(raw), 32)]
            blocks.append((entry[:4], records))
        # A zoom level's data count is its record count.
        assert struct.unpack_from("<I", data, data_offset)[0] == sum(len(r) for _, r in blocks)
        levels.append((reduction, blocks))
    return levels


def data_section(path):
    """The block count and compressed data blocks (the index holds absolute offsets)."""
    data = open(path, "rb").read()
    full_data, full_index = struct.unpack_from("<QQ", data, 16)
    return data[full_data:full_index]


def choose_levels(runs, lengths, max_zooms):
    """libBigWig makeZoomLevels: reductions from the mean interval width."""
    widths = [end - start for _, start, end, _ in runs]
    mean = int(sum(widths) / len(widths)) * 4
    if (UINT32_MAX >> 2) < mean:
        return []
    zoom = 4 * mean if mean * 4 > 10 else 10
    longest = max(lengths)
    zoom = min(zoom, longest)
    levels = []
    for _ in range(max_zooms):
        if zoom > longest:
            break
        levels.append(zoom)
        if UINT32_MAX // 4 < zoom:
            break
        zoom *= 4
    return levels


def _overlap(t0, s0, e0, t1, s1, e1):
    if t0 != t1 or e0 <= s1 or e1 <= s0:
        return 0
    return min(e0, e1) - max(s0, s1)


def reference_records(runs, zoom):
    """Blocks of zoom records [(tid, start, end, count, min, max, sum, sumsq)]."""
    f32 = np.float32
    blocks, block = [], []
    sums = [0.0, 0.0]

    def close():
        block[-1][6], block[-1][7] = float(f32(sums[0])), float(f32(sums[1]))
        sums[0] = sums[1] = 0.0

    def add(rv, value):
        sums[0] += rv * float(value)
        sums[1] += rv * float(value) ** 2

    for tid, start, end, value in runs:
        value = float(f32(value))
        while start < end:
            size = zoom if start + zoom <= UINT32_MAX else UINT32_MAX - start
            if len(block) == BLOCK_RECORDS:
                close()
                blocks.append(block)
                block = []
            elif block:
                last = block[-1]
                rv = _overlap(last[0], last[1], (last[1] + size) & UINT32_MAX, tid, start, end)
                if rv:
                    last[2] = start + rv
                    last[3] += rv
                    last[4] = min(last[4], value)
                    last[5] = max(last[5], value)
                    add(rv, value)
                    start += rv
                    continue
                close()
            rv = min(size, end - start)
            block.append([tid, start, start + rv, rv, value, value, 0.0, 0.0])
            add(rv, value)
            start += rv
    if block:
        close()
        blocks.append(block)
    return blocks


def reference_zoom_levels(runs, lengths, max_zooms):
    """Expected [(reduction, blocks)] with libBigWig's duplicate-level cut."""
    levels = []
    for zoom in choose_levels(runs, lengths, max_zooms):
        blocks = reference_records(runs, zoom)
        if levels and len(blocks) == len(levels[-1][1]):
            break
        levels.append((zoom, blocks))
    return levels


def file_runs(path):
    """Data intervals [(tid, start, end, value)] and chromosome lengths, in file order."""
    from tests.helpers.bigwig import pyBigWig
    bw = pyBigWig.open(str(path))
    try:
        chroms = list(bw.chroms().items())
        runs = [(tid, s, e, v) for tid, (name, length) in enumerate(chroms)
                for s, e, v in (bw.intervals(name) or ())]
    finally:
        bw.close()
    return runs, [length for _, length in chroms]


def assert_zoom_matches_reference(path, max_zooms):
    runs, lengths = file_runs(path)
    expected = reference_zoom_levels(runs, lengths, max_zooms)
    actual = read_zoom_levels(path)
    assert [r for r, _ in actual] == [r for r, _ in expected]
    for (reduction, got), (_, want) in zip(actual, expected):
        assert len(got) == len(want), reduction
        for (entry, records), block in zip(got, want):
            assert entry == (block[0][0], block[0][1], block[-1][0], block[-1][2])
            assert [list(r) for r in records] == [
                [*r[:4], *(float(np.float32(x)) for x in r[4:])] for r in block]
    return actual
