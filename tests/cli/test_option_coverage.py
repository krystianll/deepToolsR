"""Successful public settings routes with numerical and persistence assertions."""

import argparse
import gzip
import json

import numpy as np
from tests.helpers.bigwig import pyBigWig
import pytest

from deeptoolsr import (_raster, bigWigOperations, config,
                        deeptoolsr_list_tools, options as run_options)


def make_tracks(tmp_path):
    arrays = [np.zeros(23), np.zeros(23)]
    arrays[0][:5], arrays[0][9:14], arrays[0][18:] = 2, -1, 3
    arrays[1][3:10], arrays[1][12:20] = 6, 4
    intervals = [
        [(0, 5, 2.0), (9, 14, -1.0), (18, 23, 3.0)],
        [(3, 10, 6.0), (12, 20, 4.0)],
    ]
    paths = []
    for i, entries in enumerate(intervals):
        path = tmp_path / f"input-{i}.bw"
        with pyBigWig.open(str(path), "w") as bw:
            bw.addHeader([("chr1", 23)])
            bw.addEntries(
                ["chr1"] * len(entries),
                [s for s, _, _ in entries],
                ends=[e for _, e, _ in entries],
                values=[v for _, _, v in entries],
            )
        paths.append(str(path))
    return paths, arrays


@pytest.mark.parametrize("operation", ["sum", "mean"])
@pytest.mark.parametrize(
    "format_name,suffix",
    [("bigwig", ".bw"), ("bedgraph", ".bg"), ("bedgraph", ".bg.gz")],
)
@pytest.mark.parametrize("bins,compression,zooms", [(1, 1, 0), (7, 12, 3)])
@pytest.mark.parametrize("threads", ["1", "max"])
def test_merge_cli_settings_preserve_bin_formula(
    tmp_path, operation, format_name, suffix, bins, compression, zooms, threads
):
    paths, arrays = make_tracks(tmp_path)
    output = tmp_path / ("merged" + suffix)
    bigWigOperations.main(
        [
            "merge",
            "-b",
            *paths,
            "-o",
            str(output),
            "--operation",
            operation,
            "--binSize",
            str(bins),
            "--scale",
            "-2",
            "--numberOfProcessors",
            threads,
            "--outFileFormat",
            format_name,
            "--compressionLevel",
            str(compression),
            "--zoomLevels",
            str(zooms),
        ]
    )
    merged = arrays[0] + arrays[1]
    if operation == "mean":
        merged /= 2
    expected = np.zeros(23)
    for start in range(0, 23, bins):
        expected[start: start + bins] = -2 * np.mean(merged[start: start + bins])
    if format_name == "bigwig":
        with pyBigWig.open(str(output)) as bw:
            actual = np.nan_to_num(bw.values("chr1", 0, 23))
    else:
        actual = np.zeros(23)
        opener = gzip.open if suffix.endswith(".gz") else open
        with opener(output, "rt") as handle:
            for line in handle:
                chrom, start, end, value = line.split()
                assert chrom == "chr1"
                actual[int(start): int(end)] = float(value)
    np.testing.assert_allclose(actual, expected, rtol=6e-6, atol=1e-6)


@pytest.mark.parametrize("factor", [-2, 0, 2.5])
@pytest.mark.parametrize("compression,zooms", [(1, 0), (12, 3)])
def test_scale_cli_writer_settings_keep_values_and_missingness(
    tmp_path, factor, compression, zooms
):
    paths, _ = make_tracks(tmp_path)
    output = tmp_path / "scaled.bw"
    bigWigOperations.main(
        [
            "scale",
            "-b",
            paths[0],
            "-o",
            str(output),
            "--scaleFactor",
            str(factor),
            "--compressionLevel",
            str(compression),
            "--zoomLevels",
            str(zooms),
        ]
    )
    with pyBigWig.open(paths[0]) as source, pyBigWig.open(str(output)) as result:
        expected = np.asarray(source.values("chr1", 0, 23)) * factor
        np.testing.assert_allclose(
            result.values("chr1", 0, 23), expected, equal_nan=True
        )


@pytest.mark.parametrize("filter_name", config.RASTER_FILTERS)
@pytest.mark.parametrize("bits", [8, 16])
def test_raster_settings_roundtrip_and_native_constant_field(
    tmp_path, filter_name, bits
):
    deeptoolsr_list_tools.process_args(
        [
            "options",
            "--rasterFilter",
            filter_name,
            "--rasterBitDepth",
            str(bits),
        ]
    )
    resolved = run_options.resolve_run_options(
        argparse.Namespace(config='auto', numberOfProcessors=1))
    assert resolved.raster.filter == filter_name
    assert resolved.raster.bit_depth == bits
    stored = json.loads(config.options_path().read_text())
    assert stored["raster_filter"] == filter_name and stored["raster_bit_depth"] == bits
    # A normalized resampling kernel must preserve a constant color at edges too.
    lut = np.tile(np.array([23, 45, 67, 255], dtype=np.uint8), (256, 1))
    pixels = _raster.render_heatmap(
        np.full((4, 4), 3.25, dtype=np.float32),
        lut,
        [0, 0, 0, 0],
        0.0,
        5.0,
        9,
        7,
        resolved.raster.filter,
        resolved.raster.bit_depth,
        1,
    )
    np.testing.assert_array_equal(pixels, np.tile(lut[0], (9, 7, 1)))


def style_fields():
    def walk(value, path=()):
        for name, child in value.items():
            if isinstance(child, dict):
                yield from walk(child, path + (name,))
            else:
                yield path + (name,), child

    fields = list(walk(config.default_style_dict()))
    fields += [
        (("typography", role, "family"), None)
        for role in config.default_style_dict()["typography"]
    ]
    return fields


STYLE_FIELDS = style_fields()


@pytest.mark.parametrize(
    "path,default", STYLE_FIELDS, ids=[".".join(p) for p, _ in STYLE_FIELDS]
)
@pytest.mark.parametrize("valid", [False, True])
def test_each_persistent_style_field_resolves_without_rewriting(
    path, default, valid, capsys
):
    name = path[-1]
    if name in ("font_family", "family"):
        value = ["serif"] if valid else []
        expected = (
            ("serif",) if valid else (tuple(default) if default is not None else None)
        )
    elif name == "weight":
        value, expected = ("bold", "bold") if valid else ("invalid", default)
    elif name == "style":
        value, expected = ("italic", "italic") if valid else ("invalid", default)
    elif isinstance(default, bool):
        value, expected = (not default, not default) if valid else ("invalid", default)
    elif isinstance(default, int):
        value, expected = (default + 1, default + 1) if valid else (0, default)
    else:
        value, expected = (
            (float(default) + 0.75,) * 2 if valid else ("invalid", default)
        )
    options = {}
    node = options
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[name] = value
    config.save_options(options)
    before = config.options_path().read_bytes()
    actual = run_options.resolve_run_options(
        argparse.Namespace(config='auto', numberOfProcessors='auto'),
        plotting=True).style
    for part in path:
        actual = getattr(actual, part)
    assert actual == expected
    assert config.options_path().read_bytes() == before
    warnings = capsys.readouterr().err
    if valid:
        assert warnings == ""
    else:
        assert ".".join(path) in warnings
