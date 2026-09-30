"""Matrix option combinations checked against independent values and metadata."""

import numpy as np
from tests.helpers.bigwig import pyBigWig
import pytest

from deeptoolsr import computeMatrix, computeMatrixOperations as cmo
from deeptoolsr.matrix import MatrixHeader, OwnedMatrix
from tests.test_computeMatrixOperations import (
    _make_multigroup_matrix,
    _load,
)
from tests.helpers.parity import read_matrix


@pytest.mark.parametrize('removed_option', [
    ['--smartLabels'], ['--startLabel', 'start'], ['--endLabel', 'end']])
def test_compute_matrix_removed_inert_options_are_rejected(removed_option):
    with pytest.raises(SystemExit):
        computeMatrix.parse_arguments().parse_args([
            'scale-regions', '-S', 'signal.bw', '-R', 'regions.bed',
            '-o', 'matrix.gz', *removed_option])


@pytest.mark.parametrize("metagene", [False, True])
@pytest.mark.parametrize("blacklist", [False, True])
def test_custom_gtf_features_metagene_blacklist_and_exports(
    tmp_path, metagene, blacklist
):
    bigwig, gtf, output = (
        tmp_path / "signal.bw",
        tmp_path / "regions.gtf",
        tmp_path / "matrix.gz",
    )
    table, bed = tmp_path / "values.tab", tmp_path / "sorted.bed"
    with pyBigWig.open(str(bigwig), "w") as bw:
        bw.addHeader([("chr1", 200)])
        bw.addEntries(
            ["chr1"] * 200,
            list(range(200)),
            ends=list(range(1, 201)),
            values=[float(i) for i in range(200)],
        )
    lines = []
    for name, start, strand in [("a", 20, "+"), ("b", 100, "-")]:
        attributes = f'gene_id "{name}"; feature_id "{name}";'
        for feature, begin, end in [
            ("rna", start, start + 60),
            ("segment", start, start + 20),
            ("segment", start + 40, start + 60),
        ]:
            lines.append(
                f"chr1\ttest\t{feature}\t{begin + 1}\t{end}\t.\t{strand}\t.\t{attributes}\n"
            )
    gtf.write_text("".join(lines))
    args = [
        "scale-regions",
        "-S",
        str(bigwig),
        "-R",
        str(gtf),
        "-o",
        str(output),
        "-b",
        "0",
        "-a",
        "0",
        "--regionBodyLength",
        "20",
        "--binSize",
        "10",
        "--transcriptID",
        "rna",
        "--exonID",
        "segment",
        "--transcript_id_designator",
        "feature_id",
        "--samplesLabel",
        "Signal",
        "--sortRegions",
        "keep",
        "--quiet",
        "--outFileNameMatrix",
        str(table),
        "--outFileSortedRegions",
        str(bed),
    ]
    if metagene:
        args += ["--metagene"]
    if blacklist:
        excluded = tmp_path / "blacklist.bed"
        excluded.write_text("chr1\t100\t170\n")
        args += ["--blackListFileName", str(excluded)]
    computeMatrix.main(args)
    header, rows, values = read_matrix(output)
    expected = np.array(
        [[29.5, 69.5], [149.5, 109.5]] if metagene else [[34.5, 64.5], [144.5, 114.5]]
    )
    if blacklist:
        expected = expected[:1]
    np.testing.assert_array_equal(values, expected)
    assert table.read_text().splitlines()[2] == f"genes:{len(expected)}\tSignal\tSignal"
    np.testing.assert_allclose(np.loadtxt(table, skiprows=3, ndmin=2), expected)
    assert header["sample_labels"] == ["Signal"]
    assert header["group_boundaries"] == [0, len(expected)]
    assert [row[3] for row in rows] == ["a", "b"][: len(expected)]
    emitted = [
        line.split("\t")[3]
        for line in bed.read_text().splitlines()
        if not line.startswith("#")
    ]
    assert emitted == ["a", "b"][: len(expected)]


@pytest.mark.parametrize("policy", ["merge", "separate"])
def test_rbind_group_policy_cli_preserves_group_membership(tmp_path, policy):
    a, b, output = [str(tmp_path / name) for name in ("a.gz", "b.gz", "out.gz")]
    for path, values in [(a, [[1, 2], [3, 4]]), (b, [[5, 6], [7, 8]])]:
        _make_multigroup_matrix(
            path,
            np.array(values, dtype=np.float32),
            [0, 1, 2],
            [0, 2],
            ["g", "h"],
            ["s"],
        )
    cmo.main(["rbind", "-m", a, b, "-o", output, "--sameGroupLabels", policy])
    result = _load(output)
    expected = (
        [[1, 2], [5, 6], [3, 4], [7, 8]]
        if policy == "merge"
        else [[1, 2], [3, 4], [5, 6], [7, 8]]
    )
    np.testing.assert_array_equal(result.values, expected)
    assert list(result.header.group_labels) == (
        ["g", "h"] if policy == "merge" else ["g", "h", "g", "h"]
    )
    assert list(result.header.group_boundaries) == (
        [0, 2, 4] if policy == "merge" else [0, 1, 2, 3, 4]
    )


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("blind", [False, True])
@pytest.mark.parametrize("duplicate_input", ["first", "later"])
def test_rbind_merge_consolidates_duplicates_within_each_input(
    tmp_path, compressed, blind, duplicate_input
):
    suffix = ".mat.gz" if compressed else ".mat"
    a, b, output = [str(tmp_path / (name + suffix))
                    for name in ("a", "b", "out")]
    if duplicate_input == "first":
        _make_multigroup_matrix(
            a, np.array([[1, 2], [3, 4]], dtype=np.float32),
            [0, 1, 2], [0, 2], ["dup", "dup"], ["s"])
        _make_multigroup_matrix(
            b, np.array([[5, 6]], dtype=np.float32),
            [0, 1], [0, 2], ["other"], ["s"])
        expected_labels = ["dup", "other"]
        expected_boundaries = [0, 2, 3]
    else:
        _make_multigroup_matrix(
            a, np.array([[1, 2]], dtype=np.float32),
            [0, 1], [0, 2], ["base"], ["s"])
        _make_multigroup_matrix(
            b, np.array([[3, 4], [5, 6]], dtype=np.float32),
            [0, 1, 2], [0, 2], ["dup", "dup"], ["s"])
        expected_labels = ["base", "dup"]
        expected_boundaries = [0, 1, 3]

    arguments = ["rbind", "-m", a, b, "-o", output]
    if blind:
        arguments.append("--blind")
    cmo.main(arguments)

    result = _load(output)
    np.testing.assert_array_equal(
        result.values,
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
    assert list(result.header.group_labels) == expected_labels
    assert list(result.header.group_boundaries) == expected_boundaries


@pytest.mark.parametrize("command", ["subset", "reorder"])
def test_merge_selected_duplicate_groups_keeps_nan_cells(tmp_path, command):
    source, output = str(tmp_path / "input.gz"), str(tmp_path / "output.gz")
    data = np.array([[1, np.nan], [np.nan, 3]], dtype=np.float32)
    _make_multigroup_matrix(source, data, [0, 1, 2], [0, 2], ["g", "g"], ["s"])
    cmo.main(
        [command, "-m", source, "-o", output, "--groups", "g", "--mergeSameNamedGroups"]
    )
    result = _load(output)
    np.testing.assert_array_equal(result.values, data)
    assert result.header.group_boundaries == (0, 2)
    assert result.header.group_labels == ("g",)


@pytest.mark.parametrize(
    "using,selected",
    [
        ("all", slice(0, 4)),
        ("upstream", slice(0, 1)),
        ("body", slice(1, 3)),
        ("downstream", slice(3, 4)),
    ],
)
@pytest.mark.parametrize("statistic", ["sum", "min", "max", "mean"])
def test_each_scale_segment_and_statistic_uses_finite_values(
    tmp_path, using, selected, statistic
):
    source, output = str(tmp_path / "input.gz"), str(tmp_path / "output.gz")
    data = np.array(
        [
            [1, 3, 5, 7, 2, 4, 6, 8],
            [np.nan, 2, np.nan, 4, 0, np.nan, 6, 8],
            [np.nan] * 4 + [0] * 4,
        ],
        dtype=np.float32,
    )
    _make_multigroup_matrix(source, data, [0, 3], [0, 4, 8], ["g"], ["s1", "s2"])
    matrix = OwnedMatrix.load(source, 1)
    parameters = dict(matrix.header.parameters)
    parameters.update({"upstream": [1, 1], "body": [2, 2],
                       "downstream": [1, 1]})
    matrix.header = MatrixHeader.from_parameters(parameters)
    matrix.save(source, compressed=True, threads=1)
    cmo.main(
        [
            "transform",
            "-m",
            source,
            "-o",
            output,
            "--scale",
            "row-" + statistic,
            "--scaleUsing",
            using,
        ]
    )
    expected = data.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for row in range(3):
            for start in (0, 4):
                candidates = data[row, start: start + 4][selected]
                finite = candidates[np.isfinite(candidates)]
                divisor = getattr(np, statistic)(finite) if len(finite) else np.nan
                expected[row, start: start + 4] /= divisor
    expected[~np.isfinite(expected)] = np.nan
    np.testing.assert_allclose(
        _load(output).values,
        expected,
        rtol=3e-7,
        atol=1.1e-6,
        equal_nan=True,
    )


@pytest.mark.parametrize("command", ["info", "dataRange"])
def test_info_and_data_range_cli_report_selected_sample_statistics(
    tmp_path, capsys, command
):
    source = str(tmp_path / "input.gz")
    data = np.array([[1, np.nan, 9, 2], [3, 5, 6, np.nan]], dtype=np.float32)
    _make_multigroup_matrix(source, data, [0, 2], [0, 2, 4], ["group"], ["s1", "s2"])
    cmo.main([command, "-m", source])
    output = capsys.readouterr().out
    assert "s1" in output and "s2" in output
    if command == "dataRange":
        for line, finite in zip(output.splitlines()[1:], [[1, 3, 5], [2, 6, 9]]):
            values = list(map(float, line.split("\t")[1:]))
            expected = [
                min(finite),
                max(finite),
                np.median(finite),
                *np.percentile(finite, [10, 90]),
            ]
            np.testing.assert_allclose(values, expected, atol=1e-4)
    else:
        assert "group" in output
