from deeptoolsr.plotting.heatmap import _resolve_color_limits
import numpy as np
import argparse
import math
from pathlib import Path

import pytest

from deeptoolsr import numeric_validation as nv


MATRIX = Path(__file__).parent / "test_data" / "computeMatrixOperations.mat.gz"


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "+inf", "-inf", "1e9999"])
def test_finite_float_rejects_nonfinite_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        nv.finite_float(value)


@pytest.mark.parametrize("value", ["1e-9999", "-1e-9999"])
def test_finite_float_rejects_silent_double_underflow(value):
    with pytest.raises(argparse.ArgumentTypeError):
        nv.finite_float(value)


@pytest.mark.parametrize("value", ["3.5e38", "1e-50", "-3.5e38"])
def test_float32_scalar_rejects_overflow_and_underflow(value):
    with pytest.raises(argparse.ArgumentTypeError):
        nv.float32_finite(value)


@pytest.mark.parametrize("value", ["5.5", "5e0", "nan", "inf", str(1 << 80)])
def test_positive_int_rejects_fractional_nonfinite_and_overflow_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        nv.positive_int(value)


def test_directional_infinities_are_only_valid_for_open_filter_side():
    assert nv.lower_bound_float("-inf") == -math.inf
    assert nv.upper_bound_float("inf") == math.inf
    with pytest.raises(argparse.ArgumentTypeError):
        nv.lower_bound_float("inf")
    with pytest.raises(argparse.ArgumentTypeError):
        nv.upper_bound_float("-inf")


@pytest.mark.parametrize("value", ["-2", "0", "13"])
def test_compression_level_rejects_values_outside_minus_one_or_one_to_twelve(value):
    with pytest.raises(argparse.ArgumentTypeError):
        nv.compression_level(value)


@pytest.mark.parametrize("value,expected", [("-1", -1), ("1", 1), ("12", 12)])
def test_compression_level_accepts_supported_values(value, expected):
    assert nv.compression_level(value) == expected


def test_min_max_validation_supports_inclusive_and_strict_ranges():
    assert nv.min_max_error(1, 1, "minimum", "maximum") is None
    assert nv.min_max_error(2, 1, "minimum", "maximum")
    assert nv.min_max_error(1, 1, "minimum", "maximum", strict=True)


@pytest.mark.parametrize("extra", [
    ["--binSize", "5.5"],
    ["--binSize", "0"],
    ["--scale", "nan"],
    ["--minThreshold", "inf"],
    ["--maxThreshold", "-inf"],
])
def test_compute_matrix_rejects_invalid_numeric_arguments(tmp_path, extra):
    from deeptoolsr import computeMatrix
    args = [
        "reference-point", "-R", "regions.bed", "-S", "signal.bw",
        "-o", str(tmp_path / "out.gz"),
    ] + extra
    with pytest.raises(SystemExit):
        computeMatrix.process_args(args)


def test_compute_matrix_rejects_inverted_thresholds(tmp_path):
    from deeptoolsr import computeMatrix
    with pytest.raises(SystemExit):
        computeMatrix.process_args([
            "reference-point", "-R", "regions.bed", "-S", "signal.bw",
            "-o", str(tmp_path / "out.gz"),
            "--minThreshold", "2", "--maxThreshold", "1"])


@pytest.mark.parametrize("args", [
    ["filterValues", "-m", "in.gz", "-o", "out.gz", "--min", "nan"],
    ["filterValues", "-m", "in.gz", "-o", "out.gz", "--max", "-inf"],
    ["filterValues", "-m", "in.gz", "-o", "out.gz", "--min", "2", "--max", "1"],
    ["transform", "-m", "in.gz", "-o", "out.gz", "--add", "inf"],
    ["transform", "-m", "in.gz", "-o", "out.gz", "--scale", "nan"],
    ["transform", "-m", "in.gz", "-o", "out.gz", "--pseudocount", "inf"],
])
def test_matrix_operations_reject_invalid_numeric_arguments(tmp_path, args):
    from deeptoolsr import computeMatrixOperations
    mapped = [str(tmp_path / value) if value in ("in.gz", "out.gz") else value
              for value in args]
    with pytest.raises(SystemExit):
        computeMatrixOperations.main(mapped)


@pytest.mark.parametrize("module_name,extra", [
    ("plotProfile", ["--dpi", "0"]),
    ("plotProfile", ["--profileWidth", "nan"]),
    ("plotProfile", ["--pseudocount", "inf"]),
    ("plotProfile", ["--yMin", "nan"]),
    ("plotHeatmap", ["--dpi", "0"]),
    ("plotHeatmap", ["--heatmapWidth", "inf"]),
    ("plotHeatmap", ["--zMin", "1-2=nan"]),
    ("plotHeatmap", ["--zMax", "-inf"]),
])
def test_plot_commands_reject_invalid_numeric_arguments(
        tmp_path, module_name, extra):
    from deeptoolsr import plotHeatmap, plotProfile
    module = plotHeatmap if module_name == "plotHeatmap" else plotProfile
    args = extra + ["-m", str(MATRIX), "-o", str(tmp_path / "out.pdf")]
    with pytest.raises(SystemExit):
        module.parse_arguments().parse_args(args)


def test_recycled_limits_validate_every_used_pair():
    # Independent recycling repeats over lcm(2, 3) = 6 positions; with four
    # panels the fourth pair is (10, 5).
    assert nv.recycled_min_max_error(
        [0, 10], [5, 20, 30], 3, '--yMin', '--yMax', strict=True) is None
    error = nv.recycled_min_max_error(
        [0, 10], [5, 20, 30], 4, '--yMin', '--yMax', strict=True)
    assert error == ('--yMin must be less than --yMax for recycled pair 4 '
                     '(10, 5)')
    assert nv.recycled_min_max_error(
        ['auto', 3], [1], 2, '--zMin', '--zMax', strict=True).endswith(
            'pair 2 (3, 1)')


@pytest.mark.parametrize("module_name,extra,message", [
    ("plotProfile", ["--yMin", "0", "3", "--yMax", "1"], "pair 2 (3, 1)"),
    ("plotProfile", ["--yMin", "0", "10", "--yMax", "5", "20", "30"],
     "pair 4 (10, 5)"),
    ("plotHeatmap", ["--zMin", "0", "10", "--zMax", "5", "20", "30"],
     "pair 4 (10, 5)"),
    ("plotHeatmap", ["--zMin", "1", "--zMax", "1"], "pair 1 (1, 1)"),
    ("plotHeatmap", ["--zMin", "2", "--zMax", "1"], "pair 1 (2, 1)"),
])
def test_plots_reject_contradictory_recycled_limits(
        tmp_path, capsys, module_name, extra, message):
    # computeMatrixOperations.mat.gz has four samples: four profile panels
    # and four heatmap colour scales.
    from deeptoolsr import plotHeatmap, plotProfile
    module = plotHeatmap if module_name == "plotHeatmap" else plotProfile
    output = tmp_path / "out.png"
    with pytest.raises(SystemExit):
        module.main(["-m", str(MATRIX), "-o", str(output)] + extra)
    assert message in capsys.readouterr().err
    assert not output.exists()


# From general audit regressions.

@pytest.mark.parametrize('lower,upper,quantiles,expected', [(None, None, [0, 0, 0, 10], ([0], [10])), (None, None, [1, 1, 1, 1], ([0.95], [1.05])), (None, None, [np.nan] * 4, ([0], [1])), (['10'], None, [0, 0, 1, 1], ([10], [10.5])), (None, ['-10'], [0, 0, 1, 1], ([-10.5], [-10])), (['auto', '2'], ['4', 'auto'], [0, 1, 3, 5], ([1, 2], [4, 3]))])
def test_color_range_fallbacks_preserve_explicit_bounds(lower, upper, quantiles, expected):
    observed = _resolve_color_limits(lower, upper, quantiles)
    np.testing.assert_allclose(observed, expected)
