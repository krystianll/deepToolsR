from deeptoolsr import computeMatrix
from deeptoolsr import computeMatrixOperations
from deeptoolsr import plotHeatmap
from deeptoolsr import plotProfile


def test_plot_help_uses_r_command_names():
    heatmap_help = plotHeatmap.parse_arguments().format_help()
    profile_help = plotProfile.parse_arguments().format_help()

    assert "plotHeatmapR -m matrix.gz" in heatmap_help
    assert "help: plotHeatmapR -h / plotHeatmapR --help" in heatmap_help
    assert "plotProfileR -m matrix.gz" in profile_help
    assert "help: plotProfileR -h / plotProfileR --help" in profile_help


def test_matrix_help_uses_r_command_names():
    matrix_help = computeMatrix.parse_arguments().format_help()
    operations_help = computeMatrixOperations.parse_arguments().format_help()

    assert "$ computeMatrixR reference-point --help" in matrix_help
    assert "$ computeMatrixR scale-regions --help" in matrix_help
    assert "computeMatrixOperationsR subset" in operations_help
    assert "computeMatrixOperationsR transform -h" in operations_help
