import matplotlib.pyplot as plt
import numpy as np
import pytest

from deeptoolsr.plotHeatmap import process_args as heatmap_args
from deeptoolsr.plotProfile import process_args as profile_args
from deeptoolsr.stats import StatisticsSpec
from deeptoolsr.plotting.profile import (
    ProfilePanelElements,
    ProfilePanelSpec,
    ProfileSeriesSpec,
    # apply_panel_y_limits,
    calculate_profile_statistics,
    draw_profile_panel,
)
from deeptoolsr.plotting.rendering import (
    resolve_figure_format, save_figure_atomic)


def test_profile_statistics_are_backend_independent():
    matrix = np.arange(40.0).reshape(10, 4)
    statistics = calculate_profile_statistics(
        matrix,
        StatisticsSpec(
            average_type='mean', plot_type='std'), threads=1)
    np.testing.assert_allclose(statistics.center, matrix.mean(axis=0))
    np.testing.assert_allclose(statistics.lower,
                               matrix.mean(axis=0) - matrix.std(axis=0))
    np.testing.assert_allclose(statistics.upper,
                               matrix.mean(axis=0) + matrix.std(axis=0))


def test_profile_statistics_preserve_missing_bin_masks():
    matrix = np.array([[1.0, np.nan], [3.0, np.nan]])
    statistics = calculate_profile_statistics(
        matrix, StatisticsSpec(average_type='mean'), threads=1)
    assert not np.ma.getmaskarray(statistics.center)[0]
    assert np.ma.getmaskarray(statistics.center)[1]


def test_same_profile_panel_renderer_supports_optional_elements():
    fig, axes = plt.subplots(1, 2)
    series = [ProfileSeriesSpec(
        np.arange(20.0).reshape(5, 4), 'series', 'blue')]
    enabled = ProfilePanelSpec(
        title='sample', series=series, x_label='distance', y_label='signal')
    disabled = ProfilePanelSpec(
        title='sample', series=series, x_label='distance', y_label='signal',
        elements=ProfilePanelElements(
            title=False, x_ticks=False, x_label=False,
            y_ticks=False, y_label=False, legend=False))
    options = StatisticsSpec()
    draw_profile_panel(axes[0], enabled, options, threads=1)
    draw_profile_panel(axes[1], disabled, options, threads=1)
    try:
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(),
                                   axes[1].lines[0].get_ydata())
        assert axes[0].get_title() == 'sample'
        assert axes[1].get_title() == ''
        assert axes[0].get_xlabel() == 'distance'
        assert axes[1].get_xlabel() == ''
        assert axes[0].get_legend() is not None
        assert axes[1].get_legend() is None
    finally:
        plt.close(fig)


# def test_shared_and_independent_limits_use_one_implementation():
#     fig, axes = plt.subplots(1, 2)
#     axes[0].plot([0, 1])
#     axes[1].plot([0, 100])
#     apply_panel_y_limits(axes, [None], [None], mode='shared')
#     assert axes[0].get_ylim() == axes[1].get_ylim()
#     axes[0].set_ylim(-1, 2)
#     axes[1].set_ylim(-10, 200)
#     apply_panel_y_limits(axes, [None], [None], mode='independent')
#     try:
#         assert axes[0].get_ylim() != axes[1].get_ylim()
#     finally:
#         plt.close(fig)


@pytest.mark.parametrize('processor', [profile_args, heatmap_args])
def test_profile_and_heatmap_no_longer_accept_plotly(processor, tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    with pytest.raises(SystemExit):
        processor(['-m', str(matrix), '-o', str(tmp_path / 'plot.html'),
                   '--plotFileFormat', 'plotly'])


@pytest.mark.parametrize('extension,signature', [
    ('pdf', b'%PDF-'),
    ('png', b'\x89PNG\r\n\x1a\n'),
])
def test_atomic_figure_save_infers_format_from_final_path(
        tmp_path, extension, signature):
    destination = tmp_path / ('plot.' + extension)
    figure = plt.figure()
    try:
        save_figure_atomic(figure, destination)
    finally:
        plt.close(figure)
    assert destination.read_bytes().startswith(signature)


def test_explicit_figure_format_overrides_destination_extension():
    assert resolve_figure_format('plot.png', 'pdf') == 'pdf'


def test_extensionless_figure_output_uses_png():
    assert resolve_figure_format('plot') == 'png'


def test_profile_line_width_flows_to_line_and_legend_handle():
    from deeptoolsr.plotting.geometry import PROFILE_LINE_WIDTH_POINTS
    fig, ax = plt.subplots()
    series = [ProfileSeriesSpec(
        np.arange(20.0).reshape(5, 4), 'series', 'blue')]
    panel = ProfilePanelSpec(title='s', series=series,
                             x_label='d', y_label='v')
    options = StatisticsSpec()
    # Default preserves the historical 1.5 pt stroke.
    draw_profile_panel(ax, panel, options, threads=1)
    assert ax.lines[0].get_linewidth() == PROFILE_LINE_WIDTH_POINTS == 1.5
    # A configured width reaches both the centre line and the legend handle
    # (the legend entry is built from the line's own get_linewidth()).
    fig2, ax2 = plt.subplots()
    draw_profile_panel(ax2, panel, options, line_width=3.25, threads=1)
    assert ax2.lines[0].get_linewidth() == 3.25
    legend_line = ax2.get_legend().get_lines()[0]
    assert legend_line.get_linewidth() == 3.25
    plt.close(fig)
    plt.close(fig2)
