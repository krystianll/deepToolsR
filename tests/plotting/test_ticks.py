"""Tests for plotting ticks."""

import matplotlib.pyplot as plt
from deeptoolsr.plotting import ticks
from deeptoolsr.plotting.ticks import (alignTickLabelsForRotation,
                                       formatDistance,
                                       formatDistanceAxisLabel,
                                       getDistanceUnit, getProfileTicks,
                                       mitigateTickLabelOverlapsForAxes)
from pathlib import Path
import pytest
from deeptoolsr.plotHeatmap import parse_arguments
from deeptoolsr.plotting.heatmap import draw_sort_indicator
from deeptoolsr.matrix import Matrix
from tests.helpers.heatmap_build import render_heatmap
from tests.helpers.plot_data import plot_data


# From test_distance_labels.py.

def _parameters(upstream=0, downstream=5000, body=0, bin_size=50):
    return {
        'upstream': [upstream],
        'downstream': [downstream],
        'body': [body],
        'bin size': [bin_size],
        'unscaled 5 prime': [0],
        'unscaled 3 prime': [0],
    }


def test_reference_point_at_zero_has_no_redundant_numeric_zero():
    ticks, labels = getProfileTicks(_parameters(), 'TSS', 'TSS', 'TES', 0)
    assert labels == ['TSS', '5 kb']
    assert len(ticks) == 2


def test_auto_unit_uses_bp_below_one_kb():
    parameters = _parameters(downstream=500)
    assert getDistanceUnit(parameters, 0, 'auto') == 'bp'
    _, labels = getProfileTicks(parameters, 'TSS', 'TSS', 'TES', 0)
    assert labels == ['TSS', '500 bp']


def test_auto_unit_uses_mb_at_one_million_bp():
    parameters = _parameters(downstream=5_000_000)
    assert getDistanceUnit(parameters, 0, 'auto') == 'mb'
    _, labels = getProfileTicks(parameters, 'TSS', 'TSS', 'TES', 0)
    assert labels == ['TSS', '5 mb']


def test_unit_can_be_moved_from_ticks_to_axis():
    _, labels = getProfileTicks(_parameters(), 'TSS', 'TSS', 'TES', 0,
                                distanceUnit='kb',
                                distanceUnitLocation='axis')
    assert labels == ['TSS', '5']


def test_distance_format_omits_trailing_zero_decimal():
    assert formatDistance(5000, 'kb') == '5 kb'
    assert formatDistance(5500, 'kb') == '5.5 kb'
    assert formatDistance(-500, 'bp') == '-500 bp'
    assert formatDistance(1_500_000, 'mb') == '1.5 mb'


def test_axis_unit_replaces_legacy_parenthesized_unit():
    assert formatDistanceAxisLabel('gene distance (bp)', 'kb', 'axis') == \
        'gene distance [kb]'
    assert formatDistanceAxisLabel('gene distance (bp)', 'kb', 'ticks') == \
        'gene distance'


def test_scaled_region_omits_zero_distance_at_named_boundaries():
    _, labels = getProfileTicks(_parameters(upstream=0, downstream=5000, body=1000),
                                'TSS', 'TSS', 'TES', 0)
    assert labels == ['TSS', 'TES', '5 kb']


def test_overlapping_tick_labels_are_justified_away_from_gap():
    fig, axis = plt.subplots(figsize=(4, 2))
    axis.set_xlim(0, 10)
    axis.set_xticks([0, 0.2, 10])
    axis.set_xticklabels(['-0.5 kb', 'TSS', '10 kb'])
    assert mitigateTickLabelOverlapsForAxes([axis])
    labels = axis.get_xticklabels()
    assert labels[0].get_horizontalalignment() == 'right'
    assert labels[1].get_horizontalalignment() == 'left'
    plt.close(fig)


def test_overlap_detection_ignores_existing_endpoint_justification():
    fig, axis = plt.subplots(figsize=(4, 2))
    axis.set_xlim(0, 11)
    axis.set_xticks([0, 0.5, 11])
    axis.set_xticklabels(['-0.5', 'ncTSS', '5.5'])
    labels = axis.get_xticklabels()
    labels[0].set_horizontalalignment('left')
    labels[-1].set_horizontalalignment('right')
    assert mitigateTickLabelOverlapsForAxes([axis])
    assert labels[0].get_horizontalalignment() == 'right'
    assert labels[1].get_horizontalalignment() == 'left'
    plt.close(fig)


def test_inward_endpoint_label_reaching_its_neighbour_is_mitigated():
    # Centred, '-50 bp' and 'TSS' would fit 50 of 300 bins apart on a 5 cm
    # axis; anchored left at the axis edge, '-50 bp' reaches past the TSS tick.
    fig = plt.figure(figsize=(5 / 2.54 + 1, 2))
    axis = fig.add_axes((.5 / (5 / 2.54 + 1), .3, (5 / 2.54) / (5 / 2.54 + 1), .6))
    axis.set_xlim(0, 300)
    axis.set_xticks([0, 50, 300])
    axis.set_xticklabels(['-50 bp', 'TSS', '250 bp'], fontsize=8)
    alignTickLabelsForRotation(axis, 0)
    assert mitigateTickLabelOverlapsForAxes([axis])
    renderer = fig.canvas.get_renderer()
    boxes = [label.get_window_extent(renderer) for label in axis.get_xticklabels()]
    assert boxes[0].x1 < boxes[1].x0 and boxes[1].x1 < boxes[2].x0
    plt.close(fig)


def test_overlapping_rotated_label_is_shifted_without_realignment():
    fig, axis = plt.subplots(figsize=(4, 2))
    axis.set_xlim(0, 11)
    axis.set_xticks([0, 0.5, 11])
    axis.set_xticklabels(['-0.5', 'ncTSS', '5.5'], rotation=45,
                         rotation_mode='anchor')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    original_x = axis.get_xticklabels()[1].get_window_extent(renderer).x0
    assert mitigateTickLabelOverlapsForAxes([axis])
    assert axis.get_xticklabels()[1].get_rotation() == 45
    shifted_x = axis.get_xticklabels()[1].get_window_extent(renderer).x0
    assert shifted_x > original_x
    assert shifted_x - original_x <= 4 * fig.dpi / 72.0 + 0.01
    assert all(label.get_horizontalalignment() == 'center'
               for label in axis.get_xticklabels())
    plt.close(fig)


def test_positive_rotation_right_anchors_all_labels():
    fig, axis = plt.subplots(figsize=(4, 2))
    axis.set_xticks([0, 1, 2])
    axis.set_xticklabels(['-0.5', 'ncTSS', '5.5'], rotation=45)
    alignTickLabelsForRotation(axis, 45)
    assert all(label.get_horizontalalignment() == 'right'
               for label in axis.get_xticklabels())
    assert all(label.get_rotation_mode() == 'anchor'
               for label in axis.get_xticklabels())
    plt.close(fig)


def test_separated_tick_labels_keep_their_justification():
    fig, axis = plt.subplots(figsize=(4, 2))
    axis.set_xlim(0, 10)
    axis.set_xticks([0, 5, 10])
    axis.set_xticklabels(['-5 kb', 'TSS', '10 kb'])
    assert not mitigateTickLabelOverlapsForAxes([axis])
    assert all(label.get_horizontalalignment() == 'center'
               for label in axis.get_xticklabels())
    plt.close(fig)


def test_batched_overlap_mitigation_measures_once_and_visits_every_axis(
        monkeypatch):
    fig, axes = plt.subplots(1, 3, figsize=(6, 2))
    for axis in axes:
        axis.set_xlim(0, 10)
        axis.set_xticks([0, 0.2, 10])
        axis.set_xticklabels(['-0.5 kb', 'TSS', '10 kb'])
    passes = 0
    original_pass = ticks.draw_without_rendering

    def count_passes(figure):
        nonlocal passes
        passes += 1
        return original_pass(figure)

    monkeypatch.setattr(ticks, 'draw_without_rendering',
                        count_passes)

    def unexpected_draw(*args, **kwargs):
        raise AssertionError('measurement must not draw the Agg canvas')

    monkeypatch.setattr(fig.canvas, 'draw', unexpected_draw)
    assert mitigateTickLabelOverlapsForAxes(axes)
    assert passes == 1
    for axis in axes:
        labels = axis.get_xticklabels()
        assert labels[0].get_horizontalalignment() == 'right'
        assert labels[1].get_horizontalalignment() == 'left'
    plt.close(fig)


def test_batched_overlap_mitigation_can_reuse_current_renderer(monkeypatch):
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    for axis in axes:
        axis.set_xlim(0, 10)
        axis.set_xticks([0, 0.2, 10])
        axis.set_xticklabels(['-0.5 kb', 'TSS', '10 kb'])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def unexpected_draw(*args, **kwargs):
        raise AssertionError('an existing renderer should avoid a new draw')

    monkeypatch.setattr(fig.canvas, 'draw', unexpected_draw)
    assert mitigateTickLabelOverlapsForAxes(axes, renderer=renderer)
    plt.close(fig)


# From test_heatmap_sort_indicator.py.

def test_parser_default_and_none():
    parser = parse_arguments()
    matrix = str(Path(__file__).parent.parent / 'test_heatmapper' / 'master.mat.gz')
    assert parser.parse_args(['-m', matrix, '-out', 'plot.png']).sortIndicator == 'auto'
    assert parser.parse_args([
        '-m', matrix, '-out', 'plot.png', '--sortIndicator', 'none'
    ]).sortIndicator == 'none'


def test_sort_indicator_polygon_directions():
    figure, axes = plt.subplots(1, 2)
    descending = draw_sort_indicator(axes[0], 'descend')
    ascending = draw_sort_indicator(axes[1], 'ascend')
    assert descending.get_xy()[:3].tolist() == [[0, 1], [1, 1], [1, 0]]
    assert ascending.get_xy()[:3].tolist() == [[1, 1], [0, 0], [1, 0]]
    plt.close(figure)


def test_unsorted_directions_have_no_polygon():
    figure, axis = plt.subplots()
    assert draw_sort_indicator(axis, 'no') is None
    assert draw_sort_indicator(axis, 'keep') is None
    assert not axis.patches
    plt.close(figure)


def test_live_sort_indicator_uses_solved_role():
    matrix = Matrix.load(str(Path(__file__).parent.parent / 'test_heatmapper' /
                             'master.mat.gz'), threads=1)
    figure, solved = render_heatmap(plot_data(matrix, sort_method='descend'),
                                    whatToShow='heatmap and colorbar')
    axis = next(axis for axis in figure.axes
                if axis.get_gid() == 'indicator/1')
    assert 'indicator/1' in solved.rects
    assert axis.patches
    rect = solved.rects['indicator/1']
    assert axis.get_position().bounds == pytest.approx((
        rect.x / solved.figure_size.width,
        rect.y / solved.figure_size.height,
        rect.width / solved.figure_size.width,
        rect.height / solved.figure_size.height))
