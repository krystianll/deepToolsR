"""Tests for plotting geometry."""

from deeptoolsr.plotting import fonts as style_fonts
from deeptoolsr.plotting import geometry as style_geometry
from matplotlib.transforms import Bbox
from deeptoolsr import config, options as run_options
import argparse
from deeptoolsr.plotting.geometry import (
    Rect, bbox_to_size_points, cm_to_points, points_to_inches)
from deeptoolsr.plotting.scene import MatplotlibMeasureContext, measure_axis
from pathlib import Path
import pytest
from tests.helpers.heatmap_build import render_heatmap
from tests.helpers.plot_data import plot_data
from deeptoolsr.matrix import Matrix
from deeptoolsr.plotting.grid import (
    BelowColorbarEntry, pack_below_common, pack_colorbar_grid)
from deeptoolsr.plotting.geometry import (
    COLORBAR_LONG_EDGE_GAP_POINTS, COLORBAR_MIN_HEIGHT_POINTS,
    COLORBAR_SHORT_EDGE_GAP_POINTS, HEATMAP_COLORBAR_THICKNESS_POINTS)
from deeptoolsr.plotting.ticks import choose_colorbar_ticks
import matplotlib.pyplot as plt
import numpy as np
from deeptoolsr.plotHeatmap import process_args
from deeptoolsr.plotting.heatmap import add_minor_x_ticks, distribute_minor_ticks
import dataclasses
from deeptoolsr.plotting.geometry import GeometrySpec


# From test_plotting_geometry.py.

def test_unit_conversions_and_bbox_size():
    assert cm_to_points(2.54) == 72
    assert points_to_inches(144) == 2
    assert bbox_to_size_points(Bbox.from_bounds(0, 0, 200, 100), 100).width == 144


def test_rectangle_edges():
    rect = Rect(1, 2, 3, 4)
    assert (rect.x0, rect.x1, rect.y0, rect.y1) == (1, 4, 2, 6)


def test_hidden_parent_axes_do_not_measure_visible_child_decorations():
    context = MatplotlibMeasureContext(100)
    axis = context.measure_axis
    axis.set_axis_on()
    axis.set_xlabel('child text remains individually visible')
    axis.set_ylabel('another visible child')
    axis.set_xticks([0, .5, 1], ['left', 'middle', 'right'])
    axis.set_yticks([0, .5, 1], ['bottom', 'middle', 'top'])
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    context.figure.canvas.draw()
    assert measure_axis(axis, context.renderer).left == 0
    assert measure_axis(axis, context.renderer).right == 0
    assert measure_axis(axis, context.renderer).top == 0
    assert measure_axis(axis, context.renderer).bottom == 0
    context.close()


def test_y_tick_labels_do_not_reserve_vertical_flow_space():
    context = MatplotlibMeasureContext(100)
    axis = context.measure_axis
    axis.set_axis_on()
    axis.set_position((.2, .2, .6, .6))
    axis.set_xticks([])
    axis.set_yticks([0, 1], ['bottom boundary', 'top boundary'])
    context.figure.canvas.draw()
    measured = measure_axis(axis, context.renderer)
    assert measured.left > 0
    assert measured.top == 0
    assert measured.bottom == 0
    context.close()


def test_default_style_resolves_without_messages_and_round_trips():
    resolved, messages = style_geometry.resolve_style(None)
    assert messages == []
    assert resolved.font_family == ('sans-serif',)
    assert resolved.font_multiplier == 1.0
    assert resolved.typography.figure_title_text == style_geometry.FontSpec(
        12.0, 'normal')
    assert resolved.typography.panel_title_text == style_geometry.FontSpec(
        10.0, 'normal')
    assert resolved.typography.x_tick_label_text == style_geometry.FontSpec(
        8.0, 'normal')
    assert resolved.typography.axis_label_text == style_geometry.FontSpec(
        8.0, 'normal')
    assert resolved.geometry.figure_edge_padding == \
        style_geometry.FIGURE_EDGE_PADDING_POINTS
    assert resolved.drawing.profile_line_width == 1.5
    assert resolved.label_layout.auto_panel_title_column_gap is True
    assert resolved.label_layout.max_column_gap_points == 96.0
    assert resolved.label_layout.panel_title_max_lines == 3
    assert resolved.label_layout.panel_title_extra_row_penalty == 0.08
    assert resolved.label_layout.panel_title_balance_weight == 0.02
    assert resolved.label_layout.panel_title_orphan_weight == 0.05
    assert resolved.label_layout.axis_label_max_lines == 3
    assert resolved.label_layout.axis_label_extra_row_penalty == 0.08
    assert resolved.label_layout.axis_label_balance_weight == 0.02
    assert resolved.label_layout.axis_label_orphan_weight == 0.05
    assert resolved.label_layout.auto_axis_label_layout is True
    assert resolved.label_layout.auto_horizontal_colorbar_label_layout is True
    assert resolved.label_layout.auto_facet_label_layout is True
    assert resolved.label_layout.auto_heatmap_region_label_layout is True
    assert resolved.label_layout.heatmap_region_label_max_lines == 3
    assert resolved.label_layout.heatmap_region_label_extra_row_penalty == 0.08
    assert resolved.label_layout.heatmap_region_label_balance_weight == 0.02
    assert resolved.label_layout.heatmap_region_label_orphan_weight == 0.05
    assert resolved.label_layout.max_row_gap_points == 96.0
    assert resolved.label_layout.max_heatmap_region_gap_points == 96.0
    # The serialised defaults resolve back to exactly the same object.
    again, again_messages = style_geometry.resolve_style(
        style_geometry.default_style_dict())
    assert again_messages == []
    assert again == resolved


def test_spec_defaults_match_legacy_constants():
    # The dataclass defaults derive from the module constants, so the two can
    # never drift while the constants remain as compatibility aliases.
    geo = style_geometry.DEFAULT_GEOMETRY
    assert geo.axis_label_to_tick_labels_gap == \
        style_geometry.AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS
    assert geo.colorbar_label_min_clearance == \
        style_geometry.COLORBAR_LABEL_MIN_CLEARANCE_POINTS
    assert geo.colorbar_below_common_max_gap == \
        style_geometry.COLORBAR_BELOW_COMMON_MAX_GAP_POINTS
    assert geo.profile_row_gap == style_geometry.PROFILE_ROW_ADDITIONAL_GAP_POINTS
    assert geo.heatmap_colorbar_thickness == \
        style_geometry.HEATMAP_COLORBAR_THICKNESS_POINTS


def test_scalar_font_family_is_normalised_to_a_list():
    resolved, messages = style_geometry.resolve_style({'font_family': 'Arial'})
    assert resolved.font_family == ('Arial',)
    assert messages == []
    resolved, _ = style_geometry.resolve_style(
        {'font_family': ['Arial', 'Helvetica', 'sans-serif']})
    assert resolved.font_family == ('Arial', 'Helvetica', 'sans-serif')


def test_a_gap_may_be_zero_but_a_thickness_may_not():
    resolved, messages = style_geometry.resolve_style(
        {'geometry': {'legend_column_gap': 0}})
    assert resolved.geometry.legend_column_gap == 0.0
    assert messages == []
    resolved, messages = style_geometry.resolve_style(
        {'geometry': {'heatmap_colorbar_thickness': 0}})
    assert resolved.geometry.heatmap_colorbar_thickness == \
        style_geometry.HEATMAP_COLORBAR_THICKNESS_POINTS
    assert any('heatmap_colorbar_thickness' in msg for msg in messages)


def test_label_layout_policy_is_validated_and_resolves_custom_values():
    resolved, messages = style_geometry.resolve_style({
        'label_layout': {
            'auto_panel_title_column_gap': False,
            'max_column_gap_points': 54.0,
            'panel_title_max_lines': 4,
            'panel_title_extra_row_penalty': 0.12,
            'panel_title_balance_weight': 0.03,
            'panel_title_orphan_weight': 0.07,
            'axis_label_max_lines': 5,
            'axis_label_extra_row_penalty': 0.14,
            'axis_label_balance_weight': 0.04,
            'axis_label_orphan_weight': 0.09,
            'auto_axis_label_layout': False,
            'auto_horizontal_colorbar_label_layout': False,
            'auto_facet_label_layout': False,
            'auto_heatmap_region_label_layout': False,
            'heatmap_region_label_max_lines': 5,
            'heatmap_region_label_extra_row_penalty': 0.12,
            'heatmap_region_label_balance_weight': 0.03,
            'heatmap_region_label_orphan_weight': 0.07,
            'max_row_gap_points': 54.0,
            'max_heatmap_region_gap_points': 73.0,
        }})
    assert messages == []
    assert resolved.label_layout.auto_panel_title_column_gap is False
    assert resolved.label_layout.max_column_gap_points == 54.0
    assert resolved.label_layout.panel_title_max_lines == 4
    assert resolved.label_layout.panel_title_extra_row_penalty == 0.12
    assert resolved.label_layout.panel_title_balance_weight == 0.03
    assert resolved.label_layout.panel_title_orphan_weight == 0.07
    assert resolved.label_layout.axis_label_max_lines == 5
    assert resolved.label_layout.axis_label_extra_row_penalty == 0.14
    assert resolved.label_layout.axis_label_balance_weight == 0.04
    assert resolved.label_layout.axis_label_orphan_weight == 0.09
    assert resolved.label_layout.auto_axis_label_layout is False
    assert resolved.label_layout.auto_horizontal_colorbar_label_layout is False
    assert resolved.label_layout.auto_facet_label_layout is False
    assert resolved.label_layout.auto_heatmap_region_label_layout is False
    assert resolved.label_layout.heatmap_region_label_max_lines == 5
    assert resolved.label_layout.heatmap_region_label_extra_row_penalty == 0.12
    assert resolved.label_layout.heatmap_region_label_balance_weight == 0.03
    assert resolved.label_layout.heatmap_region_label_orphan_weight == 0.07
    assert resolved.label_layout.max_row_gap_points == 54.0
    assert resolved.label_layout.max_heatmap_region_gap_points == 73.0

    resolved, messages = style_geometry.resolve_style({
        'label_layout': {
            'auto_panel_title_column_gap': 'yes',
            'max_column_gap_points': -1,
            'panel_title_max_lines': True,
            'panel_title_extra_row_penalty': float('nan'),
            'panel_title_balance_weight': -1,
            'panel_title_orphan_weight': float('inf'),
            'axis_label_max_lines': True,
            'axis_label_extra_row_penalty': float('nan'),
            'axis_label_balance_weight': -1,
            'axis_label_orphan_weight': float('inf'),
            'auto_axis_label_layout': 'yes',
            'auto_horizontal_colorbar_label_layout': 1,
            'auto_facet_label_layout': None,
            'auto_heatmap_region_label_layout': 0,
            'heatmap_region_label_max_lines': True,
            'heatmap_region_label_extra_row_penalty': float('nan'),
            'heatmap_region_label_balance_weight': -1,
            'heatmap_region_label_orphan_weight': float('inf'),
            'max_row_gap_points': float('nan'),
            'max_heatmap_region_gap_points': -2,
        }})
    assert resolved.label_layout.auto_panel_title_column_gap is True
    assert resolved.label_layout.max_column_gap_points == 96.0
    assert resolved.label_layout.panel_title_max_lines == 3
    assert resolved.label_layout.panel_title_extra_row_penalty == 0.08
    assert resolved.label_layout.panel_title_balance_weight == 0.02
    assert resolved.label_layout.panel_title_orphan_weight == 0.05
    assert resolved.label_layout.axis_label_max_lines == 3
    assert resolved.label_layout.axis_label_extra_row_penalty == 0.08
    assert resolved.label_layout.axis_label_balance_weight == 0.02
    assert resolved.label_layout.axis_label_orphan_weight == 0.05
    assert resolved.label_layout.auto_axis_label_layout is True
    assert resolved.label_layout.auto_horizontal_colorbar_label_layout is True
    assert resolved.label_layout.auto_facet_label_layout is True
    assert resolved.label_layout.auto_heatmap_region_label_layout is True
    assert resolved.label_layout.heatmap_region_label_max_lines == 3
    assert resolved.label_layout.heatmap_region_label_extra_row_penalty == 0.08
    assert resolved.label_layout.heatmap_region_label_balance_weight == 0.02
    assert resolved.label_layout.heatmap_region_label_orphan_weight == 0.05
    assert resolved.label_layout.max_row_gap_points == 96.0
    assert resolved.label_layout.max_heatmap_region_gap_points == 96.0
    assert any('auto_panel_title_column_gap' in message
               for message in messages)
    assert any('max_column_gap_points' in message for message in messages)
    assert any('panel_title_max_lines' in message for message in messages)
    assert any('panel_title_extra_row_penalty' in message
               for message in messages)
    assert any('panel_title_balance_weight' in message for message in messages)
    assert any('panel_title_orphan_weight' in message for message in messages)
    assert any('axis_label_max_lines' in message for message in messages)
    assert any('axis_label_extra_row_penalty' in message
               for message in messages)
    assert any('axis_label_balance_weight' in message for message in messages)
    assert any('axis_label_orphan_weight' in message for message in messages)
    assert any('auto_axis_label_layout' in message for message in messages)
    assert any('auto_horizontal_colorbar_label_layout' in message
               for message in messages)
    assert any('auto_facet_label_layout' in message for message in messages)
    assert any('auto_heatmap_region_label_layout' in message
               for message in messages)
    assert any('heatmap_region_label_max_lines' in message
               for message in messages)
    assert any('heatmap_region_label_extra_row_penalty' in message
               for message in messages)
    assert any('heatmap_region_label_balance_weight' in message
               for message in messages)
    assert any('heatmap_region_label_orphan_weight' in message
               for message in messages)
    assert any('max_row_gap_points' in message for message in messages)
    assert any('max_heatmap_region_gap_points' in message
               for message in messages)


def test_label_layout_policy_round_trips_through_persistent_options(
        tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    options = config.build_persisted_options(config.load_options())
    assert options['label_layout'] == {
        'auto_panel_title_column_gap': True,
        'max_column_gap_points': 96.0,
        'panel_title_max_lines': 3,
        'panel_title_extra_row_penalty': 0.08,
        'panel_title_balance_weight': 0.02,
        'panel_title_orphan_weight': 0.05,
        'axis_label_max_lines': 3,
        'axis_label_extra_row_penalty': 0.08,
        'axis_label_balance_weight': 0.02,
        'axis_label_orphan_weight': 0.05,
        'auto_axis_label_layout': True,
        'auto_horizontal_colorbar_label_layout': True,
        'horizontal_colorbar_label_max_lines': 3,
        'horizontal_colorbar_label_extra_row_penalty': 0.08,
        'horizontal_colorbar_label_balance_weight': 0.02,
        'horizontal_colorbar_label_orphan_weight': 0.05,
        'auto_facet_label_layout': True,
        'auto_heatmap_region_label_layout': True,
        'heatmap_region_label_gap_growth': True,
        'heatmap_region_label_max_lines': 3,
        'heatmap_region_label_extra_row_penalty': 0.08,
        'heatmap_region_label_balance_weight': 0.02,
        'heatmap_region_label_orphan_weight': 0.05,
        'max_row_gap_points': 96.0,
        'max_heatmap_region_gap_points': 96.0,
        'auto_legend_label_layout': True,
        'legend_label_max_lines': 3,
    }
    options['label_layout'].update({
        'auto_panel_title_column_gap': False,
        'max_column_gap_points': 54.0,
        'panel_title_max_lines': 4,
        'panel_title_extra_row_penalty': 0.12,
        'panel_title_balance_weight': 0.03,
        'panel_title_orphan_weight': 0.07,
        'axis_label_max_lines': 5,
        'axis_label_extra_row_penalty': 0.14,
        'axis_label_balance_weight': 0.04,
        'axis_label_orphan_weight': 0.09,
        'auto_axis_label_layout': False,
        'auto_horizontal_colorbar_label_layout': False,
        'horizontal_colorbar_label_max_lines': 5,
        'horizontal_colorbar_label_extra_row_penalty': 0.16,
        'horizontal_colorbar_label_balance_weight': 0.06,
        'horizontal_colorbar_label_orphan_weight': 0.11,
        'auto_facet_label_layout': False,
        'auto_heatmap_region_label_layout': False,
        'heatmap_region_label_max_lines': 5,
        'heatmap_region_label_extra_row_penalty': 0.18,
        'heatmap_region_label_balance_weight': 0.08,
        'heatmap_region_label_orphan_weight': 0.13,
        'max_row_gap_points': 54.0,
        'max_heatmap_region_gap_points': 73.0,
    })
    config.save_options(options)

    style = run_options.resolve_run_options(
        argparse.Namespace(config='auto', numberOfProcessors='auto'),
        plotting=True).style
    assert style.label_layout.auto_panel_title_column_gap is False
    assert style.label_layout.max_column_gap_points == 54.0
    assert style.label_layout.panel_title_max_lines == 4
    assert style.label_layout.panel_title_extra_row_penalty == 0.12
    assert style.label_layout.panel_title_balance_weight == 0.03
    assert style.label_layout.panel_title_orphan_weight == 0.07
    assert style.label_layout.axis_label_max_lines == 5
    assert style.label_layout.axis_label_extra_row_penalty == 0.14
    assert style.label_layout.axis_label_balance_weight == 0.04
    assert style.label_layout.axis_label_orphan_weight == 0.09
    assert style.label_layout.auto_axis_label_layout is False
    assert style.label_layout.auto_horizontal_colorbar_label_layout is False
    assert style.label_layout.horizontal_colorbar_label_max_lines == 5
    assert style.label_layout.horizontal_colorbar_label_extra_row_penalty \
        == 0.16
    assert style.label_layout.horizontal_colorbar_label_balance_weight == 0.06
    assert style.label_layout.horizontal_colorbar_label_orphan_weight == 0.11
    assert style.label_layout.auto_facet_label_layout is False
    assert style.label_layout.auto_heatmap_region_label_layout is False
    assert style.label_layout.heatmap_region_label_max_lines == 5
    assert style.label_layout.heatmap_region_label_extra_row_penalty == 0.18
    assert style.label_layout.heatmap_region_label_balance_weight == 0.08
    assert style.label_layout.heatmap_region_label_orphan_weight == 0.13
    assert style.label_layout.max_row_gap_points == 54.0
    assert style.label_layout.max_heatmap_region_gap_points == 73.0


def test_one_malformed_entry_does_not_disturb_the_rest():
    resolved, messages = style_geometry.resolve_style({
        'font_multiplier': 'big',
        'typography': {'x_tick_label_text': {'size': -3, 'weight': 'bold'},
                       'unknown_role': {}},
        'geometry': {'figure_edge_padding': 'wide', 'unknown_gap': 1},
        'drawing': {'profile_line_width': None},
    })
    # Each bad field falls back independently; valid siblings are kept.
    assert resolved.font_multiplier == 1.0
    assert resolved.typography.x_tick_label_text.size == 8.0
    assert resolved.typography.x_tick_label_text.weight == 'bold'
    assert resolved.geometry.figure_edge_padding == \
        style_geometry.FIGURE_EDGE_PADDING_POINTS
    assert resolved.drawing.profile_line_width == 1.5
    assert any('unknown_role' in msg for msg in messages)
    assert any('unknown_gap' in msg for msg in messages)


def test_per_role_family_overrides_and_styles_validate():
    resolved, messages = style_geometry.resolve_style({
        'typography': {
            'legend_label_text': {'size': 9, 'style': 'italic',
                                  'family': 'Courier'}}})
    role = resolved.typography.legend_label_text
    assert role.size == 9.0
    assert role.style == 'italic'
    assert role.family == ('Courier',)
    assert messages == []
    # An unknown style is reported and falls back.
    resolved, messages = style_geometry.resolve_style(
        {'typography': {'legend_label_text': {'style': 'slanted'}}})
    assert resolved.typography.legend_label_text.style == 'normal'
    assert any('legend_label_text.style' in msg for msg in messages)


def test_scaled_size_applies_multiplier_without_mutating_base():
    spec = style_geometry.FontSpec(10.0, 'normal')
    assert style_geometry.scaled_size(spec, 1.5) == 15.0
    assert spec.size == 10.0


def test_role_font_carries_size_family_and_multiplier():
    style, _ = style_geometry.resolve_style(
        {'font_family': ['Arial', 'sans-serif'], 'font_multiplier': 1.5})
    fp = style_fonts.role_font(style, 'figure_title_text')
    assert fp.get_size() == 18.0            # 12 base * 1.5, applied at draw time
    assert fp.get_family() == ['Arial', 'sans-serif']
    # The stored base size is untouched by the multiplier (no compounding).
    assert style.typography.figure_title_text.size == 12.0


def test_per_role_family_overrides_global_in_adapter():
    style, _ = style_geometry.resolve_style({
        'font_family': ['Arial'],
        'typography': {'legend_label_text': {'family': 'Courier'}}})
    assert style_fonts.role_font(style, 'legend_label_text').get_family() == \
        ['Courier']
    assert style_fonts.role_font(style, 'x_tick_label_text').get_family() == \
        ['Arial']


def test_style_rc_seeds_sizes_from_the_resolved_style():
    import matplotlib
    style, _ = style_geometry.resolve_style({'font_multiplier': 2.0})
    # rc_context restores the global rcParams so this test cannot leak font
    # sizes into others.
    with matplotlib.rc_context(style_fonts.style_rc(style)):
        # 8 pt body roles scale to 16 pt and the 10 pt panel title to 20 pt; the
        # 12 pt figure title is applied per-call, not through these params.
        assert matplotlib.rcParams['font.size'] == 16.0        # x tick label
        assert matplotlib.rcParams['axes.labelsize'] == 16.0   # axis label
        assert matplotlib.rcParams['axes.titlesize'] == 20.0   # panel title
        assert matplotlib.rcParams['legend.fontsize'] == 16.0  # legend label


def test_style_rc_sets_tick_pad_from_geometry_unmultiplied():
    import matplotlib
    # tick pad is a physical geometry gap: the font multiplier must not scale it.
    style, _ = style_geometry.resolve_style(
        {'font_multiplier': 2.0, 'geometry': {'tick_label_to_tick_gap': 5.0}})
    with matplotlib.rc_context(style_fonts.style_rc(style)):
        assert matplotlib.rcParams['xtick.major.pad'] == 5.0
        assert matplotlib.rcParams['ytick.minor.pad'] == 5.0
    # The default reproduces Matplotlib's historical 3.5 pt tick pad.
    default_style, _ = style_geometry.resolve_style(None)
    assert default_style.geometry.tick_label_to_tick_gap == 3.5


def test_apply_axis_fonts_applies_complete_role_specs():
    import matplotlib
    import matplotlib.pyplot as plt

    style, messages = style_geometry.resolve_style({'typography': {
        'panel_title_text': {
            'size': 13, 'weight': 'bold', 'style': 'italic',
            'family': 'Courier'},
        'axis_label_text': {
            'size': 11, 'weight': 'semibold', 'style': 'oblique',
            'family': 'Helvetica'},
        'x_tick_label_text': {
            'size': 9, 'weight': 'bold', 'style': 'italic',
            'family': 'Arial'},
        'y_tick_label_text': {
            'size': 10, 'weight': 'light', 'style': 'oblique',
            'family': 'Courier'},
    }})
    assert messages == []
    with matplotlib.rc_context(style_fonts.style_rc(style)):
        figure, axis = plt.subplots()
        axis.set_title('panel')
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.set_xticks([0], ['x tick'])
        axis.set_yticks([0], ['y tick'])
        style_fonts.apply_axis_fonts(axis, style)

        assert axis.title.get_fontproperties().get_family() == ['Courier']
        assert axis.title.get_fontstyle() == 'italic'
        assert axis.title.get_fontweight() == 'bold'
        assert axis.xaxis.label.get_fontproperties().get_family() == ['Helvetica']
        assert axis.xaxis.label.get_fontstyle() == 'oblique'
        assert axis.get_xticklabels()[0].get_fontproperties().get_family() == ['Arial']
        assert axis.get_xticklabels()[0].get_fontweight() == 'bold'
        assert axis.get_yticklabels()[0].get_fontproperties().get_family() == ['Courier']
        assert axis.get_yticklabels()[0].get_fontstyle() == 'oblique'
        plt.close(figure)


def test_apply_colorbar_tick_fonts_uses_the_complete_colorbar_role():
    import matplotlib.pyplot as plt

    style, messages = style_geometry.resolve_style({'typography': {
        'colorbar_tick_label_text': {
            'size': 14, 'weight': 'bold', 'style': 'italic',
            'family': 'Courier'}}})
    assert messages == []
    figure, axis = plt.subplots()
    axis.set_xticks([0], ['tick'])
    style_fonts.apply_colorbar_tick_fonts(axis, style)
    label = axis.get_xticklabels()[0]
    assert label.get_fontsize() == 14
    assert label.get_fontproperties().get_family() == ['Courier']
    assert label.get_fontweight() == 'bold'
    assert label.get_fontstyle() == 'italic'
    plt.close(figure)


# From test_colorbar_grid.py.

def test_tick_helper_outer_near_edges_and_even_centre():
    positions, labels = choose_colorbar_ticks(-3.245, 2.113, 3.0)
    assert positions == [-3.0, -0.5, 2.0]        # roundest near edges, centred
    assert labels == ['-3.0', '-0.5', '2.0']
    # Outer ticks sit within 10% of each end.
    span = 2.113 - (-3.245)
    assert (-3.0 - (-3.245)) <= 0.10 * span
    assert (2.113 - 2.0) <= 0.10 * span


def test_tick_helper_prefers_zero_and_clean_values():
    assert choose_colorbar_ticks(-1.0, 1.0, 3.0)[0] == [-1.0, 0.0, 1.0]
    assert choose_colorbar_ticks(0.0, 10.0, 3.0)[0] == [0.0, 5.0, 10.0]
    assert choose_colorbar_ticks(0.0, 1.0, 3.0)[0] == [0.0, 0.5, 1.0]


def test_tick_helper_densifies_on_taller_bars():
    # A short bar keeps the sparse near-edge set; a taller one gets more,
    # evenly-spaced, round ticks (still reaching close to both ends).
    assert choose_colorbar_ticks(-2.0, 1.0, 2.8)[0] == [-2.0, -0.5, 1.0]
    dense = choose_colorbar_ticks(-2.0, 1.0, 6.0)[0]
    assert dense == [-2.0, -1.0, 0.0, 1.0]
    awkward = choose_colorbar_ticks(-3.245, 2.113, 6.0)[0]
    assert awkward == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]   # reaches the edges


def test_tick_helper_count_and_spacing_are_tunable():
    # prefer_count raises the target density...
    assert len(choose_colorbar_ticks(-2.0, 1.0, 6.0, prefer_count=7)[0]) == 7
    # ...and min_spacing_cm is the floor that a short bar hits first.
    assert len(choose_colorbar_ticks(-2.0, 1.0, 2.8, min_spacing_cm=0.5)[0]) > 3


def test_tick_helper_odd_count_and_min_spacing():
    # A narrow bar cannot hold three ticks 1 cm apart -> a single centre tick.
    positions, _ = choose_colorbar_ticks(-3.0, 2.0, 1.5)
    assert len(positions) == 1


def test_tick_helper_tiny_magnitudes_use_scientific():
    _, labels = choose_colorbar_ticks(-5e-4, 5e-4, 3.0)
    # Compact scientific form: -5e-4, not -5.0e-04; zero stays plain.
    assert labels == ['-5e-4', '0', '5e-4']


def test_tick_helper_scientific_kicks_in_at_the_third_power():
    from deeptoolsr.plotting.ticks import _format_tick_values
    # 0.005 (3rd power) reads as 5e-3; 0.05 stays an ordinary decimal.
    assert _format_tick_values([-0.005, 0.0, 0.005]) == ['-5e-3', '0', '5e-3']
    assert _format_tick_values([-0.05, 0.0, 0.05]) == ['-0.05', '0.00', '0.05']


DATA = Path(__file__).parent.parent / 'test_heatmapper'


def _render(colorMapDict, **kwargs):
    """Render master_multi without touching disk; return (figure, solution)."""
    hm = Matrix.load(str(DATA / 'master_multi.mat.gz'), threads=1)
    return render_heatmap(
        plot_data(hm), colorMapDict=colorMapDict,
        whatToShow='heatmap and colorbar', image_format='png', dpi=100,
        threads=1, **kwargs)


def _side_colorbar_axes(figure):
    return [axis for axis in figure.axes
            if (axis.get_gid() or '').startswith('colorbar/row/1/')]


def _column_of(rect):
    return round(rect.x, 3)


def test_single_colorbar_spans_full_block_height():
    rects, width = pack_colorbar_grid(
        1, block_height=400.0, base_x=100.0, top_y=500.0,
        tick_label_widths=[12.0])
    assert len(rects) == 1
    assert rects[0].height == pytest.approx(400.0)
    assert rects[0].y0 == pytest.approx(100.0)          # bottom-aligned to block
    assert rects[0].y1 == pytest.approx(500.0)          # top-aligned to block
    assert rects[0].width == pytest.approx(HEATMAP_COLORBAR_THICKNESS_POINTS)
    assert width == pytest.approx(HEATMAP_COLORBAR_THICKNESS_POINTS + 12.0)


def test_bars_fitting_one_column_share_height_and_stack_top_down():
    rects, _ = pack_colorbar_grid(
        3, block_height=400.0, base_x=0.0, top_y=400.0,
        tick_label_widths=[0.0, 0.0, 0.0])
    assert len({_column_of(rect) for rect in rects}) == 1        # one column
    assert len({round(rect.height, 3) for rect in rects}) == 1   # equal heights
    # Filled top-to-bottom: strictly descending tops, first bar flush with top.
    tops = [rect.y1 for rect in rects]
    assert tops[0] == pytest.approx(400.0)
    assert tops[0] > tops[1] > tops[2]
    assert all(rect.height >= COLORBAR_MIN_HEIGHT_POINTS for rect in rects)


def test_min_height_forces_columns_filled_column_major():
    # Block only tall enough for two min-height bars -> 2 rows; 4 bars -> 2 cols.
    block = 2 * COLORBAR_MIN_HEIGHT_POINTS + 20.0
    rects, width = pack_colorbar_grid(
        4, block_height=block, base_x=0.0, top_y=block,
        thickness=HEATMAP_COLORBAR_THICKNESS_POINTS,
        tick_label_widths=[5.0, 5.0, 5.0, 5.0])
    columns = sorted({_column_of(rect) for rect in rects})
    assert len(columns) == 2
    # Column-major fill: bars 0,1 in the first column, 2,3 in the second.
    assert _column_of(rects[0]) == _column_of(rects[1]) == columns[0]
    assert _column_of(rects[2]) == _column_of(rects[3]) == columns[1]
    assert len({round(rect.height, 3) for rect in rects}) == 1
    # Column advance clears the widest tick label plus the column gap.
    assert columns[1] - columns[0] == pytest.approx(
        HEATMAP_COLORBAR_THICKNESS_POINTS + 5.0 + COLORBAR_LONG_EDGE_GAP_POINTS)
    # Total width: two bars + two labels + one inter-column gap (no trailing gap).
    assert width == pytest.approx(
        2 * (HEATMAP_COLORBAR_THICKNESS_POINTS + 5.0) +
        COLORBAR_LONG_EDGE_GAP_POINTS)


def test_short_last_column_is_top_aligned_with_uniform_height():
    block = 2 * COLORBAR_MIN_HEIGHT_POINTS + 20.0
    rects, _ = pack_colorbar_grid(
        3, block_height=block, base_x=0.0, top_y=block,
        tick_label_widths=[0.0, 0.0, 0.0])
    # rows=2, columns=2 -> last column holds a single bar aligned to the top.
    assert len({round(rect.height, 3) for rect in rects}) == 1
    last = rects[2]
    assert last.y1 == pytest.approx(block)               # top-aligned
    assert _column_of(last) != _column_of(rects[0])      # in the second column


def _below_side_axes(figure):
    return [axis for axis in figure.axes
            if (axis.get_gid() or '').startswith('colorbar/figure/')]


def test_below_common_bars_are_unstretched_and_centred_per_row():
    # Three bars, block wide enough for two per row -> row1 has two, row2 one.
    entries = tuple(BelowColorbarEntry(left_overhang=8.0, right_overhang=8.0,
                                       tick_height=10.0) for _ in range(3))
    bars, labels, total = pack_below_common(
        entries, bar_width=120.0, thickness=9.0, block_width=300.0,
        origin_x=10.0, top_y=200.0)
    assert all(bar.width == pytest.approx(120.0) for bar in bars)  # un-stretched
    # Rows: [0,1] then [2]. Row 1 tops equal, row 2 lower.
    assert bars[0].y0 == pytest.approx(bars[1].y0)
    assert bars[2].y0 < bars[0].y0
    # Each row is centred in the block: symmetric margins about the block centre.
    block_centre = 10.0 + 300.0 / 2
    row1_centre = (bars[0].x0 + bars[1].x1) / 2
    row2_centre = bars[2].x0 + bars[2].width / 2
    assert row1_centre == pytest.approx(block_centre)
    assert row2_centre == pytest.approx(block_centre)
    assert total > 0


def test_below_common_mixed_titles_share_bar_baseline_and_label_bottoms():
    entries = (
        BelowColorbarEntry(tick_height=1.0, label_width=50.0,
                           label_height=30.0),
        BelowColorbarEntry(tick_height=20.0, label_width=50.0,
                           label_height=10.0))
    bars, labels, total = pack_below_common(
        entries, bar_width=50.0, thickness=9.0, block_width=108.0,
        origin_x=10.0, top_y=200.0, horizontal_gap=8.0,
        vertical_gap=4.0, label_gap=3.0)

    assert bars[0].y1 == pytest.approx(bars[1].y1)
    assert labels[0].y0 == pytest.approx(labels[1].y0)
    assert tuple(label.y0 - bar.y1 for label, bar in zip(labels, bars)) == \
        pytest.approx((3.0, 3.0))
    # Reserve the tallest title band and tallest tick extent independently;
    # here they belong to different bars in the same row.
    assert total == pytest.approx(30.0 + 3.0 + 9.0 + 20.0)


def test_below_common_gap_swap_and_footprint_spacing():
    # Two bars in one row: horizontal gap between footprints is the side ROW gap;
    # a footprint is bar_width + both tick overhangs.
    entries = (BelowColorbarEntry(left_overhang=5.0, right_overhang=7.0),
               BelowColorbarEntry(left_overhang=5.0, right_overhang=7.0))
    bars, _, _ = pack_below_common(
        entries, bar_width=100.0, thickness=9.0, block_width=400.0,
        origin_x=0.0, top_y=100.0)
    # Gap between the first footprint's right edge and the second's left edge.
    first_footprint_right = bars[0].x1 + 7.0        # bar right + right overhang
    second_footprint_left = bars[1].x0 - 5.0        # bar left - left overhang
    assert (second_footprint_left - first_footprint_right) == pytest.approx(
        COLORBAR_SHORT_EDGE_GAP_POINTS)


def test_below_common_tucked_labels_fit_two_bars_in_one_row():
    # Tucked end labels protrude ~0, so two bars occupy just 2*bar_width + gap.
    # Under two heatmaps (block = 2*bar_width + column gap) they share one row as
    # long as the horizontal gap fits the inter-heatmap gap -- matching legacy.
    entries = (BelowColorbarEntry(tick_height=10.0),
               BelowColorbarEntry(tick_height=10.0))
    column_gap = 8.0
    fits = pack_below_common(
        entries, bar_width=100.0, thickness=9.0,
        block_width=2 * 100.0 + column_gap, origin_x=0.0, top_y=100.0,
        horizontal_gap=column_gap)[0]
    assert fits[0].y0 == pytest.approx(fits[1].y0)          # one row
    # A protruding (non-tucked) label on each bar pushes them onto two rows.
    protruding = (BelowColorbarEntry(left_overhang=6, right_overhang=6),
                  BelowColorbarEntry(left_overhang=6, right_overhang=6))
    wrapped = pack_below_common(
        protruding, bar_width=100.0, thickness=9.0,
        block_width=2 * 100.0 + column_gap, origin_x=0.0, top_y=100.0,
        horizontal_gap=column_gap)[0]
    assert wrapped[0].y0 != wrapped[1].y0                   # protrusion counts


def _base_dict():
    return {'colorMap': None, 'colorList': None, 'colorNumber': 256,
            'missingDataColor': 'black', 'alpha': 1.0}


def _placed_colorbars(solution):
    return {role: rect for role, rect in solution.rects.items()
            if role.startswith('colorbar/')}


def test_below_common_dedups_and_draws_distinct_bars():
    colors = dict(_base_dict(), colorMap=['Reds', 'Blues'])
    figure, solution = _render(
        colors, colorbarLocation='below_common')
    # Four columns, two distinct scales -> two horizontal bars.
    assert len(_below_side_axes(figure)) == 2
    assert len(_placed_colorbars(solution)) == 2


def test_two_colormaps_produce_two_side_colorbars():
    colors = dict(_base_dict(), colorMap=['Reds', 'Blues'])
    figure, solution = _render(
        colors, colorbarLocation='right')
    # The reported bug: previously only the last scale drew one bar.
    assert len(_side_colorbar_axes(figure)) == 2
    assert len(_placed_colorbars(solution)) == 2


def test_identical_scales_share_a_single_colorbar():
    # Same colormap and (auto) scale on every column -> one shared bar.
    colors = dict(_base_dict(), colorMap=['Reds'])
    figure, solution = _render(
        colors, colorbarLocation='right')
    assert len(_side_colorbar_axes(figure)) == 1
    assert len(_placed_colorbars(solution)) == 1


def _colorbar_label_axes(figure):
    return {axis.get_gid(): axis for axis in figure.axes
            if (axis.get_gid() or '').startswith('colorbar_title/')}


def test_explicit_side_labels_are_drawn():
    colors = dict(_base_dict(), colorMap=['Reds', 'Blues'])
    figure, _ = _render(colors, colorbarLocation='right',
                        colorbarLabels=['Signal', 'Input'])
    labels = _colorbar_label_axes(figure)
    texts = {axis.texts[0].get_text() for axis in labels.values()}
    assert texts == {'Signal', 'Input'}


def test_auto_below_labels_use_sample_names():
    colors = dict(_base_dict(), colorMap=['Reds'])
    figure, _ = _render(colors, colorbarLocation='below',
                        colorbarLabels=['auto'])
    labels = _colorbar_label_axes(figure)
    assert labels                                       # one per below bar
    assert all(axis.texts[0].get_text() == 'test'
               for axis in labels.values())             # master_multi samples


def test_side_colorbar_uses_near_edge_tick_positions():
    colors = dict(_base_dict(), colorMap=['RdYlBu'])
    figure, _ = _render(colors, colorbarLocation='right',
                        zMin=['-3.245'], zMax=['2.113'])
    ticks = list(_side_colorbar_axes(figure)[0].get_yticks())
    # The chooser is wired: outer ticks are the roundest values near each end
    # (-3 / 2, not matplotlib's -2 / 2), whatever the count works out to be.
    assert min(ticks) == pytest.approx(-3.0)
    assert max(ticks) == pytest.approx(2.0)
    assert len(ticks) >= 3


def test_partial_colorlist_falls_back_to_default_colormap():
    # Only cols 1,2 get an explicit colour list; cols 3,4 use the default
    # colormap with the common scale -> two distinct side colorbars.
    colors = dict(_base_dict(), colorMap=['RdYlBu'],
                  colorList=['1,2=white,red'])
    figure, solution = _render(colors, colorbarLocation='right')
    assert len(_side_colorbar_axes(figure)) == 2
    assert len(_placed_colorbars(solution)) == 2


def test_partial_zmid_renders_without_requiring_full_coverage():
    # Only col 1 gets a midpoint; the rest simply get none (no crash, no error).
    colors = dict(_base_dict(), colorMap=['RdBu'])
    figure, _ = _render(colors, colorbarLocation='right',
                        zMin=['0'], zMax=['3'], zMid=['1=1.5'])
    assert figure is not None


def test_small_groups_render_crisp_not_antialiased():
    # master_multi has 3 regions per group; below the small-group threshold the
    # heatmaps must draw with nearest-neighbour so each region is a solid
    # rectangle rather than a vertical gradient (regression for the native
    # raster path, which previously hardcoded antialiased resampling).
    colors = dict(_base_dict(), colorMap=['Reds'])
    figure, _ = _render(colors, colorbarLocation='right')
    images = [image for axis in figure.axes for image in axis.get_images()]
    assert images  # heatmap data images exist
    assert {image.get_interpolation() for image in images} == {'nearest'}


def test_colorbar_row_labels_settle_by_clearance():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from deeptoolsr.plotting.matrix_figure import settle_horizontal_colorbar_row

    def _bar(fig, position):
        ax = fig.add_axes(position)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(['0', '50', '100'])
        return ax

    # A bar inset from the figure edges has room on both sides, so its end labels
    # stay centred (the -3 case: padding to the side lets it centre).
    fig = plt.figure(figsize=(6, 0.5), dpi=100)
    ax = _bar(fig, [0.25, 0.3, 0.5, 0.4])
    settle_horizontal_colorbar_row([ax], fig)
    align = {label.get_text(): label.get_horizontalalignment()
             for label in ax.get_xticklabels() if label.get_text()}
    assert align['0'] == 'center' and align['100'] == 'center'
    plt.close(fig)

    # A near-full-width bar (ticks just inside the edges) has no room to centre
    # its end labels but tucking them flush fits, so they tuck rather than drop.
    fig2 = plt.figure(figsize=(6, 0.5), dpi=100)
    ax2 = _bar(fig2, [0.015, 0.3, 0.97, 0.4])
    settle_horizontal_colorbar_row([ax2], fig2)
    align2 = {label.get_text(): label.get_horizontalalignment()
              for label in ax2.get_xticklabels() if label.get_text()}
    assert align2['0'] == 'left' and align2['100'] == 'right'
    plt.close(fig2)

    # Two bars whose rectangles nearly touch: tucking their facing labels still
    # collides, so an end tick is dropped (mitigation wins over tucking).
    fig3 = plt.figure(figsize=(6, 0.5), dpi=100)
    axa = fig3.add_axes([0.1, 0.3, 0.38, 0.4])
    axa.set_xlim(0, 10)
    axa.set_xticks([0, 10])
    axa.set_xticklabels(['-100', '100'])
    axb = fig3.add_axes([0.485, 0.3, 0.38, 0.4])
    axb.set_xlim(0, 10)
    axb.set_xticks([0, 10])
    axb.set_xticklabels(['-100', '100'])
    settle_horizontal_colorbar_row([axa, axb], fig3)
    remaining = sum(len([label for label in ax.get_xticklabels()
                         if label.get_text()])
                    for ax in (axa, axb))
    assert remaining < 4   # started with 4 labels; the collision dropped one
    plt.close(fig3)


# From test_heatmap_aspect_ratio.py.

def test_heatmap_summary_best_defaults_to_below(monkeypatch, tmp_path):
    # The complete rendering path is covered by integration tests; assert the
    # public default remains "best", which the builder resolves to "below".
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png')])
    assert args.legendLocation == 'best'


def test_heatmap_aspect_ratio_argument(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                         '--aspectRatio', '0.75'])
    assert args.heatmapAspectRatio == 0.75


def test_heatmap_aspect_ratio_sets_solved_block_width():
    matrix = Matrix.load(str(Path(__file__).parent.parent / 'test_heatmapper' /
                             'master.mat.gz'), threads=1)
    _, solved = render_heatmap(plot_data(matrix),
                               whatToShow='heatmap and colorbar',
                               heatmapHeight=8, heatmapAspectRatio=0.75)
    assert solved.rects['cell/1/1/block/1'].width == \
        cm_to_points(8 * 0.75)


def test_minor_tick_argument_and_positions(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                         '--minorTickMarks', '2'])
    assert args.minorTickMarks == 2

    fig, axis = plt.subplots()
    axis.set_xticks([0, 2, 8])
    add_minor_x_ticks(axis, 2)
    np.testing.assert_allclose(axis.get_xticks(minor=True),
                               [1, 5])
    plt.close(fig)


def test_fixed_minor_tick_count_is_total_and_proportional():
    ticks = distribute_minor_ticks([-5, 0, 10], 13)
    assert len(ticks) == 13
    np.testing.assert_allclose(ticks,
                               [-4, -3, -2, -1,
                                1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_auto_minor_ticks_are_space_aware_and_preserve_major_ticks():
    narrow, narrow_axis = plt.subplots(figsize=(2, 2))
    wide, wide_axis = plt.subplots(figsize=(8, 2))
    for axis in (narrow_axis, wide_axis):
        axis.set_xlim(0, 10)
        axis.set_xticks([0, 0.5, 10])
        add_minor_x_ticks(axis, 'auto')
        assert not set(axis.get_xticks(minor=True)) & set(axis.get_xticks())
    assert len(wide_axis.get_xticks(minor=True)) > len(
        narrow_axis.get_xticks(minor=True))
    plt.close(narrow)
    plt.close(wide)


def test_summary_plot_arguments(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args([
        '-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
        '--aspectRatioSummaryPlot', '1.5',
        '--yAxisLabelSummaryPlot', 'summary signal',
        '--averageTypeSummaryPlot', 'geom_mean',
        '--plotTypeSummaryPlot', 'std',
        '--yMinSummaryPlot', '-2', '--yMaxSummaryPlot', '3',
        '--pseudocountSummaryPlot', '0', '--trimPercSummaryPlot', '0.1',
        '--legendLocationSummaryPlot', 'below',
        '--arrangeSamples', '1,2', '3,4',
        '--yAxisVisibilitySummaryPlot', 'outer_merged',
        '--yAxisLimitsSummaryPlot', 'per_y_label', '--colorbarLocation', 'bottom',
        '--colorsSummaryPlot', 'navy', 'orange'])
    assert args.profileAspectRatio == 1.5
    assert args.yAxisLabel == ['summary signal']
    assert args.colors == ['navy', 'orange']
    assert args.averageType == 'geom_mean'
    assert args.plotType == 'std'
    assert args.yMin == [-2.0]
    assert args.yMax == [3.0]
    assert args.pseudocount == 0
    assert args.trim_perc == 0.1
    assert args.legendLocation == 'below'
    assert args.arrangeSamples == ['1,2', '3,4']
    assert args.yAxisVisibility == 'outer_merged'
    assert args.yAxisLimits == 'per_y_label'
    assert args.colorbarLocation == 'bottom'


def test_heatmap_region_count_argument(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                         '--showRegionCounts'])
    assert args.showRegionCounts


def test_heatmap_size_arguments(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                         '--heatmapWidth', '4', '--heatmapHeight', '8'])
    assert args.cellWidth == 4
    assert args.heatmapHeight == 8


# From test_geometry_spec_migration.py.

def test_packer_gaps_flow_through_the_spec():
    # A three-into-one-column grid: a larger short-edge (row) gap adds two gaps
    # of vertical space, so the boxes shrink; the packer sourced the gap from
    # the spec rather than the module constant.
    tall_spec = GeometrySpec()
    rects_default, _ = pack_colorbar_grid(3, 300, 0.0, 300.0)
    bigger_gap = dataclasses.replace(
        tall_spec, colorbar_short_edge_gap=tall_spec.colorbar_short_edge_gap + 10)
    rects_bigger, _ = pack_colorbar_grid(3, 300, 0.0, 300.0, geometry=bigger_gap)
    assert rects_bigger[0].height < rects_default[0].height


def test_explicit_kwarg_still_overrides_the_spec():
    # An explicit keyword beats the spec, preserving the existing call surface.
    from_kwarg, _ = pack_colorbar_grid(1, 300, 0.0, 300.0, thickness=20)
    assert from_kwarg[0].width == 20
