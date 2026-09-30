"""Tests for plotting legend."""

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from dataclasses import replace
import pytest
from deeptoolsr.plotting.scene import MatplotlibMeasureContext
from deeptoolsr.plotting.legend import (
    LegendEntrySpec, draw_legend_in_axes, measure_legend)
from deeptoolsr.plotting.geometry import DEFAULT_GEOMETRY


# From test_legend_layout.py.

def test_short_key_stays_on_one_row():
    context = MatplotlibMeasureContext()
    measured = measure_legend(context, [LegendEntrySpec('short', 'red')],
                              'above', 200)
    assert measured.ncols == 1


def test_keys_wrap_and_unbreakable_key_can_widen():
    context = MatplotlibMeasureContext()
    entries = [LegendEntrySpec('key {}'.format(i), 'red') for i in range(4)]
    one = measure_legend(context, entries[:1], 'above', 100)
    wrapped = measure_legend(context, entries, 'above', one.size.width * 2.2)
    assert 1 <= wrapped.ncols < 4
    wide = measure_legend(
        context, [LegendEntrySpec('x' * 100, 'red')], 'below', 20)
    assert wide.size.width > 20


def test_multiline_height_and_repeat_draw_stability():
    context = MatplotlibMeasureContext(dpi=120)
    single = measure_legend(context, [LegendEntrySpec('one', 'red')],
                            'right')
    multiline = measure_legend(context, [LegendEntrySpec('one\ntwo', 'red')],
                               'right')
    repeat = measure_legend(context, [LegendEntrySpec('one\ntwo', 'red')],
                            'right')
    assert multiline.size.height > single.size.height
    assert abs(multiline.size.width - repeat.size.width) <= 72 / 120
    assert abs(multiline.size.height - repeat.size.height) <= 72 / 120


def test_explicit_font_properties_match_measurement_and_drawing():
    entries = [LegendEntrySpec('wide legend entry', 'red')]
    font = FontProperties(
        family=['Courier'], size=15, weight='bold', style='italic')
    context = MatplotlibMeasureContext()
    default = measure_legend(context, entries, 'right')
    styled = measure_legend(
        context, entries, 'right', font_properties=font)
    assert styled.size != default.size

    figure, axis = plt.subplots()
    legend = draw_legend_in_axes(
        axis, entries, styled, font_properties=font)
    text = legend.get_texts()[0]
    assert text.get_fontproperties().get_family() == ['Courier']
    assert text.get_fontsize() == 15
    assert text.get_fontweight() == 'bold'
    assert text.get_fontstyle() == 'italic'
    context.close()
    plt.close(figure)


def test_point_based_internal_spacing_matches_measurement_and_drawing():
    geometry = replace(
        DEFAULT_GEOMETRY,
        legend_border_padding=7,
        legend_handle_to_text_gap=9,
        legend_entry_row_gap=5,
        legend_column_gap=17)
    font = FontProperties(size=10)
    entries = [LegendEntrySpec('one', 'red'),
               LegendEntrySpec('two', 'blue')]
    context = MatplotlibMeasureContext(dpi=100)
    measured = measure_legend(
        context, entries, 'above', 1000, font_properties=font,
        geometry=geometry)

    figure, axis = plt.subplots(dpi=100)
    legend = draw_legend_in_axes(
        axis, entries, measured, font_properties=font,
        geometry=geometry)
    figure.canvas.draw()
    rendered = legend.get_window_extent(figure.canvas.get_renderer())
    rendered_width = rendered.width * 72 / figure.dpi
    rendered_height = rendered.height * 72 / figure.dpi

    assert legend.borderpad == pytest.approx(7 / 10)
    assert legend.handletextpad == pytest.approx(9 / 10)
    assert legend.labelspacing == pytest.approx(5 / 10)
    assert legend.columnspacing == pytest.approx(17 / 10)
    assert rendered_width == pytest.approx(measured.size.width, abs=.1)
    assert rendered_height == pytest.approx(measured.size.height, abs=.1)
    context.close()
    plt.close(figure)


def test_legend_spacing_points_do_not_scale_with_label_font():
    geometry = replace(DEFAULT_GEOMETRY, legend_border_padding=8)
    entries = [LegendEntrySpec('one', 'red')]
    context = MatplotlibMeasureContext()
    ten = measure_legend(
        context, entries, 'right', font_properties=FontProperties(size=10),
        geometry=geometry)
    twenty = measure_legend(
        context, entries, 'right', font_properties=FontProperties(size=20),
        geometry=geometry)
    # Doubling the font enlarges the text, but the two 8-point border paddings
    # remain 16 physical points rather than doubling with it.
    assert twenty.size.width - ten.size.width < ten.size.width
    context.close()


def test_labels_wrap_between_words_when_one_column_is_too_wide():
    context = MatplotlibMeasureContext()
    entries = [LegendEntrySpec('log2FC [treated/control] [n = 7,619]', 'red')]
    single = measure_legend(context, entries, 'above')
    budget = single.size.width * 0.6
    wrapped = measure_legend(context, entries, 'above', budget, max_lines=3)
    assert wrapped.size.width <= budget
    lines = wrapped.labels[0].split('\n')
    assert len(lines) > 1
    # Bracketed units stay whole and words are never cut.
    assert all(line.count('[') == line.count(']') for line in lines)
    assert ' '.join(lines) == entries[0].label
    assert measure_legend(context, entries, 'above', budget).labels == ()
