"""Tests for plotting matrix figure."""
from tests.helpers.parity import original

from tests.helpers.plot_data import plot_data
from tests.helpers.heatmap_build import render_heatmap
from tests.helpers.profile_build import build_profile_case

from pathlib import Path
from matplotlib.figure import Figure
import argparse
import matplotlib
from matplotlib.axes import Axes
import pytest
from deeptoolsr import options as run_options
from deeptoolsr.plotting.geometry import cm_to_points
from deeptoolsr.plotting import geometry as style_geometry
from deeptoolsr.plotting import fonts as style_fonts
from deeptoolsr.plotting.geometry import (
    HEATMAP_TO_ROW_HEADER_GAP_POINTS,
    ROW_HEADER_TO_Y_LABEL_GAP_POINTS,
    SORT_INDICATOR_GAP_POINTS)
import numpy as np
from deeptoolsr.plotting.heatmap import (
    midpoint_exponent,
    shift_cmap_midpoint,
)
from deeptoolsr.plotHeatmap import process_args as heatmap_process_args
from dataclasses import replace
import matplotlib.pyplot as plt
from deeptoolsr.matrix import merge_groups_by_key
from deeptoolsr.plotProfile import process_args
from deeptoolsr.plotting.series import series_slot
from deeptoolsr.plotting.geometry import DEFAULT_GEOMETRY
from tests.test_plot_baselines import PROFILE_CASES, _render_profile
import csv
import json
from deeptoolsr.matrix import MatrixHeader, OwnedMatrix
from deeptoolsr import plotProfile as plot_profile_module
from deeptoolsr import config
from deeptoolsr.plotting import matrix_figure as profile_plotting
from deeptoolsr.plotting import profile as profile_computation
from deeptoolsr.plotting.profile import write_profile_table
from deeptoolsr.plotting.geometry import (
    AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS,
    COMMON_AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS,
    EXTERNAL_LEGEND_TO_CONTENT_GAP_POINTS,
    X_AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS)
from deeptoolsr.matrix import Matrix, RowLayout, cluster
from deeptoolsr.prepare import PlotData, resolve_labels
from deeptoolsr.plotting import grid, matrix_figure
from deeptoolsr.plotting.geometry import Insets, Size
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.font_manager import FontProperties
from deeptoolsr import plotHeatmap, plotProfile
from deeptoolsr.plotting import text_layout
from deeptoolsr.plotting.geometry import resolve_style
from deeptoolsr.plotting.text_layout import TextMeasurer, measure_matplotlib_text
from tests import test_plot_baselines as baselines
from deeptoolsr.plotting.rendering import (
    restore_axes_images, suspend_axes_images)


# From plotting/test_matrix_figure.py.

DATA = Path(__file__).resolve().parents[1] / 'test_heatmapper' / 'master.mat.gz'


def test_builder_returns_figure_without_saving(monkeypatch):
    def unexpected_save(*_args, **_kwargs):
        raise AssertionError('figure builder saved an output')

    monkeypatch.setattr(Figure, 'savefig', unexpected_save)
    matrix = Matrix.load(str(DATA), threads=1)
    figure, solved = render_heatmap(plot_data(matrix), threads=1)
    assert figure.axes
    assert solved.figure_size.width > 0
    assert solved.figure_size.height > 0


def test_builder_places_every_role_once():
    matrix = Matrix.load(str(DATA), threads=1)
    figure, solved = render_heatmap(plot_data(matrix), threads=1)
    gids = [axis.get_gid() for axis in figure.axes]
    assert all(gids)
    assert len(gids) == len(set(gids))
    assert set(gids) == set(solved.rects)
    assert any(role.startswith('cell/') and '/block/' in role
               for role in solved.rects)
    assert any(role.startswith('colorbar/') for role in solved.rects)


# From test_heatmap_slot_ownership.py.

HEATMAP_SLOT_DATA = Path(__file__).parent.parent / 'test_heatmapper'


def _rects(solution, prefix):
    return tuple(rect for role, rect in sorted(solution.rects.items())
                 if role.startswith(prefix))


def _block_rects(solution):
    return tuple(rect for role, rect in sorted(solution.rects.items())
                 if '/block/' in role)


def _title_rects(solution):
    return tuple(rect for role, rect in sorted(solution.rects.items())
                 if role.endswith('/title') and role.startswith('cell/'))


def _heatmap_slot_texts(figure, prefix):
    return tuple(axis.texts[0].get_text() for axis in sorted(
        figure.axes, key=lambda axis: axis.get_gid() or '')
        if (axis.get_gid() or '').startswith(prefix) and axis.texts)


def _line_count(text):
    return text.count('\n') + 1


def _render(matrix_name, **kwargs):
    matrix = Matrix.load(str(HEATMAP_SLOT_DATA / matrix_name), threads=1)
    group_labels = kwargs.pop('group_labels', None)
    sort_method = kwargs.pop('sort_method', None)
    colors = kwargs.pop('colorMapDict', {
        'colorMap': ['Reds'], 'colorList': None, 'colorNumber': 256,
        'missingDataColor': 'black', 'alpha': 1.0})
    original_rc = matplotlib.rcParams.copy()
    try:
        what_to_show = kwargs.pop('whatToShow', 'heatmap and colorbar')
        dpi = kwargs.pop('dpi', 100)
        kwargs.setdefault('style', run_options.resolve_run_options(
            argparse.Namespace(config='auto', numberOfProcessors=1),
            plotting=True).style)
        kwargs.setdefault('threads', 1)
        return render_heatmap(
            plot_data(matrix, group_labels=group_labels,
                      sort_method=sort_method), colorMapDict=colors,
            whatToShow=what_to_show, image_format='png', dpi=dpi,
            **kwargs)
    finally:
        matplotlib.rcParams.update(original_rc)


def _assert_axis_rect(axis, rect, solution):
    actual = axis.get_position().bounds
    expected = (rect.x / solution.figure_size.width,
                rect.y / solution.figure_size.height,
                rect.width / solution.figure_size.width,
                rect.height / solution.figure_size.height)
    assert actual == pytest.approx(expected, abs=1e-7)


def _assert_tight_bbox_inside_figure(axis, figure, renderer):
    box = axis.get_tightbbox(renderer)
    canvas = figure.bbox
    assert box.x0 >= canvas.x0 - 1
    assert box.y0 >= canvas.y0 - 1
    assert box.x1 <= canvas.x1 + 1
    assert box.y1 <= canvas.y1 + 1


def test_heatmap_decorations_are_measured_at_requested_panel_width(
        monkeypatch):
    measured_widths = []
    original = matrix_figure.measure_axis

    def record_width(axis, renderer):
        if axis.get_xlabel():
            measured_widths.append(
                axis.get_window_extent(renderer).width * 72 / axis.figure.dpi)
        return original(axis, renderer)

    monkeypatch.setattr(matrix_figure, 'measure_axis', record_width)
    _render('master_multi.mat.gz',
            whatToShow='heatmap and colorbar', heatmapWidth=3,
            heatmapHeight=6, xAxisLabel='distance from TSS [kb]')
    assert measured_widths
    assert measured_widths == pytest.approx(
        [cm_to_points(3)] * len(measured_widths), abs=.1)


def test_sample_title_slots_are_not_selected_as_right_colorbar(
        tmp_path, monkeypatch):
    figure, solution = _render(
        'master_multi.mat.gz', plotTitle='global title')
    title_axes = sorted(
        (axis for axis in figure.axes
         if (axis.get_gid() or '').startswith('cell/1/') and (axis.get_gid() or '').endswith('/title')),
        key=lambda axis: axis.get_gid())
    # 'best' with one colour scale puts one right bar on the row.
    colorbar = next(axis for axis in figure.axes
                    if axis.get_gid() == 'colorbar/row/1/1')
    assert len(title_axes) == len(_title_rects(solution))
    for axis, rect in zip(title_axes, _title_rects(solution)):
        _assert_axis_rect(axis, rect, solution)
    _assert_axis_rect(colorbar, _rects(solution, 'colorbar/')[0], solution)
    assert all(axis.get_position().bounds != colorbar.get_position().bounds
               for axis in title_axes)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    _assert_tight_bbox_inside_figure(colorbar, figure, renderer)
    color_box = colorbar.get_tightbbox(renderer)
    assert all(not color_box.overlaps(axis.get_tightbbox(renderer))
               for axis in title_axes)


def test_sample_title_slots_survive_multiple_below_colorbars(
        tmp_path, monkeypatch):
    colors = {
        'colorMap': ['Reds', 'Blues'], 'colorList': None,
        'colorNumber': 256, 'missingDataColor': 'black', 'alpha': 1.0}
    figure, solution = _render(
        'master_multi.mat.gz', colorMapDict=colors)
    title_axes = sorted(
        (axis for axis in figure.axes
         if (axis.get_gid() or '').startswith('cell/1/') and (axis.get_gid() or '').endswith('/title')),
        key=lambda axis: axis.get_gid())
    colorbars = sorted(
        (axis for axis in figure.axes
         if (axis.get_gid() or '').startswith('colorbar/cell/')),
        key=lambda axis: axis.get_gid())
    assert len(title_axes) == len(_title_rects(solution))
    assert len(colorbars) > 1
    for axis, rect in zip(title_axes, _title_rects(solution)):
        _assert_axis_rect(axis, rect, solution)
    color_union_x0 = min(axis.get_position().x0 for axis in colorbars)
    color_union_x1 = max(axis.get_position().x1 for axis in colorbars)
    colorbar_rects = _rects(solution, 'colorbar/')
    assert color_union_x0 == pytest.approx(
        min(rect.x0 for rect in colorbar_rects) / solution.figure_size.width)
    assert color_union_x1 == pytest.approx(
        max(rect.x1 for rect in colorbar_rects) / solution.figure_size.width)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tight = [axis.get_tightbbox(renderer) for axis in colorbars]
    for axis in colorbars:
        _assert_tight_bbox_inside_figure(axis, figure, renderer)
    for left, right in zip(tight, tight[1:]):
        assert left.x1 <= right.x0 + 1
    assert all(not box.overlaps(title.get_tightbbox(renderer))
               for box in tight for title in title_axes)


def test_multiline_region_label_preserves_vertical_rotation_and_font(
        tmp_path, monkeypatch):
    matrix = Matrix.load(str(HEATMAP_SLOT_DATA / 'master_multi.mat.gz'), threads=1)
    group_labels = ['first line\nsecond line'
                    for _ in matrix.header.group_labels]
    original_rc = matplotlib.rcParams.copy()
    try:
        figure, _ = render_heatmap(
            plot_data(matrix, group_labels=group_labels), colorMapDict={
                'colorMap': ['Reds'], 'colorList': None, 'colorNumber': 256,
                'missingDataColor': 'black', 'alpha': 1.0},
            whatToShow='heatmap and colorbar', image_format='png', dpi=100,
            threads=1)
        style = run_options.resolve_run_options(
            argparse.Namespace(config='auto', numberOfProcessors=1),
            plotting=True).style
        expected_label_size = style_fonts.role_font(
            style, 'axis_label_text').get_size_in_points()
    finally:
        matplotlib.rcParams.update(original_rc)
    label_axes = [axis for axis in figure.axes
                  if (axis.get_gid() or '').startswith('label/stack/')]
    assert label_axes
    for axis in label_axes:
        text = axis.texts[0]
        assert text.get_rotation() == 270
        assert text.get_horizontalalignment() == 'center'
        assert text.get_verticalalignment() == 'center'
        assert '\n' in text.get_text()
        assert text.get_fontsize() == pytest.approx(
            expected_label_size)


@pytest.mark.parametrize('location', ['left', 'right'])
def test_automatic_region_labels_wrap_and_render_in_selected_side_slots(
        monkeypatch, location):
    labels = [
        'a very long first region label with enough words to wrap vertically',
        'a second region label with enough words to wrap vertically']
    figure, solution = _render(
        'master_multi.mat.gz', group_labels=labels,
        whatToShow='heatmap and colorbar', heatmapHeight=3,
        regionLabelLocation=location)

    selected = _heatmap_slot_texts(figure, 'label/stack/')
    assert len(selected) == len(labels)
    assert all(selected)
    assert any(_line_count(text) > 1 for text in selected)
    axes = sorted(
        (axis for axis in figure.axes
         if (axis.get_gid() or '').startswith('label/stack/')),
        key=lambda axis: axis.get_gid())
    assert len(axes) == len(labels)
    for axis, rect in zip(axes, _rects(solution, 'label/stack/')):
        _assert_axis_rect(axis, rect, solution)
        assert axis.texts[0].get_rotation() == (
            270 if location == 'right' else 90)
        if location == 'left':
            assert rect.x1 < _block_rects(solution)[0].x0
        else:
            assert rect.x0 > _block_rects(solution)[-1].x1
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in axes:
        _assert_tight_bbox_inside_figure(axis, figure, renderer)


def test_automatic_region_candidates_keep_hard_breaks_and_empty_labels(
        monkeypatch):
    labels = ['first region\nexplicit section', '']
    figure, solution = _render(
        'master_multi.mat.gz', group_labels=labels,
        whatToShow='heatmap and colorbar', regionLabelLocation='left')

    selected = _heatmap_slot_texts(figure, 'label/stack/')
    assert len(selected) == 1
    assert selected[0].split('\n')[0] == 'first region'
    assert 'explicit section' in selected[0]
    axes = [axis for axis in figure.axes
            if (axis.get_gid() or '').startswith('label/stack/')]
    assert [axis.get_gid() for axis in axes] == ['label/stack/1/1/1']
    assert axes[0].texts[0].get_text() == selected[0]


@pytest.mark.parametrize('location', ['left', 'right'])
def test_region_label_rows_align_toward_heatmap_with_different_widths(
        monkeypatch, location):
    figure, solution = _render(
        'master_multi.mat.gz',
        group_labels=['short', 'long region label\nsecond line'],
        whatToShow='heatmap and colorbar', regionLabelLocation=location)

    first, second = _rects(solution, 'label/stack/')
    assert first.width < second.width
    if location == 'left':
        assert first.x1 == pytest.approx(second.x1)
    else:
        assert first.x0 == pytest.approx(second.x0)
    axes = [axis for axis in figure.axes
            if (axis.get_gid() or '').startswith('label/stack/')]
    assert len(axes) == len(_rects(solution, 'label/stack/'))


def test_long_heatmap_y_label_wraps_and_is_contained_by_final_figure(
        monkeypatch):
    text = ('a common heatmap vertical axis label with many descriptive '
            'words ' * 4).strip()
    figure, solution = _render(
        'master_multi.mat.gz',
        whatToShow='plot, heatmap and colorbar', heatmapYAxisLabel=text,
        heatmapWidth=3, heatmapHeight=6)

    label_axis = next(axis for axis in figure.axes
                      if axis.get_gid() == 'label/heatmap_y/1/1')
    assert _line_count(label_axis.texts[0].get_text()) > 1
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    _assert_tight_bbox_inside_figure(label_axis, figure, renderer)


def test_disabling_heatmap_y_layout_keeps_literal_text_and_contains_it(
        monkeypatch):
    text = 'a long literal heatmap y label\nwith an explicit second line'
    style, messages = style_geometry.resolve_style({
        'label_layout': {'auto_axis_label_layout': False}})
    assert messages == []
    figure, solution = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar',
        heatmapYAxisLabel=text, style=style)

    label_axis = next(axis for axis in figure.axes
                      if axis.get_gid() == 'label/heatmap_y/1/1')
    assert label_axis.texts[0].get_text() == text
    figure.canvas.draw()
    _assert_tight_bbox_inside_figure(
        label_axis, figure, figure.canvas.get_renderer())


def test_automatic_heatmap_y_labels_keep_explicit_breaks_and_skip_empty(
        monkeypatch):
    figure, solution = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar',
        heatmapYAxisLabel=['first line\nexplicit second line', ''])

    selected = _heatmap_slot_texts(figure, 'label/heatmap_y/')
    assert selected
    assert any('explicit second line' in text and '\n' in text
               for text in selected)
    slots = [axis for axis in figure.axes
             if (axis.get_gid() or '').startswith('label/heatmap_y/')]
    assert slots
    assert all(axis.texts[0].get_text() for axis in slots)


def test_disabling_automatic_region_labels_preserves_literal_text(
        monkeypatch):
    labels = ['a deliberately long region label stays on one line',
              'second\nexplicit label']
    style, messages = style_geometry.resolve_style({
        'label_layout': {'auto_heatmap_region_label_layout': False}})
    assert messages == []
    figure, solution = _render(
        'master_multi.mat.gz', group_labels=labels,
        whatToShow='heatmap and colorbar', regionLabelLocation='right',
        style=style)

    assert _heatmap_slot_texts(figure, 'label/stack/') == tuple(labels)
    rendered = [axis.texts[0].get_text() for axis in figure.axes
                if (axis.get_gid() or '').startswith('label/stack/')]
    assert rendered == labels


@pytest.mark.parametrize('show_indicator', [False, True])
def test_heatmap_left_decorations_use_exact_configured_gaps(
        monkeypatch, show_indicator):
    options = {'sortIndicator': 'auto', 'sort_method': 'descend'} \
        if show_indicator else {}
    figure, solution = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar',
        heatmapYAxisLabel=['signal'], regionLabelLocation='left', **options)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    scale = 72.0 / figure.dpi
    region_axis = next(axis for axis in figure.axes
                       if axis.get_gid() == 'label/stack/1/1/1')
    common_axis = next(axis for axis in figure.axes
                       if axis.get_gid() == 'label/heatmap_y/1/1')
    region_box = region_axis.texts[0].get_window_extent(renderer)
    common_box = common_axis.texts[0].get_window_extent(renderer)

    heatmap_left = _block_rects(solution)[0].x0
    indicator = solution.rects.get('indicator/1')
    if show_indicator:
        assert heatmap_left - indicator.x1 == pytest.approx(
            SORT_INDICATOR_GAP_POINTS)
        left_text_anchor = indicator.x0
    else:
        assert indicator is None
        left_text_anchor = heatmap_left
    assert left_text_anchor - region_box.x1 * scale == pytest.approx(
        HEATMAP_TO_ROW_HEADER_GAP_POINTS, abs=.25)
    assert (_rects(solution, 'label/stack/')[0].x0 -
            solution.rects['label/heatmap_y/1/1'].x1) == pytest.approx(
                ROW_HEADER_TO_Y_LABEL_GAP_POINTS)
    assert region_box.x0 * scale - common_box.x1 * scale == pytest.approx(
        ROW_HEADER_TO_Y_LABEL_GAP_POINTS, abs=.5)


def test_summary_columns_have_equal_safe_gaps_and_no_tick_overlap(
        tmp_path, monkeypatch):
    figure, solution = _render(
        'master_multi.mat.gz',
        whatToShow='plot, heatmap and colorbar', label_rotation=45)
    summaries = sorted(
        (axis for axis in figure.axes
         if (axis.get_gid() or '').endswith('/profile')),
        key=lambda axis: axis.get_gid())
    assert len(summaries) > 1
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    data_gaps = [right.get_window_extent(renderer).x0 -
                 left.get_window_extent(renderer).x1
                 for left, right in zip(summaries, summaries[1:])]
    assert max(data_gaps) - min(data_gaps) < 1
    tight = [axis.get_tightbbox(renderer) for axis in summaries]
    for left, right in zip(tight, tight[1:]):
        assert left.x1 <= right.x0 + 1


def test_per_group_heatmap_keeps_genomic_x_axis_labels(monkeypatch):
    figure, _ = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar',
        perGroup=True, xAxisLabel=['genomic distance'])
    structural_labels = [text.get_text()
                         for axis in figure.axes
                         if (axis.get_gid() or '').startswith('label/x/')
                         for text in axis.texts]
    assert structural_labels
    assert set(structural_labels) == {'genomic distance'}
    assert not set(structural_labels) & {'group1.bed', 'group2.bed'}


def test_auto_axis_layout_draws_selected_wrap_once_without_local_duplicate(
        monkeypatch):
    figure, solution = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar',
        heatmapWidth=2,
        xAxisLabel='very long genomic distance from the transcription start site label')
    assert _heatmap_slot_texts(figure, 'label/x/')
    assert any('\n' in text for text in _heatmap_slot_texts(figure, 'label/x/'))
    slots = [axis for axis in figure.axes
             if (axis.get_gid() or '').startswith('label/x/')]
    assert len(slots) == len(_heatmap_slot_texts(figure, 'label/x/'))
    assert all(text.replace('\n', ' ') ==
               'very long genomic distance from the transcription start '
               'site label' for text in _heatmap_slot_texts(figure, 'label/x/'))
    assert not [axis for axis in figure.axes
                if axis.xaxis.get_visible() and axis.get_xlabel()]


def test_auto_ordinary_xlabels_keep_axis_clearance_and_legacy_origin(
        monkeypatch):
    measurements = []
    call_counts = {}
    original_measure = matrix_figure.measure_axis

    def record_measure(axis, renderer):
        key = id(axis)
        call_counts[key] = call_counts.get(key, 0) + 1
        measured = original_measure(axis, renderer)
        if (call_counts[key] >= 2 and '/block/' in (axis.get_gid() or '') and
                axis.xaxis.get_visible()):
            measurements.append(
                measured.bottom * 72.0 / axis.figure.dpi)
        return measured

    monkeypatch.setattr(matrix_figure, 'measure_axis', record_measure)
    automatic_figure, automatic = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar',
        xAxisLabel='distance from TSS')
    assert _heatmap_slot_texts(automatic_figure, 'label/x/')
    assert all('\n' not in text for text in _heatmap_slot_texts(automatic_figure, 'label/x/'))
    assert measurements
    tick_envelope = (_block_rects(automatic)[-1].y0 - measurements[0])
    assert tick_envelope - _rects(automatic, 'label/x/')[0].y1 == \
        pytest.approx(style_geometry.DEFAULT_GEOMETRY.axis_label_to_tick_labels_gap)

    # A one-line automatic label has the same outer reservation as the
    # historical local-label route, up to renderer rounding.
    style, messages = style_geometry.resolve_style({
        'label_layout': {'auto_axis_label_layout': False}})
    assert messages == []
    legacy_figure, legacy = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar',
        xAxisLabel='distance from TSS', style=style)
    assert automatic.figure_size.height == pytest.approx(
        legacy.figure_size.height, abs=.5)
    assert min(rect.y0 for rect in _block_rects(automatic)) == pytest.approx(
        min(rect.y0 for rect in _block_rects(legacy)), abs=.5)


def test_outer_merged_xlabels_keep_common_clearance(monkeypatch):
    measurements = []
    call_counts = {}
    original_measure = matrix_figure.measure_axis

    def record_measure(axis, renderer):
        key = id(axis)
        call_counts[key] = call_counts.get(key, 0) + 1
        measured = original_measure(axis, renderer)
        if (call_counts[key] >= 2 and '/block/' in (axis.get_gid() or '') and
                axis.xaxis.get_visible()):
            measurements.append(
                measured.bottom * 72.0 / axis.figure.dpi)
        return measured

    monkeypatch.setattr(matrix_figure, 'measure_axis', record_measure)
    figure, solution = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar',
        xAxisVisibility='outer_merged', xAxisLabel='distance from TSS')
    assert len(_rects(solution, 'label/x/')) == 1
    assert measurements
    tick_envelope = _block_rects(solution)[-1].y0 - measurements[0]
    assert tick_envelope - _rects(solution, 'label/x/')[0].y1 == \
        pytest.approx(
            style_geometry.DEFAULT_GEOMETRY.
            common_axis_label_to_tick_labels_gap)


def test_outer_merged_xlabels_policy_off_preserves_source_metadata(
        monkeypatch):
    style, messages = style_geometry.resolve_style({
        'label_layout': {'auto_axis_label_layout': False}})
    assert messages == []
    original_set_xlabel = Axes.set_xlabel

    def style_source_axis_label(axis, text, *args, **kwargs):
        result = original_set_xlabel(axis, text, *args, **kwargs)
        if text == 'distance from TSS':
            axis.xaxis.label.set_fontsize(13)
            axis.xaxis.label.set_color('purple')
            axis.xaxis.label.set_rotation(17)
        return result

    monkeypatch.setattr(Axes, 'set_xlabel', style_source_axis_label)
    figure, solution = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar',
        xAxisVisibility='outer_merged', xAxisLabel='distance from TSS',
        style=style)

    assert _heatmap_slot_texts(figure, 'label/x/') == ('distance from TSS',)
    slot = next(axis for axis in figure.axes
                if (axis.get_gid() or '').startswith('label/x/'))
    source_axes = [axis for axis in figure.axes
                   if '/block/' in (axis.get_gid() or '') and axis.xaxis.get_visible()]
    assert source_axes
    source_label = source_axes[-1].xaxis.label
    source_font = source_label.get_fontproperties()
    slot_text = slot.texts[0]
    slot_font = slot_text.get_fontproperties()
    assert slot_text.get_text() == 'distance from TSS'
    assert slot_font.get_family() == source_font.get_family()
    assert slot_font.get_size_in_points() == pytest.approx(
        source_font.get_size_in_points())
    assert slot_font.get_weight() == source_font.get_weight()
    assert slot_text.get_color() == source_label.get_color()
    assert slot_text.get_rotation() == source_label.get_rotation()


def test_show_all_nonbottom_xlabels_do_not_reserve_outer_label_band(
        monkeypatch):
    """Only bottom heatmap owners feed the outer X-decoration measurement.

    ``show_all`` keeps the local labels attached to the non-bottom heatmap
    axes (their axes remain hidden, as before).  Those labels must not be
    measured a second time as part of the outer structural band; otherwise a
    multi-row heatmap pays the local X-label height twice.
    """
    calls = {}
    original_measure = matrix_figure.measure_axis

    def record_measure(axis, renderer):
        key = id(axis)
        calls[key] = calls.get(key, 0) + 1
        return original_measure(axis, renderer)

    monkeypatch.setattr(matrix_figure, 'measure_axis', record_measure)
    figure, solution = _render(
        'master_multi.mat.gz',
        whatToShow='plot, heatmap and colorbar', xAxisVisibility='show_all',
        xAxisLabel='distance from TSS')

    heatmap_axes = [axis for axis in figure.axes
                    if '/block/' in (axis.get_gid() or '')]
    nonbottom = [axis for axis in heatmap_axes
                 if axis.get_xlabel() and not axis.xaxis.get_visible()]
    bottom = [axis for axis in heatmap_axes
              if axis.xaxis.get_visible()]
    assert nonbottom
    assert all(axis.xaxis.label.get_visible() for axis in nonbottom)
    # The first call is the per-column inset measurement.  A non-bottom local
    # xlabel must not receive the second call used for the outer decoration;
    # bottom owners do, and their labels are drawn once in structural slots.
    assert all(calls[id(axis)] == 1 for axis in nonbottom)
    assert all(calls[id(axis)] == 2 for axis in bottom)
    slots = [axis for axis in figure.axes
             if (axis.get_gid() or '').startswith('label/x/')]
    assert len(slots) == len(_heatmap_slot_texts(figure, 'label/x/')) == len(bottom)
    assert _heatmap_slot_texts(figure, 'label/x/') == ('distance from TSS',) * len(bottom)

    # Summary ``show_all`` labels change only the upper summary decoration;
    # the bottom heatmap edge remains the single shared structural band.
    _, outer_solution = _render(
        'master_multi.mat.gz',
        whatToShow='plot, heatmap and colorbar', xAxisVisibility='outer',
        xAxisLabel='distance from TSS')
    assert min(rect.y0 for rect in _block_rects(solution)) == pytest.approx(
        min(rect.y0 for rect in _block_rects(outer_solution)), abs=1e-6)


def test_disabling_axis_layout_keeps_ordinary_labels_on_their_original_axes(
        monkeypatch):
    style, messages = style_geometry.resolve_style({
        'label_layout': {'auto_axis_label_layout': False}})
    assert messages == []
    figure, solution = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar',
        xAxisLabel='distance from TSS', style=style)
    local = [axis.get_xlabel() for axis in figure.axes
             if axis.xaxis.get_visible() and axis.get_xlabel()]
    assert local
    assert set(local) == {'distance from TSS'}
    assert _heatmap_slot_texts(figure, 'label/x/') == ()
    assert not any((axis.get_gid() or '').startswith('label/x/')
                   for axis in figure.axes)


def test_wide_summary_legend_and_long_structural_text_stay_on_canvas(
        monkeypatch):
    matrix = Matrix.load(str(HEATMAP_SLOT_DATA / 'master_multi.mat.gz'), threads=1)
    sample_labels = [
        'irreducibly wide sample legend entry number {}'.format(index)
        for index, _ in enumerate(matrix.header.sample_labels)]
    group_labels = [
        'very long\nmultiline region label {}'.format(index)
        for index, _ in enumerate(matrix.header.group_labels)]
    figure, solution = render_heatmap(
        plot_data(matrix, group_labels=group_labels,
                  sample_labels=sample_labels), colorMapDict={
            'colorMap': ['Reds'], 'colorList': None, 'colorNumber': 256,
            'missingDataColor': 'black', 'alpha': 1.0},
        whatToShow='plot, heatmap and colorbar', image_format='png', dpi=100,
        legendLocation='below',
        plotTitle='a global title deliberately much wider than the data block',
        heatmapYAxisLabel='a very long\ncommon vertical y label',
        yAxisLabel='a very long\nsummary vertical y label',
        threads=1)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        _assert_tight_bbox_inside_figure(axis, figure, renderer)
    assert all(rect.width > 0 for rect in _block_rects(solution))


def test_rotated_region_label_is_not_double_counted_as_bottom_decoration(
        monkeypatch):
    _, baseline = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar')
    labels = ['long rotated region label\nsecond line'
              for _ in range(2)]
    figure, decorated = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar',
        group_labels=labels)
    baseline_bottom = min(rect.y0 for rect in _block_rects(baseline))
    decorated_bottom = min(rect.y0 for rect in _block_rects(decorated))
    assert decorated_bottom == pytest.approx(baseline_bottom, abs=1)
    assert [(rect.width, rect.height) for rect in _block_rects(decorated)] == \
        pytest.approx([(rect.width, rect.height)
                       for rect in _block_rects(baseline)])
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        _assert_tight_bbox_inside_figure(axis, figure, renderer)


def test_hidden_summary_xaxis_does_not_inflate_heatmap_bottom_slack(
        monkeypatch):
    figure, solution = _render(
        'master_multi.mat.gz',
        whatToShow='plot, heatmap and colorbar')
    # Actual visible heatmap x ticks plus physical padding are roughly 31pt;
    # the hidden summary xlabel formerly inflated this to more than 80pt.
    assert min(rect.y0 for rect in _block_rects(solution)) < 50
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        _assert_tight_bbox_inside_figure(axis, figure, renderer)


def test_high_output_dpi_uses_canvas_dpi_for_layout_measurement(monkeypatch):
    figure, solution = _render(
        'master_multi.mat.gz', dpi=300,
        whatToShow='plot, heatmap and colorbar',
        yAxisLabel='log2FC', xAxisLabel='distance from TSS',
        minorTickMarks=4, sortIndicator='auto', sort_method='descend')
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        _assert_tight_bbox_inside_figure(axis, figure, renderer)
    # The summary Y label is structural, exactly as on a profile-only
    # figure; automatic heatmap X labels are structural too.
    assert [text.get_text() for axis in figure.axes
            if (axis.get_gid() or '').startswith('label/y/')
            for text in axis.texts] == ['log2FC']
    structural_xlabels = [text for axis in figure.axes
                          if (axis.get_gid() or '').startswith('label/x/')
                          for text in axis.texts]
    assert structural_xlabels
    assert all(text.get_window_extent(renderer).y0 >= figure.bbox.y0 - 1
               for text in structural_xlabels)
    assert _rects(solution, 'colorbar/')[0].width == pytest.approx(9.0)
    assert solution.rects['indicator/1'].width == pytest.approx(9.0)


def test_outer_merged_summary_y_label_uses_structural_slot(monkeypatch):
    figure, _solution = _render(
        'master_multi.mat.gz',
        whatToShow='plot, heatmap and colorbar',
        yAxisLabel=['signal'], yAxisVisibility='outer_merged')
    assert any(axis.get_gid() == 'label/y/1/1'
               for axis in figure.axes)


def test_per_column_heatmap_y_labels_split_summary_axis_runs(monkeypatch):
    figure, _solution = _render(
        'master_multi.mat.gz',
        whatToShow='plot, heatmap and colorbar',
        heatmapYAxisLabel=['RNA', 'RNA', 'ChIP', 'ChIP'],
        yAxisLabel=['signal'],
        yAxisVisibility='outer', yAxisLimits='per_y_label',
        legendLocation='none')
    heatmap_label_slots = {
        axis.get_gid() for axis in figure.axes
        if (axis.get_gid() or '').startswith('label/heatmap_y/')}
    # One label per run, keyed by the column of the cell it starts at.
    assert heatmap_label_slots == {
        'label/heatmap_y/1/1', 'label/heatmap_y/1/3'}
    summaries = sorted(
        (axis for axis in figure.axes
         if (axis.get_gid() or '').endswith('/profile')),
        key=lambda axis: axis.get_gid())
    assert len(summaries) == 4
    # Summary Y ownership is the profile rule: heatmap Y labels below do
    # not split it.
    assert [any(label.get_visible() for label in axis.get_yticklabels())
            for axis in summaries] == [True, False, False, False]
    assert summaries[0].get_ylim() == pytest.approx(summaries[1].get_ylim())
    assert summaries[2].get_ylim() == pytest.approx(summaries[3].get_ylim())


def test_right_region_labels_precede_side_colorbar(monkeypatch):
    figure, solution = _render(
        'master_multi.mat.gz',
        whatToShow='heatmap and colorbar')
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    scale = 72.0 / figure.dpi
    region_axes = [
        axis for axis in figure.axes
        if (axis.get_gid() or '').startswith('label/stack/')]
    # 'best' with one colour scale puts one right bar on the row.
    colorbar = next(axis for axis in figure.axes
                    if axis.get_gid() == 'colorbar/row/1/1')
    region_boxes = [axis.texts[0].get_window_extent(renderer)
                    for axis in region_axes]
    for axis, box in zip(region_axes, region_boxes):
        column = axis.get_gid().split('/')[3]
        block = solution.rects[f'cell/1/{column}/block/1']
        assert box.x0 * scale > block.x1
        assert box.x1 <= colorbar.get_window_extent(renderer).x0 + 1
    assert all(axis.texts[0].get_rotation() == 270
               for axis in region_axes)


def test_hidden_region_label_reserves_no_gap_with_show_region_counts(
        monkeypatch):
    """--regionsLabel "" hides a group's label; --showRegionCounts must not
    turn that back into a non-empty ' [n = ...]' string and resurrect the
    region-label column/gap that the empty string was meant to remove."""
    def region_layout_width(group_labels, show_region_counts):
        matrix = Matrix.load(str(HEATMAP_SLOT_DATA / 'master_multi.mat.gz'), threads=1)
        original_rc = matplotlib.rcParams.copy()
        try:
            _, solution = render_heatmap(
                plot_data(matrix, group_labels=group_labels,
                          show_counts=show_region_counts),
                colorMapDict={'colorMap': ['Reds'], 'colorList': None,
                              'colorNumber': 256, 'missingDataColor': 'black',
                              'alpha': 1.0},
                regionsLabel=group_labels, whatToShow='heatmap and colorbar',
                image_format='png', dpi=100, threads=1)
        finally:
            matplotlib.rcParams.update(original_rc)
        return solution.figure_size.width

    group_count = 2
    hidden = [''] * group_count
    width_hidden_no_counts = region_layout_width(hidden, False)
    width_hidden_with_counts = region_layout_width(hidden, True)
    named = ['GroupA', 'GroupB']
    width_named_with_counts = region_layout_width(named, True)

    assert width_hidden_with_counts == pytest.approx(width_hidden_no_counts)
    assert width_hidden_with_counts < width_named_with_counts


def test_merged_outer_x_axes_render_with_resolved_geometry(monkeypatch):
    # Regression: the per-sample loop in the 'outer_merged' x-axis path binds a
    # local name `geometry` (a SampleGeometry). The resolved GeometrySpec passed
    # to grid.solve must not share that name, or the solve receives a
    # SampleGeometry and raises AttributeError: 'figure_edge_padding'.
    figure, solution = _render(
        'master_multi.mat.gz',
        whatToShow='heatmap and colorbar', xAxisVisibility='outer_merged')
    assert figure is not None
    assert solution.figure_size.width > 0


def test_colorbar_roles_are_independently_wired(tmp_path, monkeypatch):
    # colorbar_title_text and colorbar_tick_label_text must actually reach the
    # colorbar title and tick labels (not silently inherit font.size).
    import json
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    (tmp_path / 'options.txt').write_text(json.dumps({'typography': {
        'colorbar_title_text': {'size': 21},
        'colorbar_tick_label_text': {'size': 17}}}))
    figure, solution = _render(
        'master_multi.mat.gz',
        whatToShow='heatmap and colorbar', zMin=[0.0], zMax=[1.0],
        colorbarLabels=['SCALE'])
    assert _heatmap_slot_texts(figure, 'colorbar_title/') == ('SCALE',)
    title_sizes, tick_sizes = [], []
    for axis in figure.axes:
        for text in axis.texts:
            if text.get_text() == 'SCALE':
                title_sizes.append(round(text.get_fontsize(), 1))
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            if label.get_text():
                tick_sizes.append(round(label.get_fontsize(), 1))
    # The colorbar title is the configured 21 pt and its ticks the configured
    # 17 pt -- neither inherits the 8 pt body font.size any more.
    assert 21.0 in title_sizes, title_sizes
    assert 17.0 in tick_sizes, tick_sizes


def test_auto_below_colorbar_titles_render_the_solved_wrap(monkeypatch):
    long_title = ('a deliberately long colour scale title that should wrap '
                  'over its served heatmap column')
    colors = {
        'colorMap': ['Reds', 'Blues'], 'colorList': None,
        'colorNumber': 256, 'missingDataColor': 'black', 'alpha': 1.0}
    figure, solution = _render(
        'master_multi.mat.gz',
        colorMapDict=colors, whatToShow='heatmap and colorbar',
        colorbarLocation='below', xAxisVisibility='all', heatmapWidth=1,
        colorbarLabels=[long_title])

    assert _heatmap_slot_texts(figure, 'colorbar_title/')
    assert any(_line_count(text) > 1
               for text in _heatmap_slot_texts(figure, 'colorbar_title/'))
    label_axes = sorted(
        (axis for axis in figure.axes
         if (axis.get_gid() or '').startswith('colorbar_title/')),
        key=lambda axis: int(axis.get_gid().rsplit('/', 1)[1]))
    assert all(text.replace('\n', ' ') == long_title
               for text in _heatmap_slot_texts(figure, 'colorbar_title/'))
    assert len(label_axes) == len(_rects(solution, 'colorbar_title/'))
    for axis, rect in zip(label_axes, _rects(solution, 'colorbar_title/')):
        _assert_axis_rect(axis, rect, solution)


def test_auto_below_common_titles_render_inside_final_grid_envelope(monkeypatch):
    long_title = ('a deliberately long common colour scale title that '
                  'wraps inside the final heatmap grid envelope')
    colors = {
        'colorMap': ['Reds', 'Blues'], 'colorList': None,
        'colorNumber': 256, 'missingDataColor': 'black', 'alpha': 1.0}
    figure, solution = _render(
        'master_multi.mat.gz',
        colorMapDict=colors, whatToShow='heatmap and colorbar',
        colorbarLocation='below_common', heatmapWidth=1,
        colorbarLabels=['short scale', long_title])

    assert _heatmap_slot_texts(figure, 'colorbar_title/')
    assert any(_line_count(text) > 1
               for text in _heatmap_slot_texts(figure, 'colorbar_title/'))
    assert any(_line_count(text) == 1
               for text in _heatmap_slot_texts(figure, 'colorbar_title/'))
    grid_left = min(rect.x0 for rect in _block_rects(solution))
    grid_right = max(rect.x1 for rect in _block_rects(solution))
    assert all(rect.x0 >= grid_left and rect.x1 <= grid_right
               for rect in _rects(solution, 'colorbar_title/') if rect is not None)
    assert all(rect.width <= grid_right - grid_left
               for rect in _rects(solution, 'colorbar_title/') if rect is not None)
    label_axes = sorted(
        (axis for axis in figure.axes
         if (axis.get_gid() or '').startswith('colorbar_title/')),
        key=lambda axis: int(axis.get_gid().rsplit('/', 1)[1]))
    assert {text.replace('\n', ' ') for text in
            _heatmap_slot_texts(figure, 'colorbar_title/')} == {
                'short scale', long_title}
    for axis, rect in zip(label_axes, _rects(solution, 'colorbar_title/')):
        _assert_axis_rect(axis, rect, solution)
    bars_by_row = {}
    for index, bar in enumerate(_rects(solution, 'colorbar/')):
        bars_by_row.setdefault(round(bar.y1, 8), []).append(index)
    mixed_rows = [indices for indices in bars_by_row.values()
                  if len({_line_count(_heatmap_slot_texts(figure, 'colorbar_title/')[index])
                          for index in indices}) > 1]
    assert mixed_rows
    for indices in bars_by_row.values():
        bars = [_rects(solution, 'colorbar/')[index] for index in indices]
        assert len({round(bar.y1, 8) for bar in bars}) == 1
        labels = [_rects(solution, 'colorbar_title/')[index] for index in indices]
        assert len({round(label.y0, 8) for label in labels}) == 1
        assert tuple(round(label.y0 - bar.y1, 8)
                     for label, bar in zip(labels, bars)) == pytest.approx(
                         (round(labels[0].y0 - bars[0].y1, 8),) * len(indices))


def test_disabling_horizontal_colorbar_layout_keeps_literal_title(monkeypatch):
    literal = ('a deliberately long colour scale title that remains literal '
               'when automatic horizontal colorbar layout is disabled')
    style, messages = style_geometry.resolve_style({
        'label_layout': {'auto_horizontal_colorbar_label_layout': False}})
    assert messages == []
    figure, solution = _render(
        'master_multi.mat.gz',
        whatToShow='heatmap and colorbar', colorbarLocation='below',
        colorbarLabels=[literal], style=style)

    assert _heatmap_slot_texts(figure, 'colorbar_title/')[0] == literal
    labels = [text.get_text() for axis in figure.axes
              if (axis.get_gid() or '').startswith('colorbar_title/')
              for text in axis.texts]
    assert labels and set(labels) == {literal}
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        if (axis.get_gid() or '').startswith('colorbar_title/'):
            _assert_tight_bbox_inside_figure(axis, figure, renderer)


@pytest.mark.parametrize('location', ['right', 'below', 'below_common'])
def test_colorbar_slots_and_ticks_fit_canvas_at_each_location(
        monkeypatch, location):
    figure, solution = _render(
        'master_multi.mat.gz',
        whatToShow='heatmap and colorbar', colorbarLocation=location,
        heatmapWidth=3,
        colorbarLabels=['a long colour scale description with several words'])
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        if (axis.get_gid() or '').startswith(
                ('colorbar_title/', 'colorbar/')):
            _assert_tight_bbox_inside_figure(axis, figure, renderer)
    assert solution.figure_size.width > 0


def test_heatmap_side_region_labels_use_their_own_font_role(
        tmp_path, monkeypatch):
    import json
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    (tmp_path / 'options.txt').write_text(json.dumps({'typography': {
        'panel_title_text': {'size': 10},
        'region_label_text': {
            'size': 18, 'weight': 'bold', 'style': 'italic',
            'family': 'Courier'}}}))
    figure, _ = _render(
        'master_multi.mat.gz', whatToShow='heatmap and colorbar')
    labels = [text for axis in figure.axes
              if (axis.get_gid() or '').startswith('label/stack/')
              for text in axis.texts if text.get_text()]
    assert labels
    for label in labels:
        assert label.get_fontsize() == 18
        assert label.get_fontweight() == 'bold'
        assert label.get_fontstyle() == 'italic'
        assert label.get_fontproperties().get_family() == ['Courier']


def test_summary_minor_ticks_match_the_heatmap(monkeypatch):
    # Regression: --minorTickMarks added minor ticks to the heatmap panels but
    # not the summary panels above them.
    figure, _ = _render(
        'master_multi.mat.gz',
        whatToShow='plot, heatmap and colorbar', minorTickMarks=4)
    summary_minor = [len(ax.xaxis.get_minorticklocs())
                     for ax in figure.axes
                     if (ax.get_gid() or '').endswith('/profile')]
    assert summary_minor and all(n == 4 for n in summary_minor), summary_minor


def test_below_colorbar_tick_density_matches_the_bar_width(monkeypatch):
    # Regression: a merged below colorbar spanning a two-column run was sized for
    # ticks against the run span (6 cm) while rendered one panel wide (3 cm), so
    # it packed 5 ticks into the narrow bar instead of the 3 that fit.
    figure, _ = _render(
        'master_multi.mat.gz', heatmapWidth=3,
        colorbarLocation='below', zMin=['1-2=-1'], zMax=['1-2=1'])
    for ax in figure.axes:
        if (ax.get_gid() or '').startswith('colorbar/cell/'):
            labels = [t.get_text() for t in ax.get_xticklabels()
                      if t.get_text()]
            assert len(labels) <= 3, (ax.get_gid(), labels)


# From test_heatmap_midpoint.py.

def test_heatmap_midpoint_argument_accepts_recycled_values(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = heatmap_process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                                 '--zMid', '0', '1.5'])

    # zMid is parsed as strings (like zMin/zMax) so explicit N,M= assignment is
    # possible; the float conversion happens during heatmap preparation.
    assert args.zMid == ['0', '1.5']


def test_shifted_colormap_keeps_normalization_linear_and_moves_middle_color():
    original = __import__('matplotlib').colormaps['bwr']
    shifted = shift_cmap_midpoint(original, vmin=-2, vcenter=0, vmax=1)
    linear_position_of_zero = (0 - (-2)) / (1 - (-2))

    np.testing.assert_allclose(shifted(linear_position_of_zero), original(0.5),
                               atol=0.01)
    # The image still uses ordinary vmin/vmax normalization, so colorbar tick
    # positions remain linear in the underlying values.
    norm = __import__('matplotlib').colors.Normalize(vmin=-2, vmax=1)
    np.testing.assert_allclose(norm([-2, -1, 0, 1]),
                               [0, 1 / 3, 2 / 3, 1])


@pytest.mark.parametrize('midpoint', [-2, 1, -3, 2])
def test_midpoint_must_be_strictly_inside_limits(midpoint):
    with pytest.raises(ValueError, match='strictly between'):
        midpoint_exponent(-2, midpoint, 1)


# From test_profile_dimensions.py.

def test_profile_dimension_arguments(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                         '--profileWidth', '6', '--profileHeight', '4'])
    assert args.cellWidth == 6
    assert args.profileHeight == 4


def test_singular_and_plural_label_aliases(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args([
        '-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
        '--sampleLabels', 'sample one', 'sample two',
        '--regionLabels', 'region one'])
    assert args.samplesLabel == ['sample one', 'sample two']
    assert args.regionsLabel == ['region one']


def test_profile_aspect_ratio_argument(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                         '--aspectRatio', '1.25'])
    assert args.profileAspectRatio == 1.25


def _series_heatmap(tmp_path, **settings):
    case = dict(PROFILE_CASES['profile_heatmap_mode'], **settings)
    _, figure = _render_profile(case, tmp_path / 'series-heatmap.png')
    axes = [axis for axis in figure.axes
            if (axis.get_gid() or '').endswith('/profile')]
    figure.canvas.draw()
    return figure, axes


def test_series_heatmap_width_only_has_exact_data_width(tmp_path):
    figure, axes = _series_heatmap(tmp_path, real_profile_width=6)
    renderer = figure.canvas.get_renderer()
    for axis in axes:
        assert axis.get_window_extent(renderer).width * 72 / figure.dpi == \
            pytest.approx(cm_to_points(6), abs=.1)


def test_series_heatmap_height_and_aspect_have_exact_data_size(tmp_path):
    figure, axes = _series_heatmap(
        tmp_path, real_profile_height=4, profile_aspect_ratio=1.5)
    renderer = figure.canvas.get_renderer()
    for axis in axes:
        box = axis.get_window_extent(renderer)
        assert box.height * 72 / figure.dpi == pytest.approx(
            cm_to_points(4), abs=.1)
        assert box.width / box.height == pytest.approx(1.5, abs=.01)


def test_series_heatmap_multiline_title_and_labels_are_contained(tmp_path):
    figure, axes = _series_heatmap(
        tmp_path, real_profile_width=5, real_profile_height=3,
        plot_title='a deliberately very long title\nwith a second line')
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        box = axis.get_tightbbox(renderer)
        assert box.x0 >= figure.bbox.x0 - 1
        assert box.y0 >= figure.bbox.y0 - 1
        assert box.x1 <= figure.bbox.x1 + 1
        assert box.y1 <= figure.bbox.y1 + 1


def test_series_heatmap_axis_label_gaps_are_unified(tmp_path):
    figure, axes = _series_heatmap(
        tmp_path, real_profile_width=5, real_profile_height=3)
    for axis in axes:
        assert axis.xaxis.labelpad == DEFAULT_GEOMETRY.axis_label_to_tick_labels_gap
        assert axis.yaxis.labelpad == DEFAULT_GEOMETRY.axis_label_to_tick_labels_gap
    plt.close(figure)


def test_profile_x_axis_label_argument(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                         '--xAxisLabel', 'distance from ncTSS'])
    assert args.xAxisLabel == ['distance from ncTSS']


def test_profile_region_count_argument(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                         '--showRegionCounts'])
    assert args.showRegionCounts


def test_profile_x_axis_label_defaults_to_generated_label(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png')])
    assert args.xAxisLabel is None


def test_external_legend_locations_are_exposed(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    for location in ('above', 'right', 'none'):
        args = process_args([
            '-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
            '--legendLocation', location])
        assert args.legendLocation == location


def test_repeated_condition_labels_share_profile_style_and_legend():
    matrix = np.vstack([
        np.tile([1.0, 2.0, 1.0], (3, 1)),
        np.tile([2.0, 3.0, 2.0], (3, 1)),
        np.tile([-1.0, -2.0, -1.0], (3, 1)),
        np.tile([-2.0, -3.0, -2.0], (3, 1)),
    ])
    hm = Matrix(MatrixHeader.from_parameters({
            'group_boundaries': [0, 3, 6, 9, 12],
            'sample_boundaries': [0, 3],
            'group_labels': ['Control', 'Treated', 'Control', 'Treated'],
            'sample_labels': ['sense and antisense'],
            'ref point': ['TSS'], 'upstream': [1], 'downstream': [1],
            'body': [0], 'bin size': [1], 'unscaled 5 prime': [0],
            'unscaled 3 prime': [0],
        }), matrix, [None] * 12, None)
    profile = build_profile_case(
        plot_data(hm), y_min=[None], y_max=[None],
        color_list=['blue', 'orange'], same_group_labels='together')
    series = _series(profile, 0)
    assert [(group, color) for group, _, _, color in series] == [
        (0, 0), (2, 0), (1, 1), (3, 1)]

    axis = profile.figure.add_subplot(111)
    seen = set()
    for group, sample, label, color in series:
        block = profile.data.layout.block(hm, group, sample)
        sub_matrix = hm.values[block.row_range[0]:block.row_range[1],
                               block.col_range[0]:block.col_range[1]]
        legend_label = label if label not in seen else '_nolegend_'
        seen.add(label)
        axis.plot(np.mean(sub_matrix, axis=0),
                  color=profile.spec.colors[color],
                  label=legend_label)
    labels = axis.get_legend_handles_labels()[1]
    colors = [line.get_color() for line in axis.lines]
    ymin, ymax = axis.get_ylim()
    plt.close(profile.figure)
    assert labels == ['Control', 'Treated']
    assert colors == ['blue', 'blue', 'orange', 'orange']
    assert ymin < 0 < ymax


def _series(profile, index):
    panel = profile.plan.panels[index]
    by_key = {item.key: item for item in profile.plan.series}
    return [(by_key[key].group, by_key[key].sample, by_key[key].label,
             series_slot(profile.plan, 0, panel.index, key) - 1)
            for key in panel.series]


def _duplicate_group_profile(mode='independent', per_group=False):
    hm = Matrix(MatrixHeader.from_parameters({
            'group_boundaries': [0, 2, 4, 6, 8],
            'sample_boundaries': [0, 3],
            'group_labels': ['Control', 'Treated', 'Control', 'Treated'],
            'sample_labels': ['sample'],
            'ref point': ['TSS'], 'upstream': [1], 'downstream': [1],
            'body': [0], 'bin size': [1], 'unscaled 5 prime': [0],
            'unscaled 3 prime': [0],
        }), np.arange(24.0).reshape(8, 3), list(range(8)), None)
    return build_profile_case(plot_data(hm), y_min=[None], y_max=[None],
                              same_group_labels=mode, per_group=per_group)


def test_independent_duplicate_groups_keep_separate_styles_and_panels():
    profile = _duplicate_group_profile('independent', per_group=True)
    assert len(profile.plan.panels) == 4
    assert [_series(profile, index)[0][3]
            for index in range(len(profile.plan.panels))] == [0, 0, 0, 0]
    plt.close(profile.figure)
    profile = _duplicate_group_profile('independent', per_group=False)
    assert [entry[3] for entry in _series(profile, 0)] == [0, 1, 2, 3]
    plt.close(profile.figure)


def test_together_duplicate_groups_share_styles_and_panels():
    profile = _duplicate_group_profile('together', per_group=True)
    assert len(profile.plan.panels) == 2
    assert [[item[0] for item in _series(profile, index)]
            for index in range(len(profile.plan.panels))] == [[0, 2], [1, 3]]
    assert profile.plan.panels[0].title == 'Control'
    assert [(group, sample) for group, sample, _, _
            in _series(profile, 0)] == [(0, 0), (2, 0)]
    plt.close(profile.figure)


def test_together_duplicate_groups_survive_count_suffixes():
    profile = _duplicate_group_profile('together', per_group=True)
    data = profile.data
    labels = replace(data.labels, groups=tuple(
        f'{key} [n = 2]' for key in data.labels.group_keys))
    # Rebuild to model CLI ordering, where counts precede series planning.
    profile = build_profile_case(replace(data, labels=labels),
                                 y_min=[None], y_max=[None],
                                 same_group_labels='together', per_group=True)
    assert [[item[0] for item in _series(profile, index)]
            for index in range(len(profile.plan.panels))] == [[0, 2], [1, 3]]
    assert profile.plan.panels[0].title == 'Control [n = 4]'
    plt.close(profile.figure)


def test_merge_duplicate_groups_concatenates_rows():
    profile = _duplicate_group_profile('merge')
    matrix = profile.data.matrix
    layout = merge_groups_by_key(profile.data.layout,
                                 profile.data.labels.group_keys)
    assert layout.group_bounds == (0, 4, 8)
    np.testing.assert_array_equal(
        matrix.values[layout.row_indices()[:4]],
        np.vstack([matrix.values[0:2], matrix.values[4:6]]))
    plt.close(profile.figure)


def test_same_group_labels_argument_defaults_to_independent(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png')])
    assert args.sameGroupLabels == 'independent'
    args = process_args(['-m', str(matrix), '-out', str(tmp_path / 'plot.png'),
                         '--sameGroupLabels', 'together'])
    assert args.sameGroupLabels == 'together'


@pytest.mark.parametrize('options', [
    ['--arrangeSamples', '1', '2'],
    ['--gridColumns', '2'],
    ['--sampleSetLabels', 'set one'],
    ['--xAxisVisibility', 'show_all'],
    ['--commonLegend'],
])
def test_series_heatmap_accepts_custom_layouts(tmp_path, options):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args(['-m', str(matrix), '-o', str(tmp_path / 'plot.png'),
                         '--plotType', 'heatmap', *options])
    assert args.plotType == 'heatmap'


# From test_profile_subplots.py.

def _profile_slot_texts(profile, prefix):
    return tuple(axis.texts[0].get_text() for axis in profile.figure.axes
                 if (axis.get_gid() or '').startswith(prefix) and axis.texts)


def _assert_structural_slots_inside(figure):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    for axis in figure.axes:
        if not ((axis.get_gid() or '').startswith(('label/', 'legend/')) or
                (axis.get_gid() or '').endswith('/title')):
            continue
        box = axis.get_tightbbox(renderer)
        assert box.x0 >= canvas.x0 - 1
        assert box.y0 >= canvas.y0 - 1
        assert box.x1 <= canvas.x1 + 1
        assert box.y1 <= canvas.y1 + 1


def test_auxiliary_profile_colorbars_share_final_canvas_envelope(
        tmp_path, monkeypatch):
    profile = build_profile_case(
        plot_data(_plot_matrix(groups=('A', 'B'))),
        str(tmp_path / 'profile-heatmap.png'),
        plot_type='heatmap', per_group=True,
        y_min=[None], y_max=[None], image_format='png')

    try:
        figure = profile.figure
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        assert profile.solution.figure_size.width == pytest.approx(
            figure.get_size_inches()[0] * 72)
        assert profile.solution.figure_size.height == pytest.approx(
            figure.get_size_inches()[1] * 72)
        for axis in figure.axes:
            box = axis.get_tightbbox(renderer)
            assert box.x0 >= figure.bbox.x0 - 1
            assert box.y0 >= figure.bbox.y0 - 1
            assert box.x1 <= figure.bbox.x1 + 1
            assert box.y1 <= figure.bbox.y1 + 1
    finally:
        plt.close(profile.figure)


def _plot_matrix(groups=('genes',), samples=('s1', 's2', 's3', 's4')):
    rows_per_group = 2
    bins_per_sample = 3
    matrix = np.arange(len(groups) * rows_per_group *
                       len(samples) * bins_per_sample,
                       dtype=float).reshape(len(groups) * rows_per_group,
                                            len(samples) * bins_per_sample)
    regions = [['chr1', [(i, i + 1)], 'r{}'.format(i), 0, '+', '0']
               for i in range(matrix.shape[0])]
    parameters = {
        'group_boundaries': list(range(0, matrix.shape[0] + 1, rows_per_group)),
        'sample_boundaries': list(range(0, matrix.shape[1] + 1, bins_per_sample)),
        'group_labels': list(groups), 'sample_labels': list(samples),
        'ref point': ['TSS'] * len(samples),
        'upstream': [0] * len(samples),
        'downstream': [3] * len(samples),
        'body': [0] * len(samples),
        'bin size': [1] * len(samples),
        'unscaled 5 prime': [0] * len(samples),
        'unscaled 3 prime': [0] * len(samples),
    }
    return Matrix(MatrixHeader.from_parameters(parameters), matrix, regions, None)


def _panel_series(profile):
    by_key = {item.key: item for item in profile.plan.series}
    return [[(by_key[key].group, by_key[key].sample,
              by_key[key].label,
              series_slot(profile.plan, 0, panel.index, key) - 1)
             for key in panel.series] for panel in profile.plan.panels]


def _grid_shape(plan):
    return (max(panel.row for panel in plan.panels) + 1,
            max(panel.column for panel in plan.panels) + 1)


def _external_legends(profile):
    return [axis.get_legend() for axis in
            profile.figure.axes[len(profile.plan.panels):]
            if axis.get_legend() is not None]


def _profile(tmp_path, groups=('genes',), **kwargs):
    subplot_samples = kwargs.pop('subplot_samples', ['1,2', '3,4'])
    profile = build_profile_case(
        plot_data(_plot_matrix(groups=groups)), str(tmp_path / 'profile.png'),
        subplot_samples=subplot_samples, **kwargs)
    return profile


def _label_style(**changes):
    base = config.resolve_style({})[0]
    return replace(base, label_layout=replace(base.label_layout, **changes))


def _heterogeneous_plot_matrix():
    matrix = np.arange(2 * 5, dtype=float).reshape(2, 5)
    regions = [['chr1', [(i, i + 1)], 'r{}'.format(i), 0, '+', '0']
               for i in range(matrix.shape[0])]
    parameters = {
        'group_boundaries': [0, 2], 'sample_boundaries': [0, 2, 5],
        'group_labels': ['genes'], 'sample_labels': ['reference', 'scaled'],
        'ref point': ['TSS', None], 'upstream': [1000, 1000],
        'downstream': [1000, 1000], 'body': [0, 1000],
        'bin size': [1000, 1000], 'unscaled 5 prime': [0, 0],
        'unscaled 3 prime': [0, 0],
    }
    return Matrix(MatrixHeader.from_parameters(parameters), matrix, regions, None)


def test_subplot_sample_panels_and_shape(tmp_path):
    profile = _profile(
        tmp_path, subplot_columns=2, subplot_labels=['exp1', 'exp2'])
    try:
        assert _grid_shape(profile.plan) == (1, 2)
        assert [panel.title for panel in profile.plan.panels] == [
            'exp1', 'exp2']
        assert [[item[1] for item in panel]
                for panel in _panel_series(profile)] == [[0, 1], [2, 3]]
    finally:
        plt.close(profile.figure)


def test_overlay_colors_are_unique_per_group_sample_in_plotting_order(tmp_path):
    profile = _profile(
        tmp_path, groups=('A', 'B'),
        subplot_samples=['3,1', '4,2'],
        subplot_columns=2, subplot_group_placement='overlay')
    try:
        first = _panel_series(profile)[0]
        second = _panel_series(profile)[1]

        # Both dimensions vary in an overlay, so every group/sample pairing
        # is a distinct visual series. Colours follow first plotting order.
        assert [(sample, color) for _, sample, _, color in first] == [
            (2, 0), (0, 1), (2, 2), (0, 3)]
        assert [(sample, color) for _, sample, _, color in second] == [
            (3, 4), (1, 5), (3, 6), (1, 7)]
    finally:
        plt.close(profile.figure)


def test_same_sample_labels_controls_colour_identity(tmp_path):
    independent = build_profile_case(
        plot_data(_plot_matrix(groups=('genes',), samples=('WT', 'WT'))),
        str(tmp_path / 'independent.png'), per_group=True)
    together = build_profile_case(
        plot_data(_plot_matrix(groups=('genes',), samples=('WT', 'WT'))),
        str(tmp_path / 'together.png'), per_group=True,
        same_sample_labels='together')
    try:
        assert [item[3] for item in
                _panel_series(independent)[0]] == [0, 1]
        assert [item[3] for item in
                _panel_series(together)[0]] == [0, 0]
    finally:
        plt.close(independent.figure)
        plt.close(together.figure)


@pytest.mark.parametrize(
    'same_sample_labels,colors,expected_entries', [
        ('independent', ['navy', 'lime'], 2),
        ('together', ['navy'], 1),
        # Matching user colours also make the two visible keys identical.
        ('independent', ['navy', 'navy'], 1),
    ])
def test_duplicate_sample_legend_requires_matching_label_and_colour(
        tmp_path, monkeypatch, same_sample_labels, colors, expected_entries):
    profile = build_profile_case(
        plot_data(_plot_matrix(groups=('genes',), samples=('WT', 'WT'))),
        str(tmp_path / 'legend.png'), per_group=True,
        same_sample_labels=same_sample_labels, color_list=colors,
        image_format='png')

    try:
        legend = profile.figure.axes[0].get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == expected_entries
        assert [text.get_text() for text in legend.get_texts()] == (
            ['WT'] * expected_entries)
    finally:
        plt.close(profile.figure)


@pytest.mark.parametrize('legend_location', ['above', 'below', 'right'])
def test_duplicate_sample_legend_dedupes_on_external_locations(
        tmp_path, monkeypatch, legend_location):
    """Identical legend keys must collapse for external legend placements
    exactly as they already do for the internal 'best' location -- the
    per-panel legend is rebuilt from re-measured entries before being drawn
    in its external slot, and that rebuild must not resurrect duplicates."""
    profile = build_profile_case(
        plot_data(_plot_matrix(groups=('genes',), samples=('WT', 'WT'))),
        str(tmp_path / 'legend.png'), per_group=True,
        color_list=['navy', 'navy'], image_format='png',
        legend_location=legend_location)

    try:
        assert len(_external_legends(profile)) == 1
        legend = _external_legends(profile)[0]
        assert [text.get_text() for text in legend.get_texts()] == ['WT']
    finally:
        plt.close(profile.figure)


@pytest.mark.parametrize(
    'placement,expected',
    [('overlay', ['exp1', 'exp2']),
     ('adjacent', ['exp1 — A', 'exp1 — B', 'exp2 — A', 'exp2 — B']),
     ('end', ['exp1 — A', 'exp2 — A', 'exp1 — B', 'exp2 — B'])])
def test_subplot_group_placement(tmp_path, placement, expected):
    profile = _profile(
        tmp_path, groups=('A', 'B'), subplot_columns=2,
        subplot_labels=['exp1', 'exp2'],
        subplot_group_placement=placement)
    try:
        assert [panel.title for panel in profile.plan.panels] == expected
    finally:
        plt.close(profile.figure)


def test_subplot_rows_determine_columns(tmp_path):
    profile = build_profile_case(
        plot_data(_plot_matrix(samples=('s1', 's2', 's3', 's4', 's5', 's6'))),
        str(tmp_path / 'profile.png'),
        subplot_samples=['1,2', '3,4', '5,6'], subplot_rows=2)
    try:
        assert _grid_shape(profile.plan) == (2, 2)
    finally:
        plt.close(profile.figure)


def test_subplot_rows_do_not_require_explicit_samples(tmp_path):
    profile = build_profile_case(
        plot_data(_plot_matrix()), str(tmp_path / 'profile.png'), subplot_rows=2)
    try:
        assert _grid_shape(profile.plan) == (2, 2)
        assert [[item[1] for item in panel]
                for panel in _panel_series(profile)] == [[0], [1], [2], [3]]
    finally:
        plt.close(profile.figure)


def test_unified_default_keeps_group_series_colours(tmp_path):
    profile = build_profile_case(
        plot_data(_plot_matrix(groups=('A', 'B'))), str(tmp_path / 'profile.png'))
    try:
        assert [[item[3] for item in panel]
                for panel in _panel_series(profile)] == [[0, 1]] * 4
        assert [[item[2] for item in panel]
                for panel in _panel_series(profile)] == [['A', 'B']] * 4
    finally:
        plt.close(profile.figure)


def test_per_group_is_unified_adjacent_preset(tmp_path):
    profile = build_profile_case(
        plot_data(_plot_matrix(groups=('A', 'B'))), str(tmp_path / 'profile.png'),
        per_group=True)
    try:
        assert [panel.title for panel in profile.plan.panels] == ['A', 'B']
        assert [panel.title for panel in profile.plan.panels] == ['A', 'B']
        assert [[item[1] for item in panel]
                for panel in _panel_series(profile)] == [
                    [0, 1, 2, 3], [0, 1, 2, 3]]
    finally:
        plt.close(profile.figure)


@pytest.mark.parametrize(
    'placement,shape,titles', [
        ('by_row', (2, 2), ['A', 'B', '', '']),
        ('by_column', (2, 2), ['exp1', 'exp2', '', ''])])
def test_structural_group_placements_override_shape(
        tmp_path, placement, shape, titles):
    profile = _profile(
        tmp_path, groups=('A', 'B'), subplot_rows=1,
        subplot_samples=['1,2', '3,4'], subplot_labels=['exp1', 'exp2'],
        subplot_group_placement=placement)
    try:
        assert _grid_shape(profile.plan) == shape
        assert [panel.title for panel in profile.plan.panels] == titles
        expected_rows = (('exp1', 'exp2') if placement == 'by_row'
                         else ('A', 'B'))
        assert profile.plan.row_labels == expected_rows
    finally:
        plt.close(profile.figure)


def test_by_row_single_group_uses_group_as_column_title(tmp_path):
    profile = _profile(
        tmp_path, groups=('genes',),
        subplot_samples=['1', '2', '3', '4'],
        subplot_labels=['ZC3H4', 'F224A', 'Y64A', 'P82'],
        subplot_group_placement='by_row')
    try:
        assert _grid_shape(profile.plan) == (4, 1)
        assert [panel.title for panel in profile.plan.panels] == [
            'genes', '', '', '']
        assert profile.plan.row_labels == (
            'ZC3H4', 'F224A', 'Y64A', 'P82')
    finally:
        plt.close(profile.figure)


@pytest.mark.parametrize('placement', ['by_row', 'by_column'])
def test_structural_group_placement_best_keeps_panel_legends(
        tmp_path, monkeypatch, placement):
    profile = _profile(
        tmp_path, groups=('A', 'B'), subplot_samples=['1,2', '3,4'],
        subplot_group_placement=placement, legend_location='best',
        image_format='png')

    try:
        data_axes = profile.figure.axes[:len(profile.plan.panels)]
        assert all(axis.get_legend() is not None for axis in data_axes)
        assert all(axis.get_legend().get_texts() for axis in data_axes)
    finally:
        plt.close(profile.figure)


@pytest.mark.parametrize('placement,scope', [
    ('by_row', 'row'), ('by_column', 'column')])
@pytest.mark.parametrize('location', ['above', 'below', 'right'])
def test_structural_group_placements_share_external_legends(
        tmp_path, monkeypatch, placement, scope, location):
    profile = _profile(
        tmp_path, groups=('A', 'B'), subplot_samples=['1,2', '3,4'],
        subplot_group_placement=placement, legend_location=location,
        image_format='png')

    try:
        slots = [axis for axis in profile.figure.axes
                 if (axis.get_gid() or '').startswith('legend/{}/'.format(scope))]
        assert len(slots) == 2
        assert all(axis.get_legend() is not None for axis in slots)
        row_labels = [axis for axis in profile.figure.axes
                      if (axis.get_gid() or '').startswith(
                          'label/row/')]
        assert len(row_labels) == 2
        assert all(axis.texts[0].get_rotation() == 270 for axis in row_labels)
    finally:
        plt.close(profile.figure)


def test_subplot_samples_reject_duplicates(tmp_path):
    with pytest.raises(ValueError, match='more than one sample set'):
        _profile(tmp_path, subplot_samples=['1,2', '2,3'])


def test_subplot_arguments(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args([
        '-m', str(matrix), '-o', str(tmp_path / 'plot.png'),
        '--arrangeSamples', '1,2', '3,4', '--gridRows', '2',
        '--xAxisVisibility', 'outer', '--yAxisVisibility', 'outer_merged',
        '--commonLegend',
        '--sampleSetGroupArrangement', 'end', '--sampleSetLabels', 'one', 'two'])
    assert args.arrangeSamples == ['1,2', '3,4']
    assert args.gridRows == 2
    assert args.xAxisVisibility == 'outer'
    assert args.yAxisVisibility == 'outer_merged'
    assert args.yAxisLimits == 'per_y_label'
    assert args.commonLegend
    assert args.sampleSetGroupArrangement == 'end'
    assert args.sampleSetLabels == ['one', 'two']


def test_same_sample_labels_argument(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args([
        '-m', str(matrix), '-o', str(tmp_path / 'plot.png'),
        '--sameSampleLabels', 'together'])
    assert args.sameSampleLabels == 'together'


def test_subplot_shape_without_explicit_sample_list(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args([
        '-m', str(matrix), '-o', str(tmp_path / 'plot.png'),
        '--gridRows', '2'])
    assert args.arrangeSamples is None
    assert args.gridRows == 2


def test_num_plots_per_row_has_been_removed(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    with pytest.raises(SystemExit):
        process_args([
            '-m', str(matrix), '-o', str(tmp_path / 'plot.png'),
            '--numPlotsPerRow', '2'])


def test_subplot_profile_renders(tmp_path):
    output = tmp_path / 'profile.png'
    build_profile_case(
        plot_data(_plot_matrix()), str(output), subplot_samples=['1,2', '3,4'],
        subplot_columns=2, subplot_labels=['exp1', 'exp2'],
        subplot_x_axes='outer', subplot_y_axes='outer',
        legend_location='none', image_format='png')

    assert output.stat().st_size > 0


def test_differently_shaped_samples_render_in_separate_panels(tmp_path):
    output = tmp_path / 'heterogeneous.png'
    profile = build_profile_case(
        plot_data(_heterogeneous_plot_matrix()), str(output),
        x_axis_label=None, y_axis_label=['RNA', 'ChIP'],
        reference_point_label=['promoter'], start_label=['start'],
        end_label=['end'], legend_location='none', image_format='png')

    assert output.stat().st_size > 0
    assert [plan.geometry.bin_count
            for plan in profile.sample_set_plans] == [2, 3]
    assert [plan.x_axis_label for plan in profile.sample_set_plans] == [
        'distance from promoter', 'gene distance']


def test_profile_data_export_uses_display_panels_and_cached_statistics(
        tmp_path, monkeypatch):
    profile = build_profile_case(
        plot_data(_plot_matrix(groups=('same', 'same'), samples=('WT', 'WT'))),
        str(tmp_path / 'profile.png'), subplot_samples=['1', '2'],
        subplot_columns=2, averagetype='mean', legend_location='none',
        image_format='png')

    try:
        # Rendering prepared the exact statistics. Export must not calculate
        # them again (particularly important for bootstrap confidence bands).
        monkeypatch.setattr(
            profile_computation, 'calculate_profile_batch',
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError('profile statistics were recomputed')))
        output = tmp_path / 'profile.tsv'
        write_profile_table(output, profile.plan, profile.prepared, profile.spec)
        with open(output, newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))

        assert len(rows) == 4
        assert [row['panel_index'] for row in rows] == ['1', '1', '2', '2']
        assert [row['series_index'] for row in rows] == ['1', '2', '1', '2']
        # Repeated visible labels are intentionally retained; occurrence IDs
        # distinguish them both within a panel and between panels.
        assert {row['panel_label'] for row in rows} == {'WT'}
        assert {row['series_label'] for row in rows} == {'same: WT'}
        assert {row['component'] for row in rows} == {'mean'}
        assert {row['valid_bin_count'] for row in rows} == {'3'}
        assert all(len(json.loads(row['breakpoint_labels'])) >= 1 for row in rows)
        assert all(len(json.loads(row['breakpoint_bins'])) >= 1 for row in rows)
        assert all(row['breakpoint_unit'] == 'bp' for row in rows)
    finally:
        plt.close(profile.figure)


def test_profile_data_export_pads_variable_bin_counts(tmp_path):
    profile = build_profile_case(
        plot_data(_heterogeneous_plot_matrix()), str(tmp_path / 'profile.png'),
        legend_location='none', image_format='png')

    try:
        output = tmp_path / 'profile.tsv'
        write_profile_table(output, profile.plan, profile.prepared, profile.spec)
        with open(output, newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        assert [row['valid_bin_count'] for row in rows] == ['2', '3']
        assert rows[0]['bin_2'] == 'nan'
        assert rows[1]['bin_2'] != 'nan'
    finally:
        plt.close(profile.figure)


@pytest.mark.parametrize(
    'placement,expected', [
        ('by_row', {
            '1': 'exp1 — A', '2': 'exp1 — B',
            '3': 'exp2 — A', '4': 'exp2 — B'}),
        ('by_column', {
            '1': 'exp1 — A', '2': 'exp2 — A',
            '3': 'exp1 — B', '4': 'exp2 — B'}),
    ])
def test_profile_data_export_keeps_logical_structural_panel_identity(
        tmp_path, placement, expected):
    profile = _profile(
        tmp_path, groups=('A', 'B'),
        subplot_samples=['1,2', '3,4'], subplot_labels=['exp1', 'exp2'],
        subplot_group_placement=placement, legend_location='none',
        image_format='png')

    try:
        # Repeated visual headings remain suppressed in the rendered grid.
        assert profile.plan.panels[2].title == ''
        assert profile.plan.panels[3].title == ''

        output = tmp_path / ('profile-' + placement + '.tsv')
        write_profile_table(output, profile.plan, profile.prepared, profile.spec)
        with open(output, newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        labels_by_panel = {}
        for row in rows:
            labels_by_panel.setdefault(row['panel_index'], set()).add(
                row['panel_label'])
        assert labels_by_panel == {
            panel: {label} for panel, label in expected.items()}
        assert all(label for labels in labels_by_panel.values()
                   for label in labels)
    finally:
        plt.close(profile.figure)


def test_profile_data_export_escapes_control_characters_in_labels(tmp_path):
    sample_label = 'sample\nline\tcolumn\\slash\rreturn'
    group_label = 'group\nline\tcolumn\\slash\rreturn'
    breakpoint_label = 'point\nline\tcolumn\\slash\rreturn'
    profile = build_profile_case(
        plot_data(_plot_matrix(groups=(group_label,), samples=(sample_label,))),
        str(tmp_path / 'profile.png'), legend_location='none',
        reference_point_label=breakpoint_label, image_format='png')
    try:
        output = tmp_path / 'profile.tsv'
        write_profile_table(output, profile.plan, profile.prepared, profile.spec)

        # The header and every record must each occupy exactly one physical
        # line, with the same number of tab-separated fields.
        physical_lines = output.read_text(encoding='utf-8').splitlines()
        assert len(physical_lines) == 2
        assert len({line.count('\t') for line in physical_lines}) == 1

        with open(output, newline='', encoding='utf-8') as handle:
            row = next(csv.DictReader(handle, delimiter='\t'))
        assert row['panel_label'] == (
            'sample\\nline\\tcolumn\\\\slash\\rreturn')
        assert row['series_label'] == (
            'group\\nline\\tcolumn\\\\slash\\rreturn')
        assert breakpoint_label in json.loads(row['breakpoint_labels'])
    finally:
        plt.close(profile.figure)


def test_profile_cli_matrix_export_is_full_plain_matrix_not_panel_expansion(
        tmp_path):
    model = _plot_matrix(groups=('genes',), samples=('s1', 's2'))
    parameters = dict(model.header.parameters)
    parameters.update({
        'min threshold': None, 'max threshold': None,
        'sort regions': 'no', 'sort using': 'mean'})
    source = OwnedMatrix.from_compute(parameters,
                                      model.values.astype(np.float32),
                                      model.regions)
    matrix_path = tmp_path / 'source.gz'
    source.save(matrix_path, compressed=True, threads=1)

    output_matrix = tmp_path / 'plotted.mat'
    plot_profile_module.main([
        '-m', str(matrix_path), '-o', str(tmp_path / 'profile.png'),
        '--outFileNameMatrix', str(output_matrix), '--arrangeSamples', '2', '1',
        '--sortRegions', 'no', '--legendLocation', 'none'])

    assert output_matrix.read_bytes().startswith(b'@')
    restored = Matrix.load(output_matrix, threads=1)
    assert restored.header.sample_labels == ('s1', 's2')
    assert restored.values.shape == source.values.shape
    np.testing.assert_array_equal(restored.values, source.values)


def test_differently_shaped_samples_cannot_share_one_panel(tmp_path):
    with pytest.raises(ValueError, match='X-axis geometries differ'):
        build_profile_case(
            plot_data(_heterogeneous_plot_matrix()), str(tmp_path / 'invalid.png'),
            subplot_samples=['1,2'])


def test_y_axis_labels_align_within_a_grid_column(tmp_path, monkeypatch):
    profile = build_profile_case(
        plot_data(_plot_matrix(samples=('small', 'large'))),
        str(tmp_path / 'aligned.png'), subplot_columns=1,
        y_axis_label=['signal'], subplot_y_axes='show_all',
        y_min=[0, 0], y_max=[10, 10000], legend_location='none',
        image_format='png')

    try:
        profile.figure.canvas.draw()
        renderer = profile.figure.canvas.get_renderer()
        centers = []
        for axis in profile.figure.axes[:2]:
            box = axis.yaxis.label.get_window_extent(renderer)
            centers.append((box.x0 + box.x1) / 2.0)
        assert centers[0] == pytest.approx(centers[1], abs=.5)
        # Alignment uses the widest tick-label envelope in the column.  Its
        # label must retain exactly the configured edge-to-edge clearance.
        widest_axis = min(
            profile.figure.axes[:2],
            key=lambda axis: min(
                label.get_window_extent(renderer).x0
                for label in axis.get_yticklabels()
                if label.get_visible() and label.get_text()))
        tick_left = min(
            label.get_window_extent(renderer).x0
            for label in widest_axis.get_yticklabels()
            if label.get_visible() and label.get_text())
        label_right = widest_axis.yaxis.label.get_window_extent(renderer).x1
        assert (tick_left - label_right) * 72.0 / profile.figure.dpi == \
            pytest.approx(
                AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS, abs=.25)
    finally:
        plt.close(profile.figure)


def test_local_x_axis_label_uses_exact_tick_label_gap(tmp_path, monkeypatch):
    profile = _profile(
        tmp_path, subplot_columns=2, x_axis_label=['distance'],
        subplot_x_axes='show_all', legend_location='none',
        image_format='png')

    try:
        profile.figure.canvas.draw()
        renderer = profile.figure.canvas.get_renderer()
        axis = profile.figure.axes[0]
        tick_bottom = min(
            label.get_window_extent(renderer).y0
            for label in axis.get_xticklabels()
            if label.get_visible() and label.get_text())
        label_top = axis.xaxis.label.get_window_extent(renderer).y1
        assert (tick_bottom - label_top) * 72.0 / profile.figure.dpi == \
            pytest.approx(
                X_AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS, abs=.25)
    finally:
        plt.close(profile.figure)


def test_outer_merged_creates_structural_axis_label_slots(tmp_path,
                                                          monkeypatch):
    profile = build_profile_case(
        plot_data(_plot_matrix()), str(tmp_path / 'merged.png'),
        subplot_samples=['1,2', '3,4'], subplot_rows=2,
        x_axis_label=['distance'], y_axis_label=['signal'],
        subplot_x_axes='outer_merged', subplot_y_axes='outer_merged',
        legend_location='none', image_format='png')

    try:
        gids = [axis.get_gid() for axis in profile.figure.axes]
        assert any((gid or '').startswith('label/x/') for gid in gids)
        assert any((gid or '').startswith('label/y/') for gid in gids)
    finally:
        plt.close(profile.figure)


@pytest.mark.parametrize('legend_location', ['above', 'below'])
def test_outer_merged_x_label_is_below_ticks_and_clear_of_row_legend(
        tmp_path, monkeypatch, legend_location):
    """The bottom decorations form one ordered, physically measured stack."""
    profile = _profile(
        tmp_path, groups=('A', 'B'),
        subplot_samples=['1,2', '3,4'],
        subplot_group_placement='by_row',
        x_axis_label=['distance'], subplot_x_axes='outer_merged',
        legend_location=legend_location, image_format='png')

    try:
        profile.figure.canvas.draw()
        renderer = profile.figure.canvas.get_renderer()
        scale = 72.0 / profile.figure.dpi
        data_axes = profile.figure.axes[:4]
        bottom_axes = data_axes[2:]
        tick_bottom = min(
            label.get_window_extent(renderer).y0
            for axis in bottom_axes for label in axis.get_xticklabels()
            if label.get_visible() and label.get_text())
        label_axis = next(
            axis for axis in profile.figure.axes
            if (axis.get_gid() or '').startswith('label/x/'))
        label_box = label_axis.texts[0].get_window_extent(renderer)
        assert (tick_bottom - label_box.y1) * scale == pytest.approx(
            COMMON_AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS, abs=.25)

        legend_axes = [
            axis for axis in profile.figure.axes
            if (axis.get_gid() or '').startswith('legend/row/')]
        assert legend_axes
        legend_boxes = [axis.get_tightbbox(renderer)
                        for axis in legend_axes]
        if legend_location == 'below':
            bottom_legend = min(legend_boxes, key=lambda box: box.y0)
            assert (label_box.y0 - bottom_legend.y1) * scale == pytest.approx(
                EXTERNAL_LEGEND_TO_CONTENT_GAP_POINTS, abs=.5)
        else:
            # Above legends belong to the top edge of their own row and must
            # never enter the bottom X-label stack.
            assert all(not label_box.overlaps(box) for box in legend_boxes)
    finally:
        plt.close(profile.figure)


def test_outer_merged_axis_label_draws_the_selected_wrapped_text(
        tmp_path, monkeypatch):
    label = ' '.join(['verylonglabel'] * 20)
    profile = build_profile_case(
        plot_data(_plot_matrix()), str(tmp_path / 'wrapped-profile.png'),
        subplot_samples=['1,2', '3,4'], subplot_rows=2,
        x_axis_label=[label], y_axis_label=['signal'],
        subplot_x_axes='outer_merged', subplot_y_axes='outer_merged',
        legend_location='none', image_format='png')

    try:
        selected = _profile_slot_texts(profile, 'label/x/')[-1]
        assert '\n' in selected
        slot = next(axis for axis in profile.figure.axes
                    if (axis.get_gid() or '').startswith('label/x/'))
        assert slot.texts[0].get_text() == selected
        assert all(axis.get_xlabel() == ''
                   for axis in profile.figure.axes[:len(profile.plan.panels)])
        _assert_structural_slots_inside(profile.figure)
    finally:
        plt.close(profile.figure)


def test_profile_structuralizes_ordinary_y_and_facet_labels_together(
        tmp_path, monkeypatch):
    long_y = ' '.join(['long-y-label'] * 16)
    long_facet = ' '.join(['long-facet-label'] * 12)
    profile = build_profile_case(
        plot_data(_plot_matrix(groups=('A', 'B'))),
        str(tmp_path / 'vertical-labels.png'),
        subplot_samples=['1,2', '3,4'], subplot_rows=2,
        subplot_group_placement='by_row',
        subplot_labels=[long_facet, long_facet],
        y_axis_label=[long_y], subplot_y_axes='show_all',
        subplot_x_axes='outer', legend_location='none', image_format='png')

    try:
        assert any('\n' in text for text in
                   _profile_slot_texts(profile, 'label/y/'))
        assert any('\n' in text for text in
                   _profile_slot_texts(profile, 'label/row/'))
        data_axes = profile.figure.axes[:len(profile.plan.panels)]
        assert all(axis.get_ylabel() == '' for axis in data_axes)
        y_slots = [axis for axis in profile.figure.axes
                   if (axis.get_gid() or '').startswith('label/y/')]
        facet_slots = [axis for axis in profile.figure.axes
                       if (axis.get_gid() or '').startswith(
                           'label/row/')]
        assert y_slots
        assert facet_slots
        assert all(slot.texts[0].get_text() for slot in y_slots + facet_slots)
        _assert_structural_slots_inside(profile.figure)
    finally:
        plt.close(profile.figure)


def test_outer_merged_axis_label_auto_off_keeps_literal_text(
        tmp_path, monkeypatch):
    label = ' '.join(['verylonglabel'] * 20)
    original_measure = profile_plotting.measure_axis

    def style_source_axis_label(axis, renderer):
        axis.xaxis.label.set_fontsize(13)
        axis.xaxis.label.set_color('purple')
        axis.xaxis.label.set_rotation(17)
        return original_measure(axis, renderer)

    monkeypatch.setattr(profile_plotting, 'measure_axis',
                        style_source_axis_label)
    profile = build_profile_case(
        plot_data(_plot_matrix()), str(tmp_path / 'literal-profile.png'),
        subplot_samples=['1,2', '3,4'], subplot_rows=2,
        x_axis_label=[label], y_axis_label=['signal'],
        subplot_x_axes='outer_merged', subplot_y_axes='outer_merged',
        legend_location='none', image_format='png',
        style=_label_style(auto_axis_label_layout=False))
    try:
        assert label in _profile_slot_texts(profile, 'label/x/')
        slot = next(axis for axis in profile.figure.axes
                    if (axis.get_gid() or '').startswith('label/x/'))
        slot_text = slot.texts[0]
        source_label = profile.figure.axes[0].xaxis.label
        source_font = source_label.get_fontproperties()
        slot_font = slot_text.get_fontproperties()
        assert slot_text.get_text() == label
        assert slot_font.get_family() == source_font.get_family()
        assert slot_font.get_size_in_points() == pytest.approx(
            source_font.get_size_in_points())
        assert slot_font.get_weight() == source_font.get_weight()
        assert slot_text.get_color() == source_label.get_color()
        assert slot_text.get_rotation() == source_label.get_rotation()
        _assert_structural_slots_inside(profile.figure)
    finally:
        plt.close(profile.figure)


def test_profile_decorations_are_measured_at_requested_panel_width(
        tmp_path, monkeypatch):
    measured_widths = []
    original = profile_plotting.measure_axis

    def record_width(axis, renderer):
        if axis.get_xlabel():
            measured_widths.append(
                axis.get_window_extent(renderer).width * 72 / axis.figure.dpi)
        return original(axis, renderer)

    monkeypatch.setattr(profile_plotting, 'measure_axis', record_width)
    _profile(
        tmp_path, subplot_columns=2, legend_location='none',
        real_profile_width=3, real_profile_height=4,
        x_axis_label='distance from TSS [kb]', image_format='png',
        style=_label_style(auto_axis_label_layout=False))
    # The automatic path structuralizes X labels before measure_axis; this
    # regression intentionally exercises the legacy axis-owned measurement
    # route used when the policy is disabled.
    assert measured_widths
    assert measured_widths == pytest.approx(
        [cm_to_points(3)] * len(measured_widths), abs=.1)


def test_profile_panel_titles_are_structuralized_and_wrapped(
        tmp_path, monkeypatch):
    title = ' '.join(['verylongpaneltitle'] * 12)
    profile = _profile(
        tmp_path, subplot_samples=['1', '2'], subplot_columns=2,
        subplot_labels=[title, title], subplot_x_axes='show_all',
        legend_location='none', image_format='png')

    try:
        title_slots = [axis for axis in profile.figure.axes
                       if (axis.get_gid() or '').endswith('/title')]
        assert len(title_slots) == 2
        selected = _profile_slot_texts(profile, 'cell/')
        assert len(selected) == 2
        assert all('\n' in text for text in selected)
        assert [axis.texts[0].get_text() for axis in title_slots] == list(selected)
        assert all(axis.get_title() == '' for axis in profile.figure.axes[:2])
    finally:
        plt.close(profile.figure)


def test_profile_ordinary_x_labels_are_structuralized_when_enabled(
        tmp_path, monkeypatch):
    label = ' '.join(['verylongxlabel'] * 12)
    profile = _profile(
        tmp_path, subplot_samples=['1', '2'], subplot_columns=2,
        subplot_x_axes='show_all', x_axis_label=[label],
        legend_location='none', image_format='png')

    try:
        selected = _profile_slot_texts(profile, 'label/x/')
        assert len(selected) == 2
        assert all('\n' in text for text in selected)
        label_slots = [axis for axis in profile.figure.axes
                       if (axis.get_gid() or '').startswith('label/x/')]
        assert len(label_slots) == 2
        assert [axis.texts[0].get_text() for axis in label_slots] == list(selected)
        assert all(axis.get_xlabel() == '' for axis in profile.figure.axes[:2])
    finally:
        plt.close(profile.figure)


def test_profile_ordinary_x_labels_auto_off_keep_axis_owned_literal_text(
        tmp_path, monkeypatch):
    label = 'literal profile X label'
    profile = _profile(
        tmp_path, subplot_samples=['1', '2'], subplot_columns=2,
        subplot_x_axes='show_all', x_axis_label=[label],
        legend_location='none', image_format='png',
        style=_label_style(auto_axis_label_layout=False))

    try:
        assert _profile_slot_texts(profile, 'label/x/') == ()
        assert not [axis for axis in profile.figure.axes
                    if (axis.get_gid() or '').startswith('label/x/')]
        assert [axis.get_xlabel() for axis in profile.figure.axes[:2]] == [
            label, label]
    finally:
        plt.close(profile.figure)


def test_profile_mixed_policy_structuralizes_literal_titles_and_x_labels(
        tmp_path, monkeypatch):
    title = 'literal panel title'
    label = ' '.join(['verylongxlabel'] * 12)
    profile = _profile(
        tmp_path, subplot_samples=['1', '2'], subplot_columns=2,
        subplot_labels=[title, title], subplot_x_axes='show_all',
        x_axis_label=[label], legend_location='none', image_format='png',
        style=_label_style(auto_panel_title_column_gap=False,
                           auto_axis_label_layout=True))

    try:
        data_axes = profile.figure.axes[:2]
        assert [axis.get_title() for axis in data_axes] == ['', '']
        assert _profile_slot_texts(profile, 'cell/') == (title, title)
        title_slots = [axis for axis in profile.figure.axes
                       if (axis.get_gid() or '').endswith('/title')]
        assert [axis.texts[0].get_text() for axis in title_slots] == [title, title]

        selected = _profile_slot_texts(profile, 'label/x/')
        assert len(selected) == 2
        assert all('\n' in text for text in selected)
        label_slots = [axis for axis in profile.figure.axes
                       if (axis.get_gid() or '').startswith('label/x/')]
        assert len(label_slots) == 2
        assert [axis.texts[0].get_text() for axis in label_slots] == list(selected)
        assert all(axis.get_xlabel() == '' for axis in data_axes)
    finally:
        plt.close(profile.figure)


def test_disabled_panel_title_preserves_whitespace(tmp_path):
    title = '  leading   inner  trailing  '
    profile = _profile(
        tmp_path, subplot_samples=['1', '2'], subplot_columns=2,
        subplot_labels=[title, title], legend_location='none',
        style=_label_style(auto_panel_title_column_gap=False))
    try:
        assert _profile_slot_texts(profile, 'cell/') == (title, title)
        slots = [axis for axis in profile.figure.axes
                 if (axis.get_gid() or '').endswith('/title')]
        assert [axis.texts[0].get_text() for axis in slots] == [title, title]
    finally:
        plt.close(profile.figure)


@pytest.mark.parametrize('automatic', (False, True), ids=('literal', 'auto'))
def test_long_facet_and_profile_y_labels_fit_canvas(tmp_path, automatic):
    facet = ' '.join(['oversized facet label'] * 6)
    y_label = ' '.join(['long profile y label'] * 10)
    profile = build_profile_case(
        plot_data(_plot_matrix(groups=('A', 'B'))),
        str(tmp_path / 'vertical-labels.png'),
        subplot_samples=['1,2', '3,4'], subplot_rows=2,
        subplot_group_placement='by_row',
        subplot_labels=[facet, facet], y_axis_label=[y_label],
        subplot_y_axes='outer_merged', subplot_x_axes='outer',
        legend_location='none', image_format='png',
        style=_label_style(auto_facet_label_layout=automatic,
                           auto_axis_label_layout=automatic))
    try:
        assert any(role.startswith('label/row/')
                   for role in profile.solution.rects)
        assert any(role.startswith('label/y/')
                   for role in profile.solution.rects)
        _assert_structural_slots_inside(profile.figure)
    finally:
        plt.close(profile.figure)


def test_profile_mixed_policy_structuralizes_titles_and_keeps_x_labels_local(
        tmp_path, monkeypatch):
    title = ' '.join(['verylongpaneltitle'] * 12)
    label = 'literal profile X label'
    profile = _profile(
        tmp_path, subplot_samples=['1', '2'], subplot_columns=2,
        subplot_labels=[title, title], subplot_x_axes='show_all',
        x_axis_label=[label], legend_location='none', image_format='png',
        style=_label_style(auto_panel_title_column_gap=True,
                           auto_axis_label_layout=False))

    try:
        data_axes = profile.figure.axes[:2]
        selected = _profile_slot_texts(profile, 'cell/')
        assert len(selected) == 2
        assert all('\n' in text for text in selected)
        title_slots = [axis for axis in profile.figure.axes
                       if (axis.get_gid() or '').endswith('/title')]
        assert len(title_slots) == 2
        assert [axis.texts[0].get_text() for axis in title_slots] == list(selected)
        assert all(axis.get_title() == '' for axis in data_axes)

        assert _profile_slot_texts(profile, 'label/x/') == ()
        assert not [axis for axis in profile.figure.axes
                    if (axis.get_gid() or '').startswith('label/x/')]
        assert [axis.get_xlabel() for axis in data_axes] == [label, label]
    finally:
        plt.close(profile.figure)


def test_profile_common_legend_remeasures_against_solved_grid_width(
        tmp_path, monkeypatch):
    common = dict(
        groups=('A', 'B'), subplot_samples=['1', '2'],
        subplot_columns=2, subplot_common_legend=True,
        legend_location='above', image_format='png')
    narrow_path = tmp_path / 'narrow'
    wide_path = tmp_path / 'wide'
    narrow_path.mkdir()
    wide_path.mkdir()
    narrow = _profile(
        narrow_path, real_profile_width=2, **common)

    wide = _profile(
        wide_path, real_profile_width=10, **common)

    try:
        assert _external_legends(narrow)
        assert _external_legends(wide)
        assert _external_legends(narrow)[0]._ncols < 4
        assert _external_legends(wide)[0]._ncols == 4
    finally:
        plt.close(narrow.figure)
        plt.close(wide.figure)


def test_profile_common_legend_second_measurement_uses_solved_grid_width(
        tmp_path, monkeypatch):
    calls = []
    original_measure_legend = profile_plotting.measure_legend

    def record_measurement(*args, **kwargs):
        maximum_width = (args[3] if len(args) > 3 else
                         kwargs.get('maximum_width_points'))
        calls.append(float(maximum_width))
        return original_measure_legend(*args, **kwargs)

    monkeypatch.setattr(profile_plotting, 'measure_legend',
                        record_measurement)
    title = ' '.join(['longtitle'] * 10)
    profile = _profile(
        tmp_path, groups=('A', 'B'), subplot_samples=['1', '2'],
        subplot_columns=2, subplot_labels=[title, title],
        subplot_common_legend=True, legend_location='above',
        real_profile_width=10, image_format='png')

    try:
        assert len(calls) >= 2
        provisional_width = 2 * cm_to_points(10)
        panels = [rect for role, rect in profile.solution.rects.items()
                  if role.endswith('/profile')]
        solved_grid_width = max(rect.x1 for rect in panels) - min(
            rect.x0 for rect in panels)
        assert calls[0] == pytest.approx(provisional_width)
        assert calls[-1] == pytest.approx(solved_grid_width)
        assert calls[-1] > provisional_width
        assert _external_legends(profile)
    finally:
        plt.close(profile.figure)


def test_profile_tick_overlap_mitigation_runs_after_exact_sizing(
        tmp_path, monkeypatch):
    widths = []
    original = profile_plotting.mitigateTickLabelOverlapsForAxes

    def record(axes, **kwargs):
        for axis in axes:
            widths.append(
                axis.get_window_extent(
                    axis.figure.canvas.get_renderer()).width *
                72 / axis.figure.dpi)
        return original(axes, **kwargs)

    monkeypatch.setattr(
        profile_plotting, 'mitigateTickLabelOverlapsForAxes', record)
    _profile(
        tmp_path, subplot_columns=2, legend_location='none',
        real_profile_width=3, real_profile_height=4,
        image_format='png')

    assert widths
    assert widths == pytest.approx(
        [cm_to_points(3)] * len(widths), abs=.1)


@pytest.mark.parametrize('mode,expect_equal', [
    ('common', True), ('per_sample_set', False), ('per_panel', False)])
def test_subplot_y_scale_mode(tmp_path, mode, expect_equal):
    profile = build_profile_case(
        plot_data(_plot_matrix()), str(tmp_path / 'unused.png'),
        subplot_samples=['1,2', '3,4'], subplot_columns=2,
        subplot_y_limits=mode, legend_location='none', image_format='png')
    limits = [axis.get_ylim() for axis in
              profile.figure.axes[:len(profile.plan.panels)]]
    try:
        assert (limits[0] == limits[1]) is expect_equal
        if mode != 'common':
            assert limits[0][0] != limits[1][0]
            assert limits[0][1] != limits[1][1]
    finally:
        plt.close(profile.figure)


def test_subplot_axes_shortcut_supersedes_individual_settings(tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    args = process_args([
        '-m', str(matrix), '-o', str(tmp_path / 'plot.png'),
        '--arrangeSamples', '1,2', '3,4',
        '--xAxisVisibility', 'show_all', '--yAxisVisibility', 'show_all',
        '--axisVisibility', 'outer_merged'])
    assert args.xAxisVisibility == 'outer_merged'
    assert args.yAxisVisibility == 'outer_merged'


def test_subplot_legends_are_local_by_default(tmp_path):
    profile = _profile(tmp_path, subplot_columns=2)
    try:
        axes = profile.figure.axes[:len(profile.plan.panels)]
        assert [text.get_text() for text in axes[0].get_legend().texts] == [
            's1', 's2']
        assert [text.get_text() for text in axes[1].get_legend().texts] == [
            's3', 's4']
    finally:
        plt.close(profile.figure)


@pytest.mark.parametrize('location', ['right', 'above', 'below'])
def test_subplot_external_legends_are_measured_after_sizing(
        tmp_path, monkeypatch, location):
    profile = _profile(
        tmp_path, subplot_columns=2, legend_location=location,
        real_profile_height=4, profile_aspect_ratio=1,
        image_format='png')
    legends = _external_legends(profile)
    assert len(legends) == 2
    assert all(legend.get_in_layout() for legend in legends)
    for axis in profile.figure.axes[:2]:
        position = axis.get_position()
        physical_aspect = (
            position.width * profile.figure.get_figwidth() /
            (position.height * profile.figure.get_figheight()))
        assert physical_aspect == pytest.approx(1, abs=0.02)


@pytest.mark.parametrize('location', ['above', 'below'])
def test_adjacent_subplot_external_legends_do_not_overlap(
        tmp_path, monkeypatch, location):
    hm = _plot_matrix(samples=(
        'short', 'a deliberately long condition',
        'another deliberately long condition', 'shorter'))
    profile = build_profile_case(
        plot_data(hm), str(tmp_path / 'profile.png'),
        subplot_samples=['1,2', '3,4'], subplot_columns=2,
        legend_location=location, real_profile_height=4,
        profile_aspect_ratio=1, image_format='png')

    profile.figure.canvas.draw()
    renderer = profile.figure.canvas.get_renderer()
    legends = _external_legends(profile)
    left = legends[0].get_window_extent(renderer)
    right = legends[1].get_window_extent(renderer)
    assert left.x1 < right.x0


@pytest.mark.parametrize('location', ['above', 'below'])
def test_horizontal_subplot_legend_wraps_before_expanding(
        tmp_path, monkeypatch, location):
    monkeypatch.setattr('tests.helpers.profile_build.save_figure_atomic',
                        lambda *args, **kwargs: None)
    hm = _plot_matrix(samples=(
        'condition alpha', 'condition beta',
        'condition gamma', 'condition delta'))
    profile = build_profile_case(
        plot_data(hm), str(tmp_path / 'profile.png'),
        subplot_samples=['1,2', '3,4'], subplot_columns=2,
        legend_location=location, real_profile_height=4,
        profile_aspect_ratio=1, image_format='png')

    assert all(legend._ncols == 1 for legend in _external_legends(profile))


@pytest.mark.parametrize('location', ['right', 'above', 'below'])
def test_external_legend_expansion_keeps_subplot_decorations_disjoint(
        tmp_path, monkeypatch, location):
    monkeypatch.setattr('tests.helpers.profile_build.save_figure_atomic',
                        lambda *args, **kwargs: None)
    hm = _plot_matrix(samples=(
        'short', 'shorter', 'condition with one exceptionally wide key',
        's4', 's5', 's6'))
    profile = build_profile_case(
        plot_data(hm), str(tmp_path / 'profile.png'),
        subplot_samples=['1,2', '3', '4,5,6'], subplot_columns=3,
        legend_location=location, real_profile_height=4,
        profile_aspect_ratio=1, image_format='png')

    profile.figure.canvas.draw()
    renderer = profile.figure.canvas.get_renderer()
    axes = profile.figure.axes[:3]
    tight = [axis.get_tightbbox(renderer) for axis in axes]
    for left, right in zip(tight, tight[1:]):
        assert left.x1 <= right.x0 + 1


def test_four_columns_with_one_extremely_wide_above_legend(tmp_path,
                                                           monkeypatch):
    hm = _plot_matrix(samples=(
        'short', 'short 2', 'short 3',
        'one exceptionally wide unbreakable legend entry ' * 3))
    profile = build_profile_case(
        plot_data(hm), str(tmp_path / 'wide.png'),
        subplot_samples=['1', '2', '3', '4'], subplot_columns=4,
        legend_location='above', real_profile_width=3,
        real_profile_height=3, image_format='png')

    profile.figure.canvas.draw()
    renderer = profile.figure.canvas.get_renderer()
    data_axes = profile.figure.axes[:4]
    widths = [axis.get_window_extent(renderer).width for axis in data_axes]
    gaps = [data_axes[i + 1].get_window_extent(renderer).x0 -
            data_axes[i].get_window_extent(renderer).x1 for i in range(3)]
    assert max(widths) - min(widths) < 1
    assert max(gaps) - min(gaps) < 1
    assert widths[0] / profile.figure.dpi * 2.54 == pytest.approx(3, abs=0.02)


@pytest.mark.parametrize('location', ['above', 'below'])
def test_two_by_two_external_legend_rows_are_disjoint(tmp_path, monkeypatch,
                                                      location):
    profile = _profile(
        tmp_path, subplot_samples=['1', '2', '3', '4'], subplot_columns=2,
        legend_location=location, real_profile_width=3,
        real_profile_height=3, image_format='png')

    profile.figure.canvas.draw()
    renderer = profile.figure.canvas.get_renderer()
    data_axes = profile.figure.axes[:4]
    top_bottom = min(axis.get_window_extent(renderer).y0
                     for axis in data_axes[:2])
    lower_top = max(axis.get_window_extent(renderer).y1
                    for axis in data_axes[2:])
    for data_axis, legend in zip(data_axes, _external_legends(profile)):
        box = legend.get_window_extent(renderer)
        if data_axis in data_axes[:2]:
            assert box.y0 >= lower_top - 1
        else:
            assert box.y1 <= top_bottom + 1


# From test_render_determinism.py.

RENDER_DETERMINISM_DATA = Path(__file__).parent.parent / 'test_heatmapper'


COLORS = {'colorMap': ['Reds'], 'colorList': None, 'colorNumber': 256,
          'missingDataColor': 'black', 'alpha': 1.0}


def _clustered(name, k, seed=None):
    matrix = Matrix.load(str(RENDER_DETERMINISM_DATA / name), threads=1)
    layout, _ = cluster(matrix, RowLayout.identity(matrix), k,
                        method='kmeans', seed=0 if seed is None else seed,
                        threads=1)
    return tuple(int(size) for size in np.diff(layout.group_bounds))


def test_kmeans_clustering_repeats_exactly():
    """The same matrix and k give the same clusters on every run.

    Region identity is what a spec refers to and what a cache keys on, so
    clustering that reshuffles between runs would make both meaningless.
    """
    assert _clustered('large_matrix.mat.gz', 4) == \
        _clustered('large_matrix.mat.gz', 4)


def test_kmeans_seed_still_selects_a_clustering():
    """Seeding pins the result without collapsing the choice.

    This is what makes the previous test meaningful: on a matrix this size
    different seeds genuinely disagree, so repeatability is the seed's doing
    rather than the data having only one answer.
    """
    # Native k-means reaches the same k=4 optimum from most seeds on
    # this matrix; seeds 10 and 11 find different ones.
    assert len({_clustered('large_matrix.mat.gz', 4, seed)
                for seed in (1, 2, 10, 11)}) > 1


def test_short_heatmap_with_many_groups_is_refused():
    """A block too short for its gaps is an error, not a negative height."""
    matrix = Matrix.load(str(RENDER_DETERMINISM_DATA / 'large_matrix.mat.gz'), threads=1)
    layout, _ = cluster(matrix, RowLayout.identity(matrix), 12,
                        method='kmeans', threads=1)
    data = PlotData(matrix, layout,
                    resolve_labels(layout, header=matrix.header))
    with pytest.raises(ValueError, match='heatmap height'):
        render_heatmap(
            data, colorMapDict=COLORS, whatToShow='heatmap and colorbar',
            image_format='png', dpi=100, heatmapHeight=3.0, threads=1)


def _panel(row, column, width=100.0, height=80.0):
    return grid.CellLayout(
        row, column, Size(width, height), (), width,
        grid.CellInsets(profile=grid.PanelInsets(Insets(), Insets())),
        grid.CellLabels())


def test_profile_grid_refuses_mixed_panel_sizes():
    """A declared panel size is honoured or refused, never quietly enlarged."""
    cells = (_panel(0, 0), _panel(0, 1, width=60.0))
    with pytest.raises(ValueError, match='same data size'):
        grid.solve(grid.GridLayoutSpec(1, 2, cells))


def test_profile_grid_accepts_uniform_panel_sizes():
    """The refusal above does not disturb the uniform case."""
    cells = (_panel(0, 0), _panel(0, 1))
    solution = grid.solve(grid.GridLayoutSpec(1, 2, cells))
    assert {(rect.width, rect.height) for role, rect in solution.rects.items()
            if role.endswith('/profile')} == {(100.0, 80.0)}


def test_grid_topology_ignores_unequal_provisional_positions(
        monkeypatch, tmp_path):
    """Declared cell positions survive shifted provisional axes."""
    original = matrix_figure._provisional_rects

    def shifted(figure, rows, columns):
        rects = list(original(figure, rows, columns))
        for index in range(columns, len(rects)):
            x, y, width, height = rects[index]
            rects[index] = (x + 0.03, y, width, height)
        return tuple(rects)

    monkeypatch.setattr(matrix_figure, '_provisional_rects', shifted)
    solution, _ = _render_profile(
        PROFILE_CASES['profile_right_legends'], tmp_path / 'shifted.png')
    rects = [solution.rects[grid.cell_profile(row, column)]
             for row in range(2) for column in range(2)]
    assert rects[0].x == pytest.approx(rects[2].x)
    assert rects[1].x == pytest.approx(rects[3].x)
    assert rects[0].y == pytest.approx(rects[1].y)
    assert rects[2].y == pytest.approx(rects[3].y)
    assert rects[1].x > rects[0].x


# From test_render_passes.py.

RENDER_PASSES_DATA = Path(__file__).parent.parent / 'test_heatmapper' / 'master_multi.mat.gz'


LABELS = ('A long figure title for measurement', 'Genes and regions',
          'Long signal description', 'long region label\nsecond line')


@pytest.mark.parametrize('tool', (plotHeatmap, plotProfile),
                         ids=('heatmap', 'profile'))
@pytest.mark.parametrize('suffix,expected', (('pdf', 0), ('png', 1)))
def test_export_uses_only_save_render(monkeypatch, tmp_path, tool,
                                      suffix, expected):
    draws = []
    original = FigureCanvasAgg.draw

    def counted(self, *args, **kwargs):
        draws.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FigureCanvasAgg, 'draw', counted)
    tool.main(['-m', str(RENDER_PASSES_DATA), '-o', str(tmp_path / ('plot.' + suffix)),
               '--plotTitle', LABELS[0], '--xAxisLabel', LABELS[1],
               '--yAxisLabel', LABELS[2], '-p', '1'])
    assert len(draws) == expected


@pytest.mark.parametrize('dpi', (100, 300))
def test_canonical_extent_matches_direct_measurement(dpi):
    figure = Figure(dpi=dpi)
    FigureCanvasAgg(figure)
    font = FontProperties(family='DejaVu Sans', size=11)
    measurer = TextMeasurer(figure.canvas.get_renderer(), figure, dpi, {})
    for label in LABELS:
        for rotation in (0, 90, -90):
            expected = measure_matplotlib_text(
                label, figure.canvas.get_renderer(), font, figure, rotation)
            assert measurer.measure(label, font, rotation) == expected


def test_text_extent_cache_reuses_figure_measurements(monkeypatch, tmp_path):
    requests = []
    uncached = []
    caches = []
    keys = set()
    original = TextMeasurer.measure
    original_measure = text_layout.measure_matplotlib_text

    def counted(self, text, font_properties, rotation=0.0):
        result = original(self, text, font_properties, rotation)
        requests.append(text)
        if not any(cache is self.cache for cache in caches):
            caches.append(self.cache)
        keys.update(self.cache)
        return result

    def measured(*args, **kwargs):
        uncached.append(args[0])
        return original_measure(*args, **kwargs)

    monkeypatch.setattr(TextMeasurer, 'measure', counted)
    monkeypatch.setattr(text_layout, 'measure_matplotlib_text', measured)
    plotProfile.main(['-m', str(RENDER_PASSES_DATA), '-o', str(tmp_path / 'profile.pdf'),
                      '--plotTitle', LABELS[0], '--xAxisLabel', LABELS[1],
                      '--yAxisLabel', LABELS[2], '-p', '1'])
    assert requests
    assert len(uncached) <= len(keys)
    assert len(requests) > len(uncached)


@pytest.mark.parametrize('kind', ('heatmap', 'profile'))
@pytest.mark.parametrize('convention', ('automatic', 'literal'))
@pytest.mark.parametrize('dpi', (100, 300))
def test_selected_candidate_size_equals_emitted_rect(
        monkeypatch, tmp_path, kind, convention, dpi):
    literal = {name: False for name in (
        'auto_panel_title_column_gap', 'auto_axis_label_layout',
        'auto_horizontal_colorbar_label_layout', 'auto_facet_label_layout',
        'auto_heatmap_region_label_layout')}
    style, messages = resolve_style(
        {'label_layout': literal if convention == 'literal' else {}})
    assert not messages
    monkeypatch.setattr(baselines, 'DPI', dpi)
    if kind == 'heatmap':
        case = dict(baselines.HEATMAP_CASES['heatmap_labels'], style=style)
    else:
        case = dict(baselines.PROFILE_CASES['profile_basic'],
                    plot_title=LABELS[0], x_axis_label=[LABELS[1]],
                    y_axis_label=[LABELS[2]],
                    style=style)
    render = (baselines._render_heatmap if kind == 'heatmap' else
              baselines._render_profile)
    grid_specs = []
    original_solve = grid.solve

    def capture_grid(spec):
        grid_specs.append(spec)
        return original_solve(spec)

    monkeypatch.setattr(grid, 'solve', capture_grid)
    solution, _ = render(case, tmp_path / f'{kind}_{convention}_{dpi}.png')
    spec = grid_specs[-1]
    candidates_by_role = {
        grid.cell_title(cell.row, cell.column): cell.labels.title_candidates
        for cell in spec.cells if cell.labels.title_candidates}
    candidates_by_role.update({
        grid.row_label(item.owner): item.candidates
        for item in spec.decorations if item.kind == 'row_label'})
    candidates_by_role.update(zip(
        grid.axis_run_roles(spec.runs, spec.cells),
        (run.candidates for run in spec.runs
         if run.axis in ('x', 'y'))))
    if kind == 'heatmap':
        stacks = [run for run in spec.runs if run.axis == 'stack']
        candidates_by_role.update({
            grid.label_role('stack', 0, 0, index): run.candidates
            for index, run in enumerate(stacks)})
        candidates_by_role.update({
            grid.label_role('heatmap_y', cell.row, cell.column):
                cell.labels.y_candidates for cell in spec.cells
            if cell.labels.y_candidates})
        candidates_by_role.update({
            grid.colorbar_title_role(item.scope, item.owner): item.candidates
            for item in spec.decorations
            if item.kind == 'colorbar' and item.candidates})
    checked = 0
    for role, candidates in candidates_by_role.items():
        rect = solution.rects.get(role)
        if rect is None:
            continue
        candidate = candidates[solution.selections.get(role, 0)]
        assert rect.width == candidate.width
        assert rect.height == candidate.height
        checked += 1
    assert checked


# From test_rendering_performance.py.

def test_suspend_axes_images_restores_each_original_visibility():
    figure, axes = plt.subplots(1, 2)
    visible_image = axes[0].imshow([[1, 2], [3, 4]])
    hidden_image = axes[1].imshow([[4, 3], [2, 1]], visible=False)

    states = suspend_axes_images(figure)

    assert not visible_image.get_visible()
    assert not hidden_image.get_visible()

    restore_axes_images(states)

    assert visible_image.get_visible()
    assert not hidden_image.get_visible()
    plt.close(figure)


# From general audit regressions.

def make_matrix(path, values, bounds=None):
    data = np.asarray(values, dtype=np.float32)
    rows, bins = data.shape
    bounds = [0, rows] if bounds is None else bounds
    labels = [f'group{i}' for i in range(len(bounds) - 1)]
    regions = [['chr1', [(i * 10, i * 10 + 5)], f'r{i}', 0, '+', '0'] for i in range(rows)]
    parameters = {'upstream': [0], 'downstream': [0], 'body': [bins], 'unscaled 5 prime': [0], 'unscaled 3 prime': [0], 'ref point': [None], 'bin size': [1], 'sort regions': 'keep', 'sort using': 'mean', 'min threshold': None, 'max threshold': None, 'sample_labels': ['sample'], 'group_labels': labels, 'sample_boundaries': [0, bins], 'group_boundaries': bounds}
    OwnedMatrix.from_compute(parameters, data, regions).save(str(path), compressed=True, threads=1)


def plot_values(kind):
    if kind == 'zero':
        return np.zeros((6, 2))
    if kind == 'constant':
        return np.ones((6, 2))
    if kind == 'sparse':
        values = np.zeros((100, 2))
        values[-1, -1] = 10
        return values
    if kind == 'missing':
        return np.full((6, 2), np.nan)
    raise AssertionError(kind)


def render(tmp_path, values, extra=(), bounds=None, upstream=False):
    source, dest = (tmp_path / 'matrix.gz', tmp_path / 'heatmap.png')
    make_matrix(source, values, bounds)
    args = ['-m', str(source), '-o', str(dest), *extra]
    try:
        if upstream:
            original('plotHeatmap', args)
        else:
            plotHeatmap.main(args)
        assert dest.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    finally:
        plt.close('all')


@pytest.mark.parametrize('kind', ['zero', 'constant', 'sparse', 'missing'])
@pytest.mark.parametrize('explicit_auto', [False, True])
def test_g02_automatic_heatmap_limits(tmp_path, kind, explicit_auto):
    render(tmp_path, plot_values(kind), ['--zMin', 'auto', '--zMax', 'auto'] if explicit_auto else [])


@pytest.mark.parametrize('kind', ['zero', 'constant', 'sparse', 'missing'])
def test_explicit_finite_heatmap_limits_work(tmp_path, kind):
    render(tmp_path, plot_values(kind), ['--zMin', '0', '--zMax', '10'])


@pytest.mark.parametrize('kind', ['zero', 'constant', 'sparse'])
def test_original_handles_degenerate_automatic_limits(tmp_path, kind):
    render(tmp_path, plot_values(kind), upstream=True)


@pytest.mark.parametrize('per_group', [False, True])
def test_programmatic_heatmap_also_omits_empty_groups(tmp_path, per_group):
    source, out = (tmp_path / 'source.gz', tmp_path / 'out.png')
    make_matrix(source, [[1, 2], [3, 4]], [0, 0, 2])
    matrix = Matrix.load(str(source), threads=1)
    data = plot_data(matrix)
    try:
        render_heatmap(data, out, perGroup=per_group, image_format='png', threads=1)
        assert out.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert data.layout.group_bounds == (0, 2)
        assert data.labels.groups == ('group1',)
        assert matrix.header.group_boundaries == (0, 0, 2)
    finally:
        plt.close('all')
