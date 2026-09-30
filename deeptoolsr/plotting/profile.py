"""Backend-independent profile models and shared profile renderers."""

import csv
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from matplotlib import colors as pltcolors
from matplotlib.transforms import ScaledTranslation
import numpy as np

from deeptoolsr.matrix import Block
from deeptoolsr.plotting.ticks import (
    getDistanceUnit, getProfileTicks)
from .geometry import (PROFILE_LINE_WIDTH_POINTS,
                       SAMPLE_LABEL_TO_PANEL_GAP_POINTS)
from deeptoolsr.stats import (
    ProfileStatistics, calculate_profile_batch, calculate_profile_statistics)


def position_sample_title(axis, gap_points=SAMPLE_LABEL_TO_PANEL_GAP_POINTS):
    """Place a panel title by its visible lower edge, not its baseline."""
    if not axis.get_title():
        return
    title = axis.title
    title.set_verticalalignment('bottom')
    title.set_position((title.get_position()[0], 1.0))
    title.set_transform(
        axis.transAxes +
        ScaledTranslation(
            0, gap_points / 72.0,
            axis.figure.dpi_scale_trans))


def legend_identity(label, color):
    """Legend keys collapse only when label and RGBA colour both match."""
    return label, tuple(pltcolors.to_rgba(color))


@dataclass(frozen=True)
class ProfileSeriesSpec:
    block: Block
    label: str
    color: Any


@dataclass(frozen=True)
class PreparedProfileSeries:
    label: str
    color: Any
    statistics: ProfileStatistics
    fill_to_zero: bool = False
    interval_alpha: float = 0.5


@dataclass(frozen=True)
class ProfilePanelElements:
    """Switchable decorations shared by profiles and heatmap summaries."""

    title: bool = True
    x_ticks: bool = True
    x_label: bool = True
    y_ticks: bool = True
    y_label: bool = True
    legend: bool = True


@dataclass(frozen=True)
class ProfilePanelSpec:
    title: str = ''
    series: Sequence[ProfileSeriesSpec] = field(default_factory=tuple)
    elements: ProfilePanelElements = field(default_factory=ProfilePanelElements)
    x_label: str = ''
    y_label: str = ''


def _matplotlib_color(color):
    if isinstance(color, np.ndarray):
        return pltcolors.to_hex(color, keep_alpha=True)
    return color


def prepare_profile_series(series_spec, options, *, threads):
    """Compute a series' statistics once, independently of any Matplotlib axis.

    This is the expensive half of drawing (averaging, confidence intervals,
    bootstrap) and is separated so callers that measure layout before the
    final render, such as the heatmap summary panels, pay that cost only once.
    The numeric statistics never depend on where the panel ends up on the
    figure, so they are computed exactly once here; only the cheap Matplotlib
    draw call (`draw_prepared_profile_series` below) happens against the
    rough-draw axes that later gets measured and repositioned in place.
    """
    return PreparedProfileSeries(
        series_spec.label, series_spec.color,
        calculate_profile_statistics(series_spec.block, options,
                                     threads=threads),
        fill_to_zero=options.plot_type == 'fill')


def prepare_profile_series_batch(series_specs, options, *, threads):
    """Attach labels and colours to one figure-wide numeric batch."""
    specs = list(series_specs)
    statistics = calculate_profile_batch(
        (spec.block for spec in specs), options, threads=threads)
    return [PreparedProfileSeries(spec.label, spec.color, stat,
                                  fill_to_zero=options.plot_type == 'fill')
            for spec, stat in zip(specs, statistics)]


def draw_prepared_profile_series(axis, prepared,
                                 line_width=PROFILE_LINE_WIDTH_POINTS):
    """Draw one already-computed series (centre line plus optional interval).

    This is the single series renderer; ``draw_profile_series`` is the
    convenience wrapper that computes the statistics first. ``line_width`` is a
    fixed physical stroke width in points (the resolved ``profile_line_width``);
    it is never scaled by the font multiplier and does not affect the
    zero/reference or interval fills.
    """
    statistics = prepared.statistics
    color = _matplotlib_color(prepared.color)
    axis.plot(statistics.x, statistics.center, color=color,
              label=prepared.label, alpha=0.9, linewidth=line_width)
    if prepared.fill_to_zero:
        axis.fill_between(statistics.x, statistics.center,
                          facecolor=color, alpha=0.6, edgecolor='none')
    if statistics.lower is not None:
        fill_color = pltcolors.colorConverter.to_rgba(
            color, prepared.interval_alpha)
        axis.fill_between(statistics.x, statistics.lower, statistics.upper,
                          facecolor=fill_color, edgecolor='none')
    if statistics.x.size:
        axis.set_xlim(0, max(statistics.x))
    return statistics


def draw_profile_series(axis, series, options,
                        line_width=PROFILE_LINE_WIDTH_POINTS, *, threads):
    """Compute a series' statistics and draw it in one step."""
    return draw_prepared_profile_series(
        axis, prepare_profile_series(series, options, threads=threads),
        line_width=line_width)


def _apply_panel_decorations(axis, panel):
    """Apply the title, tick, and axis-label switches shared by both panels."""
    elements = panel.elements
    axis.set_title(panel.title if elements.title else '')
    axis.tick_params(axis='x', which='both', bottom=elements.x_ticks,
                     labelbottom=elements.x_ticks)
    axis.tick_params(axis='y', which='both', left=True,
                     labelleft=elements.y_ticks)
    axis.set_xlabel(panel.x_label if elements.x_label else '')
    axis.set_ylabel(panel.y_label if elements.y_label else '')
    if elements.legend:
        axis.legend(frameon=False)


def draw_prepared_profile_panel(axis, panel, prepared_series,
                                line_width=PROFILE_LINE_WIDTH_POINTS):
    """Draw a panel whose statistics were prepared before rendering."""
    statistics = [draw_prepared_profile_series(axis, prepared,
                                               line_width=line_width)
                  for prepared in prepared_series]
    _apply_panel_decorations(axis, panel)
    return statistics


def draw_profile_panel(axis, panel, options,
                       line_width=PROFILE_LINE_WIDTH_POINTS, *, threads):
    """Compute every series' statistics and draw a complete panel."""
    prepared_series = prepare_profile_series_batch(
        panel.series, options, threads=threads)
    return draw_prepared_profile_panel(axis, panel, prepared_series,
                                       line_width=line_width)


def apply_panel_y_limits(axes, y_min, y_max, mode='shared'):
    """Apply shared or independent limits with explicit per-panel overrides."""
    axes = list(axes)
    if not axes:
        return
    y_min = [None] if y_min is None else y_min
    y_max = [None] if y_max is None else y_max
    global_limits = [
        min(axis.get_ylim()[0] for axis in axes),
        max(axis.get_ylim()[1] for axis in axes)]
    for index, axis in enumerate(axes):
        if mode == 'independent':
            limits = list(axis.get_ylim())
        else:
            limits = list(global_limits)
        local_min = y_min[index % len(y_min)]
        local_max = y_max[index % len(y_max)]
        if local_min is not None:
            limits[0] = float(local_min)
        if local_max is not None:
            limits[1] = float(local_max)
        if limits[0] >= limits[1]:
            limits[1] = limits[0] + 1
        axis.set_ylim(limits)


def _tabular_number(value):
    if np.ma.is_masked(value) or not np.isfinite(value):
        return 'nan'
    return '{:.15g}'.format(float(value))


def _tabular_text(value):
    return (str(value).replace('\\', '\\\\')
            .replace('\r', '\\r')
            .replace('\n', '\\n')
            .replace('\t', '\\t'))


def _profile_sample_plan(panel, plan, sample_plans, series_by_key):
    sample = series_by_key[panel.series[0]].sample
    return next(item for item in sample_plans if sample in item.samples)


def write_profile_table(path, plan, prepared, spec):
    """Write the exact displayed centre and optional ribbons as TSV."""
    series_by_key = {item.key: item for item in plan.series}
    maximum_bins = max((len(value.center)
                        for value in prepared.statistics_by_key.values()),
                       default=0)
    fixed_columns = [
        'panel_index', 'panel_label', 'series_index', 'series_label',
        'plot_type', 'component', 'breakpoint_labels', 'breakpoint_bins',
        'breakpoint_unit', 'valid_bin_count']
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(fixed_columns + [
            'bin_{}'.format(index) for index in range(maximum_bins)])
        for panel_index, panel in enumerate(plan.panels):
            sample_plan = _profile_sample_plan(
                panel, plan, prepared.sample_set_plans, series_by_key)
            sample = sample_plan.samples[0]
            breakpoints, labels = getProfileTicks(
                prepared.header_parameters, sample_plan.reference_label,
                sample_plan.start_label, sample_plan.end_label, sample,
                spec.distance_unit, 'axis')
            breakpoint_labels = json.dumps(
                list(labels), ensure_ascii=False, separators=(',', ':'))
            breakpoint_bins = json.dumps(
                [float(value) for value in breakpoints], separators=(',', ':'))
            unit = getDistanceUnit(
                prepared.header_parameters, sample, spec.distance_unit)
            for series_index, key in enumerate(panel.series):
                item = series_by_key[key]
                statistics = prepared.statistics_by_key[key]
                components = [(spec.average_type, statistics.center)]
                if statistics.lower is not None:
                    components.extend([
                        (spec.plot_type + '_low', statistics.lower),
                        (spec.plot_type + '_up', statistics.upper)])
                for component, values in components:
                    valid = len(values)
                    bins = [_tabular_number(value) for value in values]
                    bins.extend(['nan'] * (maximum_bins - valid))
                    writer.writerow([
                        panel_index + 1, _tabular_text(panel.export_label),
                        series_index + 1, _tabular_text(item.label),
                        spec.plot_type, component,
                        breakpoint_labels, breakpoint_bins, unit, valid,
                        *bins])
