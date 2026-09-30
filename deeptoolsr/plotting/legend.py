"""Figure-independent legend specifications and renderer measurement."""

from dataclasses import dataclass, replace
from typing import Optional

from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

from .geometry import DEFAULT_GEOMETRY, Size, bbox_to_size_points
from .text_layout import generate_label_candidates, measure_matplotlib_text


@dataclass(frozen=True)
class LegendEntrySpec:
    label: str
    color: object
    linestyle: str = '-'
    linewidth: float = 1.5
    marker: Optional[str] = None


@dataclass(frozen=True)
class MeasuredLegend:
    size: Size
    ncols: int
    labels: tuple = ()  # wrapped entry labels; empty keeps the entries' own


def make_legend_handles(entries):
    return [Line2D([], [], label=entry.label, color=entry.color,
                   linestyle=entry.linestyle, linewidth=entry.linewidth,
                   marker=entry.marker) for entry in entries]


def legend_spacing_kwargs(geometry=DEFAULT_GEOMETRY,
                          font_properties=None):
    """Return Matplotlib legend spacing arguments for point-based geometry.

    Matplotlib expresses these four values as fractions of the legend label
    font size. The persistent style stores physical points, so convert at the
    adapter boundary and keep spacing independent of ``font_multiplier``.
    """
    font = font_properties or FontProperties()
    font_size = font.get_size_in_points()
    return {
        'borderpad': geometry.legend_border_padding / font_size,
        'handletextpad': geometry.legend_handle_to_text_gap / font_size,
        'labelspacing': geometry.legend_entry_row_gap / font_size,
        'columnspacing': geometry.legend_column_gap / font_size,
    }


def _measure(context, entries, ncols, font_properties=None,
             geometry=DEFAULT_GEOMETRY):
    axis = context.measure_axis
    legend = axis.legend(handles=make_legend_handles(entries),
                         labels=[entry.label for entry in entries],
                         ncols=ncols, frameon=False, loc='center',
                         prop=font_properties,
                         **legend_spacing_kwargs(geometry, font_properties))
    size = bbox_to_size_points(
        legend.get_window_extent(context.renderer), context.dpi)
    legend.remove()
    return size


def _text_measure(context, font_properties):
    font = font_properties or FontProperties()
    return lambda text: measure_matplotlib_text(
        text, context.renderer, font, context.figure)


def _wrapped_labels(context, entries, budget, font_properties, max_lines):
    """Break each label into the fewest lines that fit ``budget`` points."""
    measure = _text_measure(context, font_properties)
    labels = []
    for entry in entries:
        # Legends break only between words; the narrowest wrap may overflow.
        candidates = [item for item in generate_label_candidates(
            entry.label, measure, max_lines=max_lines)
            if not item.break_penalty]
        fitting = [item for item in candidates if item.width <= budget]
        chosen = (min(fitting, key=lambda item: (
            item.line_count, item.balance_penalty + item.orphan_penalty,
            item.width)) if fitting else
            min(candidates, key=lambda item: item.width))
        labels.append(chosen.text)
    return tuple(labels)


def measure_legend(measure_context, entries, location,
                   maximum_width_points=None, font_properties=None,
                   geometry=DEFAULT_GEOMETRY, max_lines=1):
    """Measure the widest-fitting column count; wrap labels past one column.

    Above and below legends use as many columns as fit the width.  When even
    one column is too wide, labels wrap into at most ``max_lines`` lines.
    """
    entries = tuple(entries)
    if not entries:
        return MeasuredLegend(Size(0.0, 0.0), 0)
    columns = [1]
    if location in ('above', 'below'):
        columns = range(len(entries), 0, -1)
    last = None
    for ncols in columns:
        size = _measure(
            measure_context, entries, ncols, font_properties, geometry)
        last = MeasuredLegend(size, ncols)
        if maximum_width_points is None or size.width <= maximum_width_points:
            return last
    if max_lines <= 1:
        return last
    measure = _text_measure(measure_context, font_properties)
    widest = max(measure(entry.label).width for entry in entries)
    budget = maximum_width_points - (last.size.width - widest)
    labels = _wrapped_labels(measure_context, entries, budget,
                             font_properties, max_lines)
    wrapped = tuple(replace(entry, label=label)
                    for entry, label in zip(entries, labels))
    return MeasuredLegend(_measure(measure_context, wrapped, 1,
                                   font_properties, geometry), 1, labels)


def draw_legend_in_axes(legend_axis, entries, measured, font_properties=None,
                        geometry=DEFAULT_GEOMETRY):
    legend_axis.set_axis_off()
    if not entries:
        return None
    legend = legend_axis.legend(
        handles=make_legend_handles(entries),
        labels=list(measured.labels or (entry.label for entry in entries)),
        ncols=measured.ncols,
        frameon=False, loc='center', borderaxespad=0,
        prop=font_properties,
        **legend_spacing_kwargs(geometry, font_properties))
    return legend
