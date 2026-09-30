"""Renderer-aware measure, solve, and placement shared by plot scenes."""

from dataclasses import dataclass
from typing import Callable

from .geometry import Insets
from .legend import draw_legend_in_axes
from .rendering import (draw_without_rendering, restore_axes_images,
                        suspend_axes_images)
from .text_layout import literal_label_candidate, measure_label_candidates


@dataclass(frozen=True)
class ProvisionalPass:
    apply: Callable
    draw: bool | Callable = True


@dataclass(frozen=True)
class SceneInputs:
    figure: object
    data_axes: tuple
    topology: tuple
    provisional: tuple[ProvisionalPass, ...]
    measure: Callable
    solve: Callable
    place: Callable
    adjust: Callable | None = None
    place_panel: Callable | None = None
    placement_passes: tuple[ProvisionalPass, ...] = ()
    data_rects: Callable | None = None


@dataclass
class SceneContext:
    figure: object
    data_axes: tuple
    topology: tuple
    measurer: object
    style: object
    renderer: object

    @property
    def scale(self):
        return 72.0 / self.figure.dpi


def figure_fraction(rect, size):
    """Map a point-based rect into Figure.add_axes coordinates."""
    return (rect.x / size.width, rect.y / size.height,
            rect.width / size.width, rect.height / size.height)


def move_axis(axis, rect, size):
    axis.set_position(figure_fraction(rect, size))


def slot_axis(figure, rect, size, *, gid=None, axis_off=True):
    axis = figure.add_axes(figure_fraction(rect, size))
    if gid is not None:
        axis.set_gid(gid)
    if axis_off:
        axis.set_axis_off()
    return axis


def text_slot(figure, rect, size, text, *, gid=None, x=.5, y=.5,
              ha='center', va='center', axis_off=True, **kwargs):
    axis = slot_axis(figure, rect, size, gid=gid, axis_off=axis_off)
    axis.text(x, y, text, ha=ha, va=va, **kwargs)
    return axis


def legend_slot(figure, rect, size, entries, measured, *, font_properties,
                geometry, gid=None):
    axis = slot_axis(figure, rect, size, gid=gid, axis_off=False)
    legend = draw_legend_in_axes(axis, entries, measured,
                                 font_properties=font_properties,
                                 geometry=geometry)
    return axis, legend


class MatplotlibMeasureContext:
    """An Agg canvas used only for stable point measurements."""

    def __init__(self, dpi=100):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        self.dpi = dpi
        self.figure = Figure(figsize=(4, 3), dpi=dpi, layout=None)
        FigureCanvasAgg(self.figure)
        self.measure_axis = self.figure.add_axes((0, 0, 1, 1))
        self.measure_axis.set_axis_off()

    @property
    def renderer(self):
        return self.figure.canvas.get_renderer()

    def close(self):
        self.figure.clear()


def measure_axis(axis, renderer):
    """MEASURE stage: ask a real renderer how far each decoration extends
    beyond the bare data rectangle of one already-drawn Axes.

    Requires a prior draw-disabled layout pass so tick labels, axis
    labels, titles and legends have real pixel bounding boxes to query. The
    result is an `Insets` (points on each of the 4 sides) that the grid or
    heatmap solver treats as opaque input.
    """
    data = axis.get_window_extent(renderer)
    # All artists can affect horizontal containment.  Only artists belonging
    # to the horizontal axis (plus titles/legends) participate in vertical
    # flow.  A Y tick label centred on the top or bottom limit may extend a
    # few points beyond the data rectangle, but it lives beside the panel and
    # must not push an above legend away or enlarge the gap between rows.
    artists = [axis.title]
    vertical_flow_artists = [axis.title]
    if axis.xaxis.get_visible():
        x_artists = [axis.xaxis.label, axis.xaxis.get_offset_text()]
        x_artists.extend(axis.get_xticklabels(minor=False))
        x_artists.extend(axis.get_xticklabels(minor=True))
        for tick in axis.xaxis.get_major_ticks() + axis.xaxis.get_minor_ticks():
            x_artists.extend((tick.tick1line, tick.tick2line))
        artists.extend(x_artists)
        vertical_flow_artists.extend(x_artists)
    if axis.yaxis.get_visible():
        artists.extend((axis.yaxis.label, axis.yaxis.get_offset_text()))
        artists.extend(axis.get_yticklabels(minor=False))
        artists.extend(axis.get_yticklabels(minor=True))
        for tick in axis.yaxis.get_major_ticks() + axis.yaxis.get_minor_ticks():
            artists.extend((tick.tick1line, tick.tick2line))
    legend = axis.get_legend()
    if legend is not None:
        artists.append(legend)
        vertical_flow_artists.append(legend)
    boxes = []
    vertical_boxes = []
    for artist in artists:
        if artist is None or not artist.get_visible():
            continue
        if hasattr(artist, 'get_text') and not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.width > 0 and box.height > 0:
            boxes.append(box)
            if artist in vertical_flow_artists:
                vertical_boxes.append(box)
    if not boxes:
        return Insets()
    return Insets(
        left=max([0.0] + [data.x0 - box.x0 for box in boxes]),
        right=max([0.0] + [box.x1 - data.x1 for box in boxes]),
        top=max([0.0] + [box.y1 - data.y1 for box in vertical_boxes]),
        bottom=max([0.0] + [data.y0 - box.y0 for box in vertical_boxes]))


def axis_insets(axis, renderer, dpi, *, measure_fn=measure_axis):
    raw = measure_fn(axis, renderer)
    scale = 72.0 / dpi
    return Insets(raw.left * scale, raw.right * scale,
                  raw.top * scale, raw.bottom * scale)


def visible_tick_inset(axis, renderer, dpi, direction):
    box = axis.get_window_extent(renderer)
    labels = (axis.get_yticklabels() if direction == 'y'
              else axis.get_xticklabels())
    visible = [label for label in labels
               if label.get_visible() and label.get_text()]
    if not visible:
        return 0.0
    edge = (box.x0 if direction == 'y' else box.y0)
    return (edge - min(label.get_window_extent(renderer).x0
                       if direction == 'y' else
                       label.get_window_extent(renderer).y0
                       for label in visible)) * 72.0 / dpi


def label_candidates(text, font, rotation, maximum_lines, context, *, auto):
    if auto:
        return measure_label_candidates(
            text, context.renderer, font, context.figure, maximum_lines,
            rotation, measurer=context.measurer)
    size = context.measurer.measure(text, font, rotation)
    return (literal_label_candidate(text, size),)


def finish_scene(inputs, measurer, style):
    """Settle provisional artists, measure, solve, and place; never save."""
    figure = inputs.figure
    context = SceneContext(figure, inputs.data_axes, inputs.topology,
                           measurer, style, figure.canvas.get_renderer())
    images = suspend_axes_images(figure)
    try:
        for provisional in inputs.provisional:
            provisional.apply(context)
            if provisional.draw:
                draw_without_rendering(figure)
                context.renderer = figure.canvas.get_renderer()
                measurer.renderer = context.renderer
        measured = inputs.measure(context)
        solution = inputs.solve(context, measured)
        if inputs.adjust is not None:
            solution = inputs.adjust(context, measured, solution)
        figure.set_layout_engine(None)
        figure.set_size_inches(solution.figure_size.width / 72.0,
                               solution.figure_size.height / 72.0)
        if inputs.data_axes:
            rects = (inputs.data_rects(solution)
                     if inputs.data_rects is not None else
                     (panel.data_rect for panel in solution.panels))
            for index, (axis, rect) in enumerate(zip(inputs.data_axes, rects)):
                move_axis(axis, rect, solution.figure_size)
                if inputs.place_panel is not None:
                    inputs.place_panel(context, measured, solution,
                                       index, axis)
        for placement in inputs.placement_passes:
            placement.apply(context, measured, solution)
            draw = (placement.draw(context, measured, solution)
                    if callable(placement.draw) else placement.draw)
            if draw:
                draw_without_rendering(figure)
                context.renderer = figure.canvas.get_renderer()
                measurer.renderer = context.renderer
        inputs.place(context, measured, solution)
        return solution
    finally:
        restore_axes_images(images)
