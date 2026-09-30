"""Build every matrix figure (profiles, heatmaps or both) on the unified grid."""

from dataclasses import dataclass, field, replace

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from deeptoolsr.prepare import PlotData
from deeptoolsr.plotting.ticks import (
    fit_ticks, alignTickLabelsForRotation, formatDistanceAxisLabel, getDistanceUnit,
    getProfileTicks, mitigateTickLabelOverlapsForAxes)
from . import fonts, grid
from .geometry import (CM_PER_INCH, POINTS_PER_INCH, Insets, Size,
                       cm_to_points,
                       COLORBAR_LABEL_MIN_CLEARANCE_POINTS,
                       HEATMAP_SMALL_GROUP_MAX_ROWS,
                       )
from .heatmap import (
    make_colormaps, resolve_colorbar_position,
    colorbar_mappable, shift_cmap_midpoint, draw_heatmap_image,
    get_plot_ticks, add_minor_x_ticks, style_horizontal_colorbar,
    style_vertical_colorbar, draw_sort_indicator,
    flush_deferred_heatmap_rasters)
from .grid import BelowColorbarEntry
from .legend import LegendEntrySpec, legend_spacing_kwargs, measure_legend
from .matrix_plan import tool_option_spellings
from .profile import (PreparedProfileSeries, ProfilePanelElements,
                      ProfilePanelSpec,
                      legend_identity,
                      draw_prepared_profile_panel,
                      position_sample_title)
from .rendering import draw_without_rendering
from .scene import (MatplotlibMeasureContext, measure_axis,
                    ProvisionalPass, SceneInputs, axis_insets, finish_scene,
                    label_candidates, legend_slot, move_axis,
                    text_slot,
                    visible_tick_inset)
from .series import (apply_grouped_y_limits, contiguous_runs,
                     effective_per_group,
                     geometry_for_sample, has_explicit_assignment, label_runs,
                     recycle, series_slot)
from .text_layout import (TextMeasurer, literal_label_candidate,
                          measure_label_candidates)


@dataclass(frozen=True)
class _LabelStyle:
    """Artist properties kept separate from measured grid labels."""

    font_properties: object
    color: object
    rotation: float


def _legend_lines(style):
    """The most lines a legend label may wrap into (1 keeps it literal)."""
    layout = style.label_layout
    return layout.legend_label_max_lines if layout.auto_legend_label_layout \
        else 1


@dataclass(frozen=True)
class _MeasuredDecorations:
    """Solver decorations and the artists used to draw them."""

    decorations: tuple
    legends: tuple
    row_styles: dict


def _measure_decorations(context, *, row_texts, panel_entries,
                         positions, legend_scope, common_legend,
                         legend_location, font, width, geometry, extra=()):
    """Build row labels and shared external legends for every figure."""
    decorations = list(extra)
    row_styles = {}
    style = context.style
    row_font = fonts.role_font(style, 'panel_title_text')
    for row, text in enumerate(row_texts):
        if not text:
            continue
        font_properties = row_font.copy()
        candidates = label_candidates(
            text, font_properties, -90.0,
            style.label_layout.panel_title_max_lines, context,
            auto=style.label_layout.auto_facet_label_layout)
        decorations.append(grid.Decoration(
            'row_label', 'row', row, 'right', text=text,
            candidates=candidates))
        row_styles[row] = _LabelStyle(font_properties, None, -90.0)
    legends = []
    if legend_location in ('above', 'below', 'right') and panel_entries:
        if common_legend:
            owners = (('figure', 0, tuple(panel_entries)),)
        elif legend_scope in ('row', 'column'):
            count = max(position[0 if legend_scope == 'row' else 1]
                        for position in positions) + 1
            owners = tuple((legend_scope, owner, tuple(
                index for index, position in enumerate(positions)
                if position[0 if legend_scope == 'row' else 1] == owner))
                for owner in range(count))
        else:
            owners = ()
        measure_context = MatplotlibMeasureContext(dpi=context.figure.dpi)
        for scope, owner, indices in owners:
            entries = _legend_entries(_unique_legend_keys(
                key for index in indices
                for key in panel_entries.get(index, ())))
            if not entries:
                continue
            maximum = width
            if scope in ('figure', 'row'):
                maximum *= max((position[1] for position in positions),
                               default=0) + 1
            measured = measure_legend(
                measure_context, entries, legend_location, maximum,
                font_properties=font, geometry=geometry,
                max_lines=_legend_lines(style))
            decorations.append(grid.Decoration(
                'legend', scope, owner, legend_location,
                measured.size, measured.ncols))
            legends.append((scope, owner, entries, measured))
        measure_context.close()
    return _MeasuredDecorations(
        tuple(decorations), tuple(legends), row_styles)


@dataclass
class _GridState:
    """Measurement and solve state of one figure build.

    Drawing facts (plan, spec, artists, grid shape) live on the canvas that
    every grid pass receives beside this state. ``legend_location`` starts
    as the requested location and becomes ``above`` when a ``best`` legend
    does not fit its panel.
    """

    legend_location: str
    panel_entries: dict
    axis_spans: list = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    title_font: object = None
    cell_labels: dict = field(default_factory=dict)
    title_order: list = field(default_factory=list)
    title_styles: dict = field(default_factory=dict)
    row_labels: list = field(default_factory=list)
    row_styles: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)
    run_styles: list = field(default_factory=list)
    common_entries: tuple = ()
    legend_records: list = field(default_factory=list)
    cells: list = field(default_factory=list)
    legend_measured: dict = field(default_factory=dict)
    common_measured: object = None
    grid_spec: object = None
    external: bool = False


def _prepare_axis_padding(canvas, context):
    geometry = canvas.run.style.geometry
    for axis in context.data_axes:
        axis.xaxis.labelpad = geometry.axis_label_to_tick_labels_gap
        axis.yaxis.labelpad = geometry.axis_label_to_tick_labels_gap
        position_sample_title(axis, geometry.column_header_to_panel_gap)
    for axis in canvas.heatmap_axes:
        axis.xaxis.labelpad = geometry.axis_label_to_tick_labels_gap


def _panel_size(canvas, context):
    """Panel size in points: the solved sizes, else the provisional axes'
    measured extent."""
    box = context.data_axes[0].get_window_extent(context.renderer)
    spec = canvas.spec
    width = (cm_to_points(spec.cell_width) if spec.cell_width is not None
             else box.width * context.scale)
    height = (cm_to_points(spec.profile_height)
              if spec.profile_height is not None
              else box.height * context.scale)
    return width, height


def _set_provisional_positions(canvas, state, context):
    figure = context.figure
    state.title_font = (canvas.main_title.get_fontproperties().copy()
                        if canvas.main_title is not None else None)
    if not context.data_axes:
        # Heatmap stacks alone: the grid sizes blocks from the cell width.
        state.width, state.height = cm_to_points(canvas.spec.cell_width), 0.0
        return
    width, height = state.width, state.height = _panel_size(canvas, context)
    figure.set_layout_engine(None)
    figure.set_size_inches(canvas.grid_cols * width / 72.0,
                           canvas.grid_rows * height / 72.0)
    for axis, (row, column) in zip(context.data_axes, canvas.positions):
        axis.set_position((column / canvas.grid_cols,
                           (canvas.grid_rows - 1 - row) / canvas.grid_rows,
                           1.0 / canvas.grid_cols, 1.0 / canvas.grid_rows))
    _reposition_provisional_stacks(canvas)


def _style_data_axes(canvas, context):
    for axes in (context.data_axes, canvas.heatmap_axes):
        mitigateTickLabelOverlapsForAxes(axes, renderer=context.renderer)
        for axis in axes:
            fonts.apply_axis_fonts(axis, canvas.run.style)


def _align_axis_labels(canvas, state, context):
    axes = context.data_axes
    renderer = context.renderer
    gap = canvas.run.style.geometry.axis_label_to_tick_labels_gap
    for column in range(canvas.grid_cols):
        column_axes = [axis for axis, position in zip(axes, canvas.positions)
                       if position[1] == column]
        tick_inset = max((visible_tick_inset(
            axis, renderer, context.figure.dpi, 'y') for axis in column_axes),
            default=0.0)
        for axis in column_axes:
            label = axis.yaxis.label
            if label.get_text():
                label_x = -(tick_inset + gap) / state.width
                axis.yaxis.set_label_coords(label_x, 0.5,
                                            transform=axis.transAxes)
    for row in range(canvas.grid_rows):
        row_axes = [axis for axis, position in zip(axes, canvas.positions)
                    if position[0] == row]
        tick_inset = max((visible_tick_inset(
            axis, renderer, context.figure.dpi, 'x') for axis in row_axes),
            default=0.0)
        for axis in row_axes:
            label = axis.xaxis.label
            if label.get_text():
                label_y = -(tick_inset + gap) / state.height
                axis.xaxis.set_label_coords(0.5, label_y,
                                            transform=axis.transAxes)


def _collect_structural_labels(canvas, state, context):
    style = canvas.run.style
    auto_titles = style.label_layout.auto_panel_title_column_gap
    auto_axis = style.label_layout.auto_axis_label_layout
    for cell, owner in zip(canvas.plan.cells, canvas.positions):
        if not cell.title:
            continue
        font = fonts.role_font(style, 'panel_title_text')
        candidates = label_candidates(
            cell.title, font, 0.0,
            style.label_layout.panel_title_max_lines, context,
            auto=auto_titles)
        state.cell_labels[owner] = grid.CellLabels(
            title_text=cell.title, title_candidates=candidates,
            title_order=len(state.title_order))
        state.title_order.append(owner)
        state.title_styles[owner] = _LabelStyle(font, None, 0.0)
    for axis in context.data_axes:
        axis.set_title('')
    if auto_axis:
        for direction in ('y', 'x'):
            owners = {panel for span in state.axis_spans
                      if span['axis'] == direction
                      for panel in span['panels']}
            # A profile's X label above a stack stays on its axis; the grid
            # reserves it between the profile and the stack.
            local = ({cell.index - 1 for cell in canvas.plan.cells
                      if cell.blocks} if direction == 'x' else set())
            getter = 'get_{}label'.format(direction)
            setter = 'set_{}label'.format(direction)
            for index, axis in enumerate(context.data_axes):
                text = getattr(axis, getter)()
                if text and index not in owners and index not in local:
                    state.axis_spans.append({
                        'axis': direction, 'text': text,
                        'panels': (index,), 'label_to_tick_gap':
                        style.geometry.axis_label_to_tick_labels_gap})
                    owners.add(index)
            for index in owners:
                getattr(context.data_axes[index], setter)('')


def _legend_fits(legend, renderer):
    legend_box = legend.get_window_extent(renderer)
    axis_box = legend.axes.get_window_extent(renderer)
    return (legend_box.x0 >= axis_box.x0 - .5 and
            legend_box.x1 <= axis_box.x1 + .5 and
            legend_box.y0 >= axis_box.y0 - .5 and
            legend_box.y1 <= axis_box.y1 + .5)


def _settle_legend_location(state, context):
    """A 'best' legend that does not fit inside its panel moves above every
    panel instead of spilling over a neighbour."""
    if state.legend_location == 'best':
        legends = [legend for legend in
                   (axis.get_legend() for axis in context.data_axes)
                   if legend is not None]
        if any(not _legend_fits(legend, context.renderer)
               for legend in legends):
            for legend in legends:
                legend.remove()
            state.legend_location = 'above'
    state.external = state.legend_location in ('above', 'below', 'right')


def _unique_legend_keys(keys):
    """The first (handle, entry) key per displayed label and RGBA colour."""
    unique = {}
    for handle, entry in keys:
        unique.setdefault(legend_identity(entry.label, entry.color),
                          (handle, entry))
    return tuple(unique.values())


def _legend_entries(keys):
    return tuple(entry for _, entry in keys)


def _panel_legend_keys(axis):
    """One panel's legend keys, deduplicated once at their source."""
    handles, labels = axis.get_legend_handles_labels()
    return _unique_legend_keys(
        (handle, LegendEntrySpec(label, handle.get_color(),
                                 handle.get_linestyle(),
                                 handle.get_linewidth(),
                                 handle.get_marker()))
        for handle, label in zip(handles, labels)
        if label != '_nolegend_')


def _profile_panel_insets(canvas, state, context, measure_context, index,
                          axis):
    """Measure one profile panel, reserving its own external legend."""
    renderer, scale = context.renderer, context.scale
    width, height = state.width, state.height
    geometry = canvas.run.style.geometry
    insets = axis_insets(axis, renderer, context.figure.dpi,
                         measure_fn=measure_axis)
    entries = (() if (canvas.spec.common_legend or
                      canvas.legend_scope is not None or not state.external)
               else _legend_entries(state.panel_entries.get(index, ())))
    measured = None
    if entries:
        # A panel's legend fits the panel, so equal labels wrap equally.
        measured = measure_legend(
            measure_context, entries, state.legend_location, width,
            font_properties=canvas.legend_font, geometry=geometry,
            max_lines=_legend_lines(canvas.run.style))
    extents = insets
    if measured is not None:
        size = measured.size
        if state.legend_location == 'above':
            overhang = max(0, (size.width - width) / 2)
            extents = Insets(max(insets.left, overhang),
                             max(insets.right, overhang),
                             insets.top +
                             geometry.external_legend_to_content_gap +
                             size.height, insets.bottom)
        elif state.legend_location == 'below':
            overhang = max(0, (size.width - width) / 2)
            extents = Insets(max(insets.left, overhang),
                             max(insets.right, overhang), insets.top,
                             insets.bottom +
                             geometry.external_legend_to_content_gap +
                             size.height)
        else:
            extra = max(0, (size.height - height) / 2)
            extents = Insets(insets.left,
                             insets.right +
                             geometry.right_legend_to_content_gap +
                             size.width,
                             max(insets.top, extra),
                             max(insets.bottom, extra))
    state.legend_measured[index] = measured
    panel = grid.PanelInsets(
        insets, extents,
        (axis.yaxis.label.get_window_extent(renderer).width * scale
         if axis.get_ylabel() else 0.0),
        visible_tick_inset(axis, renderer, context.figure.dpi, 'y'),
        (axis.xaxis.label.get_window_extent(renderer).height * scale
         if axis.get_xlabel() else 0.0),
        None if measured is None else measured.size)
    legend = (None if measured is None else
              grid.Decoration('legend', 'cell', index, state.legend_location,
                              measured.size, measured.ncols))
    flags = dict(
        show_x_tick_labels=any(t.get_visible() for t in axis.get_xticklabels()),
        show_x_label=bool(axis.get_xlabel()),
        show_y_tick_labels=any(t.get_visible() for t in axis.get_yticklabels()),
        show_y_label=bool(axis.get_ylabel()))
    return panel, legend, flags


def _stack_insets(canvas, context, cell):
    """Measure a cell's heatmap blocks and its heatmap Y label."""
    index = cell.index - 1
    axes = tuple(canvas.block_axes[index, rank]
                 for rank in range(len(cell.blocks)))
    blocks = tuple(axis_insets(axis, context.renderer, context.figure.dpi,
                               measure_fn=measure_axis)
                   for axis in axes)
    stack = Insets(*(max(getattr(item, side) for item in blocks)
                     for side in ('left', 'right', 'top', 'bottom')))
    text = canvas.cell_y_labels[index]
    candidates = (_cell_label_candidates(
        text, 'axis_label_text', context,
        canvas.run.style.label_layout.axis_label_max_lines, 90)
        if text else ())
    return grid.PanelInsets(stack, stack), blocks, candidates


def _measure_cells(canvas, state, context):
    """Lay out every cell: its profile panel, its heatmap stack, or both."""
    measure_context = MatplotlibMeasureContext(dpi=context.figure.dpi)
    size = Size(state.width, state.height)
    for cell, (row, column) in zip(canvas.plan.cells, canvas.positions):
        index = cell.index - 1
        axis = canvas.profile_axes.get(index)
        profile, legend, flags = grid.PanelInsets(), None, {}
        if axis is not None:
            profile, legend, flags = _profile_panel_insets(
                canvas, state, context, measure_context, index, axis)
        stack, blocks, y_candidates = ((grid.PanelInsets(), (), ())
                                       if not cell.blocks else
                                       _stack_insets(canvas, context, cell))
        labels = replace(state.cell_labels.get((row, column),
                                               grid.CellLabels()),
                         y_candidates=y_candidates)
        layout = grid.CellLayout(
            row, column, size if axis is not None else None, (), state.width,
            grid.CellInsets(profile, stack, blocks), labels, legend,
            blocks=cell.blocks, **flags)
        if cell.blocks:
            layout = replace(layout, stack=grid.stack_heights(
                layout, canvas.group_sizes,
                cm_to_points(canvas.spec.heatmap_height),
                canvas.run.style.geometry.heatmap_group_gap))
        state.cells.append(layout)
    measure_context.close()


def _measure_labels(canvas, state, context):
    renderer, scale = context.renderer, context.scale
    title_box = (canvas.main_title.get_window_extent(renderer)
                 if canvas.main_title is not None else None)
    title_height = title_box.height * scale if title_box is not None else 0.0
    title_width = title_box.width * scale if title_box is not None else 0.0
    for span in state.axis_spans:
        source_axis = context.data_axes[span['panels'][0]]
        source_label = (source_axis.yaxis.label if span['axis'] == 'y'
                        else source_axis.xaxis.label)
        rotation = source_label.get_rotation()
        font = source_label.get_fontproperties().copy()
        if canvas.run.style.label_layout.auto_axis_label_layout:
            candidates = measure_label_candidates(
                span['text'], renderer, font, context.figure,
                canvas.run.style.label_layout.axis_label_max_lines, rotation,
                measurer=context.measurer)
        else:
            candidates = label_candidates(
                span['text'], font, rotation, 1, context, auto=False)
        state.runs.append(grid.LabelRun(
            span['axis'], tuple(span['panels']), candidates,
            span['label_to_tick_gap']))
        state.run_styles.append(_LabelStyle(
            font, source_label.get_color(), rotation))
    return title_height, title_width


def _stack_decorations(canvas, context):
    """Colorbars and one sort indicator per row of heatmap stacks."""
    if not canvas.spec.show_heatmap:
        return ()
    rows = sorted({cell.row for cell in canvas.plan.cells if cell.blocks})
    indicators = tuple(grid.Decoration('indicator', 'row', row, 'left')
                       for row in rows if canvas.show_indicator)
    return _cell_colorbar_decorations(canvas, context) + indicators


def _measure_grid(canvas, state, context):
    # Structural heatmap X labels leave their axes before stacks are measured.
    heatmap_runs = (_heatmap_x_runs(canvas, context)
                    if canvas.spec.show_heatmap else [])
    _measure_cells(canvas, state, context)
    title_height, title_width = _measure_labels(canvas, state, context)
    x_decoration = 0.0
    if canvas.spec.show_heatmap:
        for run, style in heatmap_runs:
            state.runs.append(run)
            state.run_styles.append(style)
        state.runs.extend(_stack_runs(canvas, context))
        x_decoration = max((measure_axis(axis, context.renderer).bottom *
                            context.scale for axis in canvas.heatmap_axes
                            if axis.xaxis.get_visible()), default=0.0)
    # A series heatmap's colorbar sits beside its own image.
    bars = tuple(
        grid.Decoration('colorbar', 'cell', index, 'right',
                        insets=axis_insets(axis, context.renderer,
                                           context.figure.dpi,
                                           measure_fn=measure_axis))
        for index, axis in canvas.series_colorbar_axes.items())
    measured = _measure_decorations(
        context, row_texts=canvas.row_labels,
        panel_entries=state.panel_entries,
        positions=tuple((cell.row, cell.column) for cell in state.cells),
        legend_scope=canvas.legend_scope,
        common_legend=canvas.spec.common_legend,
        legend_location=state.legend_location,
        font=canvas.legend_font, width=state.width,
        geometry=canvas.run.style.geometry,
        extra=bars + _stack_decorations(canvas, context))
    state.row_labels = [item for item in measured.decorations
                        if item.kind == 'row_label']
    state.row_styles = measured.row_styles
    state.legend_records = [record for record in measured.legends
                            if record[0] != 'figure']
    common = next(((entries, value) for scope, _, entries, value
                   in measured.legends if scope == 'figure'), None)
    state.common_entries, state.common_measured = (
        common if common is not None else ((), None))
    title = (grid.MeasuredLabelSpec((literal_label_candidate(
        canvas.spec.plot_title, Size(title_width, title_height)),))
        if canvas.spec.plot_title else None)
    state.grid_spec = grid.GridLayoutSpec(
        canvas.grid_rows, canvas.grid_cols, tuple(state.cells),
        measured.decorations, tuple(state.runs), title=title,
        x_decoration_height=x_decoration,
        geometry=canvas.run.style.geometry,
        label_layout=canvas.run.style.label_layout)
    return state


def _solve_grid(canvas, state, context):
    solution = grid.solve(state.grid_spec)
    if (canvas.spec.common_legend and state.external and
            state.common_entries and
            state.legend_location in ('above', 'below')):
        rects = [solution.rects[grid.cell_profile(cell.row, cell.column)]
                 for cell in state.grid_spec.cells]
        final_grid_width = max(rect.x1 for rect in rects) - min(
            rect.x0 for rect in rects)
        # The legend's columns and wrapping depend on the solved grid width;
        # solve again only when re-measuring at that width changes it.
        final_context = MatplotlibMeasureContext(dpi=context.figure.dpi)
        measured = measure_legend(
            final_context, state.common_entries, state.legend_location,
            final_grid_width, font_properties=canvas.legend_font,
            geometry=canvas.run.style.geometry,
            max_lines=_legend_lines(canvas.run.style))
        final_context.close()
        if measured == state.common_measured:
            return solution
        state.common_measured = measured
        state.grid_spec = replace(
            state.grid_spec, decorations=tuple(
                replace(item, size=state.common_measured.size,
                        ncols=state.common_measured.ncols)
                if item.kind == 'legend' and item.scope == 'figure'
                else item for item in state.grid_spec.decorations))
        solution = grid.solve(state.grid_spec)
    return solution


def _place_cell_legend(canvas, state, context, solution, index):
    role = grid.legend_role('cell', index)
    rect = solution.rects.get(role)
    if rect is not None:
        legend_slot(context.figure, rect, solution.figure_size,
                    _legend_entries(state.panel_entries[index]),
                    state.legend_measured[index],
                    font_properties=canvas.legend_font,
                    geometry=canvas.run.style.geometry, gid=role)


def _place_cell_titles(canvas, state, context, solution):
    for row, column in state.title_order:
        role = grid.cell_title(row, column)
        rect = solution.rects.get(role)
        if rect is None:
            continue
        title = state.cell_labels[(row, column)]
        style = state.title_styles[(row, column)]
        displayed = title.title_candidates[
            solution.selections.get(role, 0)].text
        text_slot(context.figure, rect, solution.figure_size, displayed,
                  gid=role,
                  rotation=style.rotation,
                  multialignment='center',
                  fontproperties=style.font_properties, color=style.color)


def _place_shared_slots(canvas, state, context, solution):
    for scope, owner, entries, measured in state.legend_records:
        role = grid.legend_role(scope, owner)
        rect = solution.rects.get(role)
        if rect is None:
            continue
        legend_slot(
            context.figure, rect, solution.figure_size, entries, measured,
            font_properties=canvas.legend_font,
            geometry=canvas.run.style.geometry, gid=role)
    for facet in state.row_labels:
        role = grid.row_label(facet.owner)
        rect = solution.rects.get(role)
        if rect is None:
            continue
        style = state.row_styles[facet.owner]
        text = facet.candidates[solution.selections.get(role, 0)].text
        text_slot(context.figure, rect, solution.figure_size, text,
                  gid=role,
                  rotation=style.rotation,
                  fontproperties=style.font_properties, color=style.color)
    roles = grid.axis_run_roles(state.grid_spec.runs,
                                state.grid_spec.cells)
    axis_runs = ((span, style) for span, style in
                 zip(state.runs, state.run_styles) if span.axis in ('x', 'y'))
    for (span, style), role in zip(axis_runs, roles):
        rect = solution.rects.get(role)
        if rect is None:
            continue
        text_slot(context.figure, rect, solution.figure_size,
                  span.candidates[solution.selections.get(role, 0)].text,
                  gid=role,
                  rotation=style.rotation, multialignment='center',
                  fontproperties=style.font_properties, color=style.color)


def _place_figure_slots(canvas, state, context, solution):
    common_role = grid.legend_role('figure', 0)
    if (canvas.spec.common_legend and state.external and
            state.common_entries and common_role in solution.rects):
        legend_slot(
            context.figure, solution.rects[common_role],
            solution.figure_size, state.common_entries,
            state.common_measured, font_properties=canvas.legend_font,
            geometry=canvas.run.style.geometry, gid=common_role)
    if canvas.main_title is not None:
        canvas.main_title.remove()
    if grid.figure_title() in solution.rects:
        text_slot(context.figure, solution.rects[grid.figure_title()],
                  solution.figure_size, canvas.spec.plot_title,
                  gid=grid.figure_title(), fontproperties=state.title_font)


def _place_slots(canvas, state, context, solution):
    for index, axis in canvas.series_colorbar_axes.items():
        role = grid.colorbar_role('cell', index)
        move_axis(axis, solution.rects[role], solution.figure_size)
    _place_cell_titles(canvas, state, context, solution)
    _place_shared_slots(canvas, state, context, solution)
    if canvas.spec.show_heatmap:
        _place_stack_text(canvas, context, state.grid_spec, solution)
    _place_figure_slots(canvas, state, context, solution)


def _finish_grid(canvas, state, axes, measurer):
    """Measure, solve and place one figure's cells on the role-keyed grid."""
    provisional = (
        ProvisionalPass(lambda ctx: _prepare_axis_padding(canvas, ctx)),
        ProvisionalPass(lambda ctx: _set_provisional_positions(
            canvas, state, ctx)),
        ProvisionalPass(lambda ctx: _style_data_axes(canvas, ctx)),
        ProvisionalPass(lambda ctx: _align_axis_labels(canvas, state, ctx)),
        ProvisionalPass(lambda ctx: _collect_structural_labels(
            canvas, state, ctx),
            draw=canvas.run.style.label_layout.auto_axis_label_layout),
        ProvisionalPass(lambda ctx: _settle_legend_location(state, ctx)))
    stacks = canvas.spec.show_heatmap
    inputs = SceneInputs(
        canvas.figure, tuple(axes), canvas.positions,
        provisional,
        lambda ctx: _measure_grid(canvas, state, ctx),
        lambda ctx, measured: _solve_grid(canvas, state, ctx),
        lambda ctx, measured, solved: _place_slots(canvas, state, ctx, solved),
        adjust=((lambda ctx, measured, solved: _retick_solved_colorbars(
            canvas, state, ctx, solved)) if stacks else None),
        data_rects=lambda solved: (
            solved.rects[grid.cell_profile(cell.row, cell.column)]
            for cell in state.grid_spec.cells),
        place_panel=(lambda ctx, measured, solved, index, axis:
                     _place_cell_legend(canvas, state, ctx, solved, index)),
        placement_passes=((
            ProvisionalPass(
                lambda ctx, measured, solved: _place_stack_axes(
                    canvas, solved),
                draw=bool(canvas.colorbar_axes) and
                canvas.colorbar_position not in ('side', 'side_common')),
            ProvisionalPass(
                lambda ctx, measured, solved: _settle_below_colorbars(
                    canvas, ctx), draw=False)) if stacks else ()))
    return finish_scene(inputs, measurer, canvas.run.style)


class _MatrixCanvas:
    """One figure's plan, preparation, artists and derived drawing state."""

    def __init__(self, matrix, layout, labels, plan, spec, prepared, run,
                 extents, raster_renderer=None):
        self.data = PlotData(matrix, layout, labels, run.threads,
                             spec.distance_unit, spec.distance_unit_location)
        self.plan, self.spec, self.prepared = plan, spec, prepared
        self.run, self.extents = run, extents
        self.raster_renderer = raster_renderer
        self.series_by_key = {item.key: item for item in plan.series}
        cells = plan.cells
        self.panel_count = len(cells)
        self.cell_rows = max(cell.row for cell in cells) + 1
        self.cell_columns = max(cell.column for cell in cells) + 1
        rows, cols = self.cell_rows, self.cell_columns
        if (spec.placement not in ('by_row', 'by_column') and
                spec.grid_columns is None and spec.grid_rows is not None):
            rows = min(spec.grid_rows, self.panel_count)
        self.provisional_rows, self.provisional_cols = rows, cols
        if (spec.plot_type == 'heatmap' and effective_per_group(spec)
                and spec.grid_columns is None and spec.grid_rows is None):
            # Each group owns a vertical series-heatmap cell, so the plan's
            # row labels name no drawn row.
            self.grid_rows, self.grid_cols = self.panel_count, 1
            self.positions = tuple((index, 0)
                                   for index in range(self.panel_count))
            self.row_labels = ()
        else:
            self.grid_rows, self.grid_cols = self.cell_rows, self.cell_columns
            self.positions = tuple((cell.row, cell.column) for cell in cells)
            self.row_labels = tuple(plan.row_labels)
        self.legend_font = fonts.role_font(
            run.style, 'legend_label_text')
        # A by_column right legend would span the stacks between its rows;
        # with heatmaps each cell repeats it beside its own profile instead.
        self.legend_scope = (
            'row' if spec.placement == 'by_row' else
            'column' if spec.placement == 'by_column' and not (
                spec.show_heatmap and spec.legend_location == 'right')
            else None)
        self.sort_method = layout.sort.method if layout.sort else None
        self.show_indicator = (
            spec.show_heatmap and spec.sort_indicator == 'auto' and
            self.sort_method in ('ascend', 'descend'))
        self.group_sizes = tuple(np.diff(layout.group_bounds))
        self.profile_axes = {}
        self.series_colorbar_axes = {}
        self.heatmap_axes = []
        self.block_axes = {}
        self.colorbar_axes = []
        self.colorbar_position = None
        if spec.show_heatmap:
            _setup_colors(self)
            profile = spec.profile_height / 2.54 if spec.show_profile else 0
            size = (cols * (spec.cell_width / 2.54 + .5) + 1,
                    rows * (spec.heatmap_height / 2.54 + profile + .5) + 1)
        else:
            size = (cols * spec.cell_width / 2.54,
                    rows * spec.profile_height / 2.54)
        self.figure = Figure(figsize=size, layout=None)
        FigureCanvasAgg(self.figure)
        self.main_title = self.figure.suptitle(
            spec.plot_title,
            fontproperties=fonts.role_font(run.style, 'figure_title_text')
        ) if spec.plot_title else None
        if self.main_title is not None:
            self.main_title.set_in_layout(False)
        self.x_signatures = tuple(_x_axis_signature(self, index)
                                  for index in range(self.panel_count))


def _panel_series(state, panel):
    """Prepared series for one profile panel, whatever lies below it."""
    plan, prepared = state.plan, state.prepared
    result = []
    for key in panel.series:
        item = state.series_by_key[key]
        color = prepared.series_colors[
            series_slot(plan, 0, panel.index, key) - 1]
        result.append(PreparedProfileSeries(
            item.label, color, prepared.statistics_by_key[key],
            fill_to_zero=state.spec.plot_type == 'fill'))
    return result


def _profile_panel_series(state, plot):
    """Return resolved series with zero-based drawing colour positions."""
    panel = state.plan.cells[plot].profile
    return [(state.series_by_key[key].group,
             state.series_by_key[key].sample,
             state.series_by_key[key].label,
             series_slot(state.plan, 0, panel.index, key) - 1)
            for key in panel.series]


def _profile_panel_title(state, plot):
    return state.plan.cells[plot].title


def _profile_panel_plan(state, plot):
    panel = state.plan.cells[plot].profile
    sample = state.series_by_key[panel.series[0]].sample
    return next(plan for plan in state.prepared.sample_set_plans if sample in plan.samples)


def _profile_panel_ticks(state, plot):
    plan = _profile_panel_plan(state, plot)
    sample = plan.samples[0]
    return getProfileTicks(
        state.data.matrix.header.parameters, plan.reference_label, plan.start_label, plan.end_label,
        sample, state.spec.distance_unit, state.spec.distance_unit_location)


def _profile_panel_distance_axis_label(state, plot):
    plan = _profile_panel_plan(state, plot)
    unit = getDistanceUnit(state.data.matrix.header.parameters, plan.samples[0], state.spec.distance_unit)
    return formatDistanceAxisLabel(
        plan.x_axis_label, unit, state.spec.distance_unit_location)


def _merge_parallel_spans(state, spans, axis_name):
    merged = []
    for span in spans:
        if not merged:
            merged.append(span)
            continue
        previous = merged[-1]
        previous_positions = [state.positions[index]
                              for index in previous['panels']]
        positions = [state.positions[index] for index in span['panels']]
        if axis_name == 'y':
            previous_parallel = sorted({col for _, col
                                        in previous_positions})
            current_parallel = sorted({col for _, col in positions})
            # A heatmap stack separates vertically neighbouring profiles.
            adjacent = (not state.spec.show_heatmap and
                        max(row for row, _ in previous_positions) + 1 ==
                        min(row for row, _ in positions))
        else:
            previous_parallel = sorted({row for row, _
                                        in previous_positions})
            current_parallel = sorted({row for row, _ in positions})
            adjacent = (max(col for _, col in previous_positions) + 1 ==
                        min(col for _, col in positions))
        if (previous['text'] == span['text'] and adjacent and
                previous_parallel == current_parallel):
            previous['panels'] += span['panels']
        else:
            merged.append(span)
    return merged


def _apply_subplot_axes(state, axes, y_labels, y_scales):
    """Apply semantic outer-axis ownership after Y scales are resolved.

    Decides, per row/column, which panel actually shows its Y or X tick
    labels and axis label versus deferring to a shared neighbour: an
    inner panel's decorations are hidden (`tick_params(..., labelleft=
    False)` / `labelbottom=False`) whenever its Y scale (``y_scales``)
    and text match a panel already showing them, and adjacent matching
    spans are recorded so the SOLVE stage can later reserve one merged
    outer label instead of one per panel (``outer_merged`` mode).
    """
    by_position = {position: index
                   for index, position in enumerate(state.positions)}
    y_spans = []
    x_spans = []

    # Y ownership is resolved over contiguous compatible panels in each
    # row. Labels may match while tick labels remain necessary because the
    # resolved numerical scales differ.
    for row in range(state.grid_rows):
        indices = [by_position[(row, column)]
                   for column in range(state.grid_cols)
                   if (row, column) in by_position]
        for label, run in contiguous_runs(
                indices, lambda index: y_labels[index]):
            if (state.spec.y_axis_visibility == 'outer_merged' and label):
                y_spans.append({'axis': 'y', 'text': label,
                                'panels': tuple(run),
                                'label_to_tick_gap':
                                state.run.style.geometry.common_axis_label_to_tick_labels_gap})
            for offset, index in enumerate(run):
                axis = axes[index]
                show_label = (state.spec.y_axis_visibility == 'show_all' or
                              offset == 0)
                if state.spec.y_axis_visibility == 'outer_merged':
                    show_label = False
                axis.set_ylabel(label if show_label else '')
                if (state.spec.y_axis_visibility != 'show_all' and offset > 0 and
                        y_scales[index] == y_scales[run[0]]):
                    axis.tick_params(axis='y', labelleft=False)

    # X ownership is resolved vertically in each column. Geometry and all
    # text decorations must match before an inner label can be removed.
    # A heatmap stack below a profile carries that cell's X axis instead.
    for index, cell in enumerate(state.plan.cells):
        if cell.blocks and state.spec.x_axis_visibility != 'show_all':
            axes[index].set_xlabel('')
            axes[index].tick_params(axis='x', labelbottom=False)
    for column in range(state.grid_cols):
        indices = [by_position[(row, column)]
                   for row in range(state.grid_rows)
                   if (row, column) in by_position and
                   not state.plan.cells[by_position[(row, column)]].blocks]
        for _signature, run in contiguous_runs(
                indices, lambda index: state.x_signatures[index]):
            owner = run[-1]
            label = _profile_panel_distance_axis_label(state, owner)
            if state.spec.x_axis_visibility == 'outer_merged' and label:
                x_spans.append({'axis': 'x', 'text': label,
                                'panels': tuple(run),
                                'label_to_tick_gap':
                                state.run.style.geometry.common_axis_label_to_tick_labels_gap})
            for index in run:
                axis = axes[index]
                if state.spec.x_axis_visibility == 'outer_merged':
                    axis.set_xlabel('')
                elif state.spec.x_axis_visibility == 'show_all' or index == owner:
                    axis.set_xlabel(
                        _profile_panel_distance_axis_label(state, index))
                else:
                    axis.set_xlabel('')
                    axis.tick_params(axis='x', labelbottom=False)

    return (_merge_parallel_spans(state, y_spans, 'y') +
            _merge_parallel_spans(state, x_spans, 'x'))


def _draw_internal_legends(state, axes):
    """Draw the legends that live inside panels; return every panel's keys.

    External legends (above, below, right) are drawn only in their solved
    slots, so here they need just their keys.
    """
    keys = {index: _panel_legend_keys(axis)
            for index, axis in enumerate(axes)}
    location = state.spec.legend_location
    if location in ('none', 'above', 'below', 'right'):
        return keys
    owners = (((axes[0], _unique_legend_keys(
        key for panel in keys.values() for key in panel)),)
        if state.spec.common_legend and axes else
        ((axis, keys[index]) for index, axis in enumerate(axes)))
    kwargs = legend_spacing_kwargs(state.run.style.geometry,
                                   state.legend_font)
    for axis, owned in owners:
        if owned:
            axis.legend([handle for handle, _ in owned],
                        [entry.label for _, entry in owned],
                        loc=location.replace('-', ' '), ncol=1,
                        prop=state.legend_font, frameon=False,
                        markerscale=0.5, **kwargs)
    return keys


def _provisional_rects(figure, rows, columns):
    """Positions from the figure's subplot margins and spacing."""
    params = figure.subplotpars
    cell_h = ((params.top - params.bottom) /
              (rows + params.hspace * (rows - 1)))
    cell_w = ((params.right - params.left) /
              (columns + params.wspace * (columns - 1)))
    height, width = cell_h, cell_w
    heights = np.cumsum(np.column_stack((
        [0] + [params.hspace * cell_h] * (rows - 1),
        [height] * rows)).flat)
    widths = np.cumsum(np.column_stack((
        [0] + [params.wspace * cell_w] * (columns - 1),
        [width] * columns)).flat)
    tops, bottoms = (params.top - heights).reshape((-1, 2)).T
    lefts, rights = (params.left + widths).reshape((-1, 2)).T
    return tuple(
        (lefts[column], bottoms[row],
         rights[column] - lefts[column],
         tops[row] - bottoms[row])
        for row in range(rows) for column in range(columns))


def _draw_profile(state, plot, provisional):
    """Draw one cell's profile panel; the same code with or without a stack.

    Only the content differs by kind: lines, or a series heatmap image with
    its colorbar.
    """
    row, col = state.positions[plot]
    panel = state.plan.cells[plot].profile
    image = panel.kind == 'series_heatmap'
    rect, bar_rect = (_series_provisional_rects(provisional) if image
                      else (provisional, None))
    ax = state.figure.add_axes(rect)
    ax.set_gid(grid.cell_profile(row, col))
    title = _profile_panel_title(state, plot)
    ax.set_title(title)
    if image:
        _draw_series_image(state, plot, ax, bar_rect)
    else:
        draw_prepared_profile_panel(
            ax,
            ProfilePanelSpec(
                title=title,
                elements=ProfilePanelElements(
                    title=True, x_ticks=True, x_label=False,
                    y_ticks=True, y_label=False, legend=False)),
            _panel_series(state, panel),
            line_width=state.run.style.drawing.profile_line_width)
    total_width = state.data.layout.block(
        state.data.matrix, *_profile_panel_series(state, plot)[-1][:2]
    ).shape[1]
    xticks, xtickslabel = _profile_panel_ticks(state, plot)
    tick_positions = fit_ticks(xticks, total_width - 1, total_width)
    ax.axes.set_xticks(tick_positions)
    # Optional dashed guide lines rising from every tick mark, mirroring
    # plotHeatmap's --linesAtTickMarks. The profile X axis has no 0.5
    # pixel offset, so the tick positions are used directly.
    if state.spec.lines_at_tick_marks:
        for x in tick_positions:
            ax.axvline(x=x, color='black', linewidth=0.5, dashes=(3, 2))
    ax.axes.set_xticklabels(xtickslabel, rotation=state.spec.label_rotation)
    if state.spec.minor_tick_marks != 'none' and len(tick_positions) > 1:
        add_minor_x_ticks(ax, state.spec.minor_tick_marks)
    ax.axes.set_xlabel(_profile_panel_distance_axis_label(state, plot))
    # align the first and last label
    # such that they don't fall off
    # the heatmap sides
    alignTickLabelsForRotation(ax, state.spec.label_rotation)
    return ax


def _resolve_profile_axes(state, axes):
    """Share Y limits, then decide which panels show which axis labels.

    A series heatmap's Y axis names its image rows: it has no axis label,
    and its row names are its scale.
    """
    if state.spec.plot_type == 'heatmap':
        return _apply_subplot_axes(
            state, axes, ('',) * len(axes),
            [_series_row_labels(state, index) for index in range(len(axes))])
    panel_set_ids = [_profile_panel_plan(state, index).index
                     for index in range(len(axes))]
    panel_y_labels = [state.prepared.sample_set_plans[index].y_axis_label
                      for index in panel_set_ids]
    resolved_limits = apply_grouped_y_limits(
        axes, panel_set_ids, panel_y_labels,
        state.prepared.y_min, state.prepared.y_max, state.spec.y_axis_limits)
    return _apply_subplot_axes(state, axes, panel_y_labels, resolved_limits)


def _draw_matrix(state):
    """Draw profiles and heatmap stacks provisionally, then solve the grid."""
    axes = []
    if state.spec.show_profile:
        provisional = _provisional_rects(
            state.figure, state.grid_rows, state.grid_cols)
        for plot, (row, col) in enumerate(state.positions):
            axes.append(_draw_profile(
                state, plot, provisional[row * state.grid_cols + col]))
        state.profile_axes = dict(enumerate(axes))
    if state.spec.show_heatmap:
        _draw_stacks(state)
    panel_entries = _draw_internal_legends(state, axes)
    merged_axis_labels = _resolve_profile_axes(state, axes) if axes else ()
    measurer = TextMeasurer(
        state.figure.canvas.get_renderer(), state.figure, state.figure.dpi,
        state.extents)
    grid_state = _GridState(
        state.spec.legend_location, panel_entries,
        [dict(span) for span in merged_axis_labels])
    return _finish_grid(state, grid_state, axes, measurer)


def _series_provisional_rects(cell):
    """Split the provisional cell using the historical 0.92/0.08 widths."""
    x, y, width, height = cell
    unit = width / 1.025
    return ((x, y, unit * .92, height),
            (x + unit * .945, y, unit * .08, height))


def _series_row_labels(state, plot):
    """A series heatmap's image rows, named like line-profile series: by
    what varies between them (group, sample or both)."""
    return tuple(state.series_by_key[key].label
                 for key in state.plan.cells[plot].profile.series)


def _series_panel_values(state, plot):
    """A series heatmap's image rows and its sample set's Y range."""
    cell = state.plan.cells[plot]
    y_min, y_max = state.prepared.y_min, state.prepared.y_max
    series = [state.prepared.statistics_by_key[key].center
              for key in cell.profile.series]
    return (np.vstack(series), y_min[cell.sample_set % len(y_min)],
            y_max[cell.sample_set % len(y_max)])


def _draw_series_image(state, plot, axis, bar_rect):
    """Draw one series heatmap image, its colorbar and its row names."""
    panel = state.plan.cells[plot].profile
    values, lower, upper = _series_panel_values(state, plot)
    image = axis.imshow(
        values, interpolation='nearest', aspect='auto', vmin=lower,
        vmax=upper,
        cmap=state.prepared.series_colors[series_slot(
            state.plan, 0, panel.index) - 1])
    bar = state.figure.add_axes(bar_rect)
    bar.set_gid(grid.colorbar_role('cell', plot))
    state.figure.colorbar(image, cax=bar)
    state.series_colorbar_axes[plot] = bar
    axis.tick_params(axis='y', which='both', left=False, right=False)
    labels = _series_row_labels(state, plot)
    ymin, ymax = axis.get_ylim()
    positions, distance = np.linspace(
        ymin, ymax, len(labels), retstep=True, endpoint=False)
    axis.set_yticks([value + float(distance) / 2 for value in positions])
    axis.set_yticklabels(labels[::-1])
    axis.set_ylim([ymin, ymax])


def build_matrix_figure(matrix, layout, labels, plan, spec, prepared, run,
                        extents, raster_renderer=None):
    """Build any matrix figure on the role-keyed grid."""
    canvas = _MatrixCanvas(matrix, layout, labels, plan, spec, prepared, run,
                           extents, raster_renderer)
    return canvas.figure, _draw_matrix(canvas)


# Heatmap stacks: blocks, colorbars and sort indicators.


def _setup_colors(state):
    spec, prepared = state.spec, state.prepared
    explicit = (bool(prepared.color_options['colorList']) and
                has_explicit_assignment(spec.color_list))
    state.cmap = make_colormaps(prepared.color_options, explicit)
    state.colorbar_position = resolve_colorbar_position(
        spec, state.cmap, prepared)
    state.group_count = len(state.data.labels.groups)
    state.sample_count = len(state.data.labels.samples)


def _heatmap_distance_axis_label(state, sample):
    plan = state.prepared.plan_by_sample[sample]
    unit = getDistanceUnit(
        state.data.matrix.header.parameters, sample,
        state.data.distance_unit)
    return formatDistanceAxisLabel(
        plan.x_axis_label, unit, state.data.distance_unit_location)


def _cell_provisional(state, cell, part, block=0):
    """Give artists distinct scratch positions before the measured solve."""
    rows, columns = state.cell_rows, state.cell_columns
    width = (cm_to_points(state.spec.cell_width) /
             (state.figure.get_figwidth() * POINTS_PER_INCH) / .94)
    height = .88 / rows
    column_gap = max(0.0, (.88 - columns * width) / max(columns - 1, 1))
    x = .06 + cell.column * (width + column_gap)
    y = .06 + (rows - cell.row - 1) * height
    if part == 'profile':
        return x, y + height * .64, width * .94, height * .28
    if part == 'bar':
        return x, y + height * .02, width * .94, height * .055
    if part == 'figure_bar':
        return .92, .16, .025, .68
    if part == 'indicator':
        return .015, y + height * .12, .02, height * .46
    heights = grid.stack_heights(
        cell, state.group_sizes, cm_to_points(state.spec.heatmap_height),
        state.run.style.geometry.heatmap_group_gap)
    total = sum(heights) + (len(heights) - 1) * \
        state.run.style.geometry.heatmap_group_gap
    above = sum(heights[:block]) + block * \
        state.run.style.geometry.heatmap_group_gap
    fraction = heights[block] / total
    stack_height = .50 if state.spec.show_profile else .75
    return (x, y + height * (.12 + stack_height *
            (1 - (above + heights[block]) / total)),
            width * .94, height * stack_height * fraction)


def _x_axis_signature(state, index):
    """What a cell's bottom X axis shows; neighbours merge axes when equal.

    The axis belongs to the cell's lowest element: its heatmap stack, else
    its profile.
    """
    cell, prepared = state.plan.cells[index], state.prepared
    if cell.blocks:
        sample = cell.samples[-1]
        ticks, labels = get_plot_ticks(
            state.data, sample, prepared.reference_point_labels,
            prepared.start_labels, prepared.end_labels)
        text = _heatmap_distance_axis_label(state, sample)
        geometry = geometry_for_sample(
            state.data.matrix.header.sample_boundaries,
            state.data.matrix.header.parameters, sample)
    else:
        geometry = _profile_panel_plan(state, index).geometry
        ticks, labels = _profile_panel_ticks(state, index)
        text = _profile_panel_distance_axis_label(state, index)
    return (geometry.compatibility_key(), text, tuple(ticks), tuple(labels))


def _cell_x_owners(state):
    if state.spec.x_axis_visibility == 'show_all':
        return {cell.index - 1 for cell in state.plan.cells}
    owners = set()
    for column in range(state.cell_columns):
        cells = sorted((cell for cell in state.plan.cells
                        if cell.column == column), key=lambda cell: cell.row)
        for _, run in contiguous_runs(
                cells, lambda cell: state.x_signatures[cell.index - 1]):
            owners.add(run[-1].index - 1)
    return owners


def _draw_cell_block(state, cell, rank, owners):
    spec, prepared = state.spec, state.prepared
    block_id = cell.blocks[rank]
    group, sample = block_id.group, block_id.sample
    block = state.data.layout.block(state.data.matrix, group, sample)
    axis = state.figure.add_axes(
        _cell_provisional(state, cell, 'block', rank))
    axis.set_gid(grid.cell_block(cell.row, cell.column, rank))
    if not spec.box_around_heatmaps:
        for side in ('top', 'right', 'bottom', 'left'):
            axis.spines[side].set_visible(False)
    rows, columns = block.shape
    interpolation = (('bilinear' if rows >= HEATMAP_SMALL_GROUP_MAX_ROWS
                      else 'nearest') if spec.interpolation_method == 'auto'
                     else spec.interpolation_method)
    color = cell.scale % len(state.cmap)
    lower = cell.scale % len(prepared.z_min)
    upper = cell.scale % len(prepared.z_max)
    middle = (None if prepared.z_mid is None else
              cell.scale % len(prepared.z_mid))
    image_cmap = state.cmap[color]
    if middle is not None and prepared.z_mid[middle] is not None:
        image_cmap = shift_cmap_midpoint(
            image_cmap, prepared.z_min[lower], prepared.z_mid[middle],
            prepared.z_max[upper])
    image = draw_heatmap_image(
        axis, block, image_cmap, prepared.z_min[lower],
        prepared.z_max[upper], spec.alpha, interpolation, rows, columns,
        spec.dpi, deferred=state.deferred_rasters,
        raster=state.run.raster, threads=state.run.threads,
        group_digest=state.data.layout.group_digest(group), sample=sample)
    if prepared.regions_length_in_bins[sample] is not None:
        x_lim, y_lim = axis.get_xlim(), axis.get_ylim()
        positions = prepared.regions_length_in_bins[sample][group]
        axis.plot(positions, np.arange(len(positions)), '--',
                  color='black', linewidth=.5, dashes=(3, 2))
        axis.set_xlim(x_lim)
        axis.set_ylim(y_lim)
    axis.set_yticks([])
    axis.get_xaxis().set_visible(False)
    axis.set_xlabel(_heatmap_distance_axis_label(state, sample))
    if spec.lines_at_tick_marks:
        ticks, _ = get_plot_ticks(
            state.data, sample, prepared.reference_point_labels,
            prepared.start_labels, prepared.end_labels)
        for value in fit_ticks([value + .5 for value in ticks], columns,
                               columns):
            axis.axvline(x=value, color='black', linewidth=.5,
                         dashes=(3, 2))
    if rank == len(cell.blocks) - 1 and cell.index - 1 in owners:
        _cell_bottom_ticks(state, axis, sample, columns)
    return axis, image


def _cell_bottom_ticks(state, axis, sample, columns):
    spec, prepared = state.spec, state.prepared
    axis.get_xaxis().set_visible(True)
    ticks, labels = get_plot_ticks(
        state.data, sample, prepared.reference_point_labels,
        prepared.start_labels, prepared.end_labels)
    axis.set_xticks(fit_ticks([value + .5 for value in ticks], columns,
                              columns))
    axis.set_xticklabels(labels, size=8, rotation=spec.label_rotation,
                         rotation_mode='anchor')
    if spec.minor_tick_marks != 'none' and len(axis.get_xticks()) > 1:
        add_minor_x_ticks(axis, spec.minor_tick_marks)
    axis.set_xlabel(_heatmap_distance_axis_label(state, sample))
    alignTickLabelsForRotation(axis, spec.label_rotation)
    axis.get_xaxis().set_tick_params(
        which='both', top=False, direction='out')


def _draw_cell_blocks(state):
    state.block_axes = {}
    state.block_images = {}
    state.heatmap_axes = []
    state.deferred_rasters = []
    owners = _cell_x_owners(state)
    for cell in state.plan.cells:
        for rank in range(len(cell.blocks)):
            axis, image = _draw_cell_block(state, cell, rank, owners)
            state.block_axes[cell.index - 1, rank] = axis
            state.block_images[cell.index - 1, rank] = image
            state.heatmap_axes.append(axis)


def _cell_color_identity(state, cell):
    index = cell.scale
    prepared = state.prepared
    color = index % len(state.cmap)
    low = prepared.z_min[index % len(prepared.z_min)]
    high = prepared.z_max[index % len(prepared.z_max)]
    mid = (None if prepared.z_mid is None else
           prepared.z_mid[index % len(prepared.z_mid)])
    sampled = tuple(tuple(float(value) for value in state.cmap[color](point))
                    for point in (0.0, 0.5, 1.0))
    return sampled, low, high, mid


def _cell_bar_groups(state):
    cells = state.plan.cells
    identities = tuple(_cell_color_identity(state, cell) for cell in cells)
    if state.colorbar_position == 'below':
        if state.spec.x_axis_visibility != 'outer_merged':
            return tuple((index, index) for index in range(len(cells)))
        groups = []
        start = 0
        for index in range(1, len(cells) + 1):
            if (index == len(cells) or cells[index].row != cells[start].row or
                    identities[index] != identities[start]):
                groups.append((start, index - 1))
                start = index
        return tuple(groups)
    groups = []
    seen = set()
    for index, identity in enumerate(identities):
        key = ((cells[index].row, identity)
               if state.colorbar_position == 'side' else identity)
        if key not in seen:
            seen.add(key)
            groups.append((index, index))
    return tuple(groups)


def _draw_cell_colorbars(state):
    state.colorbar_axes = []
    state.colorbar_cbars = []
    state.colorbar_groups = _cell_bar_groups(state)
    state.colorbar_records = []
    if not state.spec.show_heatmap:
        return
    row_counts = {}
    for index, (start, _) in enumerate(state.colorbar_groups):
        cell = state.plan.cells[start]
        _, low, high, mid = _cell_color_identity(state, cell)
        color = cell.scale % len(state.cmap)
        below = state.colorbar_position not in ('side', 'side_common')
        kind = 'bar' if below else 'figure_bar'
        axis = state.figure.add_axes(_cell_provisional(state, cell, kind))
        if state.colorbar_position == 'below':
            scope, owner, ordinal = 'cell', start, 0
        elif state.colorbar_position == 'side':
            scope, owner = 'row', cell.row
            ordinal = row_counts.get(owner, 0)
            row_counts[owner] = ordinal + 1
        else:
            scope, owner, ordinal = 'figure', index, 0
        role = grid.colorbar_role(scope, owner, ordinal)
        axis.set_gid(role)
        bar = state.figure.colorbar(
            colorbar_mappable(state.cmap, color, low, high, mid), cax=axis,
            orientation='horizontal' if below else 'vertical')
        if below:
            style_horizontal_colorbar(
                bar, low, high, state.spec.cell_width, state.run.style)
        else:
            style_vertical_colorbar(
                bar, low, high, state.spec.heatmap_height, state.run.style)
        state.colorbar_axes.append(axis)
        state.colorbar_cbars.append(bar)
        state.colorbar_records.append((start, scope, owner, ordinal, role))


def _draw_cell_indicator(state):
    state.indicator_axes = {}
    if not state.show_indicator:
        return
    for row in range(state.cell_rows):
        cell = next((cell for cell in state.plan.cells
                     if cell.row == row and cell.blocks), None)
        if cell is None:
            continue
        axis = state.figure.add_axes(_cell_provisional(
            state, cell, 'indicator'))
        axis.set_gid(grid.indicator_role(row))
        axis.set_axis_off()
        draw_sort_indicator(axis, state.sort_method, 1.0)
        state.indicator_axes[row] = axis


def _heatmap_y_texts(state):
    """Heatmap Y labels, shown once per run of equal labels in a row."""
    requested = tuple(
        state.prepared.resolved_heatmap_y_labels[cell.sample_set]
        for cell in state.plan.cells)
    labels = [''] * len(requested)
    for row in range(state.cell_rows):
        cells = tuple(cell for cell in state.plan.cells if cell.row == row)
        for name, run in contiguous_runs(
                cells, lambda cell: requested[cell.index - 1]):
            owners = (run if state.spec.y_axis_visibility == 'show_all'
                      else run[:1])
            for owner in owners:
                labels[owner.index - 1] = name
    return tuple(labels)


def _reposition_provisional_stacks(state):
    """Keep provisional blocks at the cell width after a canvas resize."""
    for cell in state.plan.cells:
        for rank in range(len(cell.blocks)):
            state.block_axes[cell.index - 1, rank].set_position(
                _cell_provisional(state, cell, 'block', rank))


def _draw_stacks(state):
    """Draw every heatmap block, colorbar and sort indicator provisionally."""
    _draw_cell_blocks(state)
    _draw_cell_colorbars(state)
    _draw_cell_indicator(state)
    (state.raster_renderer or flush_deferred_heatmap_rasters)(
        state.deferred_rasters, state.run.threads)
    state.cell_y_labels = _heatmap_y_texts(state)


def _cell_label_candidates(text, role, context, maximum, rotation=0):
    font = fonts.role_font(context.style, role)
    return label_candidates(
        text, font, rotation, maximum, context,
        auto=context.style.label_layout.auto_axis_label_layout)


def _heatmap_x_runs(state, context):
    """Structural X labels under heatmap stacks, with their text styles."""
    automatic = state.run.style.label_layout.auto_axis_label_layout
    merged = state.spec.x_axis_visibility == 'outer_merged'
    if not (automatic or merged):
        return []
    runs = []
    last_key = None
    for column in range(state.cell_columns):
        # Only the previous column's run may continue: a column without one
        # breaks the shared bottom boundary.
        previous_key, last_key = last_key, None
        cells = sorted((cell for cell in state.plan.cells
                        if cell.column == column and cell.blocks),
                       key=lambda item: item.row)
        for _, members in contiguous_runs(
                cells, lambda item: state.x_signatures[item.index - 1]):
            source = state.block_axes[members[-1].index - 1,
                                      len(members[-1].blocks) - 1]
            text = source.get_xlabel()
            if not text:
                continue
            last_key = state.x_signatures[members[0].index - 1]
            indices = tuple(item.index - 1 for item in members)
            for member in members:
                state.block_axes[member.index - 1,
                                 len(member.blocks) - 1].set_xlabel('')
            if merged and state.cell_rows == 1 and previous_key == last_key:
                # One row shares a bottom boundary across columns.
                previous, style = runs[-1]
                runs[-1] = (replace(previous, panel_indices=(
                    *previous.panel_indices, *indices)), style)
                continue
            font = source.xaxis.label.get_fontproperties().copy()
            rotation = source.xaxis.label.get_rotation()
            runs.append((grid.LabelRun(
                'x', indices,
                label_candidates(
                    text, font, rotation,
                    state.run.style.label_layout.axis_label_max_lines,
                    context, auto=automatic),
                (state.run.style.geometry.common_axis_label_to_tick_labels_gap
                 if merged else
                 state.run.style.geometry.axis_label_to_tick_labels_gap)),
                _LabelStyle(font, source.xaxis.label.get_color(), rotation)))
    return runs


def _stack_runs(state, context):
    """Region labels beside the stacks; equal neighbours share one label."""
    runs = []
    for run in label_runs(state.plan.cells, 'stack'):
        for block, text in enumerate(run.labels):
            if text:
                runs.append(grid.LabelRun(
                    'stack', tuple(index - 1 for index in run.cells),
                    _cell_label_candidates(
                        text, 'region_label_text', context,
                        state.run.style.label_layout.heatmap_region_label_max_lines,
                        -90 if state.spec.region_label_location == 'right'
                        else 90), block_index=block,
                    location=state.spec.region_label_location))
    return runs


def _scale_label(state, cell):
    """Name a cell's colour scale: its group set or its sample set."""
    if effective_per_group(state.spec):
        return ', '.join(state.data.labels.groups[group]
                         for group in cell.groups)
    return state.prepared.sample_set_plans[cell.sample_set].subplot_label


def _cell_colorbar_decorations(state, context):
    if not state.spec.show_heatmap:
        return ()
    result = []
    spellings = tool_option_spellings(
        state.spec.tool, state.spec.invoked_spellings)
    titles = recycle(
        state.spec.colorbar_labels, len(state.colorbar_groups),
        option=spellings['colorbarLabels'], domain='colorbar', text=True)
    for index, (start, end) in enumerate(state.colorbar_groups):
        _, scope, owner, ordinal, _ = state.colorbar_records[index]
        axis = state.colorbar_axes[index]
        insets = axis_insets(axis, context.renderer, context.figure.dpi)
        location = ('right' if state.colorbar_position in
                    ('side', 'side_common') else 'below')
        title = titles[index]
        if title == 'auto':
            title = ',\n'.join(dict.fromkeys(
                _scale_label(state, state.plan.cells[position])
                for position in range(start, end + 1)))
        candidates = (_cell_label_candidates(
            title, 'colorbar_title_text', context,
            state.run.style.label_layout.horizontal_colorbar_label_max_lines)
            if title and title != 'none' else ())
        entry = (BelowColorbarEntry(
            max(0.0, insets.left), max(0.0, insets.right),
            max(0.0, insets.bottom), 0.0, 0.0)
            if state.colorbar_position == 'below_common' else None)
        result.append(grid.Decoration(
            'colorbar', scope, owner,
            location, candidates=candidates,
            covered_cells=(start, end) if scope == 'cell' else None,
            insets=insets, tick_label_width=insets.right,
            below_entry=entry, ordinal=ordinal))
    return tuple(result)


def _place_stack_axes(state, solved):
    size = solved.figure_size
    for cell in state.plan.cells:
        for rank in range(len(cell.blocks)):
            move_axis(state.block_axes[cell.index - 1, rank],
                      solved.rects[grid.cell_block(cell.row, cell.column,
                                                   rank)], size)
    for index, axis in enumerate(state.colorbar_axes):
        start, _, _, _, role = state.colorbar_records[index]
        move_axis(axis, solved.rects[role], size)
        if state.colorbar_position in ('side', 'side_common'):
            _, low, high, _ = _cell_color_identity(
                state, state.plan.cells[start])
            rect = solved.rects[role]
            style_vertical_colorbar(
                state.colorbar_cbars[index], low, high,
                rect.height / POINTS_PER_INCH * CM_PER_INCH,
                state.run.style)
    for row, axis in state.indicator_axes.items():
        move_axis(axis, solved.rects[grid.indicator_role(row)], size)


def _settle_below_colorbars(state, context):
    if state.colorbar_position not in ('below', 'below_common'):
        return
    rows = {}
    for axis in state.colorbar_axes:
        y = round(axis.get_window_extent(context.renderer).y0)
        rows.setdefault(y, []).append(axis)
    for axes in rows.values():
        settle_horizontal_colorbar_row(
            axes, context.figure,
            clearance_points=state.run.style.geometry.
            colorbar_label_min_clearance)
        for axis in axes:
            fonts.apply_colorbar_tick_fonts(axis, state.run.style)


def _place_colorbar_titles(state, context, measured, solved):
    for decoration in measured.decorations:
        if decoration.kind == 'colorbar' and decoration.candidates:
            role = grid.colorbar_title_role(
                decoration.scope, decoration.owner, decoration.ordinal)
            if role in solved.rects:
                candidate = decoration.candidates[
                    solved.selections.get(role, 0)]
                text_slot(context.figure, solved.rects[role],
                          solved.figure_size, candidate.text, gid=role,
                          fontproperties=fonts.role_font(
                              state.run.style, 'colorbar_title_text'))


def _place_cell_stack_labels(state, context, measured, solved):
    for row in range(measured.rows):
        for ordinal, (_, runs) in enumerate(
                grid.stack_run_groups(measured, row)):
            for block, run in enumerate(runs):
                role = grid.label_role('stack', row, ordinal, block)
                if role not in solved.rects:
                    continue
                candidate = run.candidates[solved.selections.get(role, 0)]
                text_slot(context.figure, solved.rects[role],
                          solved.figure_size, candidate.text, gid=role,
                          rotation=-90 if run.location == 'right' else 90,
                          fontproperties=fonts.role_font(
                              state.run.style, 'region_label_text'))


def _place_cell_y_labels(state, context, measured, solved):
    for cell in measured.cells:
        role = grid.label_role('heatmap_y', cell.row, cell.column)
        if role in solved.rects:
            candidate = cell.labels.y_candidates[
                solved.selections.get(role, 0)]
            text_slot(context.figure, solved.rects[role],
                      solved.figure_size, candidate.text,
                      gid=role, rotation=90,
                      fontproperties=fonts.role_font(
                          state.run.style, 'axis_label_text'))


def _place_stack_text(state, context, measured, solved):
    _place_colorbar_titles(state, context, measured, solved)
    _place_cell_stack_labels(state, context, measured, solved)
    _place_cell_y_labels(state, context, measured, solved)


def _retick_solved_colorbars(state, scene, context, solved):
    """Pack side bars using tick labels chosen at their solved height."""
    if state.colorbar_position not in ('side', 'side_common') or \
            not state.colorbar_axes:
        return solved
    current = scene.grid_spec
    for _ in range(4):
        replacements = []
        for index, (start, _, _, _, role) in enumerate(
                state.colorbar_records):
            rect = solved.rects[role]
            _, low, high, _ = _cell_color_identity(
                state, state.plan.cells[start])
            style_vertical_colorbar(
                state.colorbar_cbars[index], low, high,
                rect.height / POINTS_PER_INCH * CM_PER_INCH,
                state.run.style)
        draw_without_rendering(context.figure)
        context.renderer = context.figure.canvas.get_renderer()
        context.measurer.renderer = context.renderer
        bar_index = 0
        for decoration in current.decorations:
            if decoration.kind != 'colorbar' or decoration.location != 'right':
                replacements.append(decoration)
                continue
            axis = state.colorbar_axes[bar_index]
            bar_index += 1
            insets = axis_insets(
                axis, context.renderer, context.figure.dpi)
            replacements.append(replace(
                decoration, insets=insets,
                tick_label_width=insets.right))
        updated = replace(current, decorations=tuple(replacements))
        if updated.decorations == current.decorations:
            return solved
        current = scene.grid_spec = updated
        solved = grid.solve(current)
    raise RuntimeError('right colorbar ticks did not settle after packing')


def _drop_colorbar_end_tick(axis, which):
    """Remove the leftmost (``which==0``) or rightmost (``-1``) tick+label."""
    positions = list(axis.get_xticks())
    texts = [label.get_text() for label in axis.get_xticklabels()]
    if len(positions) < 2:
        return False
    del positions[which]
    del texts[which]
    axis.set_xticks(positions)
    axis.set_xticklabels(texts)
    return True


def _horizontal_colorbar_cells(axes, renderer):
    cells = []
    for axis in sorted(axes, key=lambda a: a.get_window_extent(renderer).x0):
        labels = [label for label in axis.get_xticklabels()
                  if label.get_text()]
        if labels:
            cells.append({
                'axis': axis, 'bar': axis.get_window_extent(renderer),
                'left': labels[0], 'right': labels[-1],
                'left_extent': labels[0].get_window_extent(renderer),
                'right_extent': labels[-1].get_window_extent(renderer)})
    return cells


def _settle_colorbar_edges(cells, figure_width, clearance):
    changed = False
    first = cells[0]
    if first['left_extent'].x0 < clearance:
        if first['left'].get_horizontalalignment() != 'left':
            first['left'].set_horizontalalignment('left')
        else:
            _drop_colorbar_end_tick(first['axis'], 0)
        changed = True
    last = cells[-1]
    if last['right_extent'].x1 > figure_width - clearance:
        if last['right'].get_horizontalalignment() != 'right':
            last['right'].set_horizontalalignment('right')
        else:
            _drop_colorbar_end_tick(last['axis'], -1)
        changed = True
    return changed


def _settle_colorbar_pair(left_cell, right_cell, clearance):
    if (left_cell['right_extent'].x1 + clearance
            <= right_cell['left_extent'].x0):
        return False
    left_protrusion = left_cell['right_extent'].x1 - left_cell['bar'].x1
    right_protrusion = right_cell['bar'].x0 - right_cell['left_extent'].x0
    left_tuckable = left_cell['right'].get_horizontalalignment() != 'right'
    right_tuckable = right_cell['left'].get_horizontalalignment() != 'left'
    if left_tuckable and (left_protrusion >= right_protrusion
                          or not right_tuckable):
        left_cell['right'].set_horizontalalignment('right')
    elif right_tuckable:
        right_cell['left'].set_horizontalalignment('left')
    elif len(left_cell['axis'].get_xticks()) >= \
            len(right_cell['axis'].get_xticks()):
        _drop_colorbar_end_tick(left_cell['axis'], -1)
    else:
        _drop_colorbar_end_tick(right_cell['axis'], 0)
    return True


def settle_horizontal_colorbar_row(
        axes, figure, clearance_points=COLORBAR_LABEL_MIN_CLEARANCE_POINTS):
    """Centre/tuck end ticks until adjacent bars and figure edges clear."""
    axes = [axis for axis in axes if axis.get_xticks() is not None]
    if not axes:
        return
    clearance = clearance_points * figure.dpi / 72.0
    for axis in axes:
        for label in axis.get_xticklabels():
            label.set_horizontalalignment('center')
    budget = 2 + sum(len(axis.get_xticks()) for axis in axes)
    for _ in range(budget):
        draw_without_rendering(figure)
        cells = _horizontal_colorbar_cells(
            axes, figure.canvas.get_renderer())
        if not cells:
            return
        changed = _settle_colorbar_edges(
            cells, figure.bbox.width, clearance)
        for left, right in zip(cells, cells[1:]):
            changed = _settle_colorbar_pair(left, right, clearance) or changed
        if not changed:
            return


# Heatmap measurement and placement on the unified grid.
