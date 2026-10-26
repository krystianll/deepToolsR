"""Describe plot commands without rendering them."""

import json
import math
import sys

from deeptoolsr import options as run_options
from deeptoolsr.cli_errors import use_utf8_stdio
from deeptoolsr.matrix import (
    SourceKey, read_header)
from deeptoolsr.parserCommon import per_target_options
from deeptoolsr.plotting import (
    DEFAULT_HEATMAP_COLORMAP, DEFAULT_SERIES_COLORMAP)
from deeptoolsr.plotting.colorbars import plan_colorbars
from deeptoolsr.plotting.matrix_spec import project_matrix_spec
from deeptoolsr.plotting.matrix_plan import (
    resolve_matrix_plan, tool_option_spellings)
from deeptoolsr.plotting.series import (
    SAMPLE_SET_TEXTS, UNKNOWN_COUNT, y_limit_keys)
from deeptoolsr.plotting.sizes import SIZE_OPTIONS, size_defaults
from deeptoolsr.prepare import (
    DataError, OptionError, PlotData, _common_spec, collecting_problems,
    command_labels, group_numbers, option_error, parse_command, predict_layout, warning_text)


TOOLS = ('plotHeatmapR', 'plotProfileR', 'plotMatrixR')


def _geometry(geometry):
    return {'mode': geometry.mode,
            'reference_point': geometry.reference_point,
            'upstream': geometry.upstream, 'downstream': geometry.downstream,
            'body': geometry.body,
            'unscaled_5_prime': geometry.unscaled_5_prime,
            'unscaled_3_prime': geometry.unscaled_3_prime,
            'bin_size': geometry.bin_size, 'bin_count': geometry.bin_count}


def _sample_set(plan, y_scale):
    values = {'subplot_title': plan.subplot_label,
              'x_axis_label': plan.x_axis_label,
              'y_axis_label': plan.y_axis_label,
              'heatmap_y_axis_label': plan.heatmap_y_axis_label,
              'reference_label': plan.reference_label,
              'start_label': plan.start_label, 'end_label': plan.end_label}
    return {
        'index': plan.index + 1,
        'samples': [sample + 1 for sample in plan.samples],
        'groups': (None if plan.groups is None
                   else [group + 1 for group in plan.groups]),
        'geometry': (None if plan.geometry_clash
                     else _geometry(plan.geometry)),
        'y_scale': y_scale,
        'texts': {name: {'auto': plan.automatic[name],
                         'value': values[name]}
                  for name in SAMPLE_SET_TEXTS}}


def _colorbar(index, bar, position, limits):
    scale = bar.scale
    set_index = bar.sample_sets[0]
    return {
        'index': index + 1, 'position': position,
        'row': None if bar.row is None else bar.row + 1,
        'cells': [cell + 1 for cell in bar.cells],
        'sample_sets': [item + 1 for item in bar.sample_sets],
        'scale': {'color_map': scale.color_map,
                  'color_list': (None if scale.color_list is None
                                 else list(scale.color_list)),
                  'z_min': scale.z_min, 'z_max': scale.z_max,
                  'z_mid': scale.z_mid},
        'z_range': (None if limits is None else {
            'auto': list(limits[0]),
            'value': [limits[1][set_index % len(limits[1])],
                      limits[2][set_index % len(limits[2])]]}),
        'title': {'auto': None, 'value': bar.title,
                  'auto_text': bar.auto_title}}


def _series_defaults(plan):
    """Hex colours of the plot's default series palette, loaded on first use.

    Only a series colour domain needs them, so only it imports Matplotlib.
    """
    palette = []

    def colour(position):
        if not palette:
            from matplotlib.colors import to_hex
            from deeptoolsr.plotting.prepared import default_series_colors
            palette.extend(to_hex(value)
                           for value in default_series_colors(plan))
        return palette[position - 1]
    return colour


def _slot(slot, kind, titles, series_labels, series_default):
    """One colour slot, with the name it shows and its automatic colour.

    A series slot is named by its legend label; a colour-map slot by its
    panel (series heatmap) or its first heatmap cell. A series slot's
    automatic colour is ``series_default(position)``, a hex string.
    """
    if kind == 'series':
        label = series_labels[slot.series[0]]
        automatic = series_default(slot.position)
    elif slot.panels:
        label = titles['panel'][slot.panels[0]]
        automatic = DEFAULT_SERIES_COLORMAP
    else:
        label = titles['cell'][slot.cells[0]]
        automatic = DEFAULT_HEATMAP_COLORMAP
    return {'position': slot.position, 'series': list(slot.series),
            'panels': list(slot.panels), 'cells': list(slot.cells),
            'label': label, 'automatic': automatic}


def _size_defaults(args, tool, spec):
    spellings = tool_option_spellings(tool)
    found = size_defaults(
        {dest: getattr(args, dest, None) for dest in SIZE_OPTIONS},
        show_profile=spec.show_profile, show_heatmap=spec.show_heatmap)
    return [{'option': spellings[dest], 'value': value,
             'derived': (None if not sources else 'from ' + ', '.join(
                 spellings[source] for source in sources))}
            for dest, (value, sources) in found.items() if dest in spellings]


def _pair(limits):
    """``[low, high]`` as JSON numbers, or ``None`` when not finite."""
    if limits is None or not all(map(math.isfinite, limits)):
        return None
    return [float(value) for value in limits]


def _y_limits(plan, sample_plans, spec, numbers):
    """One record per group of panels sharing profile Y limits.

    ``numbers`` is ``(shared, automatic, drawn, data)`` from
    ``_drawn_y_limits``, or ``None`` when values were not read.
    """
    if not spec.show_profile or spec.plot_type == 'heatmap':
        return None
    cells, set_ids, keys = _limit_panels(plan, sample_plans, spec)
    panels = [cell.profile.index for cell in cells]
    groups = []
    for key in dict.fromkeys(keys):
        members = [index for index, other in enumerate(keys) if other == key]
        if numbers is None:
            data = auto = None
        else:
            ranges = [numbers[3][index] for index in members]
            data = (min(low for low, _ in ranges),
                    max(high for _, high in ranges))
            auto = numbers[0][key]
        groups.append({
            'index': len(groups) + 1,
            'sample_sets': sorted({set_ids[index] + 1 for index in members}),
            'panels': [panels[index] for index in members],
            'data_range': _pair(data), 'auto': _pair(auto),
            'per_panel': [
                {'panel': panels[index], 'sample_set': set_ids[index] + 1,
                 'auto': None if numbers is None else _pair(numbers[1][index]),
                 'value': None if numbers is None else _pair(numbers[2][index])}
                for index in members]})
    return groups


def _limit_panels(plan, sample_plans, spec):
    """Cells with profiles in drawing order, their sample sets and keys."""
    cells = [cell for cell in plan.cells if cell.profile]
    set_ids = [sample_plans[cell.sample_set].index for cell in cells]
    keys = y_limit_keys(set_ids, [sample_plans[index].y_axis_label
                                  for index in set_ids], spec.y_axis_limits)
    return cells, set_ids, keys


def _drawn_y_limits(plan, sample_plans, spec, scales, statistics):
    """The plot's Y limits, from its own drawing of the series off-screen.

    The automatic limits are Matplotlib's autoscaling of the drawn data on
    each panel's scale, so they are read from the same drawing, never
    rendered. Returns ``grouped_y_limits``' result and each panel's data
    range.
    """
    from matplotlib.figure import Figure
    from deeptoolsr.plotting.profile import (
        PreparedProfileSeries, draw_profile_data)
    from deeptoolsr.plotting.series import grouped_y_limits

    cells, set_ids, keys = _limit_panels(plan, sample_plans, spec)
    figure = Figure()
    axes = []
    for cell, set_id in zip(cells, set_ids):
        axis = figure.add_axes((0, 0, 1, 1))
        draw_profile_data(
            axis, [PreparedProfileSeries(
                '', 'black', statistics[key],
                fill_to_zero=spec.plot_type == 'fill')
                for key in cell.profile.series],
            scales['yScale'][set_id], spec.y_pseudolog_sigma)
        axes.append(axis)
    numbers = grouped_y_limits(
        axes, keys, [scales['yScale'][index] for index in set_ids], set_ids,
        scales['yMin'], scales['yMax'])
    return (*numbers, [tuple(axis.dataLim.intervaly) for axis in axes])


def _pseudocount(spec, statistics, automatic, key):
    """A geometric series' automatic and effective pseudocount."""
    if statistics is None or spec.average_type != 'geom_mean':
        return None
    auto = automatic[key].pseudocount
    return {'auto': auto,
            'value': spec.pseudocount if spec.pseudocount >= 0 else auto}


def _record(tool, layout, numbers, labels, plan, sample_plans, y_scales,
            colorbars, warnings, problems, limits, sizes, predicted,
            profile):
    bounds = layout.group_bounds
    titles = {'panel': {panel.index: panel.title for panel in plan.panels},
              'cell': {cell.index: cell.title for cell in plan.cells}}
    series_labels = {item.key: item.label for item in plan.series}
    series_default = _series_defaults(plan)
    spec, statistics, automatic, y_limits = profile
    return {
        'protocol': 2,
        'tool': tool,
        'values_read': limits is not False,
        'groups': [
            {'index': index + 1, 'original': original, 'key': key,
             'label': label,
             'regions': None if predicted else bounds[index + 1] - bounds[index]}
            for index, (original, key, label) in enumerate(
                zip(numbers, labels.group_keys, labels.groups))],
        'samples': [
            {'index': index + 1, 'label': label}
            for index, label in enumerate(labels.samples)],
        'panels': [
            {'index': panel.index, 'kind': panel.kind, 'title': panel.title,
             'row': panel.row + 1, 'column': panel.column + 1,
             'series': list(panel.series)}
            for panel in plan.panels],
        'row_labels': list(plan.row_labels),
        'cells': [
            {'index': cell.index, 'row': cell.row + 1,
             'column': cell.column + 1,
             'samples': [sample + 1 for sample in cell.samples],
             'groups': [group + 1 for group in cell.groups],
             'title': cell.title,
             'profile': cell.profile.index if cell.profile else None,
             'blocks': [
                 {'group': block.group + 1, 'sample': block.sample + 1}
                 for block in cell.blocks],
             'stack_labels': list(cell.stack_labels)}
            for cell in plan.cells],
        'series': [
            {'key': item.key, 'group': item.group + 1,
             'sample': item.sample + 1, 'label': item.label,
             'pseudocount': _pseudocount(spec, statistics, automatic,
                                         item.key)}
            for item in plan.series],
        'color_domains': [
            {'option': domain.option, 'kind': domain.kind,
             'slots': [_slot(slot, domain.kind, titles, series_labels,
                             series_default)
                       for slot in domain.slots]}
            for domain in plan.color_domains],
        'colorbars': ([] if colorbars is None else
                      [_colorbar(index, bar, colorbars.position,
                                 limits or None)
                       for index, bar in enumerate(colorbars.bars)]),
        'warnings': [message for _, message in warnings],
        'sample_sets': [_sample_set(item, scale)
                        for item, scale in zip(sample_plans, y_scales)],
        'y_limits': _y_limits(plan, sample_plans, spec, y_limits),
        'options': per_target_options(tool),
        'size_defaults': sizes,
        'problems': problems + [
            {'code': code, 'severity': 'warning', 'option': None,
             'target': None, 'slot': None, 'samples': None,
             'message': message}
            for code, message in warnings],
    }


def describe(tool, argv, *, force_load=False):
    """Describe a plot command without rendering or reading values unless needed."""
    from deeptoolsr.session import PlotSession

    with PlotSession(cache_bytes=0) as session:
        return session.describe([tool, *argv], force_load=force_load)


def describe_session(session, command, *, force_load=False,
                     cancelled=lambda: False, progress=None):
    """Describe using a resident matrix only when membership needs values.

    Errors that leave the rest of the command describable (an X-geometry
    clash in a sample set, an N= naming a missing target) come back as
    ``problems``; the rest raise as the plot commands do. ``cancelled`` is
    polled at the stages ``PlotSession.run`` uses and raises ``Cancelled``;
    ``progress`` receives each stage name there, as ``run`` reports it.
    """
    session.stage('parse', cancelled, progress)
    tool, *argv = command
    if tool not in TOOLS:
        raise OptionError(f'unsupported tool: {tool}')
    warnings = []
    args = parse_command(
        tool, argv, 'describe',
        emit=lambda stream, message: (
            stream == 'stderr' and
            warnings.append(('warning', message.rstrip('\n')))))
    if args.matrixFile is None:
        raise OptionError('the following arguments are required: --matrixFile/-m')
    session.apply_run_defaults(args, argv)
    path = run_options.config_path(args.config)
    raw = run_options.read_config_bytes(path)
    args.run_options = run_options.resolve_run_options_from_bytes(
        args, raw, path)
    threads = args.run_options.threads
    header = (session.matrix.header if session.matrix is not None and
              session.source_key is not None and
              session.source_key == SourceKey.of(args.matrixFile)
              else read_header(args.matrixFile, normalize=True))
    spec = _common_spec(args, header)
    # Membership that needs values is predicted from the header unless the
    # caller asks for the values (a front end describes after every edit);
    # filtering and clustering leave the group sizes unknown.
    predicted = not force_load and not spec.counts_from_header()
    if force_load:
        session.stage('load', cancelled, progress)
        matrix = session.matrix_for(args.matrixFile, threads)
        header = matrix.header
        session.stage('order', cancelled, progress)
        layout, prepared = session.layout(spec, threads)
    else:
        layout, prepared = predict_layout(header, spec)
    warnings += [(warning.kind,
                  warning_text(warning, layout,
                               regions_label=args.regionsLabel).rstrip('\n'))
                 for warning in prepared]
    figure_spec = project_matrix_spec(args, tool)
    with collecting_problems() as found:
        sizes = (len(layout.origins) * (UNKNOWN_COUNT,) if predicted
                 else None)
        labels = command_labels(args, layout, spec, header, counts=sizes)
        try:
            plan, sample_plans, scales = resolve_matrix_plan(
                header, layout, labels, figure_spec, sizes)
        except OptionError:
            raise
        except ValueError as error:
            raise option_error(error) from error
        scales = _check_colours(figure_spec, plan, scales, warnings)
        colorbars = (None if not figure_spec.show_heatmap else
                     plan_colorbars(figure_spec, plan, sample_plans, scales))
    y_scales = [{'scale': scale,
                 'sigma': (figure_spec.y_pseudolog_sigma
                           if scale == 'pseudolog' else None)}
                for scale in scales['yScale']]
    limits = False
    profile = (figure_spec, None, None, None)
    if force_load:
        data = PlotData(matrix, layout, labels, threads)
        limits = (None if colorbars is None else _z_limits(
            session, data, figure_spec, scales, cancelled, progress))
        if figure_spec.show_profile:
            profile = _profile_numbers(session, data, figure_spec, plan,
                                       sample_plans, scales)
    return _record(tool, layout, group_numbers(layout, spec), labels, plan,
                   sample_plans, y_scales, colorbars, warnings,
                   [error.problem() for error in found], limits,
                   _size_defaults(args, tool, figure_spec), predicted,
                   profile)


def _profile_numbers(session, data, spec, plan, sample_plans, scales):
    """The plot's profile statistics (session-cached) and its Y limits.

    Returns ``(spec, statistics, automatic, y_limits)``: ``automatic`` are
    the statistics with an automatic pseudocount, the drawn ones unless
    ``--pseudocount`` is typed.
    """
    from dataclasses import replace
    from deeptoolsr.stats import statistics_spec

    keys = [item.key for item in plan.series]
    pairs = [(item.group, item.sample) for item in plan.series]
    options = statistics_spec(spec)

    def batch(options):
        return dict(zip(keys, session.statistics_batch(
            data, pairs, options, data.threads)))
    statistics = batch(options)
    automatic = (batch(replace(options, pseudocount=-1))
                 if options.pseudocount >= 0 else statistics)
    y_limits = (None if spec.plot_type == 'heatmap' else _drawn_y_limits(
        plan, sample_plans, spec, scales, statistics))
    return spec, statistics, automatic, y_limits


def _z_limits(session, data, spec, scales, cancelled, progress):
    """The plot's automatic and effective heatmap limits, from one cached scan.

    Returns ``((auto low, auto high), lows, highs)``; lows and highs are
    indexed by sample set as the heatmap cells index them.
    """
    from deeptoolsr.plotting.heatmap import _limits, heatmap_scales
    from deeptoolsr.session import StagedNumeric

    numeric = StagedNumeric(session, cancelled, progress)
    _, (low,), (high,), _ = _limits(
        data, {'zMin': None, 'zMax': None, 'zMid': None}, numeric)
    _, _, lows, highs, _ = heatmap_scales(data, spec, scales, numeric)
    return (low, high), lows, highs


def _check_colours(spec, plan, scales, warnings):
    """Run the plot's colour checks; only given colours load Matplotlib.

    Returns the scale assignments with collected bad colours made automatic.
    """
    profile = spec.show_profile and spec.colors
    heatmap = spec.show_heatmap and any(scales[key]
                                        for key in ('colorMap', 'colorList'))
    if profile or heatmap:
        from deeptoolsr.plotting.prepared import (check_heatmap_colours,
                                                  series_colors)
        if profile:
            found = []
            series_colors(spec, plan, found)
            # The plot prints these as it draws (plotMatrix.py).
            warnings += [('colour', f'WARNING: {message}')
                         for message in found]
        if heatmap:
            return check_heatmap_colours(scales, spec)
    return scales


def describe_cli(tool, argv):
    """Print exactly one JSON document, or one diagnostic and an exit code."""
    try:
        record = describe(tool, argv)
    except OptionError as error:
        print(error.prefix + str(error), file=sys.stderr)
        raise SystemExit(error.exit_code) from error
    except DataError as error:
        for warning in error.warnings:
            message = (warning_text(warning, error.layout)
                       if error.layout is not None else str(warning))
            print(message.rstrip(), file=sys.stderr)
        print(error, file=sys.stderr)
        raise SystemExit(2 if error.parser_error else 1) from error
    except OSError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
    except ValueError as error:
        # The plot CLIs report invalid input in this one-line form.
        print(f'{tool} failed: {error}', file=sys.stderr)
        raise SystemExit(1) from error
    text = json.dumps(record, indent=2, ensure_ascii=False)
    try:
        text.encode('utf-8')
    except UnicodeEncodeError as error:
        print(f'{tool} failed: an argument is not valid UTF-8 text',
              file=sys.stderr)
        raise SystemExit(1) from error
    use_utf8_stdio()
    print(text)
