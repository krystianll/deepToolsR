"""Prepare heatmap statistics, colours, raster images and tick artists."""

import copy
import math

import matplotlib
import numpy as np
from matplotlib import ticker
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon

from deeptoolsr import _raster
from deeptoolsr.plotting.ticks import getProfileTicks
from deeptoolsr.matrix import projection_kwargs
from deeptoolsr.numeric_validation import recycled_min_max_error
from deeptoolsr.parallel import parallel_map
from deeptoolsr.stats import NumericProvider, scan_prepared_matrix
from deeptoolsr.plotting.ticks import choose_colorbar_ticks
from . import colormap, fonts
from .matrix_plan import tool_option_spellings
from .series import (effective_per_group, has_explicit_assignment, recycle,
                     resolve_recyclable)


HEATMAP_RASTER_OVERSAMPLE = 2.0


def _scale_domain(spec, plan):
    """Count the colour scales and name what their positions address."""
    count = max((cell.scale for cell in plan.cells), default=0) + 1
    domain = 'group' if effective_per_group(spec) else 'sample set'
    context = ('--arrangeSamples ' + ' '.join(spec.arrange_samples)
               if spec.arrange_samples else None)
    return count, domain, context


def _resolve_scale(values, count, default, *, option, domain, context):
    if values is not None and has_explicit_assignment(values):
        return resolve_recyclable(
            values, count, default=default, option=option,
            domain=domain, context=context)
    return values


def scale_assignments(spec, plan):
    """Resolve per-scale colour and limit options; needs no matrix values."""
    count, domain, context = _scale_domain(spec, plan)
    spellings = tool_option_spellings(spec.tool, spec.invoked_spellings)

    def resolve(dest, values, default):
        return _resolve_scale(values, count, default, option=spellings[dest],
                              domain=domain, context=context)
    color_list = spec.color_list
    if color_list and has_explicit_assignment(color_list):
        color_list = resolve('colorList', list(color_list), '')
    z_min = resolve('zMin', spec.z_min, 'auto')
    z_max = resolve('zMax', spec.z_max, 'auto')
    error = recycled_min_max_error(
        z_min, z_max, count, spellings['zMin'], spellings['zMax'],
        strict=True)
    if error:
        from deeptoolsr.prepare import OptionError
        raise OptionError(error)
    return {
        'colorMap': (resolve('colorMap', spec.color_map, 'RdYlBu')
                     if spec.color_map else spec.color_map),
        'colorList': color_list,
        'zMin': z_min,
        'zMax': z_max,
        'zMid': resolve('zMid', spec.z_mid, lambda index: None)}


def _limits(data, assigned, numeric: NumericProvider | None = None, *,
            mid_option='--zMid'):
    lower, upper, middle = assigned['zMin'], assigned['zMax'], assigned['zMid']

    def needs_auto(value):
        return value is None or isinstance(value, list) and 'auto' in value
    percentiles = None
    if needs_auto(lower) or needs_auto(upper):
        percentiles = _color_limit_percentiles(
            data, (0.0, 1.0, 98.0, 100.0), numeric=numeric)
        lower, upper = _resolve_color_limits(lower, upper, percentiles)
    else:
        lower = [float(value) for value in lower]
        upper = [float(value) for value in upper]
    if middle is not None:
        middle = [None if value is None or value == '' else float(value)
                  for value in middle]
        for index in range(max(len(lower), len(upper), len(middle))):
            mid = middle[index % len(middle)]
            if mid is None:
                continue
            minimum = lower[index % len(lower)]
            maximum = upper[index % len(upper)]
            if minimum is None or maximum is None:
                raise ValueError(
                    '--zMid requires finite effective zMin and zMax values')
            if not minimum < mid < maximum:
                from deeptoolsr.prepare import OptionError
                raise OptionError(
                    f'{mid_option}: {mid:g} must be strictly between zMin '
                    f'({minimum:g}) and zMax ({maximum:g})')
    return percentiles, lower, upper, middle


def heatmap_scales(data, spec, assigned,
                   numeric: NumericProvider | None = None):
    """Resolve colour maps and limits from the plan's scale assignments."""
    options = {
        'colorMap': assigned['colorMap'], 'colorList': assigned['colorList'],
        'colorNumber': spec.color_number,
        'missingDataColor': spec.missing_data_color, 'alpha': spec.alpha}
    percentiles, lower, upper, middle = _limits(
        data, assigned, numeric=numeric,
        mid_option=tool_option_spellings(
            spec.tool, spec.invoked_spellings)['zMid'])
    return options, percentiles, lower, upper, middle


def heatmap_y_labels(spec, sample_plans):
    return tuple(recycle(
        spec.heatmap_y_axis_label, len(sample_plans),
        option=tool_option_spellings(spec.tool, spec.invoked_spellings)[
            'heatmapYAxisLabel'], text=True))


def region_lengths(data):
    parameters = data.matrix.header.parameters
    result = [None] * len(parameters['upstream'])
    if ((data.layout.sort.using if data.layout.sort else None) !=
            'region_length' or
            (data.layout.sort.method if data.layout.sort else None) == 'no'):
        return tuple(result)
    for index in range(len(parameters['upstream'])):
        if parameters['ref point'][index] is None:
            continue
        groups = []
        for group in ordered_regions(data):
            lengths = []
            for region in group:
                length = (region['end'] - region['start']
                          if isinstance(region, dict) else
                          sum(end - start for start, end in region[1]))
                if parameters['ref point'][index] == 'TSS':
                    position = parameters['upstream'][index] + length
                elif parameters['ref point'][index] == 'center':
                    position = parameters['upstream'][index] + length * .5
                elif parameters['ref point'][index] == 'TES':
                    position = parameters['upstream'][index] - length
                else:
                    continue
                lengths.append(position / parameters['bin size'][index])
            groups.append(lengths)
        result[index] = groups
    return tuple(result)


def midpoint_exponent(vmin, vcenter, vmax):
    """Return the power exponent that maps *vcenter* to colour position 0.5."""
    if not vmin < vcenter < vmax:
        raise ValueError(
            'The heatmap midpoint must be strictly between zMin and zMax')
    midpoint_position = (vcenter - vmin) / (vmax - vmin)
    return np.log(0.5) / np.log(midpoint_position)


def shift_cmap_midpoint(cmap, vmin, vcenter, vmax, samples=256):
    """Shift a colormap's middle colour while retaining linear data scaling."""
    exponent = midpoint_exponent(vmin, vcenter, vmax)
    source_positions = np.linspace(0, 1, samples)
    target_positions = source_positions ** (1.0 / exponent)
    shifted = matplotlib.colors.LinearSegmentedColormap.from_list(
        '{}_midpoint'.format(cmap.name),
        list(zip(target_positions, cmap(source_positions))),
        N=getattr(cmap, 'N', samples))
    return shifted.with_extremes(bad=cmap.get_bad(),
                                 under=cmap.get_under(),
                                 over=cmap.get_over())


def make_colormaps(options, explicit):
    maps = []
    if options['colorMap']:
        for name in options['colorMap']:
            value = copy.copy(colormap(name))
            value.set_bad(options['missingDataColor'])
            maps.append(value)

    def from_list(value):
        built = matplotlib.colors.LinearSegmentedColormap.from_list(
            'my_cmap', value.replace(' ', '').split(','),
            N=options['colorNumber'])
        built.set_bad(options['missingDataColor'])
        return built

    if options['colorList'] and len(options['colorList']) > 0:
        if explicit:
            defaults = options['colorMap'] or ['RdYlBu']
            maps = []
            for column, values in enumerate(options['colorList']):
                if values:
                    maps.append(from_list(values))
                else:
                    fallback = copy.copy(colormap(
                        defaults[column % len(defaults)]))
                    fallback.set_bad(options['missingDataColor'])
                    maps.append(fallback)
        else:
            maps = [from_list(value) for value in options['colorList']]
    return maps


def resolve_colorbar_position(spec, maps, prepared):
    location = ('bottom' if spec.colorbar_location == 'below' else
                spec.colorbar_location)
    multiple = (len(maps) > 1 or len(prepared.z_min) > 1 or
                len(prepared.z_max) > 1 or
                prepared.z_mid is not None and len(prepared.z_mid) > 1)
    if location == 'below_common':
        return 'below_common'
    if location == 'right_common':
        return 'side_common'
    if location == 'right':
        return 'side'
    if location == 'bottom' or location == 'best' and multiple:
        return 'below'
    return 'side'


def colorbar_mappable(maps, index, minimum, maximum, midpoint):
    chosen = maps[index]
    if midpoint is not None:
        chosen = shift_cmap_midpoint(chosen, minimum, midpoint, maximum)
    mappable = ScalarMappable(
        norm=matplotlib.colors.Normalize(minimum, maximum), cmap=chosen)
    mappable.set_array([])
    return mappable


def _finite_scale(minimum, maximum):
    return (minimum is not None and maximum is not None and
            maximum > minimum)


def style_horizontal_colorbar(cbar, minimum, maximum, width_cm, style):
    """Use near-edge ticks and tuck horizontal end labels under the bar."""
    cbar.ax.tick_params(labelsize=fonts.scaled_point_size(
        style, 'colorbar_tick_label_text'))
    if _finite_scale(minimum, maximum):
        positions, labels = choose_colorbar_ticks(
            minimum, maximum, width_cm)
        cbar.set_ticks(positions)
        cbar.ax.set_xticklabels(labels)
        ticks = cbar.ax.get_xticklabels()
        if ticks:
            ticks[0].set_horizontalalignment('left')
            ticks[-1].set_horizontalalignment('right')
    fonts.apply_colorbar_tick_fonts(cbar.ax, style)


def style_vertical_colorbar(cbar, minimum, maximum, height_cm, style):
    """Use near-edge ticks at a colorbar's current solved height."""
    cbar.ax.tick_params(labelsize=fonts.scaled_point_size(
        style, 'colorbar_tick_label_text'))
    if _finite_scale(minimum, maximum):
        positions, labels = choose_colorbar_ticks(
            minimum, maximum, height_cm)
        cbar.set_ticks(positions)
        cbar.ax.set_yticklabels(labels)
    fonts.apply_colorbar_tick_fonts(cbar.ax, style)


def _color_limit_percentiles(data, probs,
                             numeric: NumericProvider | None = None):
    return (numeric.scan(data, tuple(probs), False, data.threads)
            if numeric is not None else scan_prepared_matrix(
                data, probs, threads=data.threads))


def ordered_regions(data):
    rows = data.layout.row_indices()
    return [[data.matrix.regions[int(row)] for row in
             rows[start:stop]]
            for start, stop in zip(data.layout.group_bounds,
                                   data.layout.group_bounds[1:])]


def get_plot_ticks(data, sample, references, starts, ends):
    return getProfileTicks(
        data.matrix.header.parameters, references[sample], starts[sample], ends[sample], sample,
        data.distance_unit, data.distance_unit_location)


def draw_sort_indicator(axis, direction, triangle_fraction=1.0):
    """Draw the effective region-sort direction in a dedicated axes."""
    if direction not in ('ascend', 'descend'):
        return None
    axis.set_axis_off()
    vertices = ([(0, 1), (triangle_fraction, 1),
                 (triangle_fraction, 0)] if direction == 'descend'
                else [(triangle_fraction, 1), (0, 0),
                      (triangle_fraction, 0)])
    polygon = Polygon(vertices, closed=True, facecolor='black',
                      edgecolor='none', transform=axis.transAxes)
    axis.add_patch(polygon)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    return polygon


def distribute_minor_ticks(major_ticks, count):
    """Distribute an exact total of ticks proportionally between landmarks."""
    major_ticks = np.asarray(major_ticks, dtype=float)
    if count <= 0 or len(major_ticks) < 2:
        return []
    order = np.argsort(major_ticks)
    major_ticks = major_ticks[order]
    gaps = np.diff(major_ticks)
    positive = gaps > 0
    if not np.any(positive):
        return []

    raw_minor_counts = gaps[positive] / np.sum(gaps[positive]) * count
    minor_counts = np.floor(raw_minor_counts).astype(int)
    remainder = count - int(np.sum(minor_counts))
    fractions = raw_minor_counts - minor_counts
    for idx in np.argsort(-fractions, kind='stable')[:remainder]:
        minor_counts[idx] += 1

    minor_ticks = []
    segment_idx = 0
    for left, right, valid in zip(major_ticks[:-1], major_ticks[1:], positive):
        if not valid:
            continue
        n_minor = minor_counts[segment_idx]
        minor_ticks.extend(np.linspace(left, right, n_minor + 2)[1:-1])
        segment_idx += 1
    return list(map(float, minor_ticks))


def add_minor_x_ticks(axis, setting='auto'):
    """Add space-aware or explicitly subdivided minor x ticks."""
    major_ticks = np.asarray(axis.get_xticks(), dtype=float)
    if setting == 'none' or setting == 0 or len(major_ticks) < 2:
        return
    if setting == 'auto':
        xmin, xmax = sorted(axis.get_xlim())
        width_pixels = max(axis.get_window_extent().width, 1)
        target_intervals = max(2, int(width_pixels / 20.0))
        locator = ticker.MaxNLocator(
            nbins=target_intervals, steps=[1, 2, 2.5, 5, 10])
        candidates = locator.tick_values(xmin, xmax)
        tolerance = max(xmax - xmin, 1) * 1e-9
        minor_ticks = [tick for tick in candidates
                       if xmin < tick < xmax and
                       not np.any(np.isclose(tick, major_ticks,
                                             rtol=0, atol=tolerance))]
    else:
        minor_ticks = distribute_minor_ticks(major_ticks, setting)
    axis.set_xticks(minor_ticks, minor=True)
    axis.tick_params(axis='x', which='minor', bottom=True,
                     length=2.5, direction='out')


def render_heatmap_rgba(request):
    """Native colour-map + downsample to a uint8 RGBA block (GIL-released).

    Pure computation with no Matplotlib calls, so a batch can run in parallel.
    """
    return _raster.render_heatmap(
        request['data'].values, request['lut'], request['bad'],
        request['vmin'], request['vmax'],
        request['out_h'], request['out_w'],
        request['raster'].filter, request['raster'].bit_depth,
        request['threads'], **projection_kwargs(request['data']))


def place_heatmap_rgba(request, rgba):
    """Draw a rendered RGBA block onto its axis (Matplotlib, main thread)."""
    drawn = request['ax'].imshow(
        rgba, aspect='auto', interpolation=request['interpolation'],
        origin='upper', extent=request['extent'], alpha=request['alpha'])
    drawn.set_rasterized(True)


def flush_deferred_heatmap_rasters(requests, threads):
    """Render every deferred native heatmap block, then place them.

    The colour-map/downsample step releases the GIL, so the blocks are rendered
    across a thread pool (one inner native thread each, never multiplying worker
    counts); the Matplotlib ``imshow`` placement then runs sequentially on the
    calling thread because Matplotlib artists are not thread-safe. Falls back to
    sequential rendering for a single block or a single-worker budget.
    """
    if not requests:
        return

    def render(request, inner_threads):
        return render_heatmap_rgba({**request, 'threads': inner_threads})

    rendered = parallel_map(render, requests, threads)
    for request, rgba in zip(requests, rendered):
        place_heatmap_rgba(request, rgba)


def _raster_output_size(ax, dpi, rows, cols):
    """Target (out_h, out_w) for a block's RGBA raster.

    Sizes to the block's own on-screen area -- its axis fraction times the
    current (scratch, 1 cm per grid unit) figure size times the output DPI --
    with a small oversample, capped by the source resolution. This replaces a
    fixed pixel ceiling, so a block never retains far more pixels than the
    requested output can show (the dominant heatmap memory cost).
    """
    figure = ax.figure
    position = ax.get_position()
    width_px = position.width * figure.get_figwidth() * dpi
    height_px = position.height * figure.get_figheight() * dpi
    out_w = int(min(cols, max(1, np.ceil(width_px * HEATMAP_RASTER_OVERSAMPLE))))
    out_h = int(min(rows, max(1, np.ceil(height_px * HEATMAP_RASTER_OVERSAMPLE))))
    return out_h, out_w


def draw_heatmap_image(ax, matrix, image_cmap, vmin, vmax, alpha,
                       interpolation_method, rows, cols, dpi, deferred=None,
                       *, raster, threads, group_digest=None, sample=None):
    """Draw one native heatmap block and return its colorbar mappable.

    When ``deferred`` is a list, the block is not rasterised here; its
    request is appended so a caller can render a whole figure's blocks in
    parallel and place them afterwards. The colorbar mappable is still returned
    immediately (it needs only the norm/cmap, not the pixels).
    """
    extent = [0, cols, rows, 0]
    data = matrix
    lut = np.ascontiguousarray(
        image_cmap(np.linspace(0, 1, 256), bytes=True)[:, :4],
        dtype=np.uint8)
    bad = (np.array(matplotlib.colors.to_rgba(image_cmap.get_bad()))
           * 255).round().astype(np.uint8).tolist()
    # Size the raster to the block's on-screen area (times a small
    # oversample) so matplotlib's uint8 -> float32 upcast and final
    # resize land on an array no larger than the output can show.
    out_h, out_w = _raster_output_size(ax, dpi, rows, cols)
    # The raster is pre-shrunk in C++ to bound memory, but is still
    # larger than the on-screen axis; let matplotlib filter that final
    # (small) resize rather than nearest-dropping it, which matches the
    # single float resample the plain imshow path does. A small group
    # asks for 'nearest' so its few regions stay crisp rectangles.
    final_interpolation = ('nearest' if interpolation_method == 'nearest'
                           else 'antialiased')
    request = {
        'ax': ax, 'data': data, 'rows': rows, 'cols': cols,
        'lut': lut, 'bad': bad, 'vmin': float(vmin), 'vmax': float(vmax),
        'out_h': out_h, 'out_w': out_w, 'extent': extent,
        'interpolation': final_interpolation, 'alpha': alpha,
        'raster': raster, 'group_digest': group_digest,
        'sample': sample,
        'threads': threads}
    if deferred is not None:
        # Establish data coordinates before deferred pixels are placed.
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        deferred.append(request)
    else:
        place_heatmap_rgba(request, render_heatmap_rgba(request))
    mappable = ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin, vmax), cmap=image_cmap)
    mappable.set_array([])
    return mappable


def _resolve_color_limits(z_min, z_max, percentiles):
    """Resolve automatic bounds once for both raster and colorbar consumers."""
    lows = ['auto'] if z_min is None else z_min
    highs = ['auto'] if z_max is None else z_max
    data_low, auto_low, auto_high, data_high = percentiles
    if np.isnan(data_low):
        data_low, auto_low, auto_high, data_high = 0., 0., 1., 1.
    resolved_low, resolved_high = [], []
    for index in range(math.lcm(len(lows), len(highs))):
        low, high = lows[index % len(lows)], highs[index % len(highs)]
        low_auto, high_auto = low == 'auto', high == 'auto'
        low = float(auto_low if low_auto else low)
        high = float(auto_high if high_auto else high)
        if high <= low and (low_auto or high_auto):
            if low_auto:
                low = float(data_low)
            if high_auto:
                high = float(data_high)
            if high <= low:
                # Stored matrices are float32; do scalar arithmetic in double.
                pad = max(abs(low), abs(high), 1.) * 0.05
                if low_auto and high_auto:
                    low, high = low - pad, high + pad
                elif low_auto:
                    low = high - pad
                else:
                    high = low + pad
        resolved_low.append(low)
        resolved_high.append(high)
    return resolved_low, resolved_high
