"""Compute a matrix figure's numbers and resolved styles before drawing."""

from dataclasses import dataclass, field

from matplotlib import colors as pltcolors
import numpy as np

from deeptoolsr.stats import (
    NumericProvider, calculate_group_statistics_batch, scan_prepared_matrix,
    statistics_spec)
from . import DEFAULT_HEATMAP_COLORMAP, DEFAULT_SERIES_COLORMAP, colormap
from .colorbars import plan_colorbars
from .heatmap import heatmap_scales, region_lengths
from .matrix_plan import tool_option_spellings
from .series import has_explicit_assignment, resolve_recyclable


@dataclass(frozen=True)
class PreparedMatrix:
    """Everything a figure needs that does not depend on its layout.

    The profile part is filled whenever profiles are drawn, with or without
    heatmap stacks, so both kinds of figure share one set of statistics,
    limits and series colours. ``series_colors`` has one entry per slot of
    the profile colour domain: a colour for line profiles, a colour map for
    a series heatmap. The heatmap part is filled only with stacks.
    """

    sample_set_plans: tuple
    header_parameters: dict
    y_min: tuple
    y_max: tuple
    y_scale: tuple
    statistics_by_key: dict = field(default_factory=dict)
    series_colors: tuple = ()
    warnings: tuple = ()
    color_options: dict | None = None
    percentiles: object = None
    z_min: tuple = ()
    z_max: tuple = ()
    z_mid: tuple | None = None
    colorbars: object = None
    regions_length_in_bins: tuple = ()


def _y_limits(data, spec, scales, run, numeric):
    """The plan's typed Y limits, with a heatmap's profile scan filling auto."""
    y_min, y_max = scales['yMin'], scales['yMax']
    if (spec.show_profile and spec.plot_type == 'heatmap' and
            (y_min == [None] or y_max == [None])):
        low, high = (numeric.scan(data, (1.0, 98.0), False, run.threads)
                     if numeric is not None else scan_prepared_matrix(
                         data, (1.0, 98.0), threads=run.threads))
        if y_min == [None]:
            y_min = [None if np.isnan(low) else low]
        if y_max == [None]:
            y_max = [None if np.isnan(high) else high]
    return tuple(y_min), tuple(y_max)


def _bad_colour(value, option, target, slot):
    """True when describe collected ``value`` as not a colour; else raise."""
    if not value or pltcolors.is_color_like(value):
        return False
    from deeptoolsr.prepare import OptionError, collected
    problem = OptionError(
        f'{option}: {value} is not a colour; use a colour name or an HTML '
        'hex string such as #eeff22', code='unknown_colour', option=option,
        target=target, slot=slot)
    if not collected(problem):
        raise problem
    return True


def _bad_colormap(name, option, target, slot):
    """True when describe collected ``name`` as no colour map; else raise."""
    try:
        colormap(name)
    except ValueError:
        from deeptoolsr.prepare import OptionError, collected
        problem = OptionError(
            f'{option}: {name} is not a colour map; use a Matplotlib or '
            'deepToolsR colour map name such as viridis',
            code='unknown_colour_map', option=option, target=target,
            slot=slot)
        if not collected(problem):
            raise problem from None
        return True
    return False


def _check_colors(values, option, target):
    """The given colours; one describe collected becomes '' (automatic)."""
    return ['' if _bad_colour(value, option, target, slot) else value
            for slot, value in enumerate(values, 1)]


def _check_colormaps(names, option, target, default):
    """The named colour maps; one describe collected becomes ``default``."""
    return tuple(colormap(default if _bad_colormap(name, option, target, slot)
                          else name)
                 for slot, name in enumerate(names, 1))


def check_heatmap_colours(assigned, spec):
    """Check --colorMap names and --colorList colours up front.

    Returns the assignments; a sample set whose colour describe collected
    as a problem keeps the automatic colour map.
    """
    spellings = tool_option_spellings(spec.tool, spec.invoked_spellings)
    result = dict(assigned)
    if assigned['colorMap']:
        result['colorMap'] = [
            DEFAULT_HEATMAP_COLORMAP if name and _bad_colormap(
                name, spellings['colorMap'], 'sample set', slot) else name
            for slot, name in enumerate(assigned['colorMap'], 1)]
    if assigned['colorList']:
        lists = [
            '' if value and any(
                _bad_colour(colour, spellings['colorList'], 'sample set', slot)
                for colour in value.replace(' ', '').split(',')) else value
            for slot, value in enumerate(assigned['colorList'], 1)]
        explicit = has_explicit_assignment(spec.color_list)
        result['colorList'] = lists if explicit or all(lists) else None
    return result


def _check_colour_count(values, slots, option, *, noun, target, warnings):
    """Require one value per slot in a plain list; extras are ignored."""
    if len(values) < slots:
        from deeptoolsr.prepare import OptionError
        raise OptionError(
            f'{option} needs exactly {slots} {noun}, one for each {target}; '
            f'{len(values)} given (or use N={noun[:-1]} to set some)')
    if len(values) > slots:
        warnings.append(
            f'{option}: {len(values)} {noun} given for {slots} '
            f'{"coloured series" if noun == "colours" else "panels"}; '
            f'the last {len(values) - slots} are ignored')
    return list(values[:slots])


def _resolve_mixed(values, count, option, *, domain, noun, target, warnings):
    """N= entries set their slots; plain entries fill the rest in order."""
    from deeptoolsr.plotting.series import assignment_parts
    plain = [value for value in values if not assignment_parts(value)]
    resolved = resolve_recyclable(
        [value for value in values if assignment_parts(value)], count,
        default=lambda index: None, option=option, domain=domain)
    free = [index for index, value in enumerate(resolved) if value is None]
    if not plain:
        return resolved
    if len(plain) < len(free):
        from deeptoolsr.prepare import OptionError
        raise OptionError(
            f'{option}: {len(plain)} plain {noun} given for {len(free)} '
            f'{target} without an N= assignment; give exactly {len(free)} '
            f'or assign them with N={noun[:-1]}')
    if len(plain) > len(free):
        warnings.append(
            f'{option}: {len(plain)} plain {noun} given for {len(free)} '
            f'{target} without an N= assignment; the last '
            f'{len(plain) - len(free)} are ignored')
    for index, value in zip(free, plain):
        resolved[index] = value
    return resolved


def default_series_colors(plan):
    """The line colours (RGBA) the series slots get without --colorsPerSample.

    One per panel when every panel draws one series, else one per slot.
    """
    panels = tuple(cell.profile for cell in plan.cells)
    single = max(len(panel.series) for panel in panels) == 1
    count = len(panels) if single else len(plan.color_domains[0].slots)
    return tuple(colormap('jet')(np.arange(count) / float(count)))


def series_colors(spec, plan, warnings):
    """Resolve the palette the profile colour slots index, once.

    Values address the colour domain's slots, numbered 1..len(slots) in
    slot order (the legend order of the distinctly coloured series; a
    series heatmap has one colour-map slot per panel).  A plain list names
    one value per slot (extras are ignored with a warning); ``N=``
    assignments set any subset of slots, and plain entries mixed with them
    fill the remaining slots in order, exactly.
    """
    option = tool_option_spellings(spec.tool, spec.invoked_spellings)['colors']
    domain = plan.color_domains[0]
    slots = len(domain.slots)
    colors = list(spec.colors or ())
    explicit = bool(colors) and has_explicit_assignment(colors)
    if domain.kind == 'colormap':
        if colors and not explicit:
            colors = _check_colour_count(
                colors, slots, option, noun='colour maps', target='panel',
                warnings=warnings)
            return _check_colormaps(colors, option, 'panel', DEFAULT_SERIES_COLORMAP)
        if explicit:
            colors = _resolve_mixed(
                colors, slots, option, domain='panel', noun='colour maps',
                target='panels', warnings=warnings)
            colors = [value or DEFAULT_SERIES_COLORMAP for value in colors]
        else:
            colors = [DEFAULT_SERIES_COLORMAP] * slots
        return _check_colormaps(colors, option, 'panel', DEFAULT_SERIES_COLORMAP)
    if explicit:
        colors = _resolve_mixed(
            colors, slots, option, domain='series', noun='colours',
            target='series', warnings=warnings)
        colors = _check_colors(colors, option, 'series')
        default = colormap('jet')(np.arange(slots) / float(slots))
        return tuple(value or tuple(default[index])
                     for index, value in enumerate(colors))
    if not colors:
        return default_series_colors(plan)
    colors = _check_colour_count(
        _check_colors(colors, option, 'series'), slots, option,
        noun='colours', target='distinctly coloured series',
        warnings=warnings)
    default = colormap('jet')(np.arange(slots) / float(slots))
    return tuple(value or tuple(default[index])
                 for index, value in enumerate(colors))


def prepare_matrix(data, plan, sample_plans, scales, spec, run,
                   numeric: NumericProvider | None = None):
    """Calculate statistics, limits and styles for one figure request."""
    statistics = {}
    warnings = []
    profile_colors = ()
    if spec.show_profile:
        pairs = tuple((item.group, item.sample) for item in plan.series)
        options = statistics_spec(spec)
        values = (numeric.statistics_batch(data, pairs, options, run.threads)
                  if numeric is not None else
                  calculate_group_statistics_batch(
                      data, pairs, options, threads=run.threads))
        statistics = {item.key: value
                      for item, value in zip(plan.series, values)}
        profile_colors = series_colors(spec, plan, warnings)
    y_min, y_max = _y_limits(data, spec, scales, run, numeric)
    common = dict(
        sample_set_plans=tuple(sample_plans),
        header_parameters=data.matrix.header.parameters, y_min=y_min, y_max=y_max,
        y_scale=tuple(scales['yScale']),
        statistics_by_key=statistics, series_colors=profile_colors,
        warnings=tuple(warnings))
    if not spec.show_heatmap:
        return PreparedMatrix(**common)
    scales = check_heatmap_colours(scales, spec)
    colorbars = plan_colorbars(spec, plan, sample_plans, scales)
    color_options, percentiles, z_min, z_max, z_mid = heatmap_scales(
        data, spec, scales, numeric=numeric)
    return PreparedMatrix(
        **common, color_options=color_options, percentiles=percentiles,
        z_min=tuple(z_min), z_max=tuple(z_max),
        z_mid=None if z_mid is None else tuple(z_mid),
        colorbars=colorbars,
        regions_length_in_bins=region_lengths(data))
