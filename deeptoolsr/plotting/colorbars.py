"""Plan heatmap colour bars from the given scale settings; reads no values.

Bars merge only on the settings as given: the colour map name or colour
list a sample set gets, and its zMin/zMax/zMid values, where 'auto' is one
token (automatic limits are one common scale for the whole matrix). Values
derived from the data never decide whether two bars merge.
"""

from dataclasses import dataclass

from . import DEFAULT_HEATMAP_COLORMAP
from .matrix_plan import tool_option_spellings
from .series import has_explicit_assignment, recycle


@dataclass(frozen=True)
class ColorbarScale:
    color_map: str | None
    color_list: tuple | None
    z_min: object
    z_max: object
    z_mid: float | None


@dataclass(frozen=True)
class Colorbar:
    """One bar: 0-based positions into ``plan.cells`` and sample sets."""

    row: int | None
    cells: tuple
    sample_sets: tuple
    scale: ColorbarScale
    title: str | None
    auto_title: str


@dataclass(frozen=True)
class ColorbarPlan:
    position: str
    bars: tuple


def _limit(value):
    if value is None or value == '':
        return None
    return 'auto' if str(value).lower() == 'auto' else float(value)


def colour_sources(color_map, color_list, explicit):
    """Each heatmap scale's colours: ``(map name, None)`` or ``(None, list)``.

    A colour list replaces the maps; with ``N=`` assignments (``explicit``)
    the scales it leaves empty keep the colour maps.
    """
    if not color_list:
        return [(name, None) for name in color_map or ()]
    defaults = color_map or [DEFAULT_HEATMAP_COLORMAP]
    return [(None, tuple(value.replace(' ', '').split(',')))
            if value or not explicit else
            (defaults[column % len(defaults)], None)
            for column, value in enumerate(color_list)]


def _set_scales(spec, scales, count):
    color_list = scales['colorList']
    colors = colour_sources(
        scales['colorMap'] or [DEFAULT_HEATMAP_COLORMAP], color_list,
        bool(color_list) and has_explicit_assignment(spec.color_list))
    lows = scales['zMin'] or ['auto']
    highs = scales['zMax'] or ['auto']
    mids = scales['zMid']
    multiple = (len(colors) > 1 or len(lows) > 1 or len(highs) > 1 or
                mids is not None and len(mids) > 1)
    return tuple(ColorbarScale(
        *colors[index % len(colors)], _limit(lows[index % len(lows)]),
        _limit(highs[index % len(highs)]),
        None if mids is None else _limit(mids[index % len(mids)]))
        for index in range(count)), multiple


def _position(location, multiple):
    if location == 'below_common':
        return 'below_common'
    if location == 'right_common':
        return 'side_common'
    if location == 'right':
        return 'side'
    if location in ('below', 'bottom') or location == 'best' and multiple:
        return 'below'
    return 'side'


def _groups(spec, cells, keys, position):
    """Return (row, served cells, cells named by an 'auto' title) per bar."""
    if position == 'below':
        # Merged x axes share a bar across equal scales; best shares one
        # across a sample set's adjacent group cells.
        if spec.x_axis_visibility == 'outer_merged':
            merge = keys
        elif spec.colorbar_location == 'best':
            merge = tuple(cell.sample_set for cell in cells)
        else:
            merge = tuple(range(len(cells)))
        groups = []
        start = 0
        for index in range(1, len(cells) + 1):
            if (index == len(cells) or cells[index].row != cells[start].row or
                    merge[index] != merge[start]):
                served = tuple(range(start, index))
                groups.append((cells[start].row, served, served))
                start = index
        return groups
    found = {}
    for index, cell in enumerate(cells):
        key = (cell.row, keys[index]) if position == 'side' else keys[index]
        found.setdefault(key, []).append(index)
    return [(cells[served[0]].row if position == 'side' else None,
             tuple(served), tuple(served)) for served in found.values()]


def plan_colorbars(spec, plan, sample_plans, scales):
    """Plan the heatmap colour bars, numbered as --colorbarLabels N= counts.

    An ``N=`` naming a missing bar raises, or is collected by describe.
    ``--colorbarLocation none`` plans no bars and checks no labels.
    """
    if spec.colorbar_location == 'none':
        return ColorbarPlan('none', ())
    cells = plan.cells
    count = max((cell.sample_set for cell in cells), default=-1) + 1
    set_scales, multiple = _set_scales(spec, scales, count)
    position = _position(spec.colorbar_location, multiple)
    keys = tuple(set_scales[cell.sample_set] for cell in cells)
    groups = _groups(spec, cells, keys, position)
    titles = recycle(
        spec.colorbar_labels, len(groups),
        option=tool_option_spellings(
            spec.tool, spec.invoked_spellings)['colorbarLabels'],
        domain='colorbar', text=True)
    bars = []
    for (row, served, named), title in zip(groups, titles):
        auto = ',\n'.join(dict.fromkeys(
            sample_plans[cells[index].sample_set].subplot_label
            for index in named))
        if title == 'auto':
            title = auto
        bars.append(Colorbar(
            row, served,
            tuple(dict.fromkeys(cells[index].sample_set for index in served)),
            keys[served[0]], title if title and title != 'none' else None,
            auto))
    return ColorbarPlan(position, tuple(bars))
