"""Plan heatmap colour bars from the given scale settings; reads no values.

Bars merge only on the settings as given: the colour map name or colour
list a sample set gets, and its zMin/zMax/zMid values, where 'auto' is one
token while automatic limits are one common scale for the whole matrix.
Outside --zAxisLimits common, automatic limits are the limit slot's own
(``limit_slots``), so bars with them never merge across slots; with
per_heatmap a cell whose blocks differ gets one bar per block, to the
right and titled by default. Values derived from the data never decide whether two bars merge.
"""

from dataclasses import dataclass, replace

from . import DEFAULT_HEATMAP_COLORMAP
from .matrix_plan import tool_option_spellings
from .series import flat_label, has_explicit_assignment, recycle


@dataclass(frozen=True)
class ColorbarScale:
    color_map: str | None
    color_list: tuple | None
    z_min: object
    z_max: object
    z_mid: float | None
    # The limit slot whose own automatic limits the bar shows, or None.
    scope: int | None = None


@dataclass(frozen=True)
class Colorbar:
    """One bar: 0-based positions into ``plan.cells`` and sample sets.

    ``block`` is the 0-based heatmap block of its one cell that the bar
    alone serves (--zAxisLimits per_heatmap), or None for whole cells.
    """

    row: int | None
    cells: tuple
    sample_sets: tuple
    scale: ColorbarScale
    title: str | None
    auto_title: str
    block: int | None = None


@dataclass(frozen=True)
class ColorbarPlan:
    position: str
    bars: tuple


def limit_slots(spec, plan, sample_count=None):
    """Each cell's colour-limit slot per heatmap block, and each slot's
    (sample set, region).

    A slot is a sample set, or with --zAxisLimits per_panel a cell, or
    with per_heatmap a block. Its region is the ``(groups, samples)`` its
    automatic limits scan: for common, None (the whole matrix) or, when the
    cells leave some of the matrix's ``sample_count`` samples out, every
    drawn heatmap; else the heatmaps of that set, cell or block. Callers
    that read only the slots may omit ``sample_count``.
    """
    cells = plan.cells
    mode = spec.z_axis_limits
    if mode == 'per_heatmap':
        slots, cell_slots = [], []
        for cell in cells:
            cell_slots.append(tuple(range(len(slots),
                                          len(slots) + len(cell.blocks))))
            slots += [(cell.sample_set, ((block.group,), (block.sample,)))
                      for block in cell.blocks]
        return tuple(cell_slots), tuple(slots)
    if mode == 'per_panel':
        return (tuple((index,) * len(cell.blocks)
                      for index, cell in enumerate(cells)),
                tuple((cell.sample_set, _region((cell,))) for cell in cells))
    count = max((cell.sample_set for cell in cells), default=-1) + 1
    common = None
    if mode == 'common' and sample_count is not None:
        # In matrix order, so arranging the same samples differently
        # scans the same region.
        drawn = tuple(tuple(sorted(part)) for part in _region(cells))
        if len(drawn[1]) < sample_count:
            common = drawn
    return (tuple((cell.sample_set,) * len(cell.blocks) for cell in cells),
            tuple((index, common if mode == 'common' else _region(
                [cell for cell in cells if cell.sample_set == index]))
                  for index in range(count)))


def _region(cells):
    return (tuple(dict.fromkeys(group for cell in cells
                                for group in cell.groups)),
            tuple(dict.fromkeys(sample for cell in cells
                                for sample in cell.samples)))


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
    result = tuple(ColorbarScale(
        *colors[index % len(colors)], _limit(lows[index % len(lows)]),
        _limit(highs[index % len(highs)]),
        None if mids is None else _limit(mids[index % len(mids)]))
        for index in range(count))
    return result, len(set(result)) > 1


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


def _groups(spec, rows, sets, keys, position):
    """Return (row, served items, items named by an 'auto' title) per bar;
    an item is a cell or one of its blocks, on grid row ``rows[item]``."""
    if position == 'below':
        # Merged x axes share a bar across equal scales; best shares one
        # across a sample set's adjacent group cells.
        if spec.x_axis_visibility == 'outer_merged':
            merge = keys
        elif spec.colorbar_location == 'best':
            merge = tuple(zip(sets, keys))
        else:
            merge = tuple(range(len(keys)))
        groups = []
        start = 0
        for index in range(1, len(keys) + 1):
            if (index == len(keys) or rows[index] != rows[start] or
                    merge[index] != merge[start]):
                served = tuple(range(start, index))
                groups.append((rows[start], served, served))
                start = index
        return groups
    found = {}
    for index, key in enumerate(keys):
        found.setdefault((rows[index], key) if position == 'side' else key,
                         []).append(index)
    return [(rows[served[0]] if position == 'side' else None,
             tuple(served), tuple(served)) for served in found.values()]


def _bar_items(spec, plan, shared):
    """Each bar candidate ``(cell, block or None, key)``: a cell, or with
    per_heatmap limits that differ within a cell, each of its blocks.

    An automatic limit outside --zAxisLimits common belongs to its slot, so
    its key carries the slot as ``scope``.
    """
    items = []
    for index, slots in enumerate(limit_slots(spec, plan)[0]):
        scale = shared[index]
        keys = [replace(scale, scope=slot) if spec.z_axis_limits != 'common'
                and 'auto' in (scale.z_min, scale.z_max) else scale
                for slot in slots] or [scale]
        if len(set(keys)) == 1:
            items.append((index, None, keys[0]))
        else:
            items += [(index, block, key) for block, key in enumerate(keys)]
    return items


def _item_title(cells, sample_plans, cell, block, headers):
    """A cell's sample-set label; a block bar names its panel and block.

    With nested headers each label joins its levels with a space, the
    block's too (``Sample A genes``).
    """
    label = flat_label(sample_plans[cells[cell].sample_set].subplot_label,
                       headers)
    if block is None:
        return label
    block_label = (' '.join(level for level in
                            cells[cell].stack_label_levels[block] if level)
                   if headers else cells[cell].stack_labels[block])
    return '\n'.join(dict.fromkeys(
        (flat_label(cells[cell].title, headers) or label, block_label)))


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
    shared = tuple(set_scales[cell.sample_set] for cell in cells)
    items = _bar_items(spec, plan, shared)
    per_block = len(items) > len(cells)
    split = len({key for _, _, key in items}) > len(set(shared))
    location = spec.colorbar_location
    position = ('side' if per_block and location == 'best' else
                _position(location, multiple or split))
    spellings = tool_option_spellings(spec.tool, spec.invoked_spellings)
    if (split and position in ('below_common', 'side_common') or
            per_block and position == 'below'):
        from deeptoolsr.prepare import OptionError, collected
        problem = OptionError(
            f'{spellings["colorbarLocation"]} {location} cannot show the '
            f'separate automatic limits of {spellings["zAxisLimits"]} '
            f'{spec.z_axis_limits}; use '
            + (f'{spellings["colorbarLocation"]} right or best, '
               if per_block else '') +
            f'{spellings["zAxisLimits"]} common, or type '
            f'{spellings["zMin"]} and {spellings["zMax"]}',
            code='colorbar_limits_not_common',
            option=spellings['zAxisLimits'])
        if not collected(problem):
            raise problem
        items = [(index, None, scale) for index, scale in enumerate(shared)]
    groups = _groups(
        spec, tuple(cells[index].row for index, _, _ in items),
        tuple(cells[index].sample_set for index, _, _ in items),
        tuple(key for _, _, key in items), position)
    titles = recycle(
        spec.colorbar_labels, len(groups),
        option=spellings['colorbarLabels'], domain='colorbar', text=True)
    bars = []
    for (row, served, named), title in zip(groups, titles):
        auto = ',\n'.join(dict.fromkeys(
            _item_title(cells, sample_plans, *items[index][:2],
                        spec.nested_headers)
            for index in named))
        block = items[served[0]][1]
        if title == 'auto' or block is not None and not spec.colorbar_labels:
            title = auto
        served_cells = tuple(dict.fromkeys(items[index][0]
                                           for index in served))
        bars.append(Colorbar(
            row, served_cells,
            tuple(dict.fromkeys(cells[index].sample_set
                                for index in served_cells)),
            items[served[0]][2], title if title and title != 'none' else None,
            auto, block))
    return ColorbarPlan(position, tuple(bars))
