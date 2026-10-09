import numpy as np
import re
from matplotlib.transforms import ScaledTranslation
from deeptoolsr.plotting.geometry import (
    COLORBAR_MIN_TICK_DISTANCE_CM, COLORBAR_PREFERRED_TICK_COUNT)
from deeptoolsr.plotting.rendering import draw_without_rendering


def fit_ticks(ticks, last, span):
    """deepTools' tick fit: when the last tick misses ``last``, stretch the
    ticks so that it lands on ``span`` (the data axis width in bins)."""
    if np.ceil(max(ticks)) != float(last):
        return [value * float(span) / max(ticks) for value in ticks]
    return list(ticks)


def _mitigate_tick_label_overlaps(ax, renderer, padding_points,
                                  max_rotated_shift_points):
    """Apply overlap corrections using an already-current renderer."""
    labels = [label for label in ax.get_xticklabels()
              if label.get_visible() and label.get_text()]
    if len(labels) < 2:
        return False
    padding_pixels = padding_points * ax.figure.dpi / 72.0
    changed = False
    if any(label.get_rotation() % 360 != 0 for label in labels):
        # Keep the conventional rotation alignment, but move a later label
        # horizontally when its rendered box overlaps the previous one.
        # Justification changes would alter the rotation anchor and can send
        # text into the plotting area.
        for left, right in zip(labels[:-1], labels[1:]):
            left_box = left.get_window_extent(renderer)
            right_box = right.get_window_extent(renderer)
            vertical_overlap = left_box.y1 > right_box.y0 and right_box.y1 > left_box.y0
            overlap = left_box.x1 + padding_pixels - right_box.x0
            if vertical_overlap and overlap > 0:
                max_shift_pixels = (max_rotated_shift_points *
                                    ax.figure.dpi / 72.0)
                shift_pixels = min(overlap, max_shift_pixels)
                right.set_transform(
                    right.get_transform() +
                    ScaledTranslation(shift_pixels / ax.figure.dpi, 0,
                                      ax.figure.dpi_scale_trans))
                changed = True
        return changed

    for left, right in zip(labels[:-1], labels[1:]):
        # Compare the labels as though both were centered on their ticks, and
        # as drawn: an endpoint label anchored inward (alignTickLabelsForRotation)
        # can reach its neighbour even when centered labels would fit (for
        # example, -50 bp beside TSS on a 300-bin axis).
        left_box = left.get_window_extent(renderer)
        right_box = right.get_window_extent(renderer)
        left_tick = ax.transData.transform((left.get_position()[0], 0))[0]
        right_tick = ax.transData.transform((right.get_position()[0], 0))[0]
        required_gap = (left_box.width + right_box.width) / 2.0 + padding_pixels
        if (right_tick - left_tick < required_gap or
                left_box.x1 + padding_pixels > right_box.x0):
            left.set_horizontalalignment('right')
            right.set_horizontalalignment('left')
            changed = True
    return changed


def mitigateTickLabelOverlapsForAxes(axes, padding_points=2,
                                     max_rotated_shift_points=4,
                                     renderer=None):
    """Mitigate X tick overlaps on many axes with one shared measurement.

    All axes must belong to the same figure. When ``renderer`` is omitted the
    figure has one draw-disabled measurement pass; callers with a current
    renderer can pass it and avoid that pass. Corrections are applied
    synchronously and the next layout/render pass observes all of them.
    """
    axes = list(axes)
    if not axes:
        return False
    figure = axes[0].figure
    if any(axis.figure is not figure for axis in axes):
        raise ValueError('all axes must belong to the same figure')
    if renderer is None:
        draw_without_rendering(figure)
        renderer = figure.canvas.get_renderer()
    changed = False
    for axis in axes:
        if _mitigate_tick_label_overlaps(
                axis, renderer, padding_points, max_rotated_shift_points):
            changed = True
    return changed


def alignTickLabelsForRotation(ax, rotation):
    """Anchor rotated X labels away from the plotting area."""
    labels = ax.get_xticklabels()
    if not labels:
        return
    if rotation > 0:
        alignment = 'right'
    elif rotation < 0:
        alignment = 'left'
    else:
        labels[0].set_horizontalalignment('left')
        labels[-1].set_horizontalalignment('right')
        return
    for label in labels:
        label.set_horizontalalignment(alignment)
        label.set_rotation_mode('anchor')


def _nice_grid_ladder(magnitude, coarsest_offset=2, depth=9):
    """Nice grid step sizes for ``magnitude``, coarsest first.

    Steps are ``{1, 2, 5} x 10^k`` sorted descending, so successive entries are
    ``... 10, 5, 2, 1, 0.5, 0.2, 0.1 ...`` -- integers before halves before
    tenths, which is the "fewest digits" ordering the tick chooser wants.
    """
    if magnitude <= 0:
        magnitude = 1.0
    exponent = int(np.floor(np.log10(magnitude)))
    steps = set()
    for power in range(exponent + coarsest_offset, exponent - depth, -1):
        for mult in (1.0, 2.0, 5.0):
            steps.add(mult * (10.0 ** power))
    return sorted(steps, reverse=True)


def _roundest_in(low, high):
    """Return the roundest (fewest-digit) number within ``[low, high]``."""
    if low > high:
        low, high = high, low
    if low <= 0.0 <= high:
        return 0.0
    midpoint = (low + high) / 2.0
    tolerance = (high - low) * 1e-9
    for step in _nice_grid_ladder(max(abs(low), abs(high))):
        candidate = round(midpoint / step) * step
        if low - tolerance <= candidate <= high + tolerance:
            return candidate
    return midpoint


def _round_number_closest_to(target, span):
    """Nice number closest to ``target``; ties break toward fewer digits."""
    best = None
    for rank, step in enumerate(_nice_grid_ladder(span)):
        candidate = round(target / step) * step
        key = (round(abs(candidate - target), 9), rank)
        if best is None or key < best[0]:
            best = (key, candidate)
    return best[1] if best is not None else target


def _format_scientific(value):
    """Compact scientific form: ``5e-4`` / ``-1.5e-3`` (no ``.0`` mantissa, no
    zero-padded exponent), with zero rendered plainly."""
    if value == 0:
        return '0'
    mantissa, _, exponent = '{:.1e}'.format(value).partition('e')
    if mantissa.endswith('.0'):
        mantissa = mantissa[:-2]
    return '{}e{}'.format(mantissa, int(exponent))


def _format_tick_values(values):
    """Format ``values`` with the fewest decimals that keep them distinct."""
    finite = [value for value in values]
    largest = max((abs(value) for value in finite), default=0.0)
    # Scientific notation once decimals get unwieldy: below 1e-2 (so 0.005 reads
    # as 5e-3 rather than a 3+-decimal figure) or above 1e5.
    if largest and (largest < 1e-2 or largest >= 1e5):
        return [_format_scientific(value) for value in finite]
    for decimals in range(0, 7):
        rounded = [round(value, decimals) for value in finite]
        exact = all(abs(r - v) < 1e-9 for r, v in zip(rounded, finite))
        distinct = len(set(rounded)) == len(rounded)
        if exact and distinct:
            labels = ['{:.{d}f}'.format(value, d=decimals) for value in finite]
            # Normalise a signed zero ("-0") to "0".
            return [label.lstrip('+') if label != '-0' else '0'
                    for label in labels]
    return ['{:g}'.format(value) for value in finite]


def _multiples_in(vmin, vmax, step):
    """Multiples of ``step`` lying within ``[vmin, vmax]``."""
    start = int(np.ceil(vmin / step - 1e-9))
    end = int(np.floor(vmax / step + 1e-9))
    return [round(index * step, 10) for index in range(start, end + 1)]


def _dense_nice_ticks(vmin, vmax, target, max_ticks):
    """Evenly-spaced round ticks, count as close to ``target`` as a nice step
    allows (and never more than ``max_ticks``).  A dense set naturally reaches
    close to both ends, so no special near-edge handling is needed here."""
    raw = (vmax - vmin) / max(1, target - 1)
    base = int(np.floor(np.log10(raw)))
    candidates = sorted({mult * 10.0 ** exp
                         for exp in range(base - 1, base + 3)
                         for mult in (1.0, 2.0, 2.5, 5.0)})
    best = None
    for step in candidates:
        ticks = _multiples_in(vmin, vmax, step)
        if len(ticks) < 2 or len(ticks) > max_ticks:
            continue
        # Closest to the target; on ties prefer the denser (larger) set.
        key = (abs(len(ticks) - target), -len(ticks))
        if best is None or key < best[0]:
            best = (key, ticks)
    if best is None:
        return [vmin, (vmin + vmax) / 2.0, vmax]
    return best[1]


def choose_colorbar_ticks(vmin, vmax, length_cm, prefer_count=None,
                          min_spacing_cm=None, edge_fraction=0.10):
    """Pick clean colorbar ticks.

    Returns ``(positions, labels)``.  The chooser aims for ``prefer_count`` ticks
    (default ``COLORBAR_PREFERRED_TICK_COUNT``) but never packs them closer than
    ``min_spacing_cm`` (default ``COLORBAR_MIN_TICK_DISTANCE_CM``) on a bar
    ``length_cm`` long, so a short bar gets fewer.  Up to three ticks the outer
    ones are the roundest numbers within ``edge_fraction`` of each end and the
    middle is the round number nearest their midpoint (so they read evenly
    spaced); for a denser bar an evenly-spaced nice-step set is used, which
    already reaches close to both ends.
    """
    if min_spacing_cm is None:
        min_spacing_cm = COLORBAR_MIN_TICK_DISTANCE_CM
    if prefer_count is None:
        prefer_count = COLORBAR_PREFERRED_TICK_COUNT
    vmin, vmax = float(vmin), float(vmax)
    if not vmax > vmin:
        return [vmin], _format_tick_values([vmin])
    span = vmax - vmin
    max_by_spacing = int(length_cm / min_spacing_cm) + 1 if length_cm else 1
    target = min(prefer_count, max(1, max_by_spacing))
    if target >= 5:
        # Only go dense once enough ticks fit that a nice step reaches close to
        # both ends on its own; below that the explicit near-edge construction
        # is what keeps an awkward range (e.g. -3.245..2.113 -> -3 not -2) tidy.
        positions = sorted(set(_dense_nice_ticks(
            vmin, vmax, target, max_by_spacing)))
        return positions, _format_tick_values(positions)
    # Sparse (<=3 effective ticks): near-edge/even-midpoint construction, odd.
    count = target if target % 2 else max(1, target - 1)
    if count == 1:
        centre = _round_number_closest_to((vmin + vmax) / 2.0, span)
        centre = min(max(centre, vmin), vmax)
        return [centre], _format_tick_values([centre])
    low = _roundest_in(vmin, vmin + edge_fraction * span)
    high = _roundest_in(vmax - edge_fraction * span, vmax)
    positions = [low]
    centre = _round_number_closest_to((low + high) / 2.0, span)
    if low < centre < high:
        positions.append(centre)
    positions.append(high)
    positions = sorted(set(positions))
    return positions, _format_tick_values(positions)


def getDistanceUnit(parameters, idx=None, requested='auto'):
    """Resolve the distance unit for one matrix sample."""
    if requested != 'auto':
        return requested
    upstream = parameters['upstream']
    downstream = parameters['downstream']
    if idx is not None:
        upstream = upstream[idx]
        downstream = downstream[idx]
    distance = max(abs(upstream), abs(downstream))
    if distance >= 1_000_000:
        return 'mb'
    if distance >= 1000:
        return 'kb'
    return 'bp'


def formatDistance(value, unit, include_unit=True):
    quotient = {'bp': 1.0, 'kb': 1000.0, 'mb': 1_000_000.0}[unit]
    scaled = float(value) / quotient
    number = '{:g}'.format(scaled)
    return '{} {}'.format(number, unit) if include_unit else number


def formatDistanceAxisLabel(label, unit, unit_location):
    """Normalize legacy units and place the selected unit on the axis."""
    label = re.sub(r'\s*[\[(](?:bp|kb|mb)[\])]\s*$', '', label,
                   flags=re.IGNORECASE)
    if unit_location == 'axis':
        return '{} [{}]'.format(label, unit) if label else '[{}]'.format(unit)
    return label


def getProfileTicks(parameters, referencePointLabel, startLabel, endLabel, idx,
                    distanceUnit='auto', distanceUnitLocation='ticks'):
    """
    returns the position and labelling of the xticks that
    correspond to the heatmap

    As of deepToolsR 3, the various parameters can be lists, in which case we then need to index things (the idx parameter)

    As of matplotlib 3 the ticks in the heatmap need to have 0.5 added to them.

    As of matplotlib 3.1 there is no longer padding added to all ticks. Reference point ticks will be adjusted by width/2
    or width for spacing and the last half of scaled ticks will be shifed by 1 bin so the ticks are at the beginning of bins.
    """
    w = parameters['bin size']
    b = parameters['upstream']
    a = parameters['downstream']
    if idx is not None:
        w = w[idx]
        b = b[idx]
        a = a[idx]

    try:
        c = parameters['unscaled 5 prime']
        if idx is not None:
            c = c[idx]
    except BaseException:
        c = 0
    try:
        d = parameters['unscaled 3 prime']
        if idx is not None:
            d = d[idx]
    except BaseException:
        d = 0
    m = parameters['body']
    if idx is not None:
        m = m[idx]

    unit = getDistanceUnit(parameters, idx, distanceUnit)
    include_unit = distanceUnitLocation == 'ticks'

    if m == 0:
        xticks = []
        xtickslabel = []
        if b > 0:
            xticks.append(0)
            xtickslabel.append(formatDistance(-b, unit, include_unit))
        xticks.append(max((b - 0.5 * w) / w, 0))
        xtickslabel.append(referencePointLabel)
        if a > 0:
            xticks.append(max((b + a - w) / w, 0))
            xtickslabel.append(formatDistance(a, unit, include_unit))
    else:
        xticks_values = [0]
        xtickslabel = []

        # only if upstream region is set, add a x tick
        if b > 0:
            xticks_values.append(b)
            xtickslabel.append(formatDistance(-b, unit, include_unit))

        xtickslabel.append(startLabel)

        # set the x tick for the body parameter, regardless if
        # upstream is 0 (not set)
        if c > 0:
            xticks_values.append(b + c)
            xtickslabel.append("")

        if d > 0:
            xticks_values.append(b + c + m)
            xtickslabel.append("")

        # We need to subtract the bin size from the last 2 point so they're placed at the beginning of the bin
        xticks_values.append(b + c + m + d - w)
        xtickslabel.append(endLabel)

        if a > 0:
            xticks_values.append(b + c + m + d + a - w)
            xtickslabel.append(formatDistance(a, unit, include_unit))

        xticks = [(k / w) for k in xticks_values]
        xticks = [max(x, 0) for x in xticks]

    return xticks, xtickslabel
