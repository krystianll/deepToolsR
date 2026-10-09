"""Resolve the one cell width and the heights of drawn panel kinds."""

from dataclasses import dataclass
from math import isclose


class SizeError(ValueError):
    """Explicit size options conflict or imply an out-of-range size."""


@dataclass(frozen=True)
class SolvedSizes:
    cell_width: float
    profile_height: float | None
    profile_aspect_ratio: float | None
    heatmap_height: float | None
    heatmap_aspect_ratio: float | None


def _check_bounds(width, width_sources, kinds, original, label):
    if not 1 <= width <= 100:
        sources = ', '.join(map(label, width_sources))
        raise SizeError(f'{sources} produce a cell width {width:g} cm; '
                        'width must be between 1 and 100 cm')
    for kind, values in kinds.items():
        height, _, shown, options, _, bounds = values
        if not shown or bounds[0] <= height <= bounds[1]:
            continue
        original_height, original_ratio = original[kind]
        sources = ((options[0],) if original_height is not None else
                   (*width_sources, options[1]) if original_ratio is not None
                   else width_sources)
        names = ', '.join(map(label, dict.fromkeys(sources)))
        raise SizeError(f'{names} produce a height {height:g} cm; '
                        f'{label(options[0])} must be between '
                        f'{bounds[0]:g} and {bounds[1]:g} cm')


def solve_sizes(*, cell_width=None, profile_height=None,
                profile_aspect_ratio=None, heatmap_height=None,
                heatmap_aspect_ratio=None, show_profile=True,
                show_heatmap=False, names=None):
    """Infer sizes, then fill ratio-kind heights, width, remaining ratios.

    Each drawn kind obeys width = height * ratio. Unshown kinds are inert.
    Defaults are profile 5 x 5 cm and heatmap 5 x 10 cm; once the width is
    known, a kind without a height or ratio takes its default ratio (1 and
    0.5). Explicit widths within 0.1% agree; the canonical ratio is
    recomputed from final sizes.
    """
    names = names or {}

    def label(key):
        return names.get(key, '--' + key)

    kinds = {
        'profile': [profile_height, profile_aspect_ratio, show_profile,
                    ('profileHeight', 'profileAspectRatio'), 5.0, (0.5, 100)],
        'heatmap': [heatmap_height, heatmap_aspect_ratio, show_heatmap,
                    ('heatmapHeight', 'heatmapAspectRatio'), 10.0, (3, 100)],
    }
    original = {'profile': (profile_height, profile_aspect_ratio),
                'heatmap': (heatmap_height, heatmap_aspect_ratio)}
    width = cell_width
    width_sources = ('cellWidth',) if width is not None else ()

    def propagate():
        nonlocal width, width_sources
        for kind, values in kinds.items():
            height, ratio, shown, options, _, _ = values
            if not shown:
                continue
            if height is not None and ratio is not None:
                implied = height * ratio
                if width is None:
                    width = implied
                    width_sources = tuple(option for option, given in zip(
                        options, original[kind]) if given is not None)
                elif not isclose(width, implied, rel_tol=0.001):
                    prior = ', '.join(map(label, width_sources))
                    current = ', '.join(map(label, options))
                    raise SizeError(f'{prior} imply width {width:g} cm, but '
                                    f'{current} imply width {implied:g} cm')
            if width is not None and height is not None:
                values[1] = width / height
            elif width is not None and ratio is not None:
                values[0] = width / ratio

    propagate()
    for values in kinds.values():
        if values[2] and values[1] is not None and values[0] is None:
            values[0] = values[4]
            propagate()
    if width is None:
        width = 5.0
        width_sources = ('cellWidth',)
        propagate()
    for values in kinds.values():
        if values[2] and values[0] is None:
            values[1] = 5.0 / values[4]  # the default ratio
            propagate()

    _check_bounds(width, width_sources, kinds, original, label)
    profile = kinds['profile']
    heatmap = kinds['heatmap']
    return SolvedSizes(
        width,
        profile[0] if show_profile else None,
        profile[1] if show_profile else None,
        heatmap[0] if show_heatmap else None,
        heatmap[1] if show_heatmap else None)
