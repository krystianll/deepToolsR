"""Plotting submodules and local colour map lookup."""


def colormap(name):
    """Resolve a deepToolsR colour map before Matplotlib's built-ins."""
    from matplotlib import colormaps
    from deeptoolsr.cm import COLORMAPS

    if name in COLORMAPS:
        return COLORMAPS[name]
    try:
        return colormaps[name]
    except KeyError as error:
        raise ValueError(str(error)) from error


def colormap_names():
    """Return accepted colour map names in sorted order."""
    from matplotlib import colormaps
    from deeptoolsr.cm import COLORMAPS

    return sorted(set(colormaps) | set(COLORMAPS))
