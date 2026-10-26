"""Plotting submodules and local colour map lookup."""

# The colour maps the plots use where no option sets one: heatmaps
# (--colorMap) and series-heatmap panels (--colorsPerSample).
DEFAULT_HEATMAP_COLORMAP = 'RdYlBu'
DEFAULT_SERIES_COLORMAP = 'RdYlBu_r'


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
    """Return every accepted colour map name, sorted (public API).

    These are the names --colorMap and a series heatmap's --colorsPerSample
    accept: Matplotlib's registered colour maps (including their ``_r``
    reversals) and deepToolsR's own. Importing Matplotlib makes the first
    call slower than ``describe``.
    """
    from matplotlib import colormaps
    from deeptoolsr.cm import COLORMAPS

    return sorted(set(colormaps) | set(COLORMAPS))
