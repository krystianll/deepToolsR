"""Small helpers for renderer-dependent layout and figure output."""

import os

import matplotlib

from deeptoolsr.path_validation import atomic_output_path


def draw_without_rendering(figure):
    """Settle artists with Agg's current renderer, without a canvas draw.

    Matplotlib's Figure.draw_without_rendering fetches a renderer by calling
    FigureCanvasAgg.draw, which allocates a full bitmap even with drawing
    disabled. Our figures already own an Agg canvas and its renderer.
    This relies on private RendererAgg._draw_disabled(), as Figure does;
    rerun tests/test_render_passes.py after a Matplotlib upgrade.
    """
    renderer = figure.canvas.get_renderer()
    with renderer._draw_disabled():
        figure.draw(renderer)


def resolve_figure_format(destination, image_format=None):
    """Return the format to pass to ``Figure.savefig``.

    Plot files are written through a temporary path whose ``.tmp`` suffix
    must not be used by Matplotlib for format inference.  Resolve the format
    from the final destination instead, while retaining the documented
    precedence of an explicit ``--plotFileFormat`` value.
    """
    if image_format:
        return image_format
    extension = os.path.splitext(os.fspath(destination))[1].lstrip('.')
    if extension:
        return extension.lower()
    return 'png'


def save_figure_atomic(figure, destination, *, dpi=None, image_format=None):
    """Save *figure* atomically without losing final-path format inference."""
    resolved_format = resolve_figure_format(destination, image_format)
    metadata = {'pdf': {'CreationDate': None},
                'svg': {'Date': None}}.get(resolved_format)
    with atomic_output_path(destination, suffix='.plot.tmp') as temporary:
        with matplotlib.rc_context({'svg.hashsalt': 'deeptoolsr'}):
            figure.savefig(temporary, dpi=dpi, format=resolved_format,
                           metadata=metadata)


def suspend_axes_images(figure):
    """Hide visible ``AxesImage`` artists and return restorable state.

    Layout measurements depend on axes decorations, not on heatmap pixels.
    Temporarily hiding images prevents every provisional layout pass from
    resampling the matrix while preserving axes limits and geometry.
    """
    states = []
    for axis in figure.axes:
        for image in axis.images:
            visible = image.get_visible()
            states.append((image, visible))
            if visible:
                image.set_visible(False)
    return states


def restore_axes_images(states):
    """Restore image visibility captured by :func:`suspend_axes_images`."""
    for image, visible in states:
        image.set_visible(visible)
