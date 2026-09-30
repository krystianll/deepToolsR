"""Matplotlib adapter for the backend-independent typography schema.

``geometry.py`` defines the style as plain, Matplotlib-free dataclasses. This
module is the only place that turns a resolved ``FontSpec`` into a Matplotlib
``FontProperties`` and supplies resolved sizes for a local ``rc_context`` for text
that still inherits its font from them. The font multiplier is applied here, at
draw time, and never written back into the stored sizes.
"""

from matplotlib.font_manager import FontProperties


def make_font_properties(font_spec, family, multiplier):
    """Convert one ``FontSpec`` to a Matplotlib ``FontProperties``.

    A role's own ``family`` overrides the global ``family``; the size is the
    base point size scaled by ``multiplier``.
    """
    resolved_family = (list(font_spec.family) if font_spec.family is not None
                       else list(family))
    return FontProperties(
        family=resolved_family,
        size=font_spec.size * multiplier,
        weight=font_spec.weight,
        style=font_spec.style)


def role_font(style, role):
    """``FontProperties`` for one named typography role of a resolved style."""
    return make_font_properties(
        getattr(style.typography, role), style.font_family,
        style.font_multiplier)


def scaled_point_size(style, role):
    """The draw-time point size for a role (base size times the multiplier)."""
    return getattr(style.typography, role).size * style.font_multiplier


def apply_text_font(text, style, role):
    """Apply the complete resolved font for ``role`` to a Text artist."""
    text.set_fontproperties(role_font(style, role))
    return text


def apply_axis_fonts(axis, style):
    """Apply complete role fonts to every standard text artist on an axis.

    Matplotlib rcParams can express role-specific sizes for these artists, but
    cannot express role-specific family/style/weight for tick labels and do not
    expose family/style settings for axis labels or titles.  Apply explicit
    ``FontProperties`` after ticks have been built so all four FontSpec fields
    reach both the measurement and final-drawing artists.
    """
    apply_text_font(axis.title, style, 'panel_title_text')
    apply_text_font(axis.xaxis.label, style, 'axis_label_text')
    apply_text_font(axis.yaxis.label, style, 'axis_label_text')
    for label in axis.get_xticklabels(which='both'):
        apply_text_font(label, style, 'x_tick_label_text')
    for label in axis.get_yticklabels(which='both'):
        apply_text_font(label, style, 'y_tick_label_text')
    apply_text_font(axis.xaxis.get_offset_text(), style, 'x_tick_label_text')
    apply_text_font(axis.yaxis.get_offset_text(), style, 'y_tick_label_text')


def apply_colorbar_tick_fonts(axis, style):
    """Apply the colorbar tick role to both orientations and offset text."""
    font = role_font(style, 'colorbar_tick_label_text')
    for label in (axis.get_xticklabels(which='both') +
                  axis.get_yticklabels(which='both')):
        label.set_fontproperties(font)
    axis.xaxis.get_offset_text().set_fontproperties(font)
    axis.yaxis.get_offset_text().set_fontproperties(font)


def style_rc(style):
    """Return the rcParams that plot text still inherits from.

    Every value is derived from the resolved style (not a Matplotlib
    default), so measurement and drawing share one deterministic base and no
    text falls back to an incidental default. Roles drawn through an explicit
    ``FontProperties`` (figure title, relocated slot labels) do not rely on
    these and are handled at their call sites.
    """
    typography = style.typography
    params = {'pdf.fonttype': 42, 'svg.fonttype': 'none'}
    params['font.family'] = list(style.font_family)
    # font.size is the inherited base; tick and axis-label roles pin their own
    # rcParams so a future change to one need not disturb the others.
    params['font.size'] = scaled_point_size(style, 'x_tick_label_text')
    params['axes.labelsize'] = scaled_point_size(style, 'axis_label_text')
    params['axes.labelweight'] = typography.axis_label_text.weight
    params['axes.titlesize'] = scaled_point_size(style, 'panel_title_text')
    params['axes.titleweight'] = typography.panel_title_text.weight
    params['xtick.labelsize'] = scaled_point_size(style, 'x_tick_label_text')
    params['ytick.labelsize'] = scaled_point_size(style, 'y_tick_label_text')
    params['legend.fontsize'] = scaled_point_size(style, 'legend_label_text')
    # Tick-to-label distance is a physical geometry gap (never multiplied by the
    # font multiplier). Setting it here, before any axis is built, applies it
    # uniformly to major/minor X and Y ticks and to colorbar axes, replacing the
    # reliance on Matplotlib's implicit default.
    tick_pad = style.geometry.tick_label_to_tick_gap
    params['xtick.major.pad'] = tick_pad
    params['xtick.minor.pad'] = tick_pad
    params['ytick.major.pad'] = tick_pad
    params['ytick.minor.pad'] = tick_pad
    return params
