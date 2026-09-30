"""Small immutable geometry primitives used by the grid solver."""

import math
from dataclasses import dataclass, fields


POINTS_PER_INCH = 72.0
CM_PER_INCH = 2.54

# Fixed physical layout distances. These are deliberately expressed in
# typographic points because they must not scale with the data-panel size.
#
# Canvas and general panel spacing
FIGURE_EDGE_PADDING_POINTS = 8.0
PLOT_TITLE_TO_CONTENT_GAP_POINTS = 8.0
ADJACENT_DECORATION_CLEARANCE_POINTS = 4.0  # default gap between objects
EXTERNAL_LEGEND_TO_CONTENT_GAP_POINTS = 4.0
RIGHT_LEGEND_TO_CONTENT_GAP_POINTS = 8.0

# Heatmap columns and vertically stacked region groups
HEATMAP_MIN_COLUMN_GAP_POINTS = 8.0
HEATMAP_DECORATION_GAP_POINTS = 4.0
HEATMAP_GROUP_GAP_POINTS = 8.0

# Color scale and sorting indicator
# The sorting indicator uses the same thickness as the colorbar so the two
# vertical guides remain visually balanced.
HEATMAP_COLORBAR_THICKNESS_POINTS = 9.0  # also used for sort indicator
HEATMAP_COLORBAR_GAP_POINTS = 8.0
SORT_INDICATOR_GAP_POINTS = 6.0

# A region group with fewer than this many rows is drawn crisply (nearest-
# neighbour, no resampling/antialiasing) so a handful of regions render as solid
# rectangles rather than smooth gradients.  This matches the historical
# deepTools threshold below which 'auto' interpolation selects 'nearest'; larger
# groups are antialiased because nearest would otherwise drop rows.
HEATMAP_SMALL_GROUP_MAX_ROWS = 1000

# Colorbar grid (multiple color scales packed beside/below the heatmaps).
# The minimum height is the colorbar rectangle only, excluding tick labels and
# any title.  Row/column gaps are measured border-to-border and widest-tick to
# next-border respectively; the label gap separates a colorbar title from the
# rectangle and is reserved only when the title is present.
COLORBAR_MIN_HEIGHT_POINTS = 2.0 / CM_PER_INCH * POINTS_PER_INCH  # ~56.7 pt (2 cm)
COLORBAR_SHORT_EDGE_GAP_POINTS = 8.0
COLORBAR_LONG_EDGE_GAP_POINTS = 8.0
COLORBAR_LABEL_GAP_POINTS = ADJACENT_DECORATION_CLEARANCE_POINTS
# Colorbar tick density. The chooser aims for COLORBAR_PREFERRED_TICK_COUNT ticks
# (kept odd so the middle is marked) but never packs them closer than
# COLORBAR_MIN_TICK_DISTANCE_CM, so a short bar drops to fewer ticks. Raise the
# count and/or lower the distance for denser ticks on tall bars.
COLORBAR_PREFERRED_TICK_COUNT = 5
COLORBAR_MIN_TICK_DISTANCE_CM = 1.0
# Minimum clearance kept between a below/below_common colorbar's end tick label
# and its facing neighbour (the adjacent bar's near label, or the reserved
# figure-edge padding for the outermost labels). Matches the heatmap axis
# clearance; a label is centred while its clearance stays at or above this and
# only tucked once it would drop below it.
COLORBAR_LABEL_MIN_CLEARANCE_POINTS = 4.0
# Two adjacent structural labels that meet in the same gutter retain this
# much visible space between their glyph boxes.
LABEL_MIN_CLEARANCE_POINTS = 4.0
# Upper bound on the below_common inter-rectangle gap. The gap tracks the heatmap
# column gap (a wider grid means a wider figure, hence more room for the bars),
# capped here so it never grows unreasonably.
COLORBAR_BELOW_COMMON_MAX_GAP_POINTS = 20.0

# Text bands to the left and above heatmaps
HEATMAP_TO_ROW_HEADER_GAP_POINTS = ADJACENT_DECORATION_CLEARANCE_POINTS
ROW_HEADER_TO_Y_LABEL_GAP_POINTS = ADJACENT_DECORATION_CLEARANCE_POINTS * 1.5
HEATMAP_TO_SUMMARY_PLOT_GAP_POINTS = 8.0

# Profile grids - also for summary plots
SAMPLE_LABEL_TO_PANEL_GAP_POINTS = ADJACENT_DECORATION_CLEARANCE_POINTS
AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS = 2.0
# Merged/common axis labels sit a little further from the shared tick-label
# envelope so their extra separation communicates that they belong to a whole
# panel run or grid rather than one nearby panel.
COMMON_AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS = 6.0
PROFILE_MINIMUM_COLUMN_GAP_POINTS = 6.0
# X and Y axis labels use the same physical clearance from their tick-label
# envelope.  Keep the historical X-named constant as an alias so callers do
# not need to know that the two directions deliberately share one setting.
X_AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS = \
    AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS
PROFILE_ROW_ADDITIONAL_GAP_POINTS = 16.0
PROFILE_DECORATION_GAP_POINTS = ADJACENT_DECORATION_CLEARANCE_POINTS  # to maybe implement - this could be the minimum gap between the profile panels in a column and the leftmost decoration of the next column
# Separation between the last data panel in a row and its rotated facet label.
PROFILE_PANEL_TO_ROW_LABEL_GAP_POINTS = ADJACENT_DECORATION_CLEARANCE_POINTS

# Distance between a tick mark and its text label. Centralising it lets the
# drivers set it explicitly and identically for major/minor X and Y ticks and
# horizontal/vertical colorbars. The default matches Matplotlib's historical
# tick pad (3.5 pt) so making it explicit is visually a no-op.
TICK_LABEL_TO_TICK_GAP_POINTS = 3.5

# Legend-internal spacing, in points, chosen to preserve the approximate physical
# spacing of the current 8-point Matplotlib legends. They remain fixed physical
# distances when font_multiplier changes; the Matplotlib adapter converts points
# to the relative (font-size-fraction) values Matplotlib expects.
LEGEND_BORDER_PADDING_POINTS = 3.2
LEGEND_HANDLE_TO_TEXT_GAP_POINTS = 6.4
LEGEND_ENTRY_ROW_GAP_POINTS = 4.0
LEGEND_COLUMN_GAP_POINTS = 16.0

# Default profile centre-line stroke width. This is a drawing property, not a
# gap, so it is never scaled by font_multiplier.
PROFILE_LINE_WIDTH_POINTS = 1.5

# Initial typography policy for each independently configurable label class.
DEFAULT_LABEL_MAX_LINES = 3
DEFAULT_LABEL_EXTRA_ROW_PENALTY = 0.08
DEFAULT_LABEL_BALANCE_WEIGHT = 0.02
DEFAULT_LABEL_ORPHAN_WEIGHT = 0.05


@dataclass(frozen=True)
class Size:
    width: float
    height: float


@dataclass(frozen=True)
class Insets:
    left: float = 0.0
    right: float = 0.0
    top: float = 0.0
    bottom: float = 0.0


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def x0(self):
        return self.x

    @property
    def x1(self):
        return self.x + self.width

    @property
    def y0(self):
        return self.y

    @property
    def y1(self):
        return self.y + self.height


def cm_to_points(value):
    return float(value) / CM_PER_INCH * POINTS_PER_INCH


def points_to_inches(value):
    return float(value) / POINTS_PER_INCH


def bbox_to_size_points(bbox, dpi):
    scale = POINTS_PER_INCH / float(dpi)
    return Size(bbox.width * scale, bbox.height * scale)


# ---------------------------------------------------------------------------
# Persistent typography / geometry / drawing / label-layout schema
#
# These dataclasses are the single, Matplotlib-independent source of truth for
# the configurable typography, geometry, drawing, and label-layout policy.
# They perform no filesystem I/O or environment reads: ``config.py`` owns
# loading, merging, and persistence, and calls ``resolve_style`` below. Sizes
# and gaps are always stored as *base* points;
# the font multiplier is a separate scalar applied only at draw time (see
# ``scaled_size``), so it can never compound into the saved sizes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FontSpec:
    """A backend-independent description of one text role's font.

    ``family`` is an ordered fallback list, or ``None`` to inherit the global
    ``font_family``. ``size`` is a base point size (never multiplier-applied).
    """

    size: float
    weight: str = "normal"
    style: str = "normal"
    family: tuple = None


@dataclass(frozen=True)
class TypographySpec:
    """Font for every named text role. Roles are assigned by visual function
    and grid location, never by the data origin of the underlying string.

    ``panel_title_text`` covers every panel-level title: profile and heatmap
    panel titles, the heatmap sample/sample-set column labels, and the row
    headers (including rotated right-side profile facet labels). Heatmap side
    region labels use their own ``region_label_text`` role.
    ``axis_label_text`` is shared by the X and Y axis labels.
    """

    figure_title_text: FontSpec
    panel_title_text: FontSpec
    region_label_text: FontSpec
    axis_label_text: FontSpec
    x_tick_label_text: FontSpec
    y_tick_label_text: FontSpec
    colorbar_title_text: FontSpec
    colorbar_tick_label_text: FontSpec
    legend_label_text: FontSpec


@dataclass(frozen=True)
class GeometrySpec:
    """Every persistent physical spacing, in typographic points.

    Defaults reference the module-level constants above so the constants (kept
    as compatibility aliases during migration) and the spec cannot drift. The
    final entries participate in layout but are dimensions rather than gaps.
    """

    figure_edge_padding: float = FIGURE_EDGE_PADDING_POINTS
    figure_title_to_content_gap: float = PLOT_TITLE_TO_CONTENT_GAP_POINTS
    external_legend_to_content_gap: float = EXTERNAL_LEGEND_TO_CONTENT_GAP_POINTS
    right_legend_to_content_gap: float = RIGHT_LEGEND_TO_CONTENT_GAP_POINTS
    heatmap_min_column_gap: float = HEATMAP_MIN_COLUMN_GAP_POINTS
    heatmap_decoration_gap: float = HEATMAP_DECORATION_GAP_POINTS
    heatmap_group_gap: float = HEATMAP_GROUP_GAP_POINTS
    heatmap_colorbar_gap: float = HEATMAP_COLORBAR_GAP_POINTS
    sort_indicator_gap: float = SORT_INDICATOR_GAP_POINTS
    colorbar_short_edge_gap: float = COLORBAR_SHORT_EDGE_GAP_POINTS
    colorbar_long_edge_gap: float = COLORBAR_LONG_EDGE_GAP_POINTS
    colorbar_title_to_bar_gap: float = COLORBAR_LABEL_GAP_POINTS
    colorbar_label_min_clearance: float = COLORBAR_LABEL_MIN_CLEARANCE_POINTS
    label_min_clearance: float = LABEL_MIN_CLEARANCE_POINTS
    colorbar_below_common_max_gap: float = COLORBAR_BELOW_COMMON_MAX_GAP_POINTS
    heatmap_to_row_header_gap: float = HEATMAP_TO_ROW_HEADER_GAP_POINTS
    row_header_to_y_label_gap: float = ROW_HEADER_TO_Y_LABEL_GAP_POINTS
    heatmap_to_summary_plot_gap: float = HEATMAP_TO_SUMMARY_PLOT_GAP_POINTS
    column_header_to_panel_gap: float = SAMPLE_LABEL_TO_PANEL_GAP_POINTS
    axis_label_to_tick_labels_gap: float = AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS
    common_axis_label_to_tick_labels_gap: float = \
        COMMON_AXIS_LABEL_TO_TICK_LABELS_GAP_POINTS
    profile_min_column_gap: float = PROFILE_MINIMUM_COLUMN_GAP_POINTS
    profile_row_gap: float = PROFILE_ROW_ADDITIONAL_GAP_POINTS
    profile_panel_to_row_header_gap: float = PROFILE_PANEL_TO_ROW_LABEL_GAP_POINTS
    tick_label_to_tick_gap: float = TICK_LABEL_TO_TICK_GAP_POINTS
    legend_border_padding: float = LEGEND_BORDER_PADDING_POINTS
    legend_handle_to_text_gap: float = LEGEND_HANDLE_TO_TEXT_GAP_POINTS
    legend_entry_row_gap: float = LEGEND_ENTRY_ROW_GAP_POINTS
    legend_column_gap: float = LEGEND_COLUMN_GAP_POINTS
    # Dimensions that participate in layout but are not gaps:
    heatmap_colorbar_thickness: float = HEATMAP_COLORBAR_THICKNESS_POINTS
    colorbar_min_height: float = COLORBAR_MIN_HEIGHT_POINTS


@dataclass(frozen=True)
class DrawingSpec:
    """Stroke/drawing properties that are not geometry gaps."""

    profile_line_width: float = PROFILE_LINE_WIDTH_POINTS


@dataclass(frozen=True)
class LabelLayoutSpec:
    """Persistent policy controls for measured label layout integrations.

    ``max_column_gap_points`` is a hard cap on the uniform common column gap
    when automatic structural-label wrapping/widening is enabled; mandatory private
    Y-decoration reservations are separate additions. If labels need more
    room, the solver leaves them overlapping rather than exceeding this limit.
    The row penalty is charged once per additional global band row and also
    contributes a normalized per-label excess-line preference; balance and
    orphan weights are weak typography preferences. The axis-label controls
    tune structural axis-label integration in heatmap and profile plots,
    including merged/non-merged X/Y labels and profile facet labels; profile
    facets and Y labels share a capped common row-gap decision while retaining
    separate visual tracks. The remaining class switches and caps are policy
    controls for horizontal colorbar title and heatmap-region-label
    integrations. Horizontal colorbar titles and heatmap region labels use
    independent line limits and typography penalties, defaulting to the same
    values as axis labels.

    ``max_row_gap_points`` caps the uniform common row gap, while
    ``max_heatmap_region_gap_points`` caps automatic expansion between
    heatmap-region rows. At either hard cap, residual overlap is preferred to
    exceeding the configured limit. Mandatory private decorations remain
    separate from the common-gap caps where the consuming layout defines them.
    Setting ``heatmap_region_label_gap_growth`` to false keeps the base group
    gaps while still choosing wrapped region labels.
    """

    auto_panel_title_column_gap: bool = True
    max_column_gap_points: float = 96.0
    panel_title_max_lines: int = DEFAULT_LABEL_MAX_LINES
    panel_title_extra_row_penalty: float = DEFAULT_LABEL_EXTRA_ROW_PENALTY
    panel_title_balance_weight: float = DEFAULT_LABEL_BALANCE_WEIGHT
    panel_title_orphan_weight: float = DEFAULT_LABEL_ORPHAN_WEIGHT
    auto_axis_label_layout: bool = True
    auto_horizontal_colorbar_label_layout: bool = True
    horizontal_colorbar_label_max_lines: int = DEFAULT_LABEL_MAX_LINES
    horizontal_colorbar_label_extra_row_penalty: float = DEFAULT_LABEL_EXTRA_ROW_PENALTY
    horizontal_colorbar_label_balance_weight: float = DEFAULT_LABEL_BALANCE_WEIGHT
    horizontal_colorbar_label_orphan_weight: float = DEFAULT_LABEL_ORPHAN_WEIGHT
    auto_facet_label_layout: bool = True
    auto_heatmap_region_label_layout: bool = True
    heatmap_region_label_gap_growth: bool = True
    heatmap_region_label_max_lines: int = DEFAULT_LABEL_MAX_LINES
    heatmap_region_label_extra_row_penalty: float = DEFAULT_LABEL_EXTRA_ROW_PENALTY
    heatmap_region_label_balance_weight: float = DEFAULT_LABEL_BALANCE_WEIGHT
    heatmap_region_label_orphan_weight: float = DEFAULT_LABEL_ORPHAN_WEIGHT
    max_row_gap_points: float = 96.0
    max_heatmap_region_gap_points: float = 96.0
    axis_label_max_lines: int = DEFAULT_LABEL_MAX_LINES
    axis_label_extra_row_penalty: float = DEFAULT_LABEL_EXTRA_ROW_PENALTY
    axis_label_balance_weight: float = DEFAULT_LABEL_BALANCE_WEIGHT
    axis_label_orphan_weight: float = DEFAULT_LABEL_ORPHAN_WEIGHT
    auto_legend_label_layout: bool = True
    legend_label_max_lines: int = DEFAULT_LABEL_MAX_LINES


@dataclass(frozen=True)
class ResolvedStyle:
    """One fully resolved style, produced once per plotting invocation."""

    font_family: tuple
    font_multiplier: float
    typography: TypographySpec
    geometry: GeometrySpec
    drawing: DrawingSpec
    label_layout: LabelLayoutSpec


# Default typography: one canonical scale shared by
# both plots on one deliberate hierarchy: the complete-figure title at 12 pt,
# panel-level titles (including column/row headers and facet labels) at 10 pt,
# and all body text -- axis labels, tick labels, colorbar text, legends -- at
# 8 pt. A role now renders at one size across both plots.
_DEFAULT_TYPOGRAPHY = {
    "figure_title_text": FontSpec(12.0, "normal"),
    "panel_title_text": FontSpec(10.0, "normal"),
    "region_label_text": FontSpec(8.0, "normal"),
    "axis_label_text": FontSpec(8.0, "normal"),
    "x_tick_label_text": FontSpec(8.0, "normal"),
    "y_tick_label_text": FontSpec(8.0, "normal"),
    "colorbar_title_text": FontSpec(8.0, "normal"),
    "colorbar_tick_label_text": FontSpec(8.0, "normal"),
    "legend_label_text": FontSpec(8.0, "normal"),
}

ROLE_NAMES = tuple(_DEFAULT_TYPOGRAPHY)

# ``sans-serif`` preserves Matplotlib's current default face; ``--fontFamily``
# prepends a concrete family (e.g. Arial) ahead of it, falling back through the
# list when that family is not installed.
DEFAULT_FONT_FAMILY = ("sans-serif",)
DEFAULT_FONT_MULTIPLIER = 1.0

DEFAULT_TYPOGRAPHY = TypographySpec(**_DEFAULT_TYPOGRAPHY)
DEFAULT_GEOMETRY = GeometrySpec()
DEFAULT_DRAWING = DrawingSpec()
DEFAULT_LABEL_LAYOUT = LabelLayoutSpec()

_GEOMETRY_FIELD_NAMES = tuple(field.name for field in fields(GeometrySpec))
# Geometry keys that describe a size rather than a gap and must be strictly
# positive; all other geometry keys are gaps and may be zero (a gap can be
# meaningfully disabled).
_POSITIVE_GEOMETRY_FIELDS = frozenset(
    {"heatmap_colorbar_thickness", "colorbar_min_height"})

_ALLOWED_WEIGHTS = frozenset({
    "ultralight", "light", "normal", "regular", "book", "medium", "roman",
    "semibold", "demibold", "demi", "bold", "heavy", "extra bold", "black"})
_ALLOWED_STYLES = frozenset({"normal", "italic", "oblique"})


def scaled_size(font_spec, multiplier):
    """Base point size scaled by ``multiplier`` for drawing."""
    return font_spec.size * float(multiplier)


def _is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _resolve_family(value, context, messages, default):
    """``default`` is the global family (a tuple) or ``None`` for a per-role
    entry that should inherit when absent or invalid."""
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return (value,)
    if (isinstance(value, (list, tuple)) and value
            and all(isinstance(item, str) and item.strip() for item in value)):
        return tuple(value)
    messages.append(
        "{}: font family must be a non-empty string or list of strings; "
        "used default".format(context))
    return default


def _resolve_positive(value, context, messages, default):
    if _is_number(value) and value > 0:
        return float(value)
    messages.append(
        "{}: must be a positive number; used default {}".format(context, default))
    return default


def _resolve_non_negative(value, context, messages, default):
    if _is_number(value) and value >= 0:
        return float(value)
    messages.append(
        "{}: must be a non-negative number; used default {}".format(
            context, default))
    return default


def _resolve_weight(value, context, messages, default):
    if isinstance(value, str) and value.strip().lower() in _ALLOWED_WEIGHTS:
        return value.strip().lower()
    if _is_number(value) and 0 <= value <= 1000:
        return int(value)
    messages.append(
        "{}: unknown font weight {!r}; used default {!r}".format(
            context, value, default))
    return default


def _resolve_font_style(value, context, messages, default):
    if isinstance(value, str) and value.strip().lower() in _ALLOWED_STYLES:
        return value.strip().lower()
    messages.append(
        "{}: unknown font style {!r}; used default {!r}".format(
            context, value, default))
    return default


def _resolve_typography(raw, messages):
    if not isinstance(raw, dict):
        messages.append("typography: expected an object; used defaults")
        raw = {}
    resolved = {}
    for role in ROLE_NAMES:
        default_spec = getattr(DEFAULT_TYPOGRAPHY, role)
        entry = raw.get(role)
        if entry is None:
            resolved[role] = default_spec
            continue
        if not isinstance(entry, dict):
            messages.append(
                "typography.{}: expected an object; used default".format(role))
            resolved[role] = default_spec
            continue
        resolved[role] = FontSpec(
            size=_resolve_positive(
                entry.get("size", default_spec.size),
                "typography.{}.size".format(role), messages, default_spec.size),
            weight=_resolve_weight(
                entry.get("weight", default_spec.weight),
                "typography.{}.weight".format(role), messages,
                default_spec.weight),
            style=_resolve_font_style(
                entry.get("style", default_spec.style),
                "typography.{}.style".format(role), messages,
                default_spec.style),
            family=_resolve_family(
                entry.get("family"), "typography.{}.family".format(role),
                messages, default_spec.family))
    for key in raw:
        if key not in _DEFAULT_TYPOGRAPHY:
            messages.append(
                "typography.{}: unknown text role; ignored".format(key))
    return TypographySpec(**resolved)


def _resolve_geometry(raw, messages):
    if not isinstance(raw, dict):
        messages.append("geometry: expected an object; used defaults")
        raw = {}
    resolved = {}
    for name in _GEOMETRY_FIELD_NAMES:
        default = getattr(DEFAULT_GEOMETRY, name)
        if name not in raw:
            resolved[name] = default
            continue
        context = "geometry.{}".format(name)
        if name in _POSITIVE_GEOMETRY_FIELDS:
            resolved[name] = _resolve_positive(
                raw[name], context, messages, default)
        else:
            resolved[name] = _resolve_non_negative(
                raw[name], context, messages, default)
    for key in raw:
        if key not in resolved:
            messages.append(
                "geometry.{}: unknown geometry setting; ignored".format(key))
    return GeometrySpec(**resolved)


def _resolve_drawing(raw, messages):
    if not isinstance(raw, dict):
        messages.append("drawing: expected an object; used defaults")
        raw = {}
    line_width = _resolve_positive(
        raw.get("profile_line_width", DEFAULT_DRAWING.profile_line_width),
        "drawing.profile_line_width", messages,
        DEFAULT_DRAWING.profile_line_width)
    for key in raw:
        if key != "profile_line_width":
            messages.append(
                "drawing.{}: unknown drawing setting; ignored".format(key))
    return DrawingSpec(profile_line_width=line_width)


def _resolve_label_layout(raw, messages):
    if not isinstance(raw, dict):
        messages.append("label_layout: expected an object; used defaults")
        raw = {}
    resolved = {}
    for field in fields(LabelLayoutSpec):
        name = field.name
        default = getattr(DEFAULT_LABEL_LAYOUT, name)
        value = raw.get(name, default)
        context = "label_layout.{}".format(name)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                messages.append(
                    "{}: expected a boolean; used default {}".format(
                        context, default))
                value = default
        elif isinstance(default, int):
            if (isinstance(value, bool) or not isinstance(value, int) or
                    value < 1):
                messages.append(
                    "{}: expected a positive integer; used default {}"
                    .format(context, default))
                value = default
        else:
            value = _resolve_non_negative(value, context, messages, default)
        resolved[name] = value
    for name in raw:
        if name not in resolved:
            messages.append(
                "label_layout.{}: unknown label-layout setting; ignored"
                .format(name))
    return LabelLayoutSpec(**resolved)


def resolve_style(raw):
    """Overlay a (possibly partial or malformed) stored style on the immutable
    defaults, validating each field independently.

    Returns ``(ResolvedStyle, messages)`` where ``messages`` is a list of
    human-readable notes about entries that could not be used; each such entry
    falls back to its own default without affecting the rest.
    """
    messages = []
    if raw is None:
        raw = {}
    elif not isinstance(raw, dict):
        messages.append("options: style section must be an object; used defaults")
        raw = {}
    resolved = ResolvedStyle(
        font_family=_resolve_family(
            raw.get("font_family", DEFAULT_FONT_FAMILY), "font_family",
            messages, DEFAULT_FONT_FAMILY),
        font_multiplier=_resolve_positive(
            raw.get("font_multiplier", DEFAULT_FONT_MULTIPLIER),
            "font_multiplier", messages, DEFAULT_FONT_MULTIPLIER),
        typography=_resolve_typography(raw.get("typography", {}), messages),
        geometry=_resolve_geometry(raw.get("geometry", {}), messages),
        drawing=_resolve_drawing(raw.get("drawing", {}), messages),
        label_layout=_resolve_label_layout(
            raw.get("label_layout", {}), messages))
    return resolved, messages


def _font_spec_to_dict(font_spec):
    data = {
        "size": font_spec.size,
        "weight": font_spec.weight,
        "style": font_spec.style,
    }
    if font_spec.family is not None:
        data["family"] = list(font_spec.family)
    return data


def default_style_dict():
    """Full expanded default schema as JSON-serialisable plain data.

    ``config.py`` writes this (deep-merged with the user's overrides) so the
    on-disk file always contains every role and gap.
    """
    return {
        "font_family": list(DEFAULT_FONT_FAMILY),
        "font_multiplier": DEFAULT_FONT_MULTIPLIER,
        "typography": {
            role: _font_spec_to_dict(getattr(DEFAULT_TYPOGRAPHY, role))
            for role in ROLE_NAMES},
        "geometry": {
            name: getattr(DEFAULT_GEOMETRY, name)
            for name in _GEOMETRY_FIELD_NAMES},
        "drawing": {"profile_line_width": DEFAULT_DRAWING.profile_line_width},
        "label_layout": {
            field.name: getattr(DEFAULT_LABEL_LAYOUT, field.name)
            for field in fields(LabelLayoutSpec)},
    }
