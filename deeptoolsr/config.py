"""Persistent user options for deepToolsR."""

import json
import os
from pathlib import Path


DEFAULTS = {
    "raster_filter": "cubic",
    "raster_bit_depth": 8,
}

RASTER_FILTERS = ("triangle", "mitchell", "catmullrom", "cubic", "box", "point")

# Top-level keys owned by the plotting style schema (typography/geometry/
# drawing/label layout). They live in the same options.txt but are validated
# and consumed separately from raster settings. Their defaults come from
# ``plotting.geometry`` (the single source of truth). ``_invalid`` is
# a reserved annotation array the writer fills and every reader ignores.
STYLE_KEYS = ("font_family", "font_multiplier", "typography", "geometry",
              "drawing", "label_layout")
INVALID_KEY = "_invalid"


def options_path():
    override = os.environ.get("DEEPTOOLSR_CONFIG_DIR")
    root = Path(override).expanduser() if override else Path.home() / ".config" / "deeptoolsr"
    return root / "options.txt"


def load_options(path=None, *, stored=None):
    options = DEFAULTS.copy()
    if stored is None:
        stored = read_stored(path)
    if isinstance(stored, dict):
        if stored.get("raster_filter") in RASTER_FILTERS:
            options["raster_filter"] = stored["raster_filter"]
        if stored.get("raster_bit_depth") in (8, 16):
            options["raster_bit_depth"] = stored["raster_bit_depth"]
        processors = stored.get("number_of_processors")
        if ((isinstance(processors, int) and not isinstance(processors, bool)
             and processors >= 1) or processors == 'max'):
            options["number_of_processors"] = processors
        budget = stored.get('ward_distance_budget_bytes')
        if (isinstance(budget, int) and not isinstance(budget, bool)
                and 1 <= budget <= (1 << 64) - 1):
            options['ward_distance_budget_bytes'] = budget
    return options


def save_options(options, path=None):
    path = Path(path) if path is not None else options_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(options, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
    return path


def _import_geometry():
    """Import the style schema lazily.

    ``plotting.geometry`` lives in a package whose ``__init__`` pulls in
    Matplotlib, so it must never be imported on the native compute/I-O hot path
    (``load_options``). Only the plotting-time and
    ``deeptoolsr options`` code paths below touch it.
    """
    from deeptoolsr.plotting import geometry
    return geometry


def read_stored(path=None):
    """Return the raw parsed options dict, or ``None`` if absent/unreadable."""
    path = Path(path) if path is not None else options_path()
    try:
        with path.open("r", encoding="utf-8") as handle:
            stored = json.load(handle)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    return stored if isinstance(stored, dict) else None


def _deep_merge(base, override):
    """Overlay ``override`` on ``base`` without losing the expanded structure.

    Nested objects are merged recursively; a leaf in ``override`` replaces the
    base leaf (so a user's field-level value, even an invalid one, is
    preserved). A non-object replacing a base object is also preserved verbatim
    so the validator can report it and the writer cannot silently destroy the
    user's malformed section. Keys present only in ``override`` (including
    unknown ones) are carried through.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict):
            if isinstance(value, dict):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def default_style_dict():
    """Full expanded default style schema (JSON-serialisable)."""
    return _import_geometry().default_style_dict()


def default_font_family():
    return _import_geometry().DEFAULT_FONT_FAMILY


def default_font_multiplier():
    return _import_geometry().DEFAULT_FONT_MULTIPLIER


def _style_portion(options):
    portion = {key: options[key] for key in STYLE_KEYS if key in options}
    # These briefly shipped as future-facing settings although R does not
    # expose legend titles. Ignore them silently in older options files.
    if isinstance(portion.get("typography"), dict):
        portion["typography"] = dict(portion["typography"])
        portion["typography"].pop("legend_title_text", None)
    if isinstance(portion.get("geometry"), dict):
        portion["geometry"] = dict(portion["geometry"])
        portion["geometry"].pop("legend_title_to_entries_gap", None)
    return portion


def resolve_style(options):
    """Return ``(ResolvedStyle, messages)`` for the supplied options."""
    geometry = _import_geometry()
    return geometry.resolve_style(_style_portion(options))


def build_persisted_options(native, font_family=None, font_multiplier=None,
                            path=None):
    """Assemble the full expanded schema to write from the ``options`` command.

    Starts from the complete defaults, overlays whatever is already on disk
    (preserving the user's valid overrides, field-level typos, and unknown
    keys), applies the requested native and font changes, and records any
    unusable style entries under ``_invalid``. This is the only writer.
    """
    stored = {key: value for key, value in (read_stored(path) or {}).items()
              if key not in (INVALID_KEY, 'native_io', 'native_compute')}
    if isinstance(stored.get("typography"), dict):
        stored["typography"].pop("legend_title_text", None)
    if isinstance(stored.get("geometry"), dict):
        stored["geometry"].pop("legend_title_to_entries_gap", None)
    merged = _deep_merge({**DEFAULTS, **default_style_dict()}, stored)
    for key in (*DEFAULTS, 'number_of_processors'):
        if key in native:
            merged[key] = native[key]
    if native.get('number_of_processors') == 'auto':
        merged.pop('number_of_processors', None)
    if font_family is not None:
        merged["font_family"] = font_family
    if font_multiplier is not None:
        merged["font_multiplier"] = font_multiplier
    _, messages = resolve_style(merged)
    if messages:
        merged[INVALID_KEY] = messages
    else:
        merged.pop(INVALID_KEY, None)
    return merged
