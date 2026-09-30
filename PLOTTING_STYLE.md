# Plot typography & geometry options

`plotMatrixR`, `plotHeatmapR` and `plotProfileR` read their fonts and
physical spacing from a persistent JSON file, `options.txt`. The look of your
figures is therefore deterministic and shared between the tools.

- **Location:** `~/.config/deeptoolsr/options.txt`, or
  `$DEEPTOOLSR_CONFIG_DIR/options.txt`. Every command also accepts
  `--config PATH` to use another file, and `--config auto` for this
  default. `deeptoolsr options --config PATH` writes that file instead.
- **Identity:** the file is identified by its content. A `deeptoolsr serve`
  worker rereads it on every request, so edits take effect on the next
  preview without a restart.
- **Who writes it:** only `deeptoolsr options`. Plotting and compute commands
  never write it — they read it, fall back to defaults for anything missing or
  invalid, and warn on stderr.

## Command-line controls

Two common settings have flags; everything else is edited in the file.

```bash
deeptoolsr options --fontFamily Arial Helvetica sans-serif
deeptoolsr options --fontMultiplier 1.25
```

- `--fontFamily` sets the global family (with fallbacks). A concrete face such
  as Arial is used only if installed; otherwise the next fallback is tried.
- `--fontMultiplier` scales **every** font size at draw time. It is stored as a
  separate number and never rewritten into the per-role sizes, so it cannot
  compound.
- Both accept `default` to restore their default (`--fontFamily default`).

Running `deeptoolsr options` (even with no flags) rewrites the full file, so
it is the way to (re)materialise it.

## The file

`options.txt` holds the **complete** schema — every text role and every gap —
so it is self-documenting. The style sections are:

- `font_family` (list) and `font_multiplier` (number).
- `typography`: one entry per text role, each `{ "size", "weight", "style" }`
  and an optional per-role `"family"` that overrides the global one. Sizes are
  base points (the multiplier is applied at draw time). Roles:
  `figure_title_text` (12 pt), `panel_title_text` (10 pt — also the sample /
  structural row / facet headers), `region_label_text` (heatmap side-region
  labels), `axis_label_text` (X and Y share this), the tick and colorbar text,
  and the legend roles — all 8 pt by default.
- `label_layout`: measured wrapping and gap limits for structural labels.
  `auto_panel_title_column_gap`, `auto_axis_label_layout`, and
  `auto_facet_label_layout` enable automatic layout for panel/sample titles,
  axis labels, and profile facet labels respectively. Axis-label layout also
  wraps heatmap Y labels against the heatmap-to-summary gap when a summary is
  shown; without a summary it balances wrapping against the nominal outer
  allowance without inventing a gap, then reserves directional overhang.
  Disable a switch to keep that label class's literal layout. Heatmap titles
  and X labels share one column-gap decision;
  profile titles and X labels do too. Profile Y and facet labels share a
  row-gap decision, while incoming-column Y decorations may add private width
  to their own column boundary.

  `max_column_gap_points` and `max_row_gap_points` default to 96 pt and
  are hard limits on the uniform common gap. If labels still do not fit,
  they may overlap; the figure is not widened without limit. Explicit
  newlines are preserved even when they exceed a `*_max_lines` setting
  (default 3). Each class also has `*_extra_row_penalty` (0.08),
  `*_balance_weight` (0.02), and `*_orphan_weight` (0.05): these trade
  extra lines against gap width and prefer balanced wraps without a very
  short last line. Neighbouring labels that both spill into a shared gap
  keep at least `label_min_clearance` (a `geometry` field, 4 pt) between
  them. Line breaks fall between words; text inside square brackets, such
  as a region count `[n = 1,234]` or a unit `[treated/control]`, breaks
  only as a last resort.

  `auto_legend_label_layout` wraps legend labels when a legend is wider
  than its space even in one column: a panel legend fits its panel, a row,
  column or common legend its span. Labels break only between words, into
  at most `legend_label_max_lines` (3) lines; an unbreakable name may still
  overflow. Legends first use as many columns as fit, and wrap only past
  one column.

  `auto_horizontal_colorbar_label_layout` enables measured wrapping for
  horizontal colorbar titles. Ordinary `below` titles join heatmap titles and
  X labels in their one capped common-column-gap decision. `below_common`
  titles wrap and pack inside the final heatmap-grid width without widening
  that grid. Disabling the switch, or using side colorbars, preserves literal
  title layout. The colorbar-specific controls default to
  `horizontal_colorbar_label_max_lines` 3,
  `horizontal_colorbar_label_extra_row_penalty` 0.08,
  `horizontal_colorbar_label_balance_weight` 0.02, and
  `horizontal_colorbar_label_orphan_weight` 0.05.

  `auto_heatmap_region_label_layout` enables measured wrapping for rotated
  heatmap region labels. Wrapped labels share one capped gap between rows;
  unequal data-row heights stay fixed when that gap grows. The region-specific
  defaults are `heatmap_region_label_max_lines` 3,
  `heatmap_region_label_extra_row_penalty` 0.08,
  `heatmap_region_label_balance_weight` 0.02, and
  `heatmap_region_label_orphan_weight` 0.05. A one-row heatmap has no gap to
  widen, so wrapping trades off against outer overhang and side-band width.
  Unequal region-label widths align toward the plot-facing edge of their
  shared side band.

  `heatmap_region_label_gap_growth` (default `true`) lets that wrapping widen
  the gaps between heatmap blocks, so a stack can grow taller than
  `--heatmapHeight`. Set it to `false` for strict heights: every gap stays
  `heatmap_group_gap`, the blocks and gaps sum to exactly the requested
  height, and region labels still wrap within their rows.
  `max_heatmap_region_gap_points` is a hard cap; if text still does not fit,
  overlap is preferred to exceeding the cap. Direct collisions between
  different label tracks (for example X versus Y labels) remain deferred.

- `geometry`: physical spacing in points, including figure padding and gaps
  between titles, legends, heatmaps, axes, ticks, and colorbars; it also holds
  colorbar size limits such as thickness and minimum height.
- `drawing`: stroke settings separate from layout geometry. `profile_line_width`
  controls the profile-line width (1.5 pt by default).

### Which gap controls which boundary

Every boundary of the figure grid uses exactly one `geometry` field. Where
the text on a boundary is measured, the gap is
`max(field, measured extent + clearance)`. Where labels are wrapped
automatically, the `label_layout` `max_*` fields cap the gap.

| Boundary | `geometry` field |
|---|---|
| canvas edge | `figure_edge_padding` |
| figure title ↔ content | `figure_title_to_content_gap` |
| cell title ↔ panel | `column_header_to_panel_gap` |
| profile ↔ heatmap stack (in a cell) | `heatmap_to_summary_plot_gap` (measured; grows for the profile's X decorations) |
| block ↔ block (in a stack, for groups and stacked samples) | `heatmap_group_gap` |
| stack ↔ sort indicator | `sort_indicator_gap` |
| stack ↔ region labels | `heatmap_to_row_header_gap` |
| region labels ↔ heatmap Y label | `row_header_to_y_label_gap` |
| heatmap ↔ colorbar | `heatmap_colorbar_gap`; inside a colorbar area, the `colorbar_*` fields |
| column ↔ column (minimum, then measured) | `heatmap_min_column_gap` if either column has a heatmap, else `profile_min_column_gap` |
| grid row ↔ grid row | `profile_row_gap`, added to the measured extents |
| panel ↔ row label | `profile_panel_to_row_header_gap` |
| axis label ↔ tick labels | `axis_label_to_tick_labels_gap`; common (merged) labels `common_axis_label_to_tick_labels_gap` |
| external legend ↔ content | `external_legend_to_content_gap`; right-hand legends `right_legend_to_content_gap` |
| tick label ↔ tick | `tick_label_to_tick_gap` (the Matplotlib tick pad) |
| stack ↔ heatmap decorations (ticks) | `heatmap_decoration_gap` |

The default `label_layout` section is:

```json
{
  "auto_panel_title_column_gap": true,
  "max_column_gap_points": 96.0,
  "panel_title_max_lines": 3,
  "panel_title_extra_row_penalty": 0.08,
  "panel_title_balance_weight": 0.02,
  "panel_title_orphan_weight": 0.05,
  "auto_axis_label_layout": true,
  "axis_label_max_lines": 3,
  "axis_label_extra_row_penalty": 0.08,
  "axis_label_balance_weight": 0.02,
  "axis_label_orphan_weight": 0.05,
  "auto_horizontal_colorbar_label_layout": true,
  "horizontal_colorbar_label_max_lines": 3,
  "horizontal_colorbar_label_extra_row_penalty": 0.08,
  "horizontal_colorbar_label_balance_weight": 0.02,
  "horizontal_colorbar_label_orphan_weight": 0.05,
  "auto_facet_label_layout": true,
  "auto_heatmap_region_label_layout": true,
  "heatmap_region_label_gap_growth": true,
  "heatmap_region_label_max_lines": 3,
  "heatmap_region_label_extra_row_penalty": 0.08,
  "heatmap_region_label_balance_weight": 0.02,
  "heatmap_region_label_orphan_weight": 0.05,
  "max_row_gap_points": 96.0,
  "max_heatmap_region_gap_points": 96.0,
  "auto_legend_label_layout": true,
  "legend_label_max_lines": 3
}
```

## Restoring defaults

- **One setting:** delete that key (or whole section) from the file. The next
  load falls back to its default, and the next `deeptoolsr options` run
  re-materialises it at the default value.
- **A font family or the multiplier:** `deeptoolsr options --fontFamily default`
  / `--fontMultiplier default`.
- **Everything:** delete `options.txt` (or its `DEEPTOOLSR_CONFIG_DIR`), then
  run `deeptoolsr options` to write a fresh default file.

## Invalid entries

A single bad value never invalidates the file. The reader keeps your original
value, uses the default for that one field in memory, and records the reason in
a reserved top-level `"_invalid"` array (also printed to stderr). Fix the value
and the note disappears on the next run; `"_invalid"` is ignored when reading.
