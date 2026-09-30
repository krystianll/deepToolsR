# Known plotting-layout issues

These are current limitations of the automatic label layout in
`plotMatrixR`, `plotHeatmapR` and `plotProfileR`, verified for 0.9.0. The
layout wraps labels and widens the gaps between cells up to fixed maximums.
When a label still does not fit at the maximum gap, it is drawn anyway and
can overlap its neighbour; nothing is truncated or hidden.

1. **Long cell titles can overlap in grid arrangements.** With
   `--arrangeSamples` placing two or more samples side by side, very long
   titles (about 70 characters per sample, joined per cell) wrap to three
   lines and still overlap the neighbouring cell's title. With `--profile
   --heatmap` the first cell's title can also overlap the Y-axis label
   (`--yAxisLabel`) at the top-left corner, because title and Y-label
   placement are not checked against each other.
   Other combinations checked with long labels drew no overlapping text:
   long X and Y labels, long `--sampleSetLabels` next to Y labels, long
   `--heatmapYAxisLabel`, and single-column `plotHeatmapR` layouts.
   *Workaround:* shorter `--samplesLabel` text, or a wider cell
   (`--cellWidth`, e.g. 9).
2. **Long region labels in multi-block stacks.** When a heatmap stack holds
   several blocks (e.g. `--profile --heatmap --arrangeSamples 1,2 3,4`) and
   each `--regionsLabel` entry is longer than its block is tall, the
   rotated region labels overlap one another and can run past the figure
   edge; the gap between blocks cannot grow enough to hold them.
   *Workaround:* shorter region labels, or a taller stack (`--heatmapHeight`,
   e.g. 30).
3. **Y-label wrapping is decided before horizontal layout.** Y-axis and
   row labels are wrapped first, then the columns and the title/X-label
   rows are laid out, in a single pass. Y and row labels are not re-wrapped
   afterwards, so they may use more lines than the final figure strictly
   needs. The result is deterministic. *Workaround:* none needed; use
   shorter labels if the wrapping is unwanted.
4. **Very large label sets are not specially optimised.** Each label row
   or column is solved by one search over its wrapping choices. A
   16-column row takes about 15 ms and a full 16×16 grid about 50 ms, but
   unusually many long, multi-word labels can make layout noticeably
   slower. *Workaround:* shorter labels reduce the number of wrapping
   choices.
