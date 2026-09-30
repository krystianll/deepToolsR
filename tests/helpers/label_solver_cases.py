"""Deterministic shared-gap solver fixtures.

The special cases pin the R1 pairwise term, a secondary-width trade-off,
and labels with mandatory newlines. The sized cases exercise two bands of
column and merged-span labels under physical gap caps.
"""

from deeptoolsr.plotting.text_layout import (  # noqa: F401 (re-exported for label_selection_capture)
    LabelCandidate, SharedGapBandSpec, optimize_shared_gap_layout)


def _candidate(text, width, height, secondary=0.0, balance=0.0):
    lines = tuple(text.split('\n'))
    return LabelCandidate(lines, width, height, len(lines), balance, 0.0), secondary


def sized_case(columns):
    """Return two explicit label bands and the solver's physical envelope."""
    top_sets, bottom_sets, top_costs, bottom_costs = [], [], [], []
    for index in range(columns):
        choices = (
            _candidate(f'column {index} description', 76.0 + index % 3,
                       10.0, 0.04),
            _candidate(f'column {index}\ndescription', 51.0 + index % 3,
                       20.0, 0.02, 0.1),
            _candidate(f'column\n{index}\ndescription', 34.0 + index % 3,
                       30.0, 0.01, 0.2))
        top_sets.append(tuple(item[0] for item in choices))
        top_costs.append(tuple(item[1] for item in choices))
        choices = (
            _candidate(f'sample {index} title', 70.0 + index % 2,
                       10.0, 0.03),
            _candidate(f'sample {index}\ntitle', 48.0 + index % 2,
                       20.0, 0.02, 0.1),
            _candidate(f'sample\n{index}\ntitle', 31.0 + index % 2,
                       30.0, 0.01, 0.2))
        bottom_sets.append(tuple(item[0] for item in choices))
        bottom_costs.append(tuple(item[1] for item in choices))
    spans = tuple((index, index) for index in range(columns))
    bands = (
        SharedGapBandSpec(tuple(top_sets), spans,
                          secondary_costs=tuple(top_costs)),
        SharedGapBandSpec(tuple(bottom_sets), spans,
                          secondary_costs=tuple(bottom_costs)))
    private = tuple(2.0 if index % 3 == 0 else 0.0
                    for index in range(columns - 1))
    return bands, dict(
        cell_extent=50.0, base_common_gap=8.0,
        base_private_gaps=private,
        original_block_extent=(50.0 * columns + 8.0 * (columns - 1) +
                               sum(private)),
        max_common_gap_points=52.0, outer_left_allowance=5.0,
        outer_right_allowance=5.0)


def r1_case():
    """Wide/narrow two-line pair with zero residual at the capped gap."""
    wide, _ = _candidate('wide\nlabel', 25.0, 20.0)
    narrow, _ = _candidate('narrow\nlabel', 15.0, 20.0)
    band = SharedGapBandSpec(((wide, narrow), (wide, narrow)),
                             ((0, 0), (1, 1)),
                             secondary_costs=((0.0, 0.1), (0.2, 0.0)))
    return (band,), dict(
        cell_extent=10.0, base_common_gap=10.0,
        base_private_gaps=(10.0,), original_block_extent=40.0,
        max_common_gap_points=10.0, outer_left_allowance=10.0,
        outer_right_allowance=10.0)


def secondary_width_case():
    wide, _ = _candidate('wide', 30.0, 10.0)
    narrow, _ = _candidate('narrow', 18.0, 10.0)
    band = SharedGapBandSpec(((wide, narrow),), ((0, 0),),
                             secondary_costs=((0.2, 0.0),))
    return (band,), dict(
        cell_extent=40.0, base_common_gap=8.0,
        base_private_gaps=(), original_block_extent=40.0,
        max_common_gap_points=8.0)


def mandatory_newline_case():
    fixed, _ = _candidate('north\nsouth', 24.0, 20.0)
    compact, _ = _candidate('east', 18.0, 10.0)
    band = SharedGapBandSpec(((fixed,), (compact,)),
                             ((0, 0), (1, 1)))
    return (band,), dict(
        cell_extent=30.0, base_common_gap=8.0,
        base_private_gaps=(2.0,), original_block_extent=70.0,
        max_common_gap_points=30.0)


def cases():
    result = {f'columns_{count}': sized_case(count)
              for count in (4, 8, 12, 16)}
    result.update(r1=r1_case(), secondary_width=secondary_width_case(),
                  mandatory_newline=mandatory_newline_case())
    return result


def grid_case(size=4):
    """A size×size grid of cells with a profile and a two-block stack."""
    from deeptoolsr.plotting import grid
    from deeptoolsr.plotting.series import StackBlock

    def choices(text):
        return tuple(_candidate(label, width, height)[0]
                     for label, width, height in (
                         (text, 90.0, 10.0),
                         (text.replace(' ', '\n', 1), 55.0, 20.0)))

    cells = tuple(
        grid.CellLayout(
            row, column, grid.Size(120.0, 60.0), (38.0, 118.0), 120.0,
            grid.CellInsets(),
            grid.CellLabels(
                title_text=f'sample {row}-{column} title',
                title_candidates=choices(f'sample {row}-{column} title'),
                title_order=row * size + column,
                y_candidates=choices('heatmap rows')),
            blocks=(StackBlock(0, column), StackBlock(1, column)))
        for row in range(size) for column in range(size))
    runs = tuple(
        grid.LabelRun('x', (row * size + column,),
                      choices('distance from TSS'))
        for row in range(size) for column in range(size))
    runs += tuple(
        grid.LabelRun('stack', (row * size + size - 1,),
                      choices(f'group {block} regions'),
                      block_index=block, location='right')
        for row in range(size) for block in range(2))
    colorbar = grid.Decoration('colorbar', 'figure', -1, 'right',
                               tick_label_width=20.0)
    return grid.GridLayoutSpec(size, size, cells, (colorbar,), runs)
