"""X-axis tick marks, tick numbers and labels across stacked panels.

Every panel that closes a stack keeps its X tick marks (major and minor);
only the owner of a run shows tick numbers and the distance label.  The
input has samples Repeat, Repeat, Third and region groups First, Other,
Last (``write_input(distinct=True)``).
"""

import pytest

from deeptoolsr import plotMatrix
from tests.helpers.capture_series_goldens import write_input


def _render(tmp_path, *flags):
    matrix = tmp_path / 'input.gz'
    write_input(matrix, distinct=True)
    built = {}
    original = plotMatrix.build_matrix_figure

    def capture(*args, **kwargs):
        figure, solution = original(*args, **kwargs)
        built['figure'] = figure
        return figure, solution

    argv = ['-m', str(matrix), '-o', str(tmp_path / 'plot.png'), '-p', '1',
            '--dpi', '50', '--sortRegions', 'no', '--minorTickMarks', '4',
            *flags]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plotMatrix, 'build_matrix_figure', capture)
        plotMatrix.main(argv)
    axes = {axis.get_gid(): axis for axis in built['figure'].axes
            if axis.get_gid()}
    axes[None] = built['figure']
    return axes


def _x_state(axis):
    """(marks, minor marks, numbers) drawn on an axis' X axis.

    Distance labels are placed as structural figure texts after solving,
    so their slots are checked by the baseline suites, not here.
    """
    if not axis.get_xaxis().get_visible():
        return (False, False, False)
    major = [tick for tick in axis.xaxis.get_major_ticks()
             if tick.tick1line.get_visible()]
    minor = [tick for tick in axis.xaxis.get_minor_ticks()
             if tick.tick1line.get_visible()]
    numbers = any(tick.label1.get_visible() and tick.label1.get_text()
                  for tick in axis.xaxis.get_major_ticks())
    return (bool(major and len(axis.get_xticks())), bool(minor),
            bool(numbers))


def _closing(axes):
    """The axis closing each cell: its last heatmap block or its profile."""
    closing = {}
    for gid, axis in axes.items():
        if gid is None:
            continue
        parts = gid.split('/')
        if parts[0] != 'cell':
            continue
        cell = (int(parts[1]), int(parts[2]))
        rank = int(parts[4]) if parts[3] == 'block' else -1
        if cell not in closing or rank > closing[cell][0]:
            closing[cell] = (rank, axis)
    return {cell: axis for cell, (_rank, axis) in sorted(closing.items())}


OWNER = (True, True, True)
DEFERRING = (True, True, False)


@pytest.mark.parametrize('content', [('--profile',), ('--heatmap',),
                                     ('--profile', '--heatmap')])
def test_default_labels_every_panel(tmp_path, content):
    axes = _render(tmp_path, *content, '--perGroup', '--gridRows', '3')
    states = set(map(_x_state, _closing(axes).values()))
    assert states == {OWNER}


@pytest.mark.parametrize('content', [('--profile',), ('--heatmap',),
                                     ('--profile', '--heatmap')])
def test_outer_rows_with_decorations_between_share_one_axis(
        tmp_path, content):
    axes = _render(tmp_path, *content, '--perGroup', '--gridRows', '3',
                   '--xAxisVisibility', 'outer')
    closing = _closing(axes)
    rows = sorted({row for row, _column in closing})
    assert len(rows) == 3
    for (row, _column), axis in closing.items():
        assert _x_state(axis) == (OWNER if row == rows[-1] else DEFERRING)


def test_outer_wrapped_grid_runs_per_column(tmp_path):
    axes = _render(tmp_path, '--profile', '--perGroup',
                   '--groupsAsSampleSets', '--gridColumns', '2',
                   '--xAxisVisibility', 'outer')
    closing = _closing(axes)
    states = {cell: _x_state(axis) for cell, axis in closing.items()}
    assert states == {(1, 1): DEFERRING, (1, 2): OWNER, (2, 1): OWNER}


def test_outer_incompatible_distance_labels_split_runs(tmp_path):
    axes = _render(tmp_path, '--profile', '--heatmap', '--perGroup',
                   '--groupsAsSampleSets', '--gridRows', '3',
                   '--xAxisVisibility', 'outer',
                   '--xAxisLabel', '1=near', '2=far', '3=far')
    closing = _closing(axes)
    assert len(closing) == 3
    states = [_x_state(axis) for axis in closing.values()]
    assert states == [OWNER, DEFERRING, OWNER]


def _boxes(figure):
    """Tick numbers, distance labels and colorbar/legend boxes in pixels."""
    renderer = figure.canvas.get_renderer()
    figure.draw(renderer)
    numbers = [label.get_window_extent(renderer)
               for axis in figure.axes
               if (axis.get_gid() or '').startswith('cell/')
               for label in axis.get_xticklabels()
               if label.get_visible() and label.get_text()]
    labels = [text.get_window_extent(renderer)
              for axis in figure.axes
              if (axis.get_gid() or '').startswith('label/x/')
              for text in axis.texts if text.get_text()]
    labels += [axis.xaxis.label.get_window_extent(renderer)
               for axis in figure.axes
               if (axis.get_gid() or '').endswith('/profile') and
               axis.get_xlabel()]
    obstacles = [axis.get_window_extent(renderer) for axis in figure.axes
                 if (axis.get_gid() or '').startswith('colorbar/')]
    obstacles += [legend.get_window_extent(renderer)
                  for legend in figure.legends]
    return numbers, labels, obstacles


@pytest.mark.parametrize('location', ['best', 'below'])
@pytest.mark.parametrize('arrangement', [
    ('--perGroup', '--gridRows', '3'),
    ('--perGroup', '--groupsAsSampleSets', '--gridColumns', '2'),
])
def test_show_all_rows_reserve_their_ticks_and_labels(
        tmp_path, location, arrangement):
    axes = _render(tmp_path, '--profile', '--heatmap', *arrangement,
                   '--colorbarLocation', location)
    closing = _closing(axes)
    numbers, labels, obstacles = _boxes(axes[None])
    assert obstacles and numbers
    for box in numbers + labels:
        for obstacle in obstacles:
            assert not box.overlaps(obstacle)
    # Every heatmap stack has its own label; the profile above a stack
    # leaves its tick numbers and label to the stack.
    assert len(labels) == len(closing)
    for gid, axis in axes.items():
        if gid and gid.endswith('/profile'):
            assert _x_state(axis) == DEFERRING
