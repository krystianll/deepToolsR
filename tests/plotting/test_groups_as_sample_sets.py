"""--perGroup is a layout shortcut; --groupsAsSampleSets promotes group panels.

The input has samples Repeat, Repeat, Third and region groups First, Other,
Last (``write_input(distinct=True)``).
"""

import pytest

from deeptoolsr import plotMatrix
from tests.helpers.capture_series_goldens import write_input


def _render(tmp_path, *flags):
    """Run plotMatrixR and capture the plan, prepared values and figure."""
    matrix = tmp_path / 'input.gz'
    write_input(matrix, distinct=True)
    built = {}
    original = plotMatrix.build_matrix_figure

    def capture(matrix, layout, labels, plan, spec, prepared, *args, **kwargs):
        figure, solution = original(
            matrix, layout, labels, plan, spec, prepared, *args, **kwargs)
        built.update(figure=figure, plan=plan, prepared=prepared)
        return figure, solution

    argv = ['-m', str(matrix), '-o', str(tmp_path / 'plot.png'), '-p', '1',
            '--dpi', '50', '--sortRegions', 'no', *flags]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plotMatrix, 'build_matrix_figure', capture)
        plotMatrix.main(argv)
    return built


def _z_max(built):
    """The upper colour limit of every cell's heatmap stack, in cell order."""
    z_max = built['prepared'].z_max
    return [z_max[cell.sample_set % len(z_max)]
            for cell in built['plan'].cells]


SPLIT = ('Repeat — First', 'Repeat — Other', 'Repeat — Last')


@pytest.mark.parametrize('flags,titles,sample_sets', [
    ((), ('Repeat', 'Repeat', 'Third'), 3),
    (('--perGroup',), ('First', 'Other', 'Last'), 1),
    (('--groupsAsSampleSets',), ('Repeat', 'Repeat', 'Third'), 3),
    (('--groupsAsSampleSets', '--sampleSetGroupArrangement', 'adjacent'),
     SPLIT * 2 + ('Third — First', 'Third — Other', 'Third — Last'), 9),
    (('--perGroup', '--groupsAsSampleSets'), ('First', 'Other', 'Last'), 3),
    (('--arrangeSamples', '1,2', '3', '--sampleSetGroupArrangement',
      'adjacent', '--groupsAsSampleSets'),
     ('Repeat, Repeat — First', 'Repeat, Repeat — Other',
      'Repeat, Repeat — Last', 'Third — First', 'Third — Other',
      'Third — Last'), 6),
])
def test_layout_and_sample_sets(tmp_path, flags, titles, sample_sets):
    built = _render(tmp_path, '--heatmap', '--profile', *flags)
    plan, prepared = built['plan'], built['prepared']
    assert tuple(cell.title for cell in plan.cells) == titles
    assert len(prepared.sample_set_plans) == sample_sets
    assert len(plan.color_domains[-1].slots) == sample_sets  # colour scales


@pytest.mark.parametrize('flags,limits', [
    (('--perGroup',), [5, 5, 5]),
    (('--perGroup', '--groupsAsSampleSets'), [5, 9, 5]),
])
def test_per_group_scales_follow_sample_sets(tmp_path, flags, limits):
    built = _render(tmp_path, '--heatmap', '--zMax', '5', '9', *flags)
    assert _z_max(built) == limits


def test_assignments_count_promoted_sets_in_drawing_order(tmp_path):
    built = _render(
        tmp_path, '--heatmap', '--profile', '--arrangeSamples', '1,3', '2',
        '--sampleSetGroupArrangement', 'end', '--groupsAsSampleSets',
        '--zMax', '2=7', '--yAxisLabel', '6=last', '--colors', '2=red')
    plan, prepared = built['plan'], built['prepared']
    assert [cell.title for cell in plan.cells[:2]] == [
        'Repeat, Third — First', 'Repeat — First']
    assert _z_max(built)[1] == 7
    assert [limit == 7 for limit in _z_max(built)].count(True) == 1
    assert [item.y_axis_label for item in prepared.sample_set_plans] == [
        ''] * 5 + ['last']
    slot = plan.color_domains[0].slots[1]
    assert (slot.series, slot.panels) == (('g1:s3',), (1,))


def test_unpromoted_series_keep_one_slot_across_sample_sets(tmp_path):
    plan = _render(
        tmp_path, '--profile', '--arrangeSamples', '1,3', '2',
        '--sampleSetGroupArrangement', 'end')['plan']
    slot = plan.color_domains[0].slots[1]
    assert (slot.series, slot.panels) == (('g1:s3', 'g2:s3', 'g3:s3'),
                                          (1, 3, 5))


@pytest.mark.parametrize('flags,slots', [
    (('--perGroup',), 3),
    (('--perGroup', '--groupsAsSampleSets'), 9),
    # Labels merged by --sameSampleLabels together keep one colour.
    (('--perGroup', '--groupsAsSampleSets', '--sameSampleLabels',
      'together'), 2),
])
def test_series_colour_slots(tmp_path, flags, slots):
    plan = _render(tmp_path, '--profile', *flags)['plan']
    assert len(plan.color_domains[0].slots) == slots


def test_short_colour_list_errors_for_promoted_sets(tmp_path, capsys):
    _render(tmp_path, '--profile', '--perGroup', '--colors', 'red', 'blue',
            'green')
    with pytest.raises(SystemExit):
        _render(tmp_path, '--profile', '--perGroup', '--groupsAsSampleSets',
                '--colors', 'red', 'blue', 'green')
    assert '--colors needs exactly 9 colours' in capsys.readouterr().err


@pytest.mark.parametrize('flags,warned', [
    ((), True),
    (('--perGroup',), False),
    (('--sampleSetGroupArrangement', 'by_row'), False),
])
def test_overlay_warning(tmp_path, capsys, flags, warned):
    _render(tmp_path, '--profile', '--groupsAsSampleSets', *flags)
    message = ('--groupsAsSampleSets has no effect because region groups are '
               'drawn in the same cell (--sampleSetGroupArrangement overlay); '
               'use adjacent, end, by_row, by_column or --perGroup to split '
               'them')
    assert (message in capsys.readouterr().err) == warned
