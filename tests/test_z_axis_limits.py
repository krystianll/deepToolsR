"""--zAxisLimits: the values automatic heatmap colour limits come from."""

import json

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io, plotHeatmap, plotMatrix
from deeptoolsr.describe import describe
from deeptoolsr.session import PlotSession

ROWS, BINS, GROUP_ROWS = 40, 10, 20


def _values():
    """Sample B is 100 times sample A; the second group 3 times the first."""
    values = np.random.default_rng(5).random((ROWS, 2 * BINS)).astype(
        np.float32)
    values[:, BINS:] *= 100
    values[GROUP_ROWS:] *= 3
    return values


@pytest.fixture
def matrix(tmp_path):
    path = tmp_path / 'matrix.gz'
    header = {
        'sample_labels': ['Sample A', 'Sample B'],
        'sample_boundaries': [0, BINS, 2 * BINS],
        'group_labels': ['GENE1', 'GENE2'],
        'group_boundaries': [0, GROUP_ROWS, ROWS],
        'bin size': [10, 10], 'upstream': [50, 50], 'body': [0, 0],
        'downstream': [50, 50], 'unscaled 5 prime': [0, 0],
        'unscaled 3 prime': [0, 0], 'ref point': ['TSS', 'TSS'],
        'min threshold': None, 'max threshold': None, 'sort regions': 'no',
        'sort using': 'mean', 'proc number': 1}
    regions = [['chr1', [(i * 100, i * 100 + 50)], f'region{i}', i, '+', '0']
               for i in range(ROWS)]
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header), regions, _values(), 1)
    return str(path)


def _argv(matrix, *extra):
    return ['-m', matrix, '--sortRegions', 'no', '--dpi', '50', *extra]


def _plot(monkeypatch, tmp_path, argv, name='plot.png'):
    """Plot; return the plan, the prepared numbers it drew, the output
    and the file rows of each drawn group."""
    seen = {}
    original = plotMatrix.build_matrix_figure

    def capture(*args, **kwargs):
        layout = args[1]
        seen['rows'] = [layout.row_indices()[start:stop] for start, stop in
                        zip(layout.group_bounds, layout.group_bounds[1:])]
        seen['plan'], seen['prepared'] = args[3], args[5]
        return original(*args, **kwargs)
    monkeypatch.setattr(plotMatrix, 'build_matrix_figure', capture)
    output = tmp_path / name
    plotHeatmap.main([*argv, '-o', str(output)])
    return seen['plan'], seen['prepared'], output, seen['rows']


def _block_limits(plan, prepared):
    """Each drawn heatmap block's limits, keyed (cell index, block)."""
    return {(cell.index - 1, block): (prepared.z_min[slot],
                                      prepared.z_max[slot])
            for cell in plan.cells
            for block, slot in enumerate(prepared.z_slots[cell.index - 1])}


def _percentiles(blocks, group_rows):
    """Exact 1st and 98th percentiles over the given heatmap blocks."""
    values = _values()
    drawn = np.zeros(values.shape, dtype=bool)
    for block in blocks:
        drawn[np.asarray(group_rows[block.group], dtype=np.int64),
              block.sample * BINS:(block.sample + 1) * BINS] = True
    return np.nanpercentile(values[drawn], [1.0, 98.0])


def _scope(plan, cell, block, mode):
    """The blocks whose values give a block's automatic limits."""
    if mode == 'per_heatmap':
        return [cell.blocks[block]]
    cells = ([cell] if mode == 'per_panel' else
             [other for other in plan.cells
              if mode == 'common' or other.sample_set == cell.sample_set])
    return [item for other in cells for item in other.blocks]


MODES = ['common', 'per_sample_set', 'per_panel', 'per_heatmap']
LAYOUTS = [(), ('--perGroup',), ('--sampleSetGroupArrangement', 'adjacent'),
           ('--perGroup', '--groupsAsSampleSets'),
           ('--arrangeGroups', '2', '--sampleSetGroupArrangement', 'end')]


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('mode', MODES)
def test_automatic_limits_are_the_scope_percentiles(
        matrix, tmp_path, monkeypatch, mode, layout):
    plan, prepared, _, rows = _plot(monkeypatch, tmp_path, _argv(
        matrix, '--zAxisLimits', mode, *layout))
    for (index, block), limits in _block_limits(plan, prepared).items():
        np.testing.assert_allclose(
            limits, _percentiles(_scope(plan, plan.cells[index], block, mode),
                                 rows), rtol=0, atol=1e-4)


@pytest.mark.parametrize('layout', [(), ('--perGroup',)])
def test_per_heatmap_gives_every_block_its_own_scale(
        matrix, tmp_path, monkeypatch, layout):
    plan, prepared, _, _ = _plot(monkeypatch, tmp_path, _argv(
        matrix, '--zAxisLimits', 'per_heatmap', *layout))
    limits = _block_limits(plan, prepared)
    assert len(limits) == len(set(limits.values())) == 4
    # Sample A's small values are no longer squeezed by sample B's scale.
    assert all(high < 10 for (index, block), (_, high) in limits.items()
               if plan.cells[index].blocks[block].sample == 0)


def test_per_panel_limits_differ_between_samples(
        matrix, tmp_path, monkeypatch):
    plan, prepared, _, _ = _plot(monkeypatch, tmp_path, _argv(
        matrix, '--zAxisLimits', 'per_panel'))
    limits = _block_limits(plan, prepared)
    assert limits[0, 0] == limits[0, 1] and limits[1, 0] == limits[1, 1]
    assert limits[0, 0][1] < 10 < limits[1, 0][1]


@pytest.mark.parametrize('mode', ['per_panel', 'per_heatmap'])
def test_typed_limits_win_over_scoped_automatic_ones(
        matrix, tmp_path, monkeypatch, mode):
    plan, prepared, _, rows = _plot(monkeypatch, tmp_path, _argv(
        matrix, '--zAxisLimits', mode,
        '--sampleSetGroupArrangement', 'adjacent',
        '--zMin', '1=-5', '--zMax', '2=500'))
    for (index, block), (low, high) in _block_limits(plan, prepared).items():
        cell = plan.cells[index]
        auto = _percentiles(_scope(plan, cell, block, mode), rows)
        assert low == (-5.0 if cell.sample_set == 0 else
                       pytest.approx(auto[0], abs=1e-4))
        assert high == (500.0 if cell.sample_set == 1 else
                        pytest.approx(auto[1], abs=1e-4))


def test_midpoint_is_checked_against_scoped_limits(matrix, tmp_path, capsys):
    argv = _argv(matrix, '--zMid', '50', '-o', str(tmp_path / 'plot.png'))
    plotHeatmap.main(argv)
    with pytest.raises(SystemExit):
        plotHeatmap.main([*argv, '--zAxisLimits', 'per_sample_set'])
    assert '--zMid: 50 must be strictly between zMin' in \
        capsys.readouterr().err


# Per-block bars cannot sit below a stack (see the test after this one).
@pytest.mark.parametrize('mode, location', [
    (mode, location) for mode in MODES for location in ('best', 'right', 'below')
    if (mode, location) != ('per_heatmap', 'below')])
@pytest.mark.parametrize('layout', LAYOUTS)
def test_bars_split_where_limits_differ(matrix, tmp_path, monkeypatch, mode,
                                        layout, location):
    argv = _argv(matrix, '--zAxisLimits', mode, '--colorbarLocation',
                 location, *layout)
    plan, prepared, _, _ = _plot(monkeypatch, tmp_path, argv)
    limits = _block_limits(plan, prepared)
    record = describe('plotHeatmapR', argv, force_load=True)
    served = [[(cell, block) for cell in bar.cells
               for block in ((bar.block,) if bar.block is not None else
                             range(len(plan.cells[cell].blocks)))]
              for bar in prepared.colorbars.bars]
    assert sorted(item for blocks in served for item in blocks) == \
        sorted(limits)
    assert len(record['colorbars']) == len(served)
    for bar, blocks in zip(record['colorbars'], served):
        assert bar['cells'] == sorted({cell + 1 for cell, _ in blocks})
        assert len({limits[item] for item in blocks}) == 1
        assert bar['z_range']['value'] == list(limits[blocks[0]])
        assert bar['z_range']['auto'] == list(limits[blocks[0]])
    if mode == 'per_heatmap':
        assert all(len(blocks) == 1 for blocks in served)
    if any(bar.block is not None for bar in prepared.colorbars.bars):
        # One bar per block sits to the right of its block.
        assert {bar['position'] for bar in record['colorbars']} == {'side'}


@pytest.mark.parametrize('height', [None, '25'])
@pytest.mark.parametrize('layout', [(), ('--perGroup',)])
def test_per_heatmap_bars_sit_in_their_block_row_in_column_order(
        matrix, tmp_path, monkeypatch, layout, height):
    seen = {}
    original = plotMatrix.build_matrix_figure

    def capture(*args, **kwargs):
        seen['plan'], seen['prepared'] = args[3], args[5]
        seen['figure'], solution = original(*args, **kwargs)
        return seen['figure'], solution
    monkeypatch.setattr(plotMatrix, 'build_matrix_figure', capture)
    sizes = ('--heatmapHeight', height) if height else ()
    plotHeatmap.main(_argv(matrix, '--zAxisLimits', 'per_heatmap', *layout,
                           *sizes, '-o', str(tmp_path / 'plot.png')))
    plan = seen['plan']
    boxes = {axis.get_gid(): axis.get_position()
             for axis in seen['figure'].axes if axis.get_gid()}
    blocks = [box for gid, box in boxes.items() if '/block/' in gid]
    rows = {}
    for ordinal, bar in enumerate(seen['prepared'].colorbars.bars):
        cell = plan.cells[bar.cells[0]]
        role = f'colorbar/row/{cell.row + 1}/{ordinal + 1}'
        box, block = boxes[role], boxes[
            f'cell/{cell.row + 1}/{cell.column + 1}/block/{bar.block + 1}']
        title = boxes[role.replace('colorbar/', 'colorbar_title/', 1)]
        assert box.x0 > max(item.x1 for item in blocks)
        # The bar and its title stay inside their block row.
        assert block.y0 - 1e-6 <= box.y0 < box.y1 <= block.y1 + 1e-6
        rows.setdefault(bar.block, []).append((cell.column, box, title))
    assert sorted(rows) == [0, 1]
    for bars in rows.values():
        assert len(bars) == 2
        (_, first, title), (_, second, below) = sorted(
            bars, key=lambda item: item[0])
        assert title.y0 >= first.y1 - 1e-6 and below.y0 >= second.y1 - 1e-6
        # Column order: top to bottom when the row is tall enough for both
        # bars and titles, else side by side.
        if height:
            assert first.x0 == pytest.approx(second.x0)
            assert below.y1 <= first.y0 + 1e-6
        else:
            assert first.x1 < second.x0


@pytest.mark.parametrize('layout, titles', [
    ((), ['Sample A\nGENE1', 'Sample A\nGENE2', 'Sample B\nGENE1',
          'Sample B\nGENE2']),
    (('--perGroup',), ['GENE1\nSample A', 'GENE1\nSample B',
                       'GENE2\nSample A', 'GENE2\nSample B']),
    # Nested headers join a block's levels (sample, group) with a space.
    (('--arrangeSamples', '1,2', '--nestedHeaders'), [
        'Sample A, Sample B\nSample A GENE1',
        'Sample A, Sample B\nSample A GENE2',
        'Sample A, Sample B\nSample B GENE1',
        'Sample A, Sample B\nSample B GENE2'])])
def test_per_heatmap_bars_are_titled_by_default(matrix, layout, titles):
    argv = ['-m', matrix, '--zAxisLimits', 'per_heatmap', *layout]
    record = describe('plotHeatmapR', argv)
    assert [bar['title']['value'] for bar in record['colorbars']] == titles
    assert [bar['title']['auto'] for bar in record['colorbars']] == titles
    given = describe('plotHeatmapR', [*argv, '--colorbarLabels', 'one',
                                      'two', 'three', 'four'])
    assert [bar['title']['value'] for bar in given['colorbars']] == [
        'one', 'two', 'three', 'four']
    hidden = describe('plotHeatmapR', [*argv, '--colorbarLabels', 'none'])
    assert all(bar['title']['value'] is None
               for bar in hidden['colorbars'])
    common = describe('plotHeatmapR', argv[:2])
    assert all(bar['title']['value'] is None
               for bar in common['colorbars'])


def test_scoped_limits_bars_keep_their_slot_settings(matrix):
    record = describe('plotHeatmapR', ['-m', matrix, '--zAxisLimits',
                                       'per_panel', '--perGroup'])
    assert [(bar['cells'], bar['scale']['z_min']) for bar in
            record['colorbars']] == [([1], 'auto'), ([2], 'auto')]
    assert all(bar['z_range'] is None for bar in record['colorbars'])
    assert record['values_read'] is False


def test_typed_limits_share_bars_in_every_mode(matrix):
    bars = [describe('plotHeatmapR', ['-m', matrix, '--zMin', '0', '--zMax',
                                      '9', '--zAxisLimits', mode,
                                      '--perGroup'])['colorbars']
            for mode in MODES]
    assert all(bar == bars[0] for bar in bars)
    assert [bar['cells'] for bar in bars[0]] == [[1, 2]]


def test_describe_reports_each_bar_scope(matrix):
    record = describe('plotHeatmapR', [
        '-m', matrix, '--sortRegions', 'no', '--zAxisLimits', 'per_panel',
        '--zMax', '2=500'], force_load=True)
    first, second = record['colorbars']
    assert second['z_range']['value'][1] == 500.0
    assert second['z_range']['auto'][1] != 500.0
    assert second['z_range']['auto'][0] == second['z_range']['value'][0]
    assert first['z_range']['auto'] == first['z_range']['value']
    assert first['z_range']['auto'][1] < second['z_range']['auto'][1]


@pytest.mark.parametrize('location', ['right_common', 'below_common'])
@pytest.mark.parametrize('mode', ['per_sample_set', 'per_panel',
                                  'per_heatmap'])
def test_common_bar_locations_reject_scoped_limits(
        matrix, tmp_path, capsys, mode, location):
    argv = ['-m', matrix, '--zAxisLimits', mode, '--colorbarLocation',
            location]
    (problem,) = describe('plotHeatmapR', argv)['problems']
    assert (problem['code'], problem['option']) == (
        'colorbar_limits_not_common', '--zAxisLimits')
    with pytest.raises(SystemExit):
        plotHeatmap.main([*argv, '-o', str(tmp_path / 'plot.png')])
    assert '--zAxisLimits common' in capsys.readouterr().err
    # Typed limits need no automatic ones, and one set has one scope.
    plotHeatmap.main([*argv, '--zMin', '0', '--zMax', '9', '-o',
                      str(tmp_path / 'typed.png')])
    assert describe('plotHeatmapR', [*argv, '--perGroup',
                                     '--zAxisLimits', 'per_sample_set'])[
        'problems'] == []


@pytest.mark.parametrize('extra', [(), ('--perGroup', '--zMid', '1'),
                                   ('--colorbarLocation', 'below_common')])
def test_common_is_the_default_drawing(matrix, tmp_path, monkeypatch, extra):
    _, _, default, _ = _plot(monkeypatch, tmp_path, _argv(matrix, *extra),
                             'default.png')
    _, _, common, _ = _plot(monkeypatch, tmp_path, _argv(
        matrix, *extra, '--zAxisLimits', 'common'), 'common.png')
    assert default.read_bytes() == common.read_bytes()


@pytest.mark.parametrize('mode', MODES[1:])
def test_typed_limits_draw_alike_in_every_mode(matrix, tmp_path, monkeypatch,
                                               mode):
    typed = ('--zMin', '0', '1', '--zMax', '2', '300')
    _, _, default, _ = _plot(monkeypatch, tmp_path, _argv(matrix, *typed),
                             'default.png')
    _, _, scoped, _ = _plot(monkeypatch, tmp_path, _argv(
        matrix, *typed, '--zAxisLimits', mode), 'scoped.png')
    assert default.read_bytes() == scoped.read_bytes()


@pytest.mark.parametrize('location', ['below', 'bottom'])
def test_per_heatmap_bars_cannot_sit_below_a_stack(matrix, tmp_path, capsys,
                                                   location):
    argv = ['-m', matrix, '--zAxisLimits', 'per_heatmap',
            '--colorbarLocation', location]
    (problem,) = describe('plotHeatmapR', argv)['problems']
    assert problem['code'] == 'colorbar_limits_not_common'
    with pytest.raises(SystemExit):
        plotHeatmap.main([*argv, '-o', str(tmp_path / 'plot.png')])
    assert '--colorbarLocation right or best' in capsys.readouterr().err
    # One block per panel needs no more than one bar per panel.
    assert describe('plotHeatmapR', [*argv, '--arrangeGroups', '1'])[
        'problems'] == []


def test_scoped_scans_are_session_cached(matrix, tmp_path):
    command = ['plotHeatmapR', *_argv(matrix, '--zAxisLimits', 'per_panel',
                                      '--perGroup')]
    with PlotSession() as session:
        first = session.run([*command, '-o', str(tmp_path / 'one.png')])
        again = session.run([*command, '-o', str(tmp_path / 'two.png')])
        common = session.run([*command, '--zAxisLimits', 'common',
                              '-o', str(tmp_path / 'three.png')])
    assert 'scan' in first.recomputed
    assert 'scan' in again.reused and 'scan' not in again.recomputed
    assert 'scan' in common.recomputed


@pytest.mark.parametrize('tool', [plotHeatmap, plotMatrix])
def test_per_heatmap_bars_that_cannot_fit_a_short_heatmap_are_a_usage_error(
        matrix, tmp_path, capsys, tool):
    argv = ['-m', matrix, '--dpi', '50', '--zAxisLimits', 'per_heatmap',
            '--arrangeSamples', '1,2']
    if tool is plotMatrix:
        argv.append('--heatmap')
    short, normal = tmp_path / 'short.png', tmp_path / 'normal.png'
    with pytest.raises(SystemExit) as stop:
        tool.main([*argv, '--heatmapHeight', '3', '-o', str(short)])
    assert stop.value.code == 2
    assert '--heatmapHeight' in capsys.readouterr().err
    assert not short.exists()
    # The same command at a normal height still draws.
    tool.main([*argv, '--heatmapHeight', '10', '-o', str(normal)])
    assert normal.stat().st_size > 0
