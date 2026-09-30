"""The pure resolver and drawn plotters agree with the pinned series goldens."""

import json
from pathlib import Path
import subprocess
import sys
from dataclasses import replace
from unittest.mock import patch

import pytest
from matplotlib.colors import to_hex
import numpy as np

from deeptoolsr import plotHeatmap, plotProfile, plotMatrix
from deeptoolsr.matrix import Matrix, MatrixHeader
from deeptoolsr.prepare import (
    heatmap_spec, prepare, profile_spec, resolve_prepared_labels)
from deeptoolsr.plotting.series import (
    Cell, LabelKind, StackBlock, apply_grouped_y_limits, build_sample_set_plans,
    label_runs, parse_sample_sets, resolve_cells, series_slot, cell_options)
from deeptoolsr.plotting.matrix_spec import project_matrix_spec
from tests.helpers.capture_series_goldens import (
    HERE, _rendered, cases, regenerate_newly_supported_case, write_input)


CASES = {name: case for name, case in cases().items() if not case.rejected}


def test_new_heatmap_series_golden_regenerates(tmp_path):
    index = (HERE / 'index.json').read_bytes()
    (tmp_path / 'index.json').write_bytes(index)
    regenerate_newly_supported_case(tmp_path)
    name = 'reject_heatmap_per_group_sets.json'
    assert (tmp_path / name).read_bytes() == (HERE / name).read_bytes()
    assert (tmp_path / 'index.json').read_bytes() == index


def _inputs(tmp_path, case):
    matrix_path = tmp_path / 'input.gz'
    write_input(matrix_path, distinct=case.source == 'distinct')
    argv = ['-m', str(matrix_path), '-o', str(tmp_path / 'plot.png'),
            '-p', '1', '--dpi', '72', '--sortRegions', 'no', *case.options]
    module = plotProfile if case.tool == 'profile' else plotHeatmap
    args = module.process_args(argv)
    matrix = Matrix.load(str(matrix_path), 1)
    spec = profile_spec(args, matrix.header) if case.tool == 'profile' else heatmap_spec(args, matrix.header)
    layout, _ = prepare(matrix, spec, 1)
    labels = resolve_prepared_labels(
        layout, spec, regions_label=args.regionsLabel,
        samples_label=args.samplesLabel, show_counts=args.showRegionCounts,
        header=matrix.header)
    sample_sets = parse_sample_sets(
        labels.samples, args.arrangeSamples, per_group=args.perGroup)
    plans = build_sample_set_plans(
        labels.samples, matrix.header.sample_boundaries,
        matrix.header.parameters, sample_sets,
        subplot_labels=args.sampleSetLabels,
        x_axis_labels=args.xAxisLabel,
        y_axis_labels=args.yAxisLabel,
        reference_labels=args.refPointLabel,
        start_labels=args.startLabel, end_labels=args.endLabel)
    sizes = tuple(b - a for a, b in zip(layout.group_bounds,
                                        layout.group_bounds[1:]))
    return module, argv, args, layout, labels, plans, sizes


@pytest.mark.parametrize('name', CASES)
def test_resolver_matches_semantic_golden(name, tmp_path):
    case = CASES[name]
    _module, _argv, args, _layout, labels, samples, sizes = _inputs(tmp_path, case)
    golden = json.loads((HERE / (name + '.json')).read_text())['semantic']
    opts = cell_options(project_matrix_spec(
        args, 'plotProfileR' if case.tool == 'profile' else 'plotHeatmapR'))
    plan = resolve_cells(sizes, samples, labels, opts)
    by_key = {series.key: series for series in plan.series}
    actual = []
    for panel in plan.panels:
        assert panel.index == len(actual) + 1
        record = {'row': panel.row, 'column': panel.column,
                  'kind': panel.kind, 'title': panel.title,
                  'export_label': panel.export_label, 'series': []}
        for key in panel.series:
            series = by_key[key]
            if case.tool == 'profile' and opts.profile_style == 'heatmap':
                # The historical line index is captured although this mode
                # actually addresses colormaps by panel, not by line.
                color_index = series.group
            else:
                color_index = series_slot(plan, 0, panel.index, key) - 1
            record['series'].append({
                'group': series.group, 'sample': series.sample,
                'label': series.label, 'color_index': color_index})
        actual.append(record)
    assert actual == golden['panels']
    assert list(plan.row_labels) == golden['row_labels']
    expected_slots = [{} for _ in plan.color_domains]
    for panel_index, panel in enumerate(golden['panels'], 1):
        if case.tool == 'profile':
            domain = 0
            for item in panel['series']:
                position = (panel_index if opts.profile_style == 'heatmap'
                            else item['color_index'] + 1)
                key = f"g{item['group'] + 1}:s{item['sample'] + 1}"
                keys, panels = expected_slots[domain].setdefault(
                    position, ([], []))
                if key not in keys:
                    keys.append(key)
                if panel_index not in panels:
                    panels.append(panel_index)
        else:
            # Heatmap tools expose only the profile panels above the stacks.
            for item in panel['series']:
                position = item['color_index'] + 1
                key = f"g{item['group'] + 1}:s{item['sample'] + 1}"
                keys, panels = expected_slots[0].setdefault(
                    position, ([], []))
                if key not in keys:
                    keys.append(key)
                if panel_index not in panels:
                    panels.append(panel_index)
    domains = plan.color_domains
    if case.tool == 'heatmap':
        # One colormap slot per sample set (group set with --perGroup),
        # naming the cells whose stacks use it.
        scales = {}
        for cell in plan.cells:
            scale = cell.groups if opts.per_group else cell.sample_set
            scales.setdefault(scale, []).append(cell.index)
        assert [(slot.position, slot.series, slot.panels, slot.cells)
                for slot in domains[-1].slots] == [
                    (position, (), (), tuple(cells))
                    for position, cells in enumerate(scales.values(), 1)]
        domains = domains[:-1]
        assert expected_slots.pop() == {}
    for domain, positions in zip(domains, expected_slots, strict=True):
        assert [(slot.position, slot.series, slot.panels)
                for slot in domain.slots] == [
                    (position, tuple(keys), tuple(panels))
                    for position, (keys, panels) in sorted(positions.items())]
    assert all(domain.option in {action.option_strings[0]
               for action in (plotProfile if case.tool == 'profile'
                              else plotHeatmap).parse_arguments()._actions
               if action.option_strings}
               for domain in plan.color_domains)
    if case.tool == 'profile' and args.colors:
        assert ([str(color) for color in args.colors]
                if opts.profile_style == 'heatmap' else
                [to_hex(color, keep_alpha=True) for color in args.colors]
                ) == golden['color_list'][:len(args.colors)]
    if case.tool == 'heatmap':
        assert list(args.colorList or args.colorMap) == golden['color_lists']['heatmap']
        if args.colors:
            assert [to_hex(color, keep_alpha=True)
                    for color in args.colors] == (
                        golden['color_lists']['summary'][:len(args.colors)])


@pytest.mark.parametrize('name', CASES)
def test_plotter_matches_rendered_golden(name, tmp_path):
    case = CASES[name]
    module, argv, *_ = _inputs(tmp_path, case)
    golden = json.loads((HERE / (name + '.json')).read_text())['rendered']
    captured = {}

    def save(figure, path, **_kwargs):
        if case.tool == 'profile':
            axes = [axis for axis in figure.axes if axis.images]
            if case.options[:2] != ('--plotType', 'heatmap'):
                axes = [axis for axis in figure.axes if axis.lines]
        else:
            axes = [axis for axis in figure.axes if
                    (axis.get_gid() or '').endswith('/profile')]
        captured['rendered'] = _rendered(
            figure, axes, colorbars=case.tool == 'heatmap')
        Path(path).write_bytes(b'')

    with patch.object(plotMatrix, 'save_figure_atomic', save):
        module.main(argv)
    assert captured['rendered'] == golden


def test_series_import_without_matplotlib():
    code = ("import sys; sys.modules['matplotlib'] = None; "
            "import deeptoolsr.plotting.series")
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_hidden_summary_colours_are_inert(tmp_path):
    case = replace(cases()['heatmap_no_summary'], options=(
        '--whatToShow', 'heatmap and colorbar', '--colorsSummaryPlot', 'red'))
    _module, argv, *_ = _inputs(tmp_path, case)
    plotHeatmap.main(argv)


def test_too_few_summary_colours_is_an_option_error(tmp_path, capsys):
    case = replace(cases()['heatmap_summary'], options=(
        '--colorsSummaryPlot', 'red'))
    _module, argv, *_ = _inputs(tmp_path, case)
    with pytest.raises(SystemExit) as error:
        plotHeatmap.main(argv)
    assert error.value.code == 2
    assert ('--colorsSummaryPlot needs exactly 3 colours, one for each '
            'distinctly coloured series; 1 given') in capsys.readouterr().err


@pytest.mark.parametrize('tool', ('profile', 'heatmap'))
def test_relabel_changes_series_plan_without_repreparing(tool, tmp_path):
    if tool == 'profile':
        case = cases()['profile_relabel_equal']
    else:
        case = replace(cases()['heatmap_summary'], source='distinct',
                       options=cases()['heatmap_summary'].options +
                       ('--regionsLabel', 'Equal', 'Other', 'Equal'))
    _module, _argv, args, layout, labels, samples, sizes = _inputs(tmp_path, case)
    matrix = Matrix.load(str(tmp_path / 'input.gz'), 1)
    spec = profile_spec(args, matrix.header) if tool == 'profile' else heatmap_spec(args, matrix.header)
    distinct = resolve_prepared_labels(
        layout, spec, regions_label=('First', 'Other', 'Last'),
        header=matrix.header)
    opts = cell_options(project_matrix_spec(
        args, 'plotProfileR' if tool == 'profile' else 'plotHeatmapR'))
    before = layout.digest()
    assert resolve_cells(sizes, samples, labels, opts) != resolve_cells(
        sizes, samples, distinct, opts)
    assert layout.digest() == before


def test_stack_order_and_cell_colours(tmp_path):
    case = replace(cases()['heatmap_summary'], options=(
        '--arrangeSamples', '1,2', '3', '--showRegionCounts'))
    _module, _argv, args, _layout, labels, samples, sizes = _inputs(
        tmp_path, case)
    opts = cell_options(project_matrix_spec(args, 'plotHeatmapR'))
    plan = resolve_cells(sizes, samples, labels, opts)
    assert [(block.group, block.sample) for block in plan.cells[0].blocks] == [
        (group, sample) for sample in (0, 1) for group in range(3)]
    # Samples and groups both vary within the stack, so blocks name both.
    assert plan.cells[0].stack_labels == tuple(
        f'{labels.samples[sample]}\n{labels.groups[group]}'
        for sample in (0, 1) for group in range(3))
    # A single-sample stack only varies by group.
    assert plan.cells[1].stack_labels == tuple(labels.groups[:3])
    assert [slot.position for slot in plan.color_domains[-1].slots] == [1, 2]
    owners = [entry[-1] for entry in sorted(
        (block.sample, block.group, cell.index, rank, cell)
        for cell in plan.cells for rank, block in enumerate(cell.blocks))]
    for panel, owner in zip(
            (item for item in plan.panels if item.kind == 'heatmap'), owners):
        assert series_slot(plan, -1, panel.index) == owner.scale + 1
    assert [(item.key, item.label) for item in plan.series[:3]] == [
        ('g1:s1', 'Equal: Repeat'), ('g1:s2', 'Equal: Repeat'),
        ('g2:s1', 'Other: Repeat')]
    assert labels.groups[:2] == ('Equal [n = 2]', 'Other [n = 3]')

    grouped = replace(opts, per_group=True, placement='adjacent')
    grouped_plan = resolve_cells(sizes, samples, labels, grouped)
    assert [(block.group, block.sample)
            for block in grouped_plan.cells[0].blocks] == [(0, 0), (0, 1)]
    assert grouped_plan.cells[0].stack_labels == (
        labels.samples[0], labels.samples[1])
    assert len(grouped_plan.cells) == 6


def test_profile_heatmap_cell_kind_keeps_legacy_panel_projection(tmp_path):
    case = cases()['profile_heatmap_colormaps']
    _module, _argv, args, _layout, labels, samples, sizes = _inputs(
        tmp_path, case)
    opts = cell_options(project_matrix_spec(args, 'plotProfileR'))
    plan = resolve_cells(sizes, samples, labels, opts)
    assert all(cell.profile.kind == 'series_heatmap' for cell in plan.cells)
    assert all(panel.kind == 'heatmap' for panel in plan.panels)
    assert all(cell.blocks == () for cell in plan.cells)


def test_label_runs_split_equal_vectors_by_row_and_column():
    def cell(index, row, column, label):
        return Cell(index, row, column, (0,), (0,), label, None, None,
                    (StackBlock(0, 0),), (label,))

    cells = (cell(1, 0, 0, 'A'), cell(2, 0, 1, 'A'),
             cell(3, 0, 2, 'B'), cell(4, 1, 0, 'A'),
             cell(5, 1, 1, 'B'), cell(6, 1, 2, 'B'))
    assert [(run.axis, run.labels, run.cells)
            for run in label_runs(cells, 'stack')] == [
        (0, ('A',), (1, 2)), (0, ('B',), (3,)),
        (1, ('A',), (4,)), (1, ('B',), (5, 6))]
    assert label_runs(cells, LabelKind(
        'left', lambda cell: cell.stack_labels)) == label_runs(cells, 'stack')
    assert [(run.axis, run.cells) for run in label_runs(
            cells, LabelKind('top', lambda cell: cell.title))] == [
        (0, (1, 4)), (1, (2,)), (1, (5,)), (2, (3, 6))]
    assert label_runs(cells, LabelKind(
        'bottom', lambda cell: cell.title)) == label_runs(
            cells, LabelKind('top', lambda cell: cell.title))
    alternating = tuple(replace(item, stack_labels=(label,)) for item, label
                        in zip(cells[:3], ('A', 'B', 'A')))
    assert [run.cells for run in label_runs(alternating, 'stack')] == [
        (1,), (2,), (3,)]
    separated = (cells[0], replace(cells[1], column=2))
    assert [run.cells for run in label_runs(separated, 'stack')] == [
        (1,), (2,)]


def _matrix_and_parameters():
    data = np.arange(2 * 9, dtype=float).reshape(2, 9)
    matrix = Matrix(MatrixHeader.from_parameters({
        'group_boundaries': [0, 2], 'sample_boundaries': [0, 2, 5, 9],
        'group_labels': ['genes'],
        'sample_labels': ['ref1', 'scaled', 'ref2']}), data,
        [['chr1', [(0, 1)], 'a', 0, '+', '0'],
         ['chr1', [(1, 2)], 'b', 0, '+', '0']], None)
    parameters = {
        'ref point': ['TSS', None, 'TES'],
        'upstream': [1000, 1000, 1000],
        'downstream': [1000, 1000, 1000],
        'body': [0, 1000, 0],
        'bin size': [1000, 1000, 500],
        'unscaled 5 prime': [0, 0, 0],
        'unscaled 3 prime': [0, 0, 0],
    }
    return matrix, parameters


def test_reference_labels_recycle_only_over_applicable_sample_sets():
    matrix, parameters = _matrix_and_parameters()
    plans = build_sample_set_plans(
        matrix.header.sample_labels, matrix.header.sample_boundaries, parameters, [(0,), (1,), (2,)],
        reference_labels=['first', 'second'],
        start_labels=['start'], end_labels=['end'])
    assert [plan.reference_label for plan in plans] == [
        'first', None, 'second']
    assert plans[1].start_label == 'start'
    assert plans[1].end_label == 'end'


def test_axis_and_subplot_labels_recycle_per_sample_set():
    matrix, parameters = _matrix_and_parameters()
    plans = build_sample_set_plans(
        matrix.header.sample_labels, matrix.header.sample_boundaries, parameters, [(0,), (1,), (2,)],
        subplot_labels=['A', 'B'], x_axis_labels=['x'],
        y_axis_labels=['RNA', 'ChIP'])
    assert [plan.subplot_label for plan in plans] == ['A', 'B', 'A']
    assert [plan.x_axis_label for plan in plans] == ['x', 'x', 'x']
    assert [plan.y_axis_label for plan in plans] == ['RNA', 'ChIP', 'RNA']


def test_different_geometry_is_allowed_between_but_not_within_sets():
    matrix, parameters = _matrix_and_parameters()
    plans = build_sample_set_plans(matrix.header.sample_labels, matrix.header.sample_boundaries, parameters, [(0,), (1,), (2,)])
    assert [plan.geometry.bin_count for plan in plans] == [2, 3, 4]
    with pytest.raises(ValueError, match='X-axis geometries differ'):
        build_sample_set_plans(matrix.header.sample_labels, matrix.header.sample_boundaries, parameters, [(0, 1), (2,)])


def test_y_limits_group_by_subplot():
    class Axis:
        def __init__(self, limits):
            self.limits = limits

        def get_ylim(self):
            return self.limits

        def set_ylim(self, limits):
            self.limits = tuple(limits)

        def axhline(self, *args, **kwargs):
            pass

    axes = [Axis((0, 1)), Axis((-1, 2)), Axis((10, 20))]
    limits = apply_grouped_y_limits(
        axes, [0, 0, 1], ['RNA', 'RNA', 'ChIP'],
        [None], [None], 'per_subplot')
    assert limits == ((-1, 2), (-1, 2), (10, 20))
