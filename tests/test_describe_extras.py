"""Describe reports slot defaults, size defaults and stage progress."""

import csv
import json

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io, plotMatrix, plotProfile
from deeptoolsr.describe import describe
from deeptoolsr.plotting import colormap_names
from deeptoolsr.plotting.matrix_spec import project_matrix_spec
from deeptoolsr.prepare import parse_command
from deeptoolsr.session import STAGES, PlotSession


@pytest.fixture
def matrix(tmp_path):
    """Three region groups and two samples."""
    path = tmp_path / 'small.gz'
    header = {
        'sample_labels': ['Sample A', 'Sample B'],
        'sample_boundaries': [0, 2, 4],
        'group_labels': ['group1', 'group2', 'group3'],
        'group_boundaries': [0, 2, 4, 6],
        'bin size': [1, 1], 'upstream': [1, 1], 'body': [0, 0],
        'downstream': [1, 1], 'unscaled 5 prime': [0, 0],
        'unscaled 3 prime': [0, 0], 'ref point': ['TSS', 'TSS'],
        'min threshold': None, 'max threshold': None, 'sort regions': 'no',
        'sort using': 'mean', 'proc number': 1}
    regions = [['chr1', [(i * 20, i * 20 + 10)], f'region{i}', i, '+', '0']
               for i in range(6)]
    values = np.arange(6 * 4, dtype=np.float32).reshape(6, 4) / 10 + 1
    _compute_matrix_io.write_matrix(str(path), json.dumps(header), regions,
                                    values, 1)
    return str(path)


def _slots(record, option):
    domain = next(item for item in record['color_domains']
                  if item['option'] == option)
    return [(slot['label'], slot['automatic']) for slot in domain['slots']]


def test_series_slots_are_named_by_their_legend_labels(matrix):
    record = describe('plotMatrixR', ['-m', matrix, '--profile'])
    labels = {item['key']: item['label'] for item in record['series']}
    domain = record['color_domains'][0]
    assert domain['kind'] == 'series'
    assert [slot['label'] for slot in domain['slots']] == [
        labels[slot['series'][0]] for slot in domain['slots']]
    assert [slot['label'] for slot in domain['slots']] == [
        'group1', 'group2', 'group3']


def _drawn_line_colours(matrix, tmp_path, monkeypatch, *options):
    """Legend label -> hex colour of each profile line the plot draws."""
    from matplotlib.colors import to_hex
    from matplotlib.figure import Figure

    drawn = {}

    def capture(figure, *args, **kwargs):
        for axes in figure.axes:
            for line in axes.get_lines():
                if not line.get_label().startswith('_'):
                    drawn[line.get_label()] = to_hex(line.get_color())
    monkeypatch.setattr(Figure, 'savefig', capture)
    plotProfile.main(['-m', matrix, '-o', str(tmp_path / 'p.png'),
                      '--dpi', '30', *options])
    return drawn


@pytest.mark.parametrize('options', [[], ['--perGroup']])
def test_series_slot_automatic_is_the_drawn_colour(matrix, tmp_path,
                                                   monkeypatch, options):
    record = describe('plotProfileR', ['-m', matrix, *options])
    slots = _slots(record, '--colorsPerSample')
    assert all(automatic.startswith('#') for _, automatic in slots)
    assert dict(slots) == _drawn_line_colours(matrix, tmp_path, monkeypatch,
                                              *options)


def test_describe_without_series_colours_loads_no_matplotlib(matrix):
    import subprocess
    import sys
    code = ('import sys; from deeptoolsr.describe import describe; '
            f'describe("plotMatrixR", ["-m", {matrix!r}, "--profile", '
            '"--plotType", "heatmap"]); '
            'print("matplotlib" in sys.modules)')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=60, check=True)
    assert result.stdout.strip() == 'False'


def test_per_group_series_slots_are_named_by_sample(matrix):
    record = describe('plotMatrixR', ['-m', matrix, '--profile', '--perGroup'])
    assert [label for label, _ in _slots(record, '--colorsPerSample')] == [
        'Sample A', 'Sample B']


def test_colormap_slots_carry_their_default_maps(matrix):
    record = describe('plotMatrixR', ['-m', matrix, '--profile',
                                      '--plotType', 'heatmap'])
    assert _slots(record, '--colorsPerSample') == [
        ('Sample A', 'RdYlBu_r'), ('Sample B', 'RdYlBu_r')]
    record = describe('plotMatrixR', ['-m', matrix, '--heatmap',
                                      '--groupsAsSampleSets'])
    titles = [cell['title'] for cell in record['cells']]
    assert _slots(record, '--colorMap') == [
        (title, 'RdYlBu') for title in titles]


@pytest.mark.parametrize('tool, options', [
    ('plotMatrixR', ['--profile']),
    ('plotMatrixR', ['--heatmap']),
    ('plotMatrixR', ['--profile', '--heatmap']),
    ('plotProfileR', []),
    ('plotHeatmapR', []),
])
def test_size_defaults_are_what_the_plot_uses(matrix, tool, options):
    argv = ['-m', matrix, *options]
    record = describe(tool, argv)
    spec = project_matrix_spec(parse_command(tool, argv, 'describe'), tool)
    solved = {'cellWidth': spec.cell_width,
              'profileHeight': spec.profile_height,
              'profileAspectRatio': spec.profile_aspect_ratio,
              'heatmapHeight': spec.heatmap_height,
              'heatmapAspectRatio': spec.heatmap_aspect_ratio}
    from deeptoolsr.plotting.matrix_plan import tool_option_spellings
    spellings = tool_option_spellings(tool)
    expected = [{'option': spellings[dest], 'value': value, 'derived': None}
                for dest, value in solved.items()
                if value is not None and dest in spellings]
    assert record['size_defaults'] == expected


def test_size_defaults_name_the_options_they_follow(matrix):
    record = describe('plotMatrixR', ['-m', matrix, '--profile',
                                      '--cellWidth', '8'])
    assert record['size_defaults'][1:] == [
        {'option': option, 'value': None, 'derived': 'from --cellWidth'}
        for option in ('--plotHeight', '--profileAspectRatio')]


def test_colormap_names_is_public_and_accepts_the_defaults():
    names = colormap_names()
    assert names == sorted(names)
    assert {'RdYlBu', 'RdYlBu_r', 'viridis'} <= set(names)


def _table_labels(matrix, tmp_path, *options):
    table = tmp_path / 'table.tsv'
    plotProfile.main(['-m', matrix, '-o', str(tmp_path / 'p.png'), '--dpi',
                      '30', '--outFileNameData', str(table), *options])
    with open(table, newline='') as handle:
        return list(dict.fromkeys(
            row['series_label']
            for row in csv.DictReader(handle, delimiter='\t')))


def test_regions_label_assignment_renames_one_group(matrix, tmp_path):
    record = describe('plotProfileR', ['-m', matrix, '--regionsLabel',
                                       '2=GENE1'])
    assert [group['label'] for group in record['groups']] == [
        'group1', 'GENE1', 'group3']
    assert record['problems'] == []
    assert _table_labels(matrix, tmp_path, '--regionsLabel', '2=GENE1') == [
        'group1', 'GENE1', 'group3']
    option = next(item for item in record['options']
                  if item['option'] == '--regionsLabel')
    assert (option['counts'], option['recycles']) == ('region group', False)


def test_regions_label_counts_groups_before_arranging(matrix, tmp_path):
    options = ['--regionsLabel', '1=GENE1', '--arrangeGroups', '3', '1']
    record = describe('plotProfileR', ['-m', matrix, *options])
    assert [group['label'] for group in record['groups']] == [
        'group3', 'GENE1']
    assert _table_labels(matrix, tmp_path, *options) == ['group3', 'GENE1']


def test_regions_label_out_of_range_is_a_problem_and_a_plot_error(
        matrix, tmp_path, capsys):
    record = describe('plotMatrixR', ['-m', matrix, '--profile',
                                      '--regionsLabel', '4=GENE1'])
    message = ('--regionsLabel: 4 names region group 4, but there are only '
               '3 region groups')
    assert [(problem['code'], problem['option'], problem['target'],
             problem['slot'], problem['message'])
            for problem in record['problems']] == [
        ('assignment_out_of_range', '--regionsLabel', 'region group', 4,
         message)]
    assert [group['label'] for group in record['groups']] == [
        'group1', 'group2', 'group3']
    with pytest.raises(SystemExit) as error:
        plotMatrix.main(['-m', matrix, '--profile', '--regionsLabel',
                         '4=GENE1', '-o', str(tmp_path / 'plot.png')])
    assert error.value.code != 0
    assert message in capsys.readouterr().err


def test_plain_regions_label_list_must_name_every_group(matrix):
    record = describe('plotProfileR', ['-m', matrix, '--regionsLabel',
                                       'GENE1', 'GENE2'])
    assert [problem['code'] for problem in record['problems']] == [
        'label_count']


def test_describe_reports_progress_in_stage_order(matrix, capsys):
    stages = []
    with PlotSession(cache_bytes=0) as session:
        session.describe(['plotMatrixR', '-m', matrix, '--heatmap',
                          '--kmeans', '2'], force_load=True,
                         progress=stages.append)
    assert stages[:3] == ['parse', 'load', 'order']
    assert 'limits' in stages
    assert stages == sorted(stages, key=STAGES.index)
    assert len(set(stages)) == len(stages)
    assert capsys.readouterr() == ('', '')
