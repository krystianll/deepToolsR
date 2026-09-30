"""Describe is a renderer-free view of the same resolved plot series."""

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from deeptoolsr.prepare import (
    OptionError, parse_command)
from deeptoolsr.describe import describe
from deeptoolsr.parserCommon import plot_parser
from tests.helpers.capture_series_goldens import cases, write_input
from tests.helpers.generate_describe_goldens import CASES as GOLDEN_CASES
from tests.helpers.generate_prepare_expected import CASES as PREPARE_CASES
from tests.plotting.test_cells import _inputs


ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / 'tests' / 'contract' / 'describe'
SERIES_INDEX = json.loads((ROOT / 'tests' / 'contract' / 'series' /
                           'index.json').read_text())
PREPARE_INDEX = json.loads((ROOT / 'tests' / 'test_data' /
                            'prepare_expected' / 'cases.json').read_text())


@pytest.mark.parametrize('name', GOLDEN_CASES)
def test_describe_golden(name, tmp_path):
    tool, source, options = GOLDEN_CASES[name]
    matrix = tmp_path / (source + '.gz')
    write_input(matrix, distinct=source == 'distinct')
    actual = describe(tool, ['-m', str(matrix), *options])
    accepted = {option for action in plot_parser(tool)._actions
                for option in action.option_strings}
    assert all(domain['option'] in accepted
               for domain in actual['color_domains'])
    assert actual == json.loads((GOLDENS / (name + '.json')).read_text())
    if name != 'kmeans':
        assert actual == describe(tool, ['-m', str(matrix), *options],
                                  force_load=True)


@pytest.mark.parametrize('name,case', [
    (name, case) for name, case in cases().items() if not case.rejected])
def test_describe_matches_series_plan(name, case, tmp_path):
    from deeptoolsr.plotting.matrix_spec import project_matrix_spec
    from deeptoolsr.plotting.series import resolve_cells

    _module, argv, args, layout, labels, samples, sizes = _inputs(tmp_path, case)
    tool = 'plotHeatmapR' if case.tool == 'heatmap' else 'plotProfileR'
    actual = describe(tool, argv)
    accepted = {option for action in plot_parser(tool)._actions
                for option in action.option_strings}
    assert all(domain['option'] in accepted
               for domain in actual['color_domains'])
    plan = resolve_cells(sizes, samples, labels, project_matrix_spec(
        args, tool).series_options)
    assert [(group['key'], group['label'], group['regions'])
            for group in actual['groups']] == list(
        zip(labels.group_keys, labels.groups, sizes))
    assert [sample['label'] for sample in actual['samples']] == list(labels.samples)
    assert actual['row_labels'] == list(plan.row_labels)
    assert actual['cells'] == [
        {'index': cell.index, 'row': cell.row + 1,
         'column': cell.column + 1,
         'samples': [sample + 1 for sample in cell.samples],
         'groups': [group + 1 for group in cell.groups],
         'title': cell.title,
         'profile': cell.profile.index if cell.profile else None,
         'blocks': [
             {'group': block.group + 1, 'sample': block.sample + 1}
             for block in cell.blocks],
         'stack_labels': list(cell.stack_labels)}
        for cell in plan.cells]
    assert [(panel['kind'], panel['title'], panel['row'], panel['column'],
             tuple(panel['series'])) for panel in actual['panels']] == [
        (panel.kind, panel.title, panel.row + 1, panel.column + 1,
         panel.series) for panel in plan.panels]
    assert [(item['key'], item['group'], item['sample'], item['label'])
            for item in actual['series']] == [
        (item.key, item.group + 1, item.sample + 1, item.label)
        for item in plan.series]
    assert [(domain['option'], domain['kind'], domain['slots'])
            for domain in actual['color_domains']] == [
        (domain.option, domain.kind,
         [{'position': slot.position, 'series': list(slot.series),
           'panels': list(slot.panels), 'cells': list(slot.cells)}
          for slot in domain.slots])
        for domain in plan.color_domains]


def _run(*argv):
    return subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                          timeout=30)


@pytest.mark.parametrize('name', [name for name, case in cases().items()
                                  if case.rejected])
def test_series_rejections_match_cli(name, tmp_path):
    case = cases()[name]
    matrix = tmp_path / 'input.gz'
    write_input(matrix, distinct=case.source == 'distinct')
    tool = 'plotHeatmapR' if case.tool == 'heatmap' else 'plotProfileR'
    args = ['-m', str(matrix), '-o', str(tmp_path / 'plot.png'), *case.options]
    cli = _run(tool, *args)
    metadata = _run('deeptoolsr', 'describe', tool, *args)
    assert metadata.returncode == cli.returncode == SERIES_INDEX[name]['status']
    assert metadata.stdout == ''
    assert cli.stderr.splitlines()[-1].removeprefix('ValueError: ') == (
        metadata.stderr.splitlines()[-1].removeprefix('ValueError: '))


@pytest.mark.parametrize('name', [name for name, case in PREPARE_CASES.items()
                                  if case[3]])
def test_prepare_rejections_match_cli(name):
    tool, options, input_name, _, _ = PREPARE_CASES[name]
    command = 'plotHeatmapR' if tool == 'heatmap' else 'plotProfileR'
    args = ['-m', str(ROOT / 'tests' / 'test_data' / 'prepare_expected' /
                      input_name), '-o', str(ROOT / '.cache' / 'describe.png'),
            *options]
    result = _run('deeptoolsr', 'describe', command, *args)
    assert result.returncode == PREPARE_INDEX[name]['status']
    assert result.stderr == PREPARE_INDEX[name]['stderr']
    assert result.stdout == ''


@pytest.mark.parametrize('tool', ('plotHeatmapR', 'plotProfileR', 'plotMatrixR'))
def test_output_paths_are_optional_and_not_probed(tool, tmp_path):
    matrix = tmp_path / 'input.gz'
    write_input(matrix)
    mode = ['--profile'] if tool == 'plotMatrixR' else []
    with patch('deeptoolsr.session.PlotSession.matrix_for', side_effect=AssertionError(
            'header-only describe loaded values')):
        without_output = describe(tool, ['-m', str(matrix), *mode])
    forbidden = tmp_path / 'missing-parent' / 'forbidden.png'
    with_output = describe(tool, ['-m', str(matrix), '-o', str(forbidden),
                                  *mode])
    assert without_output == with_output
    assert not forbidden.parent.exists()
    for mode in ('describe', 'worker'):
        flags = ['--profile'] if tool == 'plotMatrixR' else []
        assert parse_command(tool, ['-m', str(matrix), *flags],
                             mode).outFileName is None


@pytest.mark.parametrize('tool', ('plotHeatmapR', 'plotProfileR', 'plotMatrixR'))
@pytest.mark.parametrize('flag', ('--help', '--version', '--bad-option'))
@pytest.mark.parametrize('mode', ('describe', 'worker'))
def test_non_cli_parse_errors_are_values(tool, flag, mode):
    with pytest.raises(OptionError):
        parse_command(tool, ['-m', 'absent.gz', flag], mode)


@pytest.mark.parametrize('tool', ('plotHeatmapR', 'plotProfileR', 'plotMatrixR'))
@pytest.mark.parametrize('flag', ('--help', '--version'))
def test_cli_help_and_version_still_exit(tool, flag, capsys):
    with pytest.raises(SystemExit) as result:
        parse_command(tool, [flag], 'cli')
    assert result.value.code == 0
    assert capsys.readouterr().out


def test_describe_imports_no_matplotlib():
    script = '''
import sys
from deeptoolsr.deeptoolsr_list_tools import main
assert 'matplotlib' not in sys.modules
main(['describe', 'plotProfileR', '-m', sys.argv[1]])
assert 'matplotlib' not in sys.modules
'''
    result = _run(sys.executable, '-c', script, str(
        ROOT / 'tests' / 'test_data' / 'prepare_expected' / 'input.gz'))
    assert result.returncode == 0, result.stderr


def test_describe_cli_json_bytes_match_golden(tmp_path):
    matrix = tmp_path / 'input.gz'
    write_input(matrix)
    tool, _source, options = GOLDEN_CASES['heatmap_summary']
    result = _run('deeptoolsr', 'describe', tool, '-m', str(matrix), *options)
    assert result.returncode == 0, result.stderr
    assert result.stdout == (GOLDENS / 'heatmap_summary.json').read_text()


def test_empty_header_has_the_plot_error(tmp_path):
    matrix = tmp_path / 'empty.mat'
    from deeptoolsr.matrix import read_header
    source = ROOT / 'tests' / 'test_data' / 'prepare_expected' / 'input.gz'
    header = dict(read_header(source).parameters)
    header['group_boundaries'] = [0] * len(header['group_boundaries'])
    matrix.write_text('@' + json.dumps(header) + '\n')
    result = _run('deeptoolsr', 'describe', 'plotHeatmapR', '-m', str(matrix))
    assert result.returncode == 1
    assert result.stdout == ''
    assert result.stderr == ('No regions remain after filtering; cannot plot '
                             'an empty matrix.\n')


@pytest.mark.parametrize('colour', ['tab:blue', 'xkcd:sky blue', '0.3',
                                    '#12345678', 'not-a-colour', '2'])
def test_missing_data_colour_validation_matches_cli(colour, tmp_path):
    matrix = tmp_path / 'input.gz'
    write_input(matrix)
    args = ['-m', str(matrix), '-o', str(tmp_path / 'plot.png'),
            '--missingDataColor', colour]
    cli = _run('plotHeatmapR', *args)
    metadata = _run('deeptoolsr', 'describe', 'plotHeatmapR', *args)
    assert (metadata.returncode == 0) == (cli.returncode == 0)
    if cli.returncode:
        assert metadata.returncode == cli.returncode
        assert metadata.stderr.splitlines()[-1] == cli.stderr.splitlines()[-1]


def test_heatmap_colormap_slots_name_their_cells():
    matrix = str(Path(__file__).parent / 'test_heatmapper' /
                 'master_multi.mat.gz')
    base = ['-m', matrix, '--profile', '--arrangeSamples', '1,2', '3,4']
    alone = describe('plotMatrixR', base)
    stacked = describe('plotMatrixR', [*base, '--heatmap'])
    # Profiles, series and their colours do not change under a stack.
    for key in ('panels', 'series'):
        assert stacked[key] == alone[key]
    assert stacked['color_domains'][:-1] == alone['color_domains']
    colormap = stacked['color_domains'][-1]
    assert colormap['kind'] == 'colormap'
    assert [slot['cells'] for slot in colormap['slots']] == [[1], [2]]
