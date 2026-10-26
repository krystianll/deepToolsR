"""Describe reports each per-target error the plot raises, naming its option."""

import pytest

from deeptoolsr import plotMatrix
from deeptoolsr.describe import describe
from deeptoolsr.parserCommon import plot_parser
from deeptoolsr.prepare import OptionError
from tests.test_describe_protocol2 import _write


@pytest.fixture
def plain(tmp_path):
    """Three samples ('Sample A', 'Sample A', 'Sample B'), two groups."""
    return _write(tmp_path / 'plain.gz')


@pytest.fixture
def mixed(tmp_path):
    return _write(tmp_path / 'mixed.gz', mixed=True)


def _plot(command, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        plotMatrix.main([*command, '-o', str(tmp_path / 'plot.png')])
    return error.value.code, capsys.readouterr().err


COLLECTED = [
    (['--profile', '--yMax', '9=5'], 'assignment_out_of_range', '--yMax',
     'sample set', 9,
     '--yMax: 9 names sample set 9, but there are only 3 sample sets'),
    (['--profile', '--yMin', '4=0'], 'assignment_out_of_range', '--yMin',
     'sample set', 4,
     '--yMin: 4 names sample set 4, but there are only 3 sample sets'),
    (['--heatmap', '--zMin', '0', '--zMax', '1', '--zMid', '1=5'],
     'zmid_out_of_range', '--zMid', 'sample set', 1,
     '--zMid: 5 must be strictly between zMin (0) and zMax (1)'),
    (['--heatmap', '--colorMap', '2=notacolormap'], 'unknown_colour_map',
     '--colorMap', 'sample set', 2,
     '--colorMap: notacolormap is not a colour map; use a Matplotlib or '
     'deepToolsR colour map name such as viridis'),
    (['--heatmap', '--colorList', 'red,zzz'], 'unknown_colour',
     '--colorList', 'sample set', 1,
     '--colorList: zzz is not a colour; use a colour name or an HTML hex '
     'string such as #eeff22'),
    (['--profile', '--colorsPerSample', 'red', 'notacolor'],
     'unknown_colour', '--colorsPerSample', 'series', 2,
     '--colorsPerSample: notacolor is not a colour; use a colour name or an '
     'HTML hex string such as #eeff22'),
    (['--profile', '--samplesLabel', '5=x'], 'assignment_out_of_range',
     '--samplesLabel', 'sample', 5,
     '--samplesLabel: 5 names sample 5, but there are only 3 samples'),
    (['--profile', '--samplesLabel', 'a', 'b'], 'label_count',
     '--samplesLabel', 'sample', None,
     '--samplesLabel: 2 labels given for 3 samples'),
    (['--profile', '--regionsLabel', 'a', 'b', 'c', 'd', 'e'], 'label_count',
     '--regionsLabel', 'region group', None,
     '--regionsLabel: 5 labels given for 2 region groups'),
]


@pytest.mark.parametrize('extra, code, option, target, slot, message',
                         COLLECTED)
def test_describe_collects_what_the_plot_rejects(
        plain, tmp_path, capsys, extra, code, option, target, slot, message):
    command = ['-m', plain, *extra]
    problem, = describe('plotMatrixR', command)['problems']
    assert (problem['code'], problem['severity'], problem['option'],
            problem['target'], problem['slot'], problem['message']) == (
        code, 'error', option, target, slot, message)
    status, err = _plot(command, tmp_path, capsys)
    assert (status, err) == (2, message + '\n')


def test_unknown_colour_map_keeps_the_automatic_scale(plain):
    record = describe('plotMatrixR', ['-m', plain, '--heatmap',
                                      '--colorMap', 'notacolormap'])
    assert record['problems'][0]['code'] == 'unknown_colour_map'
    assert record['colorbars']


def test_bad_region_labels_leave_the_group_names(plain):
    record = describe('plotMatrixR', ['-m', plain, '--profile',
                                      '--regionsLabel', 'a', 'b', 'c'])
    assert record['problems'][0]['message'] == (
        '--regionsLabel: 3 labels given for 2 region groups')
    assert describe('plotMatrixR', ['-m', plain, '--profile'])[
        'groups'] == record['groups']


def test_zmid_with_auto_limits_is_still_checked_with_values(plain):
    record = describe('plotMatrixR', ['-m', plain, '--heatmap',
                                      '--zMid', '1000'])
    assert record['problems'] == []


@pytest.mark.parametrize('extra, option', [
    (['--yMax', 'abc'], '--yMax'),
    (['--cellWidth', '5', '--profileHeight', '5',
      '--profileAspectRatio', '2'], '--profileHeight'),
])
def test_parse_errors_name_their_option(plain, extra, option):
    with pytest.raises(OptionError) as error:
        describe('plotMatrixR', ['-m', plain, '--profile', *extra])
    assert error.value.option == option


def test_geometry_clash_message_is_a_sentence(mixed, tmp_path, capsys):
    command = ['-m', mixed, '--profile', '--arrangeSamples', '1,3', '2']
    message = (
        'Samples in sample set 1 cannot share a panel because their X-axis '
        'geometries differ: Sample A is reference-point at TSS, 1 bp '
        'upstream, 1 bp downstream, bin size 1; Sample B is scale-regions, '
        '1 bp upstream, 2 bp body, 1 bp downstream, bin size 1.')
    assert describe('plotMatrixR', command)['problems'][0][
        'message'] == message
    status, _ = _plot(command, tmp_path, capsys)
    assert status == 'Error: ' + message


@pytest.mark.parametrize('tool', ['plotHeatmapR', 'plotProfileR',
                                  'plotMatrixR'])
def test_per_group_help_says_it_titles_panels_by_group(tool, monkeypatch):
    monkeypatch.setenv('COLUMNS', '1000')
    text = ' '.join(plot_parser(tool).format_help().split())
    assert ('It arranges the panels like --arrangeSamples 1,2,...,n '
            '--sampleSetGroupArrangement adjacent, but titles each panel by '
            'its region group' in text)
    assert 'shortcut for --arrangeSamples' not in text


def test_merged_region_labels_of_the_wrong_count_name_the_option(
        plain, tmp_path, capsys):
    command = ['-m', plain, '--profile', '--sameGroupLabels', 'merge',
               '--regionsLabel', 'a', 'b', 'c']
    message = '--regionsLabel: 3 labels given for 2 region groups'
    with pytest.raises(OptionError, match=message) as error:
        describe('plotMatrixR', command)
    assert (error.value.code, error.value.option) == (
        'label_count', '--regionsLabel')
    assert _plot(command, tmp_path, capsys) == (2, message + '\n')


def test_small_group_warning_with_too_few_region_labels(tmp_path):
    """The warning names the group as the resolved labels do."""
    import json
    import numpy as np
    from deeptoolsr import _compute_matrix_io
    rows = 1001
    regions = [['chr1', [(i * 20, i * 20 + 10)], f'region{i}', i, '+', '0']
               for i in range(rows)]
    header = {
        'sample_labels': ['Sample A'], 'sample_boundaries': [0, 2],
        'group_labels': ['GENE1', 'GENE2'],
        'group_boundaries': [0, rows - 1, rows],
        'bin size': [1], 'upstream': [1], 'body': [0], 'downstream': [1],
        'unscaled 5 prime': [0], 'unscaled 3 prime': [0], 'ref point': ['TSS'],
        'min threshold': None, 'max threshold': None, 'sort regions': 'no',
        'sort using': 'mean', 'proc number': 1}
    path = str(tmp_path / 'small.gz')
    _compute_matrix_io.write_matrix(
        path, json.dumps(header), regions,
        np.ones((rows, 2), dtype=np.float32), 1)
    record = describe('plotProfileR', ['-m', path, '--filterNans', 'any_bin',
                                       '--regionsLabel', 'a'], force_load=True)
    assert 'label_count' in [item['code'] for item in record['problems']]
    assert any("Group 'GENE2' is too small" in message
               for message in record['warnings'])
