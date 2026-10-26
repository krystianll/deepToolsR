"""Describe protocol 2: sample sets, the per-target option table, problems."""

import json

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io, plotMatrix
from deeptoolsr.describe import describe, describe_cli
from deeptoolsr.parserCommon import (
    capability_tag, per_target_options, plot_parser)
from deeptoolsr.plotting.series import resolve_recyclable
from deeptoolsr.prepare import OptionError
from deeptoolsr.session import PlotSession


def _write(path, *, mixed=False):
    """Two groups; samples 1 and 2 share a label; sample 3 may be scaled."""
    values = np.arange(4 * 8, dtype=np.float32).reshape(4, 8) / 10 + 1
    regions = [['chr1', [(i * 20, i * 20 + 10)], f'region{i}', i, '+', '0']
               for i in range(4)]
    header = {
        'sample_labels': ['Sample A', 'Sample A', 'Sample B'],
        'sample_boundaries': [0, 2, 4, 8] if mixed else [0, 2, 4, 6],
        'group_labels': ['GENE1', 'GENE2'], 'group_boundaries': [0, 2, 4],
        'bin size': [1, 1, 1], 'upstream': [1, 1, 1],
        'body': [0, 0, 2 if mixed else 0], 'downstream': [1, 1, 1],
        'unscaled 5 prime': [0, 0, 0], 'unscaled 3 prime': [0, 0, 0],
        'ref point': ['TSS', 'TSS', None if mixed else 'TSS'],
        'min threshold': None, 'max threshold': None, 'sort regions': 'no',
        'sort using': 'mean', 'proc number': 1}
    if not mixed:
        values = values[:, :6]
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header), regions, values, 1)
    return str(path)


@pytest.fixture
def mixed(tmp_path):
    return _write(tmp_path / 'mixed.gz', mixed=True)


@pytest.fixture
def plain(tmp_path):
    return _write(tmp_path / 'plain.gz')


def _texts(record, index):
    return record['sample_sets'][index - 1]['texts']


def test_sample_sets_list_samples_geometry_and_texts(mixed):
    record = describe('plotMatrixR', ['-m', mixed, '--profile',
                                      '--arrangeSamples', '1,2', '3'])
    assert record['protocol'] == 2
    first, second = record['sample_sets']
    assert (first['index'], first['samples'], first['groups']) == (
        1, [1, 2], None)
    assert first['geometry'] == {
        'mode': 'reference-point', 'reference_point': 'TSS', 'upstream': 1,
        'downstream': 1, 'body': 0, 'unscaled_5_prime': 0,
        'unscaled_3_prime': 0, 'bin_size': 1, 'bin_count': 2}
    assert second['geometry']['mode'] == 'scale-regions'
    assert second['geometry']['bin_count'] == 4
    # Duplicate sample labels stay distinct by index.
    assert first['texts']['subplot_title'] == {
        'auto': 'Sample A, Sample A', 'value': 'Sample A, Sample A'}
    assert first['texts']['x_axis_label']['value'] == 'distance from TSS'
    assert first['texts']['reference_label']['value'] == 'TSS'
    assert first['texts']['start_label'] == {'auto': None, 'value': None}
    assert second['texts']['x_axis_label']['value'] == 'gene distance'
    assert second['texts']['reference_label'] == {'auto': None,
                                                  'value': None}
    assert second['texts']['start_label'] == {'auto': 'TSS', 'value': 'TSS'}
    assert second['texts']['end_label'] == {'auto': 'TES', 'value': 'TES'}


def test_override_keeps_its_automatic_value(mixed):
    record = describe('plotMatrixR', [
        '-m', mixed, '--heatmap', '--arrangeSamples', '1,2', '3',
        '--sampleSetLabels', '2=Treated', '--endLabel', 'stop',
        '--heatmapYAxisLabel', 'regions'])
    assert _texts(record, 1)['subplot_title'] == {
        'auto': 'Sample A, Sample A', 'value': 'Sample A, Sample A'}
    assert _texts(record, 2)['subplot_title'] == {
        'auto': 'Sample B', 'value': 'Treated'}
    assert _texts(record, 2)['end_label'] == {'auto': 'TES', 'value': 'stop'}
    assert _texts(record, 1)['heatmap_y_axis_label'] == {
        'auto': '', 'value': 'regions'}


def test_groups_as_sample_sets_counts_promoted_panels(plain):
    record = describe('plotMatrixR', [
        '-m', plain, '--profile', '--perGroup', '--groupsAsSampleSets',
        '--sampleSetLabels', '2=Second'])
    sets = record['sample_sets']
    assert [item['groups'] for item in sets] == [[1], [2]]
    assert [item['samples'] for item in sets] == [[1, 2, 3], [1, 2, 3]]
    assert _texts(record, 1)['subplot_title']['value'] == 'GENE1'
    assert _texts(record, 2)['subplot_title'] == {'auto': 'GENE2',
                                                  'value': 'Second'}
    assert record['problems'] == []


def test_per_group_makes_one_sample_set(plain):
    record = describe('plotProfileR', ['-m', plain, '--perGroup'])
    assert [item['samples'] for item in record['sample_sets']] == [[1, 2, 3]]
    assert record['sample_sets'][0]['groups'] is None


def test_geometry_clash_is_a_problem_and_other_sets_are_described(mixed):
    command = ['-m', mixed, '--profile', '--arrangeSamples', '1,3', '2']
    record = describe('plotMatrixR', command)
    clash, other = record['sample_sets']
    assert clash['geometry'] is None and clash['samples'] == [1, 3]
    assert other['geometry']['mode'] == 'reference-point'
    problem, = record['problems']
    assert problem['code'] == 'geometry_clash'
    assert (problem['severity'], problem['target'], problem['slot'],
            problem['samples']) == ('error', 'sample set', 1, [1, 3])
    assert problem['message'].startswith(
        'Samples in sample set 1 cannot share a panel')
    assert record['warnings'] == []


def test_plot_still_raises_the_geometry_clash(mixed, tmp_path):
    with pytest.raises(SystemExit) as error:
        plotMatrix.main(['-m', mixed, '--profile', '--arrangeSamples',
                         '1,3', '2', '-o', str(tmp_path / 'plot.png')])
    assert str(error.value.code).startswith(
        'Error: Samples in sample set 1 cannot share a panel')


def test_out_of_range_assignment_is_a_problem_with_a_fallback(plain):
    record = describe('plotMatrixR', [
        '-m', plain, '--heatmap', '--arrangeSamples', '1,2', '3',
        '--colorMap', '1-3=viridis', '--sampleSetLabels', '5=x', 'Kept'])
    problems = record['problems']
    assert [(item['code'], item['option'], item['target'], item['slot'])
            for item in problems] == [
        ('assignment_out_of_range', '--sampleSetLabels', 'sample set', 5),
        ('assignment_out_of_range', '--colorMap', 'sample set', 3)]
    assert problems[1]['message'] == (
        '--colorMap: 1-3 names sample set 3, but there are only 2 sample '
        'sets (--arrangeSamples 1,2 3)')
    assert [_texts(record, index)['subplot_title']['value']
            for index in (1, 2)] == ['Kept', 'Kept']


def test_plot_still_raises_the_out_of_range_assignment(plain, tmp_path,
                                                       capsys):
    with pytest.raises(SystemExit) as error:
        plotMatrix.main(['-m', plain, '--profile', '--sampleSetLabels',
                         '5=x', '-o', str(tmp_path / 'plot.png')])
    assert error.value.code == 2
    assert ('--sampleSetLabels: 5 names sample set 5, but there are only '
            '3 sample sets') in capsys.readouterr().err


def test_the_raised_error_carries_the_problem_fields():
    with pytest.raises(OptionError) as error:
        resolve_recyclable(['5=x'], 3, option='--sampleSetLabels',
                           domain='sample set')
    assert error.value.problem() == {
        'code': 'assignment_out_of_range', 'severity': 'error',
        'option': '--sampleSetLabels', 'target': 'sample set', 'slot': 5,
        'samples': None, 'message': str(error.value)}


def test_describe_cli_prints_problems_and_exits_zero(mixed, capsys):
    describe_cli('plotMatrixR', ['-m', mixed, '--profile',
                                 '--arrangeSamples', '1,3', '2'])
    out, err = capsys.readouterr()
    assert err == ''
    assert json.loads(out)['problems'][0]['code'] == 'geometry_clash'


def test_options_table_matches_capability_tags():
    for tool in ('plotHeatmapR', 'plotProfileR', 'plotMatrixR'):
        table = {item['option']: item for item in per_target_options(tool)}
        assert table
        for action in plot_parser(tool)._actions:
            names = [name for name in action.option_strings
                     if name.startswith('--')]
            entry = next((table[name] for name in names if name in table),
                         None)
            if entry is None:
                continue
            tag = capability_tag(action)
            assert ('explicit' in tag) == bool(entry['assignments'])
            assert ('recycled' in tag) == entry['recycles']
            assert entry['counts']
            if entry['assignments']:
                assert entry['assignments'] == ['N=', 'N,M=', 'a-b=']


def test_record_round_trips_through_json(mixed):
    record = describe('plotMatrixR', ['-m', mixed, '--profile',
                                      '--arrangeSamples', '1,3', '2'])
    assert json.loads(json.dumps(record)) == record
    assert set(record['options'][0]) == {'option', 'counts', 'recycles',
                                         'assignments'}


def test_warnings_appear_as_warning_problems(plain):
    record = describe('plotMatrixR', ['-m', plain, '--profile',
                                      '--groupsAsSampleSets'])
    assert record['warnings']
    assert [item['message'] for item in record['problems']
            if item['severity'] == 'warning'] == record['warnings']


def test_session_mode_prints_nothing(mixed, capfd):
    with PlotSession(cache_bytes=0) as session:
        record = session.describe(['plotMatrixR', '-m', mixed, '--profile',
                                   '--arrangeSamples', '1,3', '2',
                                   '--groupsAsSampleSets',
                                   '--sampleSetLabels', '9=x'])
    assert len(record['problems']) == 3
    assert capfd.readouterr() == ('', '')
