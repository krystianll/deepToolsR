"""Describe reports the plot's colour limits only when it read values."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io
from deeptoolsr import plotMatrix
from deeptoolsr.session import Cancelled, PlotSession


def _write_matrix(path):
    values = (np.arange(8 * 4, dtype=np.float32).reshape(8, 4) / 4 + 0.5)
    regions = [['chr1', [(i * 20, i * 20 + 10)], f'region{i}', i, '+', '0']
               for i in range(8)]
    header = {
        'sample_labels': ['Sample A', 'Sample B'],
        'sample_boundaries': [0, 2, 4],
        'group_labels': ['Group A', 'Group B'],
        'group_boundaries': [0, 4, 8],
        'bin size': [1, 1], 'upstream': [1, 1], 'body': [0, 0],
        'downstream': [1, 1], 'unscaled 5 prime': [0, 0],
        'unscaled 3 prime': [0, 0], 'ref point': [None, None],
        'min threshold': None, 'max threshold': None,
        'sort regions': 'no', 'sort using': 'mean', 'proc number': 1,
    }
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header, separators=(',', ':')),
        regions, values, 1)
    return str(path)


@pytest.fixture
def matrix(tmp_path):
    return _write_matrix(tmp_path / 'values.gz')


def test_header_only_describe_reads_no_values(matrix):
    with PlotSession(cache_bytes=0) as session, patch(
            'deeptoolsr.plotting.cli_io.load_plot_matrix',
            side_effect=AssertionError('values loaded')):
        record = session.describe(['plotHeatmapR', '-m', matrix])
        assert session.matrix is None
        assert 'matrix' not in session.recomputed
    assert record['values_read'] is False
    assert record['colorbars']
    assert all(bar['z_range'] is None for bar in record['colorbars'])
    assert 'y_range' not in record['sample_sets'][0]
    json.dumps(record)


def _plotted_limits(session, command):
    captured = []
    original = plotMatrix.prepare_matrix

    def capture(*args, **kwargs):
        captured.append(original(*args, **kwargs))
        return captured[-1]
    with patch.object(plotMatrix, 'prepare_matrix', capture):
        result = session.run(command)
    return captured[0], result


def test_kmeans_describe_reports_the_plotted_limits(matrix, tmp_path, capfd):
    command = ['plotHeatmapR', '-m', matrix, '--kmeans', '2']
    with PlotSession() as session:
        record = session.describe(command, force_load=True)
        assert capfd.readouterr() == ('', '')
        assert 'scan' in session.recomputed
        prepared, result = _plotted_limits(
            session, command + ['-o', str(tmp_path / 'auto.png')])
        # Describe's scan is the run's: cached, not recomputed.
        assert 'scan' in result.reused and 'scan' not in result.recomputed
    assert record['values_read'] is True
    (bar,) = record['colorbars']
    expected = [prepared.z_min[0], prepared.z_max[0]]
    assert bar['z_range'] == {'auto': expected, 'value': expected}


def test_typed_z_max_differs_from_auto(matrix, tmp_path):
    command = ['plotHeatmapR', '-m', matrix, '--kmeans', '2', '--zMax', '42']
    with PlotSession() as session:
        record = session.describe(command, force_load=True)
        prepared, _ = _plotted_limits(
            session, command + ['-o', str(tmp_path / 'typed.png')])
    (bar,) = record['colorbars']
    assert bar['z_range']['value'] == [prepared.z_min[0], 42.0]
    assert bar['z_range']['value'][1] != bar['z_range']['auto'][1]
    assert bar['z_range']['auto'][0] == bar['z_range']['value'][0]


def test_force_load_reads_values(matrix):
    with PlotSession() as session:
        header_only = session.describe(['plotHeatmapR', '-m', matrix])
        loaded = session.describe(['plotHeatmapR', '-m', matrix],
                                  force_load=True)
        assert 'matrix' in session.recomputed
    assert loaded['values_read'] is True
    assert all(len(bar['z_range']['auto']) == 2
               for bar in loaded['colorbars'])
    strip = {'values_read', 'colorbars'}

    def unread(record):
        record = {key: value for key, value in record.items()
                  if key not in strip}
        record['series'] = [dict(item, pseudocount=None)
                            for item in record['series']]
        record['y_limits'] = [
            dict(group, data_range=None, auto=None, per_panel=[
                dict(entry, auto=None, value=None)
                for entry in group['per_panel']])
            for group in record['y_limits']]
        return record
    assert unread(loaded) == unread(header_only)


def test_profile_only_describe_has_no_colorbar_limits(matrix):
    with PlotSession() as session:
        record = session.describe(['plotProfileR', '-m', matrix],
                                  force_load=True)
    assert record['values_read'] is True
    assert record['colorbars'] == []


@pytest.mark.parametrize('stage', ['load', 'order', 'limits'])
def test_cancelled_describe_stops_at_a_stage(matrix, stage):
    command = ['plotHeatmapR', '-m', matrix, '--kmeans', '2']
    with PlotSession() as session:
        reached = []
        original = session.stage

        def recording(name, cancelled, progress):
            reached.append(name)
            return original(name, cancelled, progress)
        with patch.object(session, 'stage', recording):
            with pytest.raises(Cancelled):
                session.describe(command, force_load=True,
                                 cancelled=lambda: reached[-1] == stage)
        assert reached[-1] == stage
        record = session.describe(command, force_load=True)
    assert record['values_read'] is True
    assert record['colorbars'][0]['z_range'] is not None
