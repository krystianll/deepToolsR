"""Plot CLIs retain the recorded matrix, BED and diagnostic outputs."""
from deeptoolsr.matrix import OwnedMatrix
import numpy as np
import matplotlib.pyplot as plt
from tests.helpers.parity import read_matrix
from deeptoolsr import computeMatrix
from tests.helpers.parity import original
from tests.helpers.bigwig import pyBigWig

import json
import os
import io
import sys

import pytest

from deeptoolsr import plotHeatmap
from tests.helpers.generate_prepare_expected import CASES, HERE, run_case
from tests.helpers.prepare_fixtures import (
    assert_bed_equal, assert_diagnostics_equal)


EXPECTED = json.loads((HERE / 'cases.json').read_text())


@pytest.mark.parametrize('case', CASES)
def test_plot_cli_matches_recorded_outputs(case, tmp_path):
    actual = run_case(case, tmp_path, HERE)
    assert_diagnostics_equal(actual, EXPECTED[case],
                             silhouette=case == 'kmeans_silhouette')
    if CASES[case][3]:
        return
    assert_bed_equal(tmp_path / (case + '.bed'), HERE / (case + '.bed'),
                     silhouette=case == 'kmeans_silhouette')
    suffix = '.gz' if CASES[case][4] else '.txt'
    assert (tmp_path / (case + suffix)).read_bytes() == (
        HERE / (case + suffix)).read_bytes()


def test_sort_notice_precedes_silhouette_notice(tmp_path, monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stream)
    monkeypatch.setattr(sys, 'stderr', stream)
    plotHeatmap.main([
        '-m', str(HERE / 'input.gz'), '-o', str(tmp_path / 'plot.png'),
        '-p', '1', '--kmeans', '2', '--sortUsingSamples', '1',
        '--silhouette'])
    output = stream.getvalue()
    assert output.index('Samples used for ordering') < (
        output.index('The average silhouette score'))


# From general audit regressions.

def make_matrix(path, values, bounds=None):
    data = np.asarray(values, dtype=np.float32)
    rows, bins = data.shape
    bounds = [0, rows] if bounds is None else bounds
    labels = [f'group{i}' for i in range(len(bounds) - 1)]
    regions = [['chr1', [(i * 10, i * 10 + 5)], f'r{i}', 0, '+', '0'] for i in range(rows)]
    parameters = {'upstream': [0], 'downstream': [0], 'body': [bins], 'unscaled 5 prime': [0], 'unscaled 3 prime': [0], 'ref point': [None], 'bin size': [1], 'sort regions': 'keep', 'sort using': 'mean', 'min threshold': None, 'max threshold': None, 'sample_labels': ['sample'], 'group_labels': labels, 'sample_boundaries': [0, bins], 'group_boundaries': bounds}
    OwnedMatrix.from_compute(parameters, data, regions).save(str(path), compressed=True, threads=1)


@pytest.mark.parametrize('module', ['plotHeatmap', 'plotProfile'])
def test_plot_exports_keep_populated_group_names_and_rows(tmp_path, module):
    import importlib
    command = importlib.import_module('deeptoolsr.' + module)
    source, out = (tmp_path / 'source.gz', tmp_path / 'out.gz')
    png, bed = (tmp_path / 'plot.png', tmp_path / 'out.bed')
    make_matrix(source, [[1, 2], [3, 4]], [0, 0, 2, 2])
    try:
        command.main(['-m', str(source), '-o', str(png), '--regionsLabel', 'empty1', 'retained', 'empty2', '--showRegionCounts', '--outFileNameMatrix', str(out), '--outFileSortedRegions', str(bed), '--sortRegions', 'keep'])
        header, rows, values = read_matrix(out)
        assert header['group_boundaries'] == [0, 2]
        assert header['group_labels'] == ['retained [n = 2]']
        assert [row[3] for row in rows] == ['r0', 'r1']
        np.testing.assert_array_equal(values, [[1, 2], [3, 4]])
        emitted = [line.split('\t') for line in bed.read_text().splitlines() if not line.startswith('#')]
        assert [row[3] for row in emitted] == ['r0', 'r1']
        assert all(('retained [n = 2]' in row for row in emitted))
    finally:
        plt.close('all')


@pytest.mark.parametrize('module', ['plotHeatmap', 'plotProfile'])
def test_all_rows_filtered_reports_clear_error_preserving_output(tmp_path, module):
    import importlib
    source, out = (tmp_path / 'source.gz', tmp_path / 'out.png')
    make_matrix(source, [[np.nan, np.nan]])
    out.write_bytes(b'previous plot')
    with pytest.raises(SystemExit, match='No regions remain'):
        importlib.import_module('deeptoolsr.' + module).main(
            ['-m', str(source), '-o', str(out), '--filterNans', 'all_bins'])
    assert out.read_bytes() == b'previous plot'


@pytest.mark.parametrize('module', ['plotHeatmap', 'plotProfile'])
def test_header_thresholds_are_provenance_not_plot_filters(tmp_path, module):
    # Deliberate divergence from deepTools 3.5.6, which re-applied the
    # header's min/max threshold to the stored (possibly --scale'd) values.
    # computeMatrixR already applied them before scaling, so plotting keeps
    # every stored row.
    import importlib
    source = tmp_path / 'source.gz'
    bigwig = os.path.join(os.path.dirname(__file__), 'test_heatmapper',
                          'test.bw')
    bed = tmp_path / 'regions.bed'
    bed.write_text('ch1\t100\t150\n')
    computeMatrix.main(['scale-regions', '-S', bigwig, '-R', str(bed),
                        '-m', '50', '-b', '0', '-a', '0', '-bs', '50',
                        '--minThreshold', '0.5', '--scale', '0.1',
                        '-o', str(source)])
    matrix = OwnedMatrix.load(str(source), threads=1)
    assert matrix.header.parameters['min threshold'] == 0.5
    assert matrix.values.shape[0] == 1
    assert float(matrix.values[0, 0]) < 0.5
    sorted_regions = tmp_path / 'regions.out.bed'
    importlib.import_module('deeptoolsr.' + module).main(
        ['-m', str(source), '-o', str(tmp_path / 'out.png'),
         '--outFileSortedRegions', str(sorted_regions)])
    assert len([line for line in sorted_regions.read_text().splitlines()
                if not line.startswith('#')]) == 1


# From parity audit regressions.

@pytest.fixture
def parity_matrix_inputs(tmp_path):
    paths = []
    for sample in range(2):
        path = tmp_path / f's{sample}.bw'
        with pyBigWig.open(str(path), 'w') as bw:
            bw.addHeader([('chr1', 200)])
            starts = [i for i in range(200) if not 85 <= i < 100]
            bw.addEntries(['chr1'] * len(starts), starts, ends=[i + 1 for i in starts], values=[0.0 if 40 <= i < 55 else float((i % 11 + 1) * (sample + 1)) for i in starts])
        paths.append(str(path))
    bed = tmp_path / 'regions.bed'
    bed.write_text(''.join((f'chr1\t{s}\t{e}\tr{i}\t{i}\t{strand}\n' for i, (s, e, strand) in enumerate([(2, 23, '+'), (35, 53, '-'), (80, 95, '.'), (130, 151, '-'), (183, 198, '+')]))))
    return (paths, str(bed))


@pytest.fixture
def parity_input_matrix(tmp_path, parity_matrix_inputs):
    paths, bed = parity_matrix_inputs
    output = tmp_path / 'input.gz'
    computeMatrix.main(['reference-point', '-S', *paths, '-R', bed, '-b', '10', '-a', '30', '-bs', '5', '--quiet', '-p', '1', '-o', str(output)])
    return str(output)


@pytest.mark.parametrize('per_group', [False, True])
def test_heatmap_export_parity(tmp_path, parity_input_matrix, per_group):
    paths = {}
    for kind in ('original', 'plus'):
        paths[kind] = tmp_path / f'{kind}.gz'
        args = ['-m', parity_input_matrix, '-o', str(tmp_path / f'{kind}.png'), '--outFileNameMatrix', str(paths[kind]), '--sortRegions', 'no'] + (['--perGroup'] if per_group else [])
        if kind == 'original':
            original('plotHeatmap', args)
        else:
            plotHeatmap.main(args)
    _, orows, ov = read_matrix(paths['original'])
    _, prows, pv = read_matrix(paths['plus'])
    assert orows == prows
    np.testing.assert_allclose(pv, ov, rtol=3e-07, atol=1e-06, equal_nan=True)
