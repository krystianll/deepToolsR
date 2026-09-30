"""Shared plot preparation against the pinned CLI fixture contract."""
from deeptoolsr.matrix import OwnedMatrix
from deeptoolsr.matrix import RowLayout
from tests.helpers.parity import original
import matplotlib.pyplot as plt
from deeptoolsr.matrix import remove_empty_groups
from deeptoolsr import computeMatrix
from tests.helpers.bigwig import pyBigWig
from tests.helpers.parity import read_matrix

import io
import json
from pathlib import Path

import numpy as np
import pytest

from deeptoolsr import plotHeatmap, plotProfile
from deeptoolsr.matrix import Matrix, MatrixHeader, save, save_bed
from deeptoolsr.prepare import (
    DataError, heatmap_spec, prepare, profile_spec,
    resolve_prepared_labels, warning_text,
)
from tests.helpers.generate_prepare_expected import CASES
from tests.helpers.prepare_fixtures import (
    assert_bed_equal, assert_diagnostics_equal,
)


FIXTURES = Path(__file__).parent / 'test_data' / 'prepare_expected'
MANIFEST = json.loads((FIXTURES / 'cases.json').read_text())


def _args(case, tmp_path):
    tool, options, input_name, _, compressed = CASES[case]
    module = plotHeatmap if tool == 'heatmap' else plotProfile
    suffix = '.gz' if compressed else '.txt'
    argv = ['-m', str(FIXTURES / input_name),
            '-o', str(tmp_path / (case + '.png')),
            '--outFileSortedRegions', str(tmp_path / (case + '.bed')),
            '--outFileNameMatrix', str(tmp_path / (case + suffix)),
            *options]
    args = module.parse_arguments().parse_args(argv)
    return args, heatmap_spec if tool == 'heatmap' else profile_spec


def _run(case, tmp_path):
    args, project = _args(case, tmp_path)
    matrix = Matrix.load(FIXTURES / CASES[case][2], threads=1)
    spec = project(args, matrix.header)
    layout, warnings = prepare(matrix, spec, threads=1)
    labels = resolve_prepared_labels(
        layout, spec, regions_label=args.regionsLabel,
        samples_label=args.samplesLabel,
        show_counts=args.showRegionCounts, header=matrix.header)
    return args, matrix, spec, layout, labels, warnings


ACCEPTED = [name for name in CASES if not CASES[name][3]]
REJECTED = [name for name in CASES if CASES[name][3]]


@pytest.mark.parametrize('case', ACCEPTED)
def test_prepare_reproduces_every_accepted_cli_fixture(case, tmp_path):
    args, matrix, spec, layout, labels, warnings = _run(case, tmp_path)
    compressed = CASES[case][4]
    suffix = '.gz' if compressed else '.txt'
    output_matrix = tmp_path / (case + suffix)
    save(matrix, layout, labels, output_matrix,
         compressed=compressed, threads=1)
    assert output_matrix.read_bytes() == (FIXTURES / (case + suffix)).read_bytes()
    output_bed = tmp_path / (case + '.bed')
    with output_bed.open('w', newline='', encoding='utf-8') as handle:
        save_bed(matrix, layout, labels, handle)
    assert_bed_equal(output_bed, FIXTURES / (case + '.bed'),
                     silhouette=case == 'kmeans_silhouette')

    stdout = io.StringIO()
    if spec.cluster_method == 'hierarchical':
        print('Performing hierarchical clustering.'
              'Please note that it might be very slow for large datasets.\n',
              file=stdout)
    if args.sortUsingSamples is not None and spec.row_order_consumed:
        print('Samples used for ordering within each group: ',
              [index - 1 for index in args.sortUsingSamples], file=stdout)
    actual = {
        'status': 0, 'stdout': stdout.getvalue(),
        'stderr': ''.join(warning_text(
            warning, layout, regions_label=args.regionsLabel)
            for warning in warnings),
    }
    assert_diagnostics_equal(actual, MANIFEST[case],
                             silhouette=case == 'kmeans_silhouette')


@pytest.mark.parametrize('case', REJECTED)
def test_prepare_rejects_pre_s6_cases(case, tmp_path):
    with pytest.raises((DataError, ValueError)) as caught:
        _run(case, tmp_path)
    expected = MANIFEST[case]['stderr'].splitlines()[-1]
    if case == 'reject_sort_sample':
        expected = expected.replace('--sortSamples', '--sortUsingSamples')
    assert str(caught.value) == expected.removeprefix('ValueError: ')
    if case == 'reject_quantiles':
        assert ''.join(warning_text(
            # No empty group: Empty is no longer re-filtered by the header's
            # min threshold (divergence from deepTools 3.5.6).
            warning, None) for warning in caught.value.warnings) == ''


@pytest.mark.parametrize('tool', ['heatmap', 'profile'])
def test_labels_do_not_recompute_layout(tool, tmp_path):
    matrix = Matrix.load(FIXTURES / 'input.gz', threads=1)
    module = plotHeatmap if tool == 'heatmap' else plotProfile
    project = heatmap_spec if tool == 'heatmap' else profile_spec

    def run(*extra):
        args = module.parse_arguments().parse_args([
            '-m', str(FIXTURES / 'input.gz'),
            '-o', str(tmp_path / 'plot.png'),
            '--outFileSortedRegions', str(tmp_path / 'rows.bed'), *extra])
        spec = project(args, matrix.header)
        layout, _ = prepare(matrix, spec, threads=1)
        labels = resolve_prepared_labels(
            layout, spec, regions_label=args.regionsLabel,
            samples_label=args.samplesLabel,
            show_counts=args.showRegionCounts, header=matrix.header)
        return spec, layout, labels

    base_spec, base_layout, base_labels = run()
    for extra in (
            ('--regionsLabel', 'Same', 'Same', 'Gone'),
            ('--samplesLabel', 'A', 'B', 'C'),
            ('--showRegionCounts',)):
        spec, layout, labels = run(*extra)
        assert spec == base_spec
        assert layout == base_layout
        assert layout.digest() == base_layout.digest()
        assert labels != base_labels
    if tool == 'profile':
        _, together, _ = run('--regionsLabel', 'Same', 'Same', 'Gone',
                             '--sameGroupLabels', 'together')
        _, merged, _ = run('--regionsLabel', 'Same', 'Same', 'Gone',
                           '--sameGroupLabels', 'merge')
        assert together == base_layout
        assert merged != base_layout
        # Empty stays (no header-threshold re-filter; 3.5.6 dropped it).
        assert merged.group_bounds == (0, 10, 12)


def test_prepare_does_not_compound_on_one_matrix(tmp_path):
    matrix = Matrix.load(FIXTURES / 'input.gz', threads=1)
    args_a, project = _args('sort_disjoint_median', tmp_path)
    args_b, _ = _args('kmeans_selected', tmp_path)
    spec_a = project(args_a, matrix.header)
    spec_b = project(args_b, matrix.header)
    first, _ = prepare(matrix, spec_a, threads=1)
    _ = prepare(matrix, spec_b, threads=1)
    again, _ = prepare(matrix, spec_a, threads=1)
    assert first == again
    assert first.digest() == again.digest()
    assert not matrix.values.flags.writeable


def test_small_group_warning_uses_premerge_override(tmp_path):
    header = MatrixHeader.from_parameters({
        'sample_labels': ['one'], 'sample_boundaries': [0, 1],
        'group_labels': ['A', 'B'], 'group_boundaries': [0, 1, 201],
    })
    values = np.zeros((201, 1), dtype=np.float32)
    values.flags.writeable = False
    matrix = Matrix(header, values, [], None)
    args = plotProfile.parse_arguments().parse_args([
        '-m', str(FIXTURES / 'input.gz'), '-o', str(tmp_path / 'plot.png'),
        '--sortRegions', 'no', '--regionsLabel', 'Tiny', 'Rest',
        '--sameGroupLabels', 'merge'])
    layout, warnings = prepare(matrix, profile_spec(args, header), threads=1)
    assert [warning.kind for warning in warnings] == ['small_group']
    assert warning_text(warnings[0], layout) == (
        "WARNING: Group 'Tiny' is too small for plotting, you might "
        'want to remove it. \n')


@pytest.mark.parametrize('tool', ['heatmap', 'profile'])
def test_unconsumed_profile_sort_is_skipped(tool, tmp_path):
    matrix = Matrix.load(FIXTURES / 'input.gz', threads=1)
    module = plotHeatmap if tool == 'heatmap' else plotProfile
    project = heatmap_spec if tool == 'heatmap' else profile_spec
    args = module.parse_arguments().parse_args([
        '-m', str(FIXTURES / 'input.gz'), '-o', str(tmp_path / 'plot.png'),
        '--sortRegions', 'ascend'])
    spec = project(args, matrix.header)
    layout, _ = prepare(matrix, spec, threads=1)
    assert spec.row_order_consumed is (tool == 'heatmap')
    assert layout.sort.method == ('ascend' if tool == 'heatmap' else 'keep')


# From general audit regressions.

def make_matrix(path, values, bounds=None):
    data = np.asarray(values, dtype=np.float32)
    rows, bins = data.shape
    bounds = [0, rows] if bounds is None else bounds
    labels = [f'group{i}' for i in range(len(bounds) - 1)]
    regions = [['chr1', [(i * 10, i * 10 + 5)], f'r{i}', 0, '+', '0'] for i in range(rows)]
    parameters = {'upstream': [0], 'downstream': [0], 'body': [bins], 'unscaled 5 prime': [0], 'unscaled 3 prime': [0], 'ref point': [None], 'bin size': [1], 'sort regions': 'keep', 'sort using': 'mean', 'min threshold': None, 'max threshold': None, 'sample_labels': ['sample'], 'group_labels': labels, 'sample_boundaries': [0, bins], 'group_boundaries': bounds}
    OwnedMatrix.from_compute(parameters, data, regions).save(str(path), compressed=True, threads=1)


def render(tmp_path, values, extra=(), bounds=None, upstream=False):
    source, dest = (tmp_path / 'matrix.gz', tmp_path / 'heatmap.png')
    make_matrix(source, values, bounds)
    args = ['-m', str(source), '-o', str(dest), *extra]
    try:
        if upstream:
            original('plotHeatmap', args)
        else:
            plotHeatmap.main(args)
        assert dest.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    finally:
        plt.close('all')


@pytest.mark.parametrize('route', ['kmeans', 'hclust', 'filterNans', 'input'])
def test_g03_empty_groups(tmp_path, route):
    data = np.tile([1.0, 2.0], (6, 1))
    bounds = None
    if route in ('kmeans', 'hclust'):
        extra = ['--' + route, '3']
    elif route == 'filterNans':
        data[:3] = np.nan
        bounds = [0, 3, 6]
        extra = ['--filterNans', 'any_bin']
    else:
        bounds = [0, 0, 6]
        extra = []
    render(tmp_path, data, extra, bounds)


def test_empty_group_removal_does_not_copy_values(tmp_path):
    source = tmp_path / 'source.gz'
    make_matrix(source, [[1, 2]], [0, 0, 1, 1])
    matrix = Matrix.load(str(source), threads=1)
    data = matrix.values
    layout, removed = remove_empty_groups(RowLayout.identity(matrix))
    assert removed == 2
    assert matrix.values is data
    assert layout.base_names[layout.origins[0].source] == 'group1'


# From general second audit regressions.

def matrix_args(bw, regions, output, extra=()):
    return ['reference-point', '-S', str(bw), '-R', str(regions), '-o', str(output), '-b', '0', '-a', '10', '-bs', '10', *extra]


def assert_rows(path, names=('one', 'two'), values=(2.0, 7.0)):
    header, rows, observed = read_matrix(path)
    assert [row[3] for row in rows] == list(names)
    np.testing.assert_array_equal(observed, np.asarray(values).reshape(-1, 1))
    return header


def filtered_groups(tmp_path, criterion, extra=()):
    bw, bed, out = (tmp_path / 'signal.bw', tmp_path / 'regions.bed', tmp_path / 'out.gz')
    with pyBigWig.open(str(bw), 'w') as handle:
        handle.addHeader([('chr1', 100)])
        handle.addEntries(['chr1', 'chr1'], [0, 50], ends=[50, 100], values=[0.0, 7.0])
    bed.write_text('chr1\t10\t20\tone\t0\t+\n#zero\nchr1\t60\t70\ttwo\t0\t+\n#signal\n')
    options = {'skipZeros': ['--skipZeros'], 'minimum': ['--minThreshold', '1'], 'maximum': ['--maxThreshold', '1']}
    if criterion == 'blacklist':
        blacklist = tmp_path / 'excluded.bed'
        blacklist.write_text('chr1\t10\t20\n')
        filter_args = ['--blackListFileName', str(blacklist)]
    else:
        filter_args = options[criterion]
    computeMatrix.main(matrix_args(bw, bed, out, [*filter_args, *extra]))
    if criterion == 'maximum':
        assert_rows(out, ['one'], [0])
    else:
        assert_rows(out, ['two'], [7])


@pytest.mark.parametrize('criterion', ['skipZeros', 'minimum', 'maximum', 'blacklist'])
def test_h05_filtering_a_whole_group_does_not_require_quiet(tmp_path, criterion):
    filtered_groups(tmp_path, criterion)


@pytest.mark.parametrize('criterion', ['skipZeros', 'minimum', 'maximum', 'blacklist'])
@pytest.mark.parametrize('extra', [['--quiet'], ['--sortRegions', 'no']])
def test_filtering_empty_group_controls(tmp_path, criterion, extra):
    filtered_groups(tmp_path, criterion, extra)
