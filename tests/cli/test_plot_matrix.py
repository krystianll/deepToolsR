"""The shared plot front end preserves the two established commands."""

from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from deeptoolsr import plotHeatmap, plotMatrix, plotProfile
from deeptoolsr.matrix import Matrix, OwnedMatrix
from deeptoolsr.plotting import matrix_figure
from tests.test_plot_baselines import (
    HEATMAP_CASES, MATRIX_CASES, PROFILE_CASES)


DATA = Path(__file__).parent.parent / 'test_heatmapper'
PARTIAL = Path(__file__).parent.parent / 'test_data' / 'prepare_expected' / 'input.gz'


def _case_flags(kind, case):
    flags = []
    names = {
        'plot_type': 'plotType', 'averagetype': 'averageType',
        'plot_width': 'plotWidth', 'plot_height': 'profileHeight',
        'legend_location': 'legendLocation',
        'subplot_group_placement': 'sampleSetGroupArrangement',
        'grid_columns': 'gridColumns',
        'reference_point_label': 'refPointLabel',
        'per_group': 'perGroup',
        'plot_title': 'plotTitle',
    }
    for name, value in case.items():
        if name in ('matrix', 'colorMapDict') or value is None:
            continue
        if name == 'boxAroundHeatmaps':
            flags.extend(['--boxAroundHeatmaps', 'yes' if value else 'no'])
            continue
        if value is False:
            continue
        if name == 'subplot_common_legend':
            flags.append('--commonLegend')
        elif isinstance(value, bool):
            flags.append('--' + names.get(name, name))
        else:
            option = names.get(name, name)
            if kind == 'heatmap' and option == 'heatmapYAxisLabel':
                option = 'yAxisLabel'
            if kind == 'heatmap' and option in ('plotType', 'averageType',
                                                'legendLocation'):
                option += 'SummaryPlot'
            if option == 'boxAroundHeatmaps':
                value = 'yes' if value else 'no'
            flags.extend(['--' + option, *map(str, value if isinstance(
                value, (list, tuple)) else (value,))])
    for name, value in case.get('colorMapDict', {}).items():
        if value is not None:
            flags.extend(['--' + name, *map(str, value if isinstance(
                value, (list, tuple)) else (value,))])
    return flags


@pytest.mark.parametrize('kind,name,case', [
    *[('heatmap', name, case) for name, case in HEATMAP_CASES.items()],
    *[('profile', name, case) for name, case in PROFILE_CASES.items()],
])
def test_plot_matrix_matches_wrapper_outputs(kind, name, case, tmp_path):
    wrapper = plotHeatmap if kind == 'heatmap' else plotProfile
    common = ['-m', str(DATA / case['matrix']), '-p', '1', '--dpi', '100',
              '--sortRegions', 'no', *_case_flags(kind, case)]
    matrix_flags = ['--heatmap', '--profile'] if kind == 'heatmap' else ['--profile']
    if kind == 'heatmap':
        replacements = {'--plotTypeSummaryPlot': '--plotType',
                        '--averageTypeSummaryPlot': '--averageType',
                        '--legendLocationSummaryPlot': '--legendLocation',
                        '--yAxisLabel': '--heatmapYAxisLabel',
                        '--heatmapWidth': '--cellWidth'}
        translated = [replacements.get(item, item) for item in common]
    else:
        translated = [('--cellWidth' if item == '--plotWidth' else item)
                      for item in common]
        translated += ['--yAxisLimits', 'per_sample_set']
    outputs = []
    commands = ((wrapper, common),
                (plotMatrix, [*translated, *matrix_flags]))
    for command, flags in commands:
        stem = 'wrapper' if command is wrapper else 'matrix'
        paths = [tmp_path / f'{stem}{suffix}' for suffix in
                 ('.png', '.tsv', '.mat', '.bed')]
        argv = [*flags, '-o', str(paths[0]),
                '--outFileNameMatrix', str(paths[2]),
                '--outFileSortedRegions', str(paths[3])]
        if kind == 'profile':
            argv += ['--outFileNameData', str(paths[1])]
        command.main(argv)
        outputs.append(tuple(path.read_bytes() for path in paths if path.exists()))
    assert outputs[0] == outputs[1], name


@pytest.mark.parametrize('name', [
    name for name in MATRIX_CASES if name != 'matrix_ci_summary'])
def test_new_matrix_cases_match_wrapper_outputs(name, tmp_path):
    """Equivalent wrapper spellings preserve every available file byte."""
    case = MATRIX_CASES[name]
    heatmap = '--heatmap' in case
    profile = '--profile' in case
    wrapper = plotHeatmap if heatmap else plotProfile
    wrapper_flags = [flag for flag in case
                     if flag not in ('--heatmap', '--profile')]
    if heatmap:
        spellings = {'--plotType': '--plotTypeSummaryPlot',
                     '--heatmapYAxisLabel': '--yAxisLabel'}
        wrapper_flags = [spellings.get(flag, flag) for flag in wrapper_flags]
        if not profile:
            wrapper_flags.extend(['--whatToShow', 'heatmap and colorbar'])
    # plotHeatmapR deliberately offers only lines/fill/se/std summaries;
    # matrix_ci_summary has no equivalent wrapper spelling.
    outputs = []
    for command, flags, stem in ((plotMatrix, case, 'matrix'),
                                 (wrapper, wrapper_flags, 'wrapper')):
        paths = {suffix: tmp_path / f'{stem}.{suffix}'
                 for suffix in ('png', 'mat', 'bed', 'tsv')}
        argv = ['-m', str(DATA / 'master_multi.mat.gz'), '-p', '1',
                '--dpi', '100', *flags, '-o', str(paths['png']),
                '--outFileNameMatrix', str(paths['mat']),
                '--outFileSortedRegions', str(paths['bed'])]
        if not heatmap:
            argv.extend(['--outFileNameData', str(paths['tsv'])])
        command.main(argv)
        outputs.append({suffix: path.read_bytes() for suffix, path in
                        paths.items() if path.exists()})
    assert outputs[0] == outputs[1], name


@pytest.mark.parametrize('extra', [
    ['--arrangeSamples', '1,2'], ['--sampleSetGroupArrangement', 'adjacent'],
    ['--gridColumns', '2'], ['--commonLegend'],
    ['--sameGroupLabels', 'together'], ['--sameGroupLabels', 'merge'],
    ['--sameSampleLabels', 'together'],
    ['--plotType', 'ci'],
])
def test_formerly_guarded_heatmap_modes_render_and_describe(extra, tmp_path):
    argv = ['-m', str(DATA / 'master_multi.mat.gz'),
            '-o', str(tmp_path / 'out.png'), '--profile', '--heatmap', *extra]
    plotMatrix.main(argv)
    assert (tmp_path / 'out.png').stat().st_size > 0
    from deeptoolsr.describe import describe
    record = describe('plotMatrixR', argv)
    assert record['cells']
    assert any(cell['blocks'] for cell in record['cells'])


def test_no_panel_and_series_heatmap_conflict(tmp_path, capsys):
    base = ['-m', str(DATA / 'master.mat.gz'), '-o', str(tmp_path / 'x.png')]
    with pytest.raises(SystemExit) as stopped:
        plotMatrix.main(base)
    assert stopped.value.code == 2
    assert 'choose --profile, --heatmap or both' in capsys.readouterr().err
    with pytest.raises(SystemExit) as stopped:
        plotMatrix.main([*base, '--heatmap', '--plotType', 'heatmap'])
    assert stopped.value.code == 2
    assert 'cannot be combined with --heatmap' in capsys.readouterr().err


def test_profile_heatmap_options_are_inert(tmp_path):
    base = ['-m', str(DATA / 'master.mat.gz'), '--profile', '-p', '1']
    first, second = tmp_path / 'first.png', tmp_path / 'second.png'
    plotMatrix.main([*base, '-o', str(first)])
    plotMatrix.main([*base, '-o', str(second), '--zMin', '0',
                     '--colorMap', 'viridis'])
    assert first.read_bytes() == second.read_bytes()


def test_filter_nans_is_shared_membership(tmp_path):
    beds = []
    for mode in ('--profile', '--heatmap'):
        plot, bed = tmp_path / (mode[2:] + '.png'), tmp_path / (mode[2:] + '.bed')
        plotMatrix.main(['-m', str(PARTIAL), '-o', str(plot), mode,
                         '--filterNans', 'any_bin', '--outFileSortedRegions',
                         str(bed), '-p', '1'])
        beds.append(bed.read_bytes())
    assert beds[0] == beds[1]


@pytest.mark.parametrize('choice', ['plot and heatmap', 'heatmap only'])
def test_removed_heatmap_choices_are_usage_errors(choice, capsys):
    with pytest.raises(SystemExit) as stopped:
        plotHeatmap.process_args(['-m', 'unused', '-o', 'unused.png',
                                  '--whatToShow', choice])
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert 'plot, heatmap and colorbar' in error
    assert 'heatmap and colorbar' in error


def test_plot_matrix_error_spells_its_y_min(capsys):
    with pytest.raises(SystemExit) as stopped:
        plotMatrix.process_args(['-m', 'unused', '-o', 'unused.png', '--profile',
                                 '--yMin', 'bad'])
    assert stopped.value.code == 2
    assert '--yMin' in capsys.readouterr().err


def test_heatmap_wrapper_arrangement_renders_and_describes(tmp_path):
    argv = ['-m', str(DATA / 'master_multi.mat.gz'),
            '-o', str(tmp_path / 'out.png'), '--arrangeSamples', '1,2', '3,4']
    from deeptoolsr.describe import describe
    plotHeatmap.main(argv)
    assert (tmp_path / 'out.png').stat().st_size > 0
    record = describe('plotHeatmapR', argv)
    assert [cell['samples'] for cell in record['cells']] == [[1, 2], [3, 4]]


def test_profile_height_with_heatmap_uses_derived_aspect(tmp_path):
    matrix = DATA / 'master.mat.gz'
    wrapper, shared = tmp_path / 'wrapper.png', tmp_path / 'shared.png'
    plotHeatmap.main(['-m', str(matrix), '-o', str(wrapper), '-p', '1',
                      '--heatmapWidth', '4',
                      '--aspectRatioSummaryPlot', '2'])
    plotMatrix.main(['-m', str(matrix), '-o', str(shared), '-p', '1',
                     '--profile', '--heatmap', '--cellWidth', '4',
                     '--profileHeight', '2'])
    assert wrapper.read_bytes() == shared.read_bytes()


def test_profile_only_explicit_defaults_match_profile_wrapper_bytes(tmp_path):
    base = ['-m', str(DATA / 'master_multi.mat.gz'), '-p', '1']
    outputs = []
    for stem, command, extra in (
            ('profile', plotProfile, ()),
            ('matrix', plotMatrix, ('--profile', '--sortRegions', 'ascend',
                                    '--yAxisLimits', 'per_sample_set'))):
        paths = [tmp_path / f'{stem}.{suffix}' for suffix in
                 ('png', 'tsv', 'mat', 'bed')]
        command.main([*base, *extra, '-o', str(paths[0]),
                      '--outFileNameData', str(paths[1]),
                      '--outFileNameMatrix', str(paths[2]),
                      '--outFileSortedRegions', str(paths[3])])
        outputs.append(tuple(path.read_bytes() for path in paths))
    assert outputs[0] == outputs[1]


def _capture_prepared(tmp_path, monkeypatch, stem, *flags, matrix_path=None,
                      draw=True):
    """Observe request inputs at the CLI's preparation and raster boundary."""
    captured = {'rasters': []}
    original_build = plotMatrix.build_matrix_figure
    original_flush = matrix_figure.flush_deferred_heatmap_rasters

    def build(matrix, layout, labels, plan, spec, prepared, run, extents,
              **kwargs):
        captured.update(layout=layout, plan=plan, prepared=prepared)
        if not draw:
            figure = Figure()
            FigureCanvasAgg(figure)
            return figure, None
        renderer = kwargs.get('raster_renderer')
        if renderer is not None:
            def capture_rasters(requests, threads):
                captured['rasters'].extend(requests)
                return renderer(requests, threads)

            kwargs['raster_renderer'] = capture_rasters
        return original_build(
            matrix, layout, labels, plan, spec, prepared, run, extents,
            **kwargs)

    def flush(requests, threads):
        captured['rasters'].extend(requests)
        return original_flush(requests, threads)

    argv = ['-m', str(matrix_path or DATA / 'master_multi.mat.gz'), '-p', '1',
            '-o', str(tmp_path / (stem + '.png')), *flags]
    with monkeypatch.context() as patch:
        patch.setattr(plotMatrix, 'build_matrix_figure', build)
        patch.setattr(matrix_figure, 'flush_deferred_heatmap_rasters', flush)
        plotMatrix.main(argv)
    return captured


def _raster_request_key(request):
    block = request['data']
    return (
        request['rows'], request['cols'], request['vmin'], request['vmax'],
        request['out_h'], request['out_w'], request['interpolation'],
        request['alpha'], request['lut'].tobytes(), tuple(request['bad']),
        block.row_range, block.col_range,
        None if block.rows is None else block.rows.tobytes(),
        None if block.cols is None else block.cols.tobytes())


def _statistics(prepared, plan):
    return prepared.statistics_by_key


def _interval_bytes(statistics):
    return {key: (value.center.tobytes(),
                  None if value.lower is None else value.lower.tobytes(),
                  None if value.upper is None else value.upper.tobytes())
            for key, value in statistics.items()}


def test_profile_toggle_keeps_heatmap_raster_requests_and_limits(
        tmp_path, monkeypatch):
    heatmap = _capture_prepared(tmp_path, monkeypatch, 'heatmap',
                                '--heatmap', '--sortRegions', 'no')
    both = _capture_prepared(tmp_path, monkeypatch, 'both',
                             '--heatmap', '--profile', '--sortRegions', 'no')
    assert heatmap['layout'].digest() == both['layout'].digest()
    assert heatmap['rasters'] and both['rasters']
    assert [_raster_request_key(request) for request in heatmap['rasters']] == \
        [_raster_request_key(request) for request in both['rasters']]
    for name in ('percentiles', 'z_min', 'z_max', 'z_mid'):
        left = getattr(heatmap['prepared'], name)
        right = getattr(both['prepared'], name)
        assert (left is right is None or np.array_equal(
            left, right, equal_nan=True))


@pytest.mark.parametrize('plot_type', ('ci', 'bootstrap'))
def test_heatmap_toggle_keeps_statistics_when_data_spec_is_unchanged(
        tmp_path, monkeypatch, plot_type):
    # `ci` is the analytic t interval; `bootstrap` is resampling based.
    options = ('--profile', '--plotType', plot_type, '--averageType', 'mean',
               '--sortRegions', 'keep')
    profile = _capture_prepared(tmp_path, monkeypatch, 'profile', *options,
                                draw=False)
    both = _capture_prepared(tmp_path, monkeypatch, 'both',
                             *options, '--heatmap', draw=False)
    assert profile['layout'].digest() == both['layout'].digest()
    assert _interval_bytes(_statistics(profile['prepared'], profile['plan'])) == \
        _interval_bytes(_statistics(both['prepared'], both['plan']))


def test_heatmap_toggle_changes_order_sensitive_bootstrap_intervals(
        tmp_path, monkeypatch):
    source = Matrix.load(str(DATA / 'master_multi.mat.gz'), 1)
    values = np.random.default_rng(17).lognormal(
        size=(24, source.values.shape[1])).astype(np.float32)
    values *= np.linspace(0.5, 2.0, 24, dtype=np.float32)[:, None]
    parameters = dict(source.header.parameters)
    parameters['group_boundaries'] = [0, 12, 24]
    regions = [source.regions[row % 3 + (row // 12) * 3]
               for row in range(24)]
    path = tmp_path / 'order_sensitive.gz'
    OwnedMatrix.from_compute(parameters, values, regions).save(
        path, compressed=True, threads=1)
    options = ('--profile', '--plotType', 'bootstrap',
               '--averageType', 'mean', '--sortRegions', 'descend')
    profile = _capture_prepared(
        tmp_path, monkeypatch, 'profile', *options, matrix_path=path,
        draw=False)
    both = _capture_prepared(tmp_path, monkeypatch, 'both',
                             *options, '--heatmap', matrix_path=path,
                             draw=False)
    assert profile['layout'].digest() != both['layout'].digest()
    first = _interval_bytes(_statistics(profile['prepared'], profile['plan']))
    second = _interval_bytes(_statistics(both['prepared'], both['plan']))
    assert first.keys() == second.keys()
    assert any(first[key][1:] != second[key][1:] for key in first)


def test_profile_table_is_written_with_or_without_heatmaps(tmp_path):
    matrix = str(DATA / 'master_multi.mat.gz')
    tables = []
    for flags in (('--profile',), ('--profile', '--heatmap')):
        table = tmp_path / f'{len(flags)}.tsv'
        plotMatrix.main(['-m', matrix, '-o', str(tmp_path / 'plot.png'),
                         '-p', '1', '--outFileNameData', str(table), *flags])
        tables.append(table.read_bytes())
    assert tables[0] and tables[0] == tables[1]
    with pytest.raises(SystemExit, match='needs a profile panel'):
        plotMatrix.main(['-m', matrix, '-o', str(tmp_path / 'plot.png'),
                         '--heatmap', '--outFileNameData',
                         str(tmp_path / 'none.tsv')])
    assert not (tmp_path / 'none.tsv').exists()
