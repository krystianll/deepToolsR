"""plotProfileR and plotHeatmapR expose the plotMatrixR options they draw."""

from pathlib import Path

import pytest

from deeptoolsr import plotHeatmap, plotMatrix, plotProfile
from deeptoolsr.plotting.matrix_spec import project_matrix_spec


DATA = Path(__file__).parent.parent / 'test_heatmapper'
MATRIX = str(DATA / 'master_multi.mat.gz')
PARTIAL = str(Path(__file__).parent.parent / 'test_data' / 'prepare_expected'
              / 'input.gz')


def _spec(module, tool, extra):
    args = module.process_args(['-m', MATRIX, '-o', 'unused.png', *extra])
    return args, project_matrix_spec(args, tool)


def _render(module, tmp_path, name, extra, matrix=MATRIX):
    out = tmp_path / f'{name}.png'
    module.main(['-m', matrix, '-o', str(out), '-p', '1', *extra])
    return out.read_bytes()


def test_profile_minor_tick_marks_reach_the_figure(tmp_path):
    args, spec = _spec(plotProfile, 'plotProfileR', ['--minorTickMarks', '4'])
    assert args.minorTickMarks == 4 and spec.minor_tick_marks == 4
    ticked = _render(plotProfile, tmp_path, 'profile', ['--minorTickMarks', '4'])
    assert ticked != _render(plotProfile, tmp_path, 'plain', [])
    assert ticked == _render(plotMatrix, tmp_path, 'matrix',
                             ['--profile', '--minorTickMarks', '4'])


def test_profile_filter_nans_drops_regions(tmp_path):
    beds = {}
    for name, module, extra in (
            ('keep', plotProfile, []),
            ('profile', plotProfile, ['--filterNans', 'any_bin']),
            ('matrix', plotMatrix, ['--profile', '--filterNans', 'any_bin'])):
        bed = tmp_path / f'{name}.bed'
        _render(module, tmp_path, name,
                [*extra, '--outFileSortedRegions', str(bed)], matrix=PARTIAL)
        beds[name] = bed.read_text().splitlines()
    assert len(beds['profile']) < len(beds['keep'])
    assert beds['profile'] == beds['matrix']


def test_profile_silhouette_column_in_sorted_regions(tmp_path, capsys):
    bed = tmp_path / 'regions.bed'
    _render(plotProfile, tmp_path, 'silhouette',
            ['--kmeans', '2', '--silhouette',
             '--outFileSortedRegions', str(bed)])
    header, *rows = bed.read_text().splitlines()
    assert header.split('\t')[-1] == 'silhouette'
    assert rows and all(len(row.split('\t')) == len(header.split('\t'))
                        for row in rows)
    assert 'average silhouette score' in capsys.readouterr().err


def test_heatmap_summary_height_reaches_the_figure(tmp_path):
    _, spec = _spec(plotHeatmap, 'plotHeatmapR', ['--heightSummaryPlot', '3'])
    assert spec.profile_height == 3
    tall = _render(plotHeatmap, tmp_path, 'tall', ['--heightSummaryPlot', '3'])
    assert tall != _render(plotHeatmap, tmp_path, 'plain', [])
    assert tall == _render(plotMatrix, tmp_path, 'matrix',
                           ['--profile', '--heatmap', '--profileHeight', '3'])


def test_heatmap_axis_visibility_sets_both_axes():
    args, spec = _spec(plotHeatmap, 'plotHeatmapR',
                       ['--xAxisVisibility', 'show_all',
                        '--yAxisVisibilitySummaryPlot', 'show_all',
                        '--axisVisibility', 'outer_merged'])
    assert args.xAxisVisibility == args.yAxisVisibility == 'outer_merged'
    assert spec.x_axis_visibility == spec.y_axis_visibility == 'outer_merged'


def test_heatmap_profile_table_matches_plot_matrix(tmp_path):
    tables = []
    for name, module, extra in (
            ('heatmap', plotHeatmap, []),
            ('matrix', plotMatrix, ['--profile', '--heatmap'])):
        table = tmp_path / f'{name}.tab'
        _render(module, tmp_path, name,
                [*extra, '--outFileNameData', str(table)])
        tables.append(table.read_bytes())
    assert tables[0] and tables[0] == tables[1]


def test_heatmap_profile_table_needs_the_summary_plot(tmp_path):
    table = tmp_path / 'none.tab'
    errors = []
    for module, extra in (
            (plotHeatmap, ['--whatToShow', 'heatmap and colorbar']),
            (plotMatrix, ['--heatmap'])):
        with pytest.raises(SystemExit) as stopped:
            _render(module, tmp_path, 'x',
                    [*extra, '--outFileNameData', str(table)])
        errors.append(str(stopped.value.code))
    # Each tool names its own way to show the profile.
    assert errors[0].endswith(
        "needs a profile panel (--whatToShow 'plot, heatmap and colorbar')")
    assert errors[1].endswith('needs a profile panel (--profile)')
    assert not table.exists()


def test_heatmap_help_sections():
    from deeptoolsr.parserCommon import plot_parser
    titles = [group.title for group in
              plot_parser('plotHeatmapR', full_color_help=False)._action_groups
              if group._group_actions]
    start = titles.index('Common options')
    assert titles[start:start + 3] == ['Common options', 'Heatmap options',
                                       'Summary plot options']
    summary = next(group for group in
                   plot_parser('plotHeatmapR')._action_groups
                   if group.title == 'Summary plot options')
    dests = {action.dest for action in summary._group_actions}
    assert {'outFileNameData', 'profileHeight', 'yAxisVisibility',
            'profileAspectRatio', 'colors', 'legendLocation'} <= dests
