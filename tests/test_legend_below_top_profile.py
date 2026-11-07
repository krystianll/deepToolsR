"""--legendLocation below_top_profile: 'below', except that one shared
legend hangs under the first row of profiles instead of under the heatmaps.

The input is synthetic: four samples S1-S4 and two region groups G1, G2.
"""

import numpy as np
import pytest

from deeptoolsr import plotHeatmap, plotMatrix, plotProfile
from deeptoolsr.matrix import OwnedMatrix

LOCATIONS = ('below', 'below_top_profile')
SETS = ('--arrangeSamples', '1', '2', '3', '4')


@pytest.fixture
def matrix(tmp_path):
    samples, bins, regions = 4, 6, 24
    values = (np.random.default_rng(1).random((regions, samples * bins)) +
              np.arange(regions)[:, None] / regions)
    rows = [['chr1', [(i * 10, i * 10 + 5)], f'r{i}', 0, '+', '0']
            for i in range(regions)]
    parameters = {
        'upstream': [0] * samples, 'downstream': [0] * samples,
        'body': [bins] * samples, 'unscaled 5 prime': [0] * samples,
        'unscaled 3 prime': [0] * samples, 'ref point': [None] * samples,
        'bin size': [1] * samples, 'sort regions': 'keep',
        'sort using': 'mean',
        'sample_labels': [f'S{i + 1}' for i in range(samples)],
        'group_labels': ['G1', 'G2'],
        'sample_boundaries': [i * bins for i in range(samples + 1)],
        'group_boundaries': [0, regions // 2, regions]}
    path = tmp_path / 'input.gz'
    OwnedMatrix.from_compute(parameters, values.astype(np.float32),
                             rows).save(str(path), compressed=True, threads=1)
    return path


def _option(tool):
    return ('--legendLocationSummaryPlot' if tool is plotHeatmap
            else '--legendLocation')


def _png(tmp_path, matrix, tool, location, *flags):
    out = tmp_path / f'{location}.png'
    tool.main(['-m', str(matrix), '-o', str(out), '-p', '1', '--dpi', '50',
               *flags, _option(tool), location])
    return out.read_bytes()


def _figure(tmp_path, matrix, tool, location, *flags):
    built = {}
    original = plotMatrix.build_matrix_figure

    def capture(*args, **kwargs):
        figure, solution = original(*args, **kwargs)
        built['figure'] = figure
        return figure, solution

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plotMatrix, 'build_matrix_figure', capture)
        _png(tmp_path, matrix, tool, location, *flags)
    return built['figure']


@pytest.mark.parametrize('tool,flags', [
    (plotProfile, ()),
    (plotProfile, ('--commonLegend', 'no')),
    (plotProfile, ('--perGroup', '--commonLegend', 'no')),
    (plotMatrix, ('--profile', *SETS)),
    (plotMatrix, ('--profile', *SETS, '--commonLegend', 'no')),
    (plotMatrix, ('--profile', *SETS, '--sampleSetGroupArrangement',
                  'by_row', '--commonLegend', 'yes')),
])
def test_without_heatmaps_matches_below(tmp_path, matrix, tool, flags):
    assert len({_png(tmp_path, matrix, tool, location, *flags)
                for location in LOCATIONS}) == 1


@pytest.mark.parametrize('tool,flags', [
    (plotHeatmap, ('--commonLegend', 'no')),
    (plotMatrix, ('--profile', '--heatmap', *SETS, '--commonLegend', 'no')),
    (plotMatrix, ('--profile', '--heatmap', *SETS,
                  '--sampleSetGroupArrangement', 'by_row')),
])
def test_separate_legends_match_below(tmp_path, matrix, tool, flags):
    assert len({_png(tmp_path, matrix, tool, location, *flags)
                for location in LOCATIONS}) == 1


def _boxes(figure):
    """The shared legend's box and every cell axes' box, by role."""
    renderer = figure.canvas.get_renderer()
    legends = [axis.get_legend().get_window_extent(renderer)
               for axis in figure.axes if axis.get_legend() is not None]
    assert len(legends) == 1
    axes = {axis.get_gid(): axis.get_window_extent(renderer)
            for axis in figure.axes if axis.get_legend() is None}
    return legends[0], axes


def _row(gid):
    return int(gid.split('/')[1])


SHARED = [
    (plotMatrix, ('--profile', '--heatmap', *SETS,
                  '--quantileSortedRegions', '3')),
    (plotHeatmap, ()),
    (plotMatrix, ('--profile', '--heatmap', *SETS,
                  '--sampleSetGroupArrangement', 'by_row',
                  '--commonLegend', 'yes')),
]


@pytest.mark.parametrize('tool,flags', SHARED)
def test_shared_legend_hangs_under_first_profile_row(
        tmp_path, matrix, tool, flags):
    legend, axes = _boxes(_figure(tmp_path, matrix, tool,
                                  'below_top_profile', *flags))
    profiles = {gid: box for gid, box in axes.items()
                if gid.endswith('/profile')}
    first = min(_row(gid) for gid in profiles)
    above = [box for gid, box in profiles.items() if _row(gid) == first]
    under = [box for gid, box in axes.items()
             if gid.startswith(f'cell/{first}/') and '/block/' in gid]
    assert under
    assert legend.y1 <= min(box.y0 for box in above)
    assert legend.y0 >= max(box.y1 for box in under)
    left = min(box.x0 for box in profiles.values())
    right = max(box.x1 for box in profiles.values())
    assert abs((legend.x0 + legend.x1) / 2 - (left + right) / 2) <= 1
    assert not any(legend.overlaps(box) for box in axes.values())


@pytest.mark.parametrize('tool,flags', SHARED)
def test_below_keeps_shared_legend_under_heatmaps(tmp_path, matrix, tool,
                                                  flags):
    legend, axes = _boxes(_figure(tmp_path, matrix, tool, 'below', *flags))
    assert legend.y1 <= min(box.y0 for gid, box in axes.items()
                            if '/block/' in gid)
