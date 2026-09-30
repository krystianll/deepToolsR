"""Role-keyed rendered-output baselines for the plotting engine.

Old axes → role gids (indices in the new roles are 1-based):

  heatmap data / summary-data-i → cell/1/c/block/k / cell/1/c/profile
  sample-title-slot-i → cell/1/c/title
  region-label-slot-i → label/stack/1/1/k
  heatmap-y-label-slot-c → label/heatmap_y/1/column
  merged-x-axis-label-slot → label/x/first-column/run
  summary-y-label-slot → label/y/1/1
  colorbar-below-i → colorbar/cell/start-cell
  colorbar-right-i, colorbar-below-common-i → colorbar/figure/i
  colorbar-label-i → colorbar_title/cell/start-cell (below), or
                     colorbar_title/figure/i (side or below-common)
  unnamed sort indicator / summary legend → indicator / legend/figure/1
  profile data / series-heatmap data → cell/r/c/profile
  series-heatmap colorbar → colorbar/cell/cell-index
  profile-panel-title-slot → cell/r/c/title
  shared-scope-legend-slot → legend/scope/owner
  subplot-row-label-slot-r → label/row/r
  merged-x/y-axis-label-slot → label/x or y/anchor/run
  unnamed panel/common legend → legend/cell/index or legend/figure/1
  unnamed figure title → title

The digest detects pixel changes; the role rects identify the placed axes
that moved. Label selections are read from the independently pinned solver
snapshot, never recomputed while recording a figure.

Digests depend on the renderer and Python environment, so the baseline file
keys them by platform, architecture and library versions. Tests skip when the
current environment has no digest, or fail under ``--require-pixels``.
Regenerate with::

    pytest tests/test_plot_baselines.py --rebaseline

The geometry summaries are compared on every environment, since the solver is
arithmetic over measured extents and only the measurements are renderer-bound.
"""

import hashlib
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import pytest

from tests.helpers.artist_state import artist_state
from tests.helpers.plot_data import plot_data
from tests.helpers.heatmap_build import render_heatmap
from tests.helpers.profile_build import build_profile_case
from deeptoolsr.matrix import Matrix


DATA = Path(__file__).parent / 'test_heatmapper'
BASELINE = Path(__file__).parent / 'test_baselines' / 'plot_baselines.json'
SELECTIONS = Path(__file__).parent / 'contract' / 'label_selections.json'

DEFAULT_COLORS = {'colorMap': ['Reds'], 'colorList': None, 'colorNumber': 256,
                  'missingDataColor': 'black', 'alpha': 1.0}

# One fixed DPI for every case. Measurement is FreeType-hinted and therefore
# non-linear in DPI, so a case that varied it would be recording a second
# variable rather than isolating the change under test.
DPI = 100


def _colors(**overrides):
    merged = dict(DEFAULT_COLORS)
    merged.update(overrides)
    return merged


# Cases are chosen to separate the things later phases touch: the colour-limit
# resolution (zMin/zMax/zMid), the missing-data colour and colour map that feed
# the raster LUT, the geom/stat distinction hiding inside plot_type, the panel
# geometry knobs, and the per-group partition that splits one declared height
# across several blocks.
HEATMAP_CASES = {
    'heatmap_basic': dict(
        matrix='master.mat.gz'),
    'heatmap_multi': dict(
        matrix='master_multi.mat.gz'),
    'heatmap_multi_pergroup': dict(
        matrix='master_multi.mat.gz', perGroup=True),
    'heatmap_explicit_limits': dict(
        matrix='master_multi.mat.gz', zMin=[0.0], zMax=[5.0]),
    'heatmap_zmid': dict(
        matrix='master_multi.mat.gz', zMin=[0.0], zMid=[1.0], zMax=[5.0]),
    'heatmap_missing_colour': dict(
        matrix='master_multi.mat.gz',
        colorMapDict=_colors(missingDataColor='white')),
    'heatmap_colormap': dict(
        matrix='master_multi.mat.gz',
        colorMapDict=_colors(colorMap=['viridis'])),
    'heatmap_geom_fill': dict(
        matrix='master_multi.mat.gz', plotType='fill'),
    'heatmap_stat_se': dict(
        matrix='master_multi.mat.gz', averageType='mean', plotType='se'),
    'heatmap_dimensions': dict(
        matrix='master_multi.mat.gz', heatmapWidth=4.0, heatmapHeight=12.0),
    'heatmap_labels': dict(
        matrix='master_multi.mat.gz', plotTitle='A title',
        xAxisLabel='distance', heatmapYAxisLabel='signal'),
    'heatmap_no_box': dict(
        matrix='master_multi.mat.gz', boxAroundHeatmaps=False),
    'heatmap_summary_legend': dict(
        matrix='master_multi.mat.gz', legendLocation='above'),
    'heatmap_x_show_all': dict(
        matrix='master_multi.mat.gz', xAxisVisibility='show_all'),
}

PROFILE_CASES = {
    'profile_basic': dict(
        matrix='master_multi.mat.gz'),
    'profile_pergroup': dict(
        matrix='master_multi.mat.gz', per_group=True),
    'profile_fill': dict(
        matrix='master_multi.mat.gz', plot_type='fill'),
    'profile_se': dict(
        matrix='master_multi.mat.gz', averagetype='mean', plot_type='se'),
    # The auxiliary heatmap mode uses a different finalizer from line profiles,
    # so it has its own case.
    'profile_heatmap_mode': dict(
        matrix='master_multi.mat.gz', plot_type='heatmap'),
    'profile_heatmap_mode_pergroup': dict(
        matrix='master_multi.mat.gz', plot_type='heatmap', per_group=True),
    'profile_dimensions': dict(
        matrix='master_multi.mat.gz', plot_width=4.0, plot_height=3.0),
    'profile_panel_legends': dict(
        matrix='master_multi.mat.gz', legend_location='above'),
    'profile_common_legend': dict(
        matrix='master_multi.mat.gz', legend_location='above',
        subplot_common_legend=True),
    'profile_grid_legends': dict(
        matrix='master_multi.mat.gz', legend_location='above',
        subplot_group_placement='by_row'),
    'profile_right_legends': dict(
        matrix='master_multi.mat.gz', grid_columns=2,
        legend_location='right'),
    # Pins the gap between the figure title and the profile titles.
    'profile_title': dict(
        matrix='master_multi.mat.gz', plot_title='A profile title'),
    'profile_x_runs': dict(
        matrix='master_multi.mat.gz', grid_columns=1,
        reference_point_label=['TSS', 'TES', 'center', 'peak']),
}


# Cell layouts are exercised through the public plotMatrixR command.
# These commands share the 2-group, 4-sample matrix used by the older cases.
MATRIX_CASES = {
    'matrix_multi_sample_cell': ('--heatmap', '--profile',
                                 '--arrangeSamples', '1,2', '3,4',
                                 '--colorbarLocation', 'below'),
    'matrix_pergroup_sets': ('--heatmap', '--perGroup',
                             '--arrangeSamples', '1', '2', '3', '4'),
    'matrix_by_row': ('--heatmap', '--profile',
                      '--sampleSetGroupArrangement', 'by_row'),
    'matrix_by_column': ('--heatmap',
                         '--sampleSetGroupArrangement', 'by_column'),
    'matrix_grid_columns': ('--heatmap', '--profile', '--gridColumns', '1'),
    'matrix_x_runs': ('--heatmap', '--gridColumns', '1',
                      '--refPointLabel', 'A', 'A', 'B', 'B'),
    'matrix_colorbar_scopes_right': (
        '--heatmap', '--profile', '--gridColumns', '1',
        '--colorbarLocation', 'right'),
    'matrix_colorbar_scopes_below': (
        '--heatmap', '--profile', '--gridColumns', '1',
        '--colorbarLocation', 'below'),
    'matrix_colorbar_scopes_below_common': (
        '--heatmap', '--profile', '--gridColumns', '1',
        '--colorbarLocation', 'below_common'),
    'matrix_ci_summary': ('--heatmap', '--profile', '--plotType', 'ci'),
    'matrix_label_runs': ('--heatmap', '--heatmapYAxisLabel', 'A', 'A', 'B', 'B'),
    'matrix_series_heatmap_grid': ('--profile', '--plotType', 'heatmap',
                                   '--gridColumns', '2'),
    'matrix_profile_only_equivalence': (
        '--profile', '--sortRegions', 'ascend',
        '--yAxisLimits', 'per_sample_set'),
}


def _load_matrix(name):
    return Matrix.load(str(DATA / name), threads=1)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _role_rects(figure):
    """Capture every final axes rect, requiring unique and complete roles."""
    width_points = figure.get_figwidth() * 72.0
    height_points = figure.get_figheight() * 72.0
    roles = {}
    for axis in figure.axes:
        role = axis.get_gid()
        assert role, 'placed axes has no role gid'
        assert role not in roles, 'duplicate plot role: {}'.format(role)
        bounds = axis.get_position().bounds
        roles[role] = [round(bounds[0] * width_points, 4),
                       round(bounds[1] * height_points, 4),
                       round(bounds[2] * width_points, 4),
                       round(bounds[3] * height_points, 4)]
    return roles


def _render_heatmap(case, out_path):
    options = dict(case)
    hm = _load_matrix(options.pop('matrix'))
    colors = options.pop('colorMapDict', DEFAULT_COLORS)
    what_to_show = options.pop('whatToShow', 'plot, heatmap and colorbar')
    figure, solution = render_heatmap(
        plot_data(hm), out_path, colorMapDict=colors,
        whatToShow=what_to_show, image_format='png', dpi=DPI,
        threads=1, **options)
    return solution, figure


def _render_profile(case, out_path):
    options = dict(case)
    hm = _load_matrix(options.pop('matrix'))
    # The direct-render defaults are intentionally different from CLI defaults.
    options.setdefault('y_min', [None])
    options.setdefault('y_max', [None])
    built = build_profile_case(
        plot_data(hm), out_path, image_format='png', dpi=DPI, **options)
    return built.solution, built.figure


def _render_matrix(case, out_path):
    from deeptoolsr import plotMatrix

    built = []
    original = plotMatrix.build_matrix_figure

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        built.append(result)
        return result

    argv = ['-m', str(DATA / 'master_multi.mat.gz'), '-o', str(out_path),
            '--dpi', str(DPI), '-p', '1', *case]
    with patch.object(plotMatrix, 'build_matrix_figure', capture):
        plotMatrix.main(argv)
    assert len(built) == 1
    figure, solution = built[0]
    return solution, figure


def _record(kind, case, tmp_path, name):
    out_path = tmp_path / '{}.png'.format(name)
    render = {'heatmap': _render_heatmap, 'profile': _render_profile,
              'matrix': _render_matrix}[kind]
    _, figure = render(case, out_path)
    return {
        'figure_size': [round(figure.get_figwidth() * 72.0, 4),
                        round(figure.get_figheight() * 72.0, 4)],
        'roles': _role_rects(figure),
        # Saved (rendered) tick state of visible axes; auto locators settle
        # at savefig, after the scene records' artist_state was captured.
        'artist_state': {axis.get_gid(): state for axis in figure.axes
                         if axis.axison and (state := artist_state(axis))},
        'label_selections': json.loads(SELECTIONS.read_text())['baselines'][name],
        'png_sha256': _digest(out_path),
    }


def _environment():
    return (f'{platform.system()}-{platform.machine()}-'
            f'py{sys.version_info.major}.{sys.version_info.minor}-'
            f'mpl{matplotlib.__version__}-'
            f'ft{matplotlib.ft2font.__freetype_version__}')


def _load_baseline():
    if not BASELINE.exists():
        return None
    return json.loads(BASELINE.read_text())


ALL_CASES = [('heatmap', name, case) for name, case in HEATMAP_CASES.items()]
ALL_CASES += [('profile', name, case) for name, case in PROFILE_CASES.items()]
ALL_CASES += [('matrix', name, case) for name, case in MATRIX_CASES.items()]


@pytest.fixture(scope='session')
def baseline():
    stored = _load_baseline()
    if stored is None:
        pytest.skip('no baseline recorded; run with --rebaseline')
    return stored


@pytest.mark.parametrize('kind,name,case',
                         ALL_CASES, ids=[entry[1] for entry in ALL_CASES])
def test_plot_geometry_matches_baseline(kind, name, case, baseline, tmp_path):
    """The solved layout is unchanged.

    Geometry is compared on every platform. The solver is pure arithmetic over
    measured extents, so a difference here is either a real layout change or a
    measurement change -- both of which are worth failing on.
    """
    expected = baseline['geometry'].get(name)
    if expected is None:
        pytest.skip(
            'case {!r} is not in the baseline; rebaseline'.format(name))
    actual = _record(kind, case, tmp_path, name)
    actual.pop('png_sha256')
    assert actual == expected, 'solved geometry changed for case {!r}'.format(name)


@pytest.mark.parametrize('kind,name,case',
                         ALL_CASES, ids=[entry[1] for entry in ALL_CASES])
def test_plot_pixels_match_baseline(kind, name, case, baseline, tmp_path, request):
    """The rendered pixels are unchanged.

    Skipped when the current renderer has no digest, unless the pixel gate
    requires one for every case.
    """
    current = _environment()
    expected = baseline['digests'].get(current, {}).get(name)
    if expected is None:
        message = 'no pixel digest for {!r} under {}'.format(name, current)
        if request.config.getoption('--require-pixels'):
            pytest.fail(message)
        pytest.skip(message)
    actual = _record(kind, case, tmp_path, name)
    assert actual['png_sha256'] == expected, (
        'rendered pixels changed for case {!r} while its geometry did not; '
        'look at drawing or colour mapping'.format(name))


def test_heatmap_show_all_keeps_each_summary_x_axis(tmp_path):
    _, figure = _render_heatmap(
        HEATMAP_CASES['heatmap_x_show_all'], tmp_path / 'heatmap.png')
    summaries = [axis for axis in figure.axes
                 if (axis.get_gid() or '').endswith('/profile')]
    assert len(summaries) == 4
    assert all(axis.get_xlabel() for axis in summaries)
    assert all(any(tick.get_visible() and tick.get_text()
                   for tick in axis.get_xticklabels()) for axis in summaries)


def test_profile_right_legends_keep_one_slot_per_cell(tmp_path):
    _, figure = _render_profile(
        PROFILE_CASES['profile_right_legends'], tmp_path / 'profile.png')
    assert {axis.get_gid() for axis in figure.axes
            if (axis.get_gid() or '').startswith('legend/')} == {
                'legend/cell/1', 'legend/cell/2',
                'legend/cell/3', 'legend/cell/4'}


def test_profile_distinct_x_runs_keep_each_label(tmp_path):
    _, figure = _render_profile(
        PROFILE_CASES['profile_x_runs'], tmp_path / 'profile.png')
    slots = {axis.get_gid(): axis.texts[0].get_text()
             for axis in figure.axes
             if (axis.get_gid() or '').startswith('label/x/')}
    assert slots == {f'label/x/1/{index}': f'distance from {label}'
                     for index, label in enumerate(
                         ('TSS', 'TES', 'center', 'peak'), 1)}
    panels = [axis for axis in figure.axes
              if (axis.get_gid() or '').endswith('/profile')]
    assert len(panels) == 4
    assert all(any(tick.get_visible() and tick.get_text()
                   for tick in axis.get_xticklabels()) for axis in panels)


def test_rebaseline(request, tmp_path):
    """Regenerate the baseline file. Only runs under --rebaseline."""
    if not request.config.getoption('--rebaseline'):
        pytest.skip('use --rebaseline to regenerate')
    geometry = {}
    digests = dict((_load_baseline() or {}).get('digests', {}))
    current_digests = {}
    for kind, name, case in ALL_CASES:
        record = _record(kind, case, tmp_path, name)
        current_digests[name] = record.pop('png_sha256')
        geometry[name] = record
    digests[_environment()] = current_digests
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(
        {'geometry': geometry, 'digests': digests},
        indent=2, sort_keys=True) + '\n')


def test_rebaseline_keeps_other_environment_digests(tmp_path, monkeypatch):
    baseline_path = tmp_path / 'baseline.json'
    baseline_path.write_text(json.dumps({
        'geometry': {'old': {'figure_size': [1, 2]}},
        'digests': {'other-env': {'old': 'older-digest'}},
    }))
    monkeypatch.setitem(globals(), 'BASELINE', baseline_path)
    monkeypatch.setitem(globals(), 'ALL_CASES', [('heatmap', 'new', {})])
    monkeypatch.setitem(globals(), '_record', lambda *_: {
        'png_sha256': 'new-digest', 'figure_size': [3, 4]})
    request = SimpleNamespace(config=SimpleNamespace(
        getoption=lambda option: option == '--rebaseline'))
    test_rebaseline(request, tmp_path)
    updated = json.loads(baseline_path.read_text())
    assert updated['geometry'] == {'new': {'figure_size': [3, 4]}}
    assert updated['digests']['other-env'] == {'old': 'older-digest'}
    assert updated['digests'][_environment()] == {'new': 'new-digest'}
