"""The shared cell-width solver applies before any plotting geometry."""

import pytest

from deeptoolsr.plotting.sizes import SizeError, solve_sizes
from deeptoolsr.prepare import OptionError, parse_command
from deeptoolsr.plotting.matrix_spec import project_matrix_spec
from deeptoolsr.parserCommon import plot_parser


@pytest.mark.parametrize('given,expected', [
    ({'cell_width': 8, 'profile_height': 4}, (8, 4, 2)),
    ({'cell_width': 8, 'profile_aspect_ratio': 2}, (8, 4, 2)),
    ({'profile_height': 4, 'profile_aspect_ratio': 2}, (8, 4, 2)),
])
def test_profile_two_of_three(given, expected):
    solved = solve_sizes(**given)
    assert (solved.cell_width, solved.profile_height,
            solved.profile_aspect_ratio) == expected


@pytest.mark.parametrize('given,expected', [
    ({'cell_width': 8, 'heatmap_height': 4}, (8, 4, 2)),
    ({'cell_width': 8, 'heatmap_aspect_ratio': 2}, (8, 4, 2)),
    ({'heatmap_height': 4, 'heatmap_aspect_ratio': 2}, (8, 4, 2)),
])
def test_heatmap_two_of_three(given, expected):
    solved = solve_sizes(show_profile=False, show_heatmap=True, **given)
    assert (solved.cell_width, solved.heatmap_height,
            solved.heatmap_aspect_ratio) == expected


def test_cross_kind_width_propagates():
    solved = solve_sizes(show_heatmap=True, profile_height=4,
                         profile_aspect_ratio=2, heatmap_aspect_ratio=1)
    assert (solved.cell_width, solved.heatmap_height) == (8, 8)


def test_explicit_kind_widths_conflict_with_option_names():
    with pytest.raises(SizeError) as error:
        solve_sizes(show_heatmap=True, profile_height=10,
                    profile_aspect_ratio=1, heatmap_height=4,
                    heatmap_aspect_ratio=.5)
    message = str(error.value)
    assert all(option in message for option in (
        '--profileHeight', '--profileAspectRatio',
        '--heatmapHeight', '--heatmapAspectRatio'))
    assert '10 cm' in message and '2 cm' in message


def test_relative_consistency_tolerance():
    solved = solve_sizes(cell_width=10, profile_height=10,
                         profile_aspect_ratio=1.0009)
    assert solved.profile_aspect_ratio == 1
    with pytest.raises(SizeError):
        solve_sizes(cell_width=10, profile_height=10,
                    profile_aspect_ratio=1.002)


def test_only_ratio_uses_kind_default_height_before_width():
    assert solve_sizes(profile_aspect_ratio=2).cell_width == 10
    heatmap = solve_sizes(show_profile=False, show_heatmap=True,
                          heatmap_aspect_ratio=.4)
    assert (heatmap.cell_width, heatmap.heatmap_height) == (4, 10)


def test_inert_kind_and_defaults():
    profile = solve_sizes(show_heatmap=False, heatmap_height=99,
                          heatmap_aspect_ratio=100)
    assert (profile.cell_width, profile.profile_height,
            profile.heatmap_height) == (5, 5, None)
    heatmap = solve_sizes(show_profile=False, show_heatmap=True)
    assert (heatmap.cell_width, heatmap.heatmap_height,
            heatmap.profile_height) == (5, 10, None)
    both = solve_sizes(show_heatmap=True)
    assert (both.cell_width, both.profile_height,
            both.heatmap_height) == (5, 5, 10)


@pytest.mark.parametrize('given,option', [
    ({'cell_width': 101}, '--cellWidth'),
    ({'profile_height': .4}, '--profileHeight'),
    ({'show_profile': False, 'show_heatmap': True,
      'heatmap_height': 2.9}, '--heatmapHeight'),
    ({'profile_height': 100, 'profile_aspect_ratio': 1.01},
     '--profileAspectRatio'),
])
def test_bounds_after_solving_name_input(given, option):
    with pytest.raises(SizeError, match=option):
        solve_sizes(**given)


@pytest.mark.parametrize('tool,flags,expected', [
    ('plotProfileR', [], (5, 5, None)),
    ('plotHeatmapR', [], (5, 5, 10)),
    ('plotMatrixR', ['--profile'], (5, 5, None)),
    ('plotMatrixR', ['--heatmap'], (5, None, 10)),
    ('plotMatrixR', ['--profile', '--heatmap'], (5, 5, 10)),
])
def test_tool_defaults(tool, flags, expected):
    args = parse_command(tool, ['-m', 'unused', *flags], 'worker')
    spec = project_matrix_spec(args, tool)
    assert (spec.cell_width, spec.profile_height,
            spec.heatmap_height) == expected


@pytest.mark.parametrize('flag', (
    '--profileWidth', '--plotWidth', '--heatmapWidth'))
def test_matrix_rejects_removed_width_spellings(flag):
    with pytest.raises(OptionError, match='unrecognized arguments'):
        parse_command('plotMatrixR', ['-m', 'unused', '--profile',
                                      flag, '5'], 'worker')


@pytest.mark.parametrize('flag', (
    '--profileWidth', '--plotWidth', '--heatmapWidth'))
def test_matrix_cli_rejects_removed_width_spellings(flag, capsys):
    with pytest.raises(SystemExit) as error:
        parse_command('plotMatrixR', ['-m', 'unused', '-o', 'unused.png',
                                      '--profile', flag, '5'], 'cli')
    assert error.value.code == 2
    assert f'unrecognized arguments: {flag} 5' in capsys.readouterr().err


@pytest.mark.parametrize('wrapper,flag,mode', [
    ('plotProfileR', '--plotWidth', '--profile'),
    ('plotHeatmapR', '--heatmapWidth', '--heatmap'),
])
def test_wrapper_width_spelling_matches_matrix(wrapper, flag, mode):
    legacy = project_matrix_spec(parse_command(
        wrapper, ['-m', 'unused', flag, '6'], 'worker'), wrapper)
    shared = project_matrix_spec(parse_command(
        'plotMatrixR', ['-m', 'unused', mode, '--cellWidth', '6'],
        'worker'), 'plotMatrixR')
    assert legacy.cell_width == shared.cell_width == 6


def test_matrix_help_groups_own_their_options():
    parser = plot_parser('plotMatrixR', full_color_help=False)
    groups = {group.title: group for group in parser._action_groups}
    assert {'Required arguments', 'Output options',
            'Clustering arguments', 'Common options',
            'Profile options', 'Heatmap options'} <= set(groups)
    assert 'Optional arguments' not in groups
    ownership = {
        'Common options': (
            '--profile', '--heatmap', '--cellWidth', '--sortRegions',
            '--sortUsing', '--sortUsingSamples', '--quantileSortedRegions',
            '--clusterUsingSamples', '--arrangeSamples', '--commonLegend',
            '--sampleSetLabels', '--sampleSetGroupArrangement',
            '--gridColumns', '--gridRows', '--regionsLabel',
            '--samplesLabel', '--showRegionCounts', '--sameGroupLabels',
            '--sameSampleLabels', '--xAxisLabel', '--xAxisVisibility',
            '--axisVisibility', '--distanceUnit', '--distanceUnitLocation',
            '--startLabel', '--endLabel', '--refPointLabel',
            '--labelRotation', '--linesAtTickMarks', '--minorTickMarks',
            '--filterNans', '--plotTitle', '--perGroup',
            '--plotFileFormat', '--verbose', '--nanAfterEnd'),
        'Profile options': (
            '--plotType', '--averageType', '--pseudocount', '--trim_perc',
            '--ci_level', '--bootstrapReplicates', '-p',
            '--profileHeight', '--profileAspectRatio', '--yAxisLabel',
            '--yAxisVisibility', '--yAxisLimits', '--yMin', '--yMax',
            '--colorsPerSample', '--legendLocation'),
        'Heatmap options': (
            '--heatmapHeight', '--heatmapAspectRatio',
            '--heatmapYAxisLabel', '--regionLabelLocation',
            '--interpolationMethod', '--colorbarLocation',
            '--colorbarLabels', '--sortIndicator', '--colorMap', '--alpha',
            '--colorList', '--colorNumber', '--missingDataColor',
            '--boxAroundHeatmaps', '--zMin', '--zMax', '--zMid',
            '--silhouette'),
    }
    for title, options in ownership.items():
        actual = {option for action in groups[title]._group_actions
                  for option in action.option_strings}
        assert set(options) <= actual, title
    primary = {
        title: {action.option_strings[0]
                for action in groups[title]._group_actions
                if action.option_strings}
        for title in ownership}
    assert primary['Common options'] - set(ownership['Common options']) == {
        '--help', '--version'}
    assert primary['Profile options'] - set(ownership['Profile options']) == {
        '--plotHeight'}
    assert primary['Heatmap options'] == set(ownership['Heatmap options'])


def test_known_width_gives_unset_kinds_their_default_ratio():
    # --heatmapHeight 6 --aspectRatio 0.618 fixes the width; the profile then
    # follows its default 1:1 ratio instead of its default 5 cm height.
    solved = solve_sizes(show_heatmap=True, heatmap_height=6,
                         heatmap_aspect_ratio=0.618)
    assert solved.profile_height == pytest.approx(solved.cell_width)
    assert solve_sizes(cell_width=8).profile_height == 8
    assert solve_sizes(show_profile=False, show_heatmap=True,
                       cell_width=8).heatmap_height == 16
    assert (solve_sizes(show_heatmap=True).profile_height,
            solve_sizes(show_heatmap=True).heatmap_height) == (5, 10)
