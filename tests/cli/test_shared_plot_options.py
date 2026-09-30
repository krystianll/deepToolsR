"""Shared plot destinations preserve each wrapper's public spelling."""

import pytest

from deeptoolsr.prepare import OptionError, parse_command
from deeptoolsr.plotting.matrix_spec import project_matrix_spec


@pytest.mark.parametrize('tool, flag', [
    ('plotHeatmapR', '--yMinSummaryPlot'),
    ('plotProfileR', '--yMin'),
])
def test_bad_y_min_names_the_invoked_spelling(tool, flag):
    with pytest.raises(OptionError, match=flag):
        parse_command(tool, ['-m', 'unused', flag, 'bad'], 'worker')


@pytest.mark.parametrize('flag', ['--yAxisLabel', '-y'])
def test_heatmap_y_label_aliases_share_one_destination(flag):
    args = parse_command('plotHeatmapR',
                         ['-m', 'unused', flag, 'signal'], 'worker')
    assert args.heatmapYAxisLabel == ['signal']
    assert args.yAxisLabel == ['']
    spec = project_matrix_spec(args, 'plotHeatmapR')
    assert spec.heatmap_y_axis_label == ['signal']


def test_figure_specs_use_shared_destinations_and_derived_sizes():
    heatmap = parse_command('plotHeatmapR',
                            ['-m', 'unused', '--heatmapWidth', '4',
                             '--aspectRatioSummaryPlot', '2'], 'worker')
    heatmap_spec = project_matrix_spec(heatmap, 'plotHeatmapR')
    assert (heatmap_spec.show_profile, heatmap_spec.show_heatmap) == (True, True)
    assert (heatmap_spec.cell_width, heatmap_spec.profile_height) == (4, 2)
    assert heatmap_spec.profile_aspect_ratio == 2

    profile = parse_command('plotProfileR', ['-m', 'unused'], 'worker')
    profile_spec = project_matrix_spec(profile, 'plotProfileR')
    assert (profile_spec.show_profile, profile_spec.show_heatmap) == (True, False)
    assert (profile_spec.cell_width, profile_spec.profile_height) == (5, 5)


def test_heatmap_parser_exposes_shared_arrangement_options():
    args = parse_command('plotHeatmapR', [
        '-m', 'unused', '--arrangeSamples', '1,2',
        '--sampleSetLabels', 'Pair', '--sampleSetGroupArrangement', 'adjacent',
        '--gridColumns', '2', '--sameGroupLabels', 'together',
        '--sameSampleLabels', 'together', '--commonLegend',
        '--ci_level', '0.9', '--bootstrapReplicates', '40'], 'worker')
    assert args.arrangeSamples == ['1,2']
    assert args.sampleSetLabels == ['Pair']
    assert args.sampleSetGroupArrangement == 'adjacent'
    assert args.gridColumns == 2
    assert args.sameGroupLabels == args.sameSampleLabels == 'together'
    assert args.commonLegend
    assert args.ci_level == 0.9
    assert args.bootstrapReplicates == 40
