"""Tests for explicit N,M=value assignment and a-b ranges."""

from deeptoolsr.parserCommon import (
    apply_capability_tags, capability_tag)
import argparse
import pytest

from deeptoolsr.plotting.series import (
    has_explicit_assignment, parse_index_spec, parse_sample_sets, recycle,
    resolve_recyclable)
from deeptoolsr.prepare import OptionError
from deeptoolsr.prepare import parse_command
from deeptoolsr import plotMatrix
from deeptoolsr.describe import describe
from deeptoolsr.session import PlotSession


class _Matrix:
    def __init__(self, labels):
        self.sample_labels = list(labels)


# --- parse_index_spec ------------------------------------------------------

def test_index_spec_commas_and_ranges():
    assert parse_index_spec('1,2', 4) == [0, 1]
    assert parse_index_spec('3-4', 4) == [2, 3]
    assert parse_index_spec('1,3-5', 5) == [0, 2, 3, 4]


def test_index_spec_reverse_range_and_bounds():
    assert parse_index_spec('3-1', 4) == [2, 1, 0]
    with pytest.raises(ValueError):
        parse_index_spec('5', 4)


# --- has_explicit_assignment ----------------------------------------------

def test_detects_explicit_prefix_only():
    assert has_explicit_assignment(['1,2=red', 'blue'])
    assert has_explicit_assignment(['3-4=x'])
    assert not has_explicit_assignment(['red', 'blue'])
    assert not has_explicit_assignment(['1,2:red'])
    assert not has_explicit_assignment([r'1\=red'])


def test_out_of_range_assignment_names_option_and_domain():
    with pytest.raises(OptionError, match=(
            r'--colorList: 1-3 names sample set 3, but there are only '
            r'2 sample sets')):
        resolve_recyclable(['1-3=#fff'], 2, option='--colorList',
                           domain='sample set')


def test_colon_is_literal_and_escaped_equals_is_text():
    assert resolve_recyclable(['1:red', '2:blue'], 2) == ['1:red', '2:blue']
    assert resolve_recyclable([r'1\=x', r'A\\B', r'C\nD'], 3,
                              text=True) == ['1=x', r'A\B', 'C\nD']


def test_text_escapes_decode_after_assignment_detection():
    args = parse_command('plotProfileR', [
        '-m', 'tests/test_data/prepare_expected/input.gz',
        '--plotTitle', r'Top\nBottom', '--regionsLabel', r'1\=A',
        '--samplesLabel', r'1\=Control', '2=Case'], 'worker')
    assert args.plotTitle == 'Top\nBottom'
    assert args.regionsLabel == ['1=A']
    assert args.samplesLabel == [r'1\=Control', '2=Case']
    assert resolve_recyclable(args.samplesLabel, 3, text=True,
                              default=lambda index: str(index)) == [
                                  '1=Control', 'Case', '1=Control']


# --- resolve_recyclable ----------------------------------------------------

def test_plain_recycling_when_no_explicit():
    assert resolve_recyclable(['a', 'b'], 4) == ['a', 'b', 'a', 'b']


def test_mixed_explicit_and_recycled_pool():
    # --zMin 1,2=5 0 -> 5 on sets 1,2; 0 recycled elsewhere.
    assert resolve_recyclable(['1,2=5', '0'], 4, default='auto') == \
        ['5', '5', '0', '0']


def test_explicit_ranges_cover_all():
    assert resolve_recyclable(['1,2=blue,red', '3-4=black,white'], 4) == \
        ['blue,red', 'blue,red', 'black,white', 'black,white']


def test_unassigned_without_pool_uses_default():
    # No prefix-less pool -> columns 3,4 fall back to the default.
    assert resolve_recyclable(['1,2=5'], 4, default='auto') == \
        ['5', '5', 'auto', 'auto']


def test_default_may_be_callable():
    result = resolve_recyclable(['2=x'], 3, default=lambda i: 'd{}'.format(i))
    assert result == ['d0', 'x', 'd2']


# --- recycle delegates to explicit ----------------------------------------

def test_recycle_supports_explicit_with_defaults_fallback():
    # Position 3 (index 2) is unassigned and has no pool -> its default.
    out = recycle(['1,2=GroupA'], 3, defaults=('d0', 'd1', 'd2'))
    assert out == ('GroupA', 'GroupA', 'd2')


def test_recycle_plain_behaviour_unchanged():
    assert recycle(['x'], 3) == ('x', 'x', 'x')
    assert recycle(None, 2, defaults=('a', 'b')) == ('a', 'b')


def test_samples_label_assignment_uses_sample_domain(tmp_path):
    matrix = 'tests/test_data/prepare_expected/input.gz'
    command = ['--profile', '-m', matrix, '-o', str(tmp_path / 'plot.png'),
               '--samplesLabel', '2=Case']
    record = describe('plotMatrixR', command)
    assert [sample['label'] for sample in record['samples']] == [
        'one', 'Case', 'three']
    plotMatrix.main(command)
    assert (tmp_path / 'plot.png').is_file()


def test_samples_label_out_of_range_names_invoked_alias(tmp_path, capsys):
    command = ['--profile', '-m', 'tests/test_data/prepare_expected/input.gz',
               '-o', str(tmp_path / 'plot.png'), '--sampleLabels', '4=Case']
    with pytest.raises(OptionError, match=(
            r'--sampleLabels: 4 names sample 4, but there are only 3 samples')):
        describe('plotMatrixR', command)
    with pytest.raises(SystemExit) as error:
        plotMatrix.main(command)
    assert error.value.code == 2
    assert ('--sampleLabels: 4 names sample 4, but there are only 3 samples'
            in capsys.readouterr().err)


def test_sample_set_assignment_out_of_range_names_invoked_alias(tmp_path):
    command = ['--heatmap', '-m', 'tests/test_data/prepare_expected/input.gz',
               '-o', str(tmp_path / 'plot.png'), '--arrangeSamples', '1,2',
               '3', '--colorMap', '1-3=viridis']
    with pytest.raises(OptionError, match=(
            r'--colorMap: 1-3 names sample set 3, but there are only '
            r'2 sample sets')):
        describe('plotMatrixR', command)


def test_y_limit_assignment_checks_sample_set_count(tmp_path, capsys):
    command = ['--profile', '-m', 'tests/test_data/prepare_expected/input.gz',
               '-o', str(tmp_path / 'plot.png'), '--yMin', '4=0']
    with pytest.raises(SystemExit) as error:
        plotMatrix.main(command)
    assert error.value.code == 2
    assert ('--yMin: 4 names sample set 4, but there are only 3 sample sets'
            in capsys.readouterr().err)


def test_numeric_colon_alias_is_rejected(tmp_path, capsys):
    command = ['--heatmap', '-m',
               'tests/test_data/prepare_expected/input.gz',
               '-o', str(tmp_path / 'plot.png'), '--zMin', '1:0']
    with pytest.raises(SystemExit) as error:
        plotMatrix.main(command)
    assert error.value.code == 2
    assert '1:0' in capsys.readouterr().err


def test_equivalent_color_spellings_have_same_scene_digest(tmp_path):
    base = ['plotMatrixR', '--profile', '-m',
            'tests/test_data/prepare_expected/input.gz',
            '-o', str(tmp_path / 'plot.png')]
    # The Empty group is plotted now (no header-threshold re-filter).
    colors = ['red', 'blue', 'green']
    with PlotSession(cache_bytes=0) as session:
        first = session.run(base + ['--colorsPerSample', *colors])
        second = session.run(base + ['--colors', *colors])
    assert first.scene_digest == second.scene_digest


def _line_colours(tmp_path, monkeypatch, *options):
    """Rendered series colours, in legend order (one per colour slot)."""
    from matplotlib.colors import to_hex
    built = {}
    original = plotMatrix.build_matrix_figure

    def capture(*args, **kwargs):
        figure, solution = original(*args, **kwargs)
        built['figure'] = figure
        return figure, solution

    monkeypatch.setattr(plotMatrix, 'build_matrix_figure', capture)
    plotMatrix.main(['--profile', '-m', MULTI,
                     '-o', str(tmp_path / 'plot.png'), *options])
    figure = built['figure']
    legends = [axis.get_legend() for axis in figure.axes
               if axis.get_legend()] + list(figure.legends)
    colours = []
    for legend in legends:
        for handle in legend.legend_handles:
            colour = to_hex(handle.get_color())
            if colour not in colours:
                colours.append(colour)
    return colours


def _default_line_colours(count):
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    return [to_hex(colormaps['jet'](index / float(count)))
            for index in range(count)]


# master_multi.mat.gz without --arrangeSamples overlays two group series
# (two colour slots) in every panel.
@pytest.mark.parametrize('assignment, expected', [
    (('1=red', 'blue'), ['#ff0000', '#0000ff']),
    (('2=blue', 'red'), ['#ff0000', '#0000ff']),
    (('2=blue',), [_default_line_colours(2)[0], '#0000ff']),
])
def test_line_colour_assignment_addresses_colour_slots(
        tmp_path, monkeypatch, assignment, expected):
    assert _line_colours(tmp_path, monkeypatch, '--colorsPerSample',
                         *assignment) == expected


def test_line_colour_assignment_colours_one_legend_entry(
        tmp_path, monkeypatch):
    from matplotlib.colors import to_hex
    base = ('--quantiles', '2', '--arrangeSamples', '1,2', '3,4')
    colours = _line_colours(tmp_path, monkeypatch, *base,
                            '--colorsPerSample', '3=red')
    defaults = _default_line_colours(8)
    assert len(colours) == 8
    assert colours[2] == to_hex('red')
    assert [c for i, c in enumerate(colours) if i != 2] == [
        c for i, c in enumerate(defaults) if i != 2]


@pytest.mark.parametrize('options, message', [
    ((), '--colorsPerSample: 3 names series 3, but there are only 2 '
     'coloured series'),
    (('--quantiles', '2', '--arrangeSamples', '1,2', '3,4'),
     '--colorsPerSample: 9 names series 9, but there are only 8 '
     'coloured series'),
])
def test_line_colour_assignment_out_of_range_names_slot_count(
        tmp_path, capsys, options, message):
    index = message.split()[1]
    command = ['--profile', '-m', MULTI, '-o', str(tmp_path / 'plot.png'),
               *options, '--colorsPerSample', f'{index}=red']
    with pytest.raises(SystemExit) as error:
        plotMatrix.main(command)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


# --- parse_sample_sets ranges ---------------------------------------------

def test_sample_sets_accept_ranges():
    matrix = _Matrix(['s1', 's2', 's3', 's4'])
    assert parse_sample_sets(matrix.sample_labels, ['1-2', '3,4']) == ((0, 1), (2, 3))


def test_sample_sets_range_reports_duplicates():
    matrix = _Matrix(['s1', 's2', 's3'])
    with pytest.raises(ValueError):
        parse_sample_sets(matrix.sample_labels, ['1-2', '2-3'])


def test_sample_sets_hyphenated_names_still_work():
    matrix = _Matrix(['wt-1', 'wt-2'])
    assert parse_sample_sets(matrix.sample_labels, ['wt-1', 'wt-2']) == ((0,), (1,))


# --- capability tags (PR E) ------------------------------------------------


def _action(dest, nargs=None):
    parser = argparse.ArgumentParser(add_help=False)
    return parser.add_argument('--' + dest, dest=dest, nargs=nargs, help='x')


def test_capability_tag_single_valued_is_global():
    assert capability_tag(_action('heatmapHeight')) == '[global]'


def test_capability_tag_multiple_explicit_recycled():
    assert capability_tag(_action('colorList', nargs='+')) == \
        '[multiple|explicit|recycled]'
    assert capability_tag(_action('arrangeSamples', nargs='+')) == '[multiple]'
    assert capability_tag(_action('colorbarLabels', nargs='+')) == \
        '[multiple|explicit|recycled]'


def test_apply_tags_prefixes_and_is_idempotent_and_skips_suppressed():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zMin', dest='zMin', nargs='+', help='limit')
    parser.add_argument('--perGroup', dest='perGroup', action='store_true',
                        help='flag')
    parser.add_argument('--hidden', dest='hidden', help=argparse.SUPPRESS)
    apply_capability_tags(parser)
    by_dest = {a.dest: a.help for a in parser._actions}
    assert by_dest['zMin'].startswith('[multiple|explicit|recycled] ')
    assert by_dest['perGroup'].startswith('[global] ')
    assert by_dest['hidden'] == argparse.SUPPRESS      # untouched
    # Second application does not double-prefix.
    apply_capability_tags(parser)
    zmin_help = {a.dest: a.help for a in parser._actions}['zMin']
    assert zmin_help.count('[multiple') == 1


MULTI = 'tests/test_heatmapper/master_multi.mat.gz'


def _plot_matrix(tmp_path, *options):
    plotMatrix.main(['-m', MULTI, '-o', str(tmp_path / 'plot.png'),
                     *options])


def _plot_matrix_error(tmp_path, capsys, *options):
    with pytest.raises(SystemExit) as error:
        _plot_matrix(tmp_path, *options)
    assert error.value.code == 2
    return capsys.readouterr().err


def test_plain_line_colours_need_exactly_one_per_series(tmp_path, capsys):
    _plot_matrix(tmp_path, '--profile', '--colorsPerSample', 'red', 'blue')
    err = _plot_matrix_error(tmp_path, capsys, '--profile',
                             '--colorsPerSample', 'red')
    assert ('--colorsPerSample needs exactly 2 colours, one for each '
            'distinctly coloured series; 1 given (or use N=colour to set '
            'some)') in err


def test_too_many_plain_line_colours_warn_and_are_ignored(tmp_path, capsys):
    _plot_matrix(tmp_path, '--profile', '--colors', 'red', 'blue', 'green')
    assert ('WARNING: --colors: 3 colours given for 2 coloured series; '
            'the last 1 are ignored') in capsys.readouterr().err


def test_mixed_line_colours_fill_remaining_series_in_order(tmp_path, capsys):
    base = ('--profile', '--quantiles', '2', '--arrangeSamples', '1,2', '3,4')
    _plot_matrix(tmp_path, *base, '--colorsPerSample', '1=red', *['blue'] * 7)
    assert 'WARNING' not in capsys.readouterr().err
    err = _plot_matrix_error(tmp_path, capsys, *base, '--colorsPerSample',
                             '1=red', 'blue')
    assert ('--colorsPerSample: 1 plain colours given for 7 series without '
            'an N= assignment; give exactly 7 or assign them with '
            'N=colour') in err
    _plot_matrix(tmp_path, *base, '--colorsPerSample', '1=red',
                 *['blue'] * 8)
    assert ('8 plain colours given for 7 series without an N= assignment; '
            'the last 1 are ignored') in capsys.readouterr().err


def test_line_colour_count_follows_arranged_series(tmp_path, capsys):
    err = _plot_matrix_error(
        tmp_path, capsys, '--profile', '--quantiles', '2',
        '--arrangeSamples', '1,2', '3,4', '--plotColors', 'red', 'blue')
    assert '--plotColors needs exactly 8 colours' in err
    assert '2 given' in err


def test_assigned_line_colours_set_a_subset(tmp_path):
    _plot_matrix(tmp_path, '--profile', '--colorsPerSample', '2=red')


def test_series_heatmap_maps_need_exactly_one_per_panel(tmp_path, capsys):
    base = ('--profile', '--plotType', 'heatmap', '--perGroup')
    panels = describe('plotMatrixR', [
        *base, '-m', MULTI, '-o', str(tmp_path / 'plot.png')])['panels']
    count = len(panels)
    assert count > 1
    _plot_matrix(tmp_path, *base, '--colorsPerSample', *(['Reds'] * count))
    _plot_matrix(tmp_path, *base, '--colorsPerSample', '1=Blues')
    err = _plot_matrix_error(tmp_path, capsys, *base,
                             '--colorsPerSample', 'Reds')
    assert (f'--colorsPerSample needs exactly {count} colour maps, one '
            'for each panel; 1 given (or use N=colour map to set some)'
            ) in err
    _plot_matrix(tmp_path, *base, '--colorsPerSample',
                 *(['Reds'] * (count + 1)))
    assert (f'{count + 1} colour maps given for {count} panels; the last 1 '
            'are ignored') in capsys.readouterr().err


def test_mixed_series_heatmap_maps_fill_remaining_panels(tmp_path, capsys):
    base = ('--profile', '--plotType', 'heatmap')  # 4 panels
    _plot_matrix(tmp_path, *base, '--colorsPerSample', '1=Blues',
                 *['Reds'] * 3)
    assert 'WARNING' not in capsys.readouterr().err
    err = _plot_matrix_error(tmp_path, capsys, *base, '--colorsPerSample',
                             '1=Blues', 'Reds')
    assert ('--colorsPerSample: 1 plain colour maps given for 3 panels '
            'without an N= assignment; give exactly 3 or assign them with '
            'N=colour map') in err
    _plot_matrix(tmp_path, *base, '--colorsPerSample', '1=Blues',
                 *['Reds'] * 4)
    assert ('4 plain colour maps given for 3 panels without an N= '
            'assignment; the last 1 are ignored') in capsys.readouterr().err


@pytest.mark.parametrize('options, message', [
    (('--colorMap', 'notamap'),
     '--colorMap: notamap is not a colour map; use a Matplotlib or '
     'deepToolsR colour map name such as viridis'),
    (('--colorMap', '1=notamap'), '--colorMap: notamap is not a colour map'),
    (('--colorList', 'red,notacolour'),
     '--colorList: notacolour is not a colour; use a colour name'),
    (('--zMin', '0', '--zMax', '1', '--zMid', '5'),
     '--zMid: 5 must be strictly between zMin (0) and zMax (1)'),
])
def test_heatmap_colour_options_are_option_errors(
        tmp_path, capsys, options, message):
    err = _plot_matrix_error(tmp_path, capsys, '--heatmap', *options)
    assert message in err
