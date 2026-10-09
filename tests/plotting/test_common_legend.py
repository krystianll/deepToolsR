"""--commonLegend auto|yes|no and the per-sample-set 'best' colourbars.

The input has samples Repeat, Repeat, Third and region groups First, Other,
Last (``write_input(distinct=True)``).
"""

import pytest

from deeptoolsr import parserCommon
from deeptoolsr.plotting.legend import LegendEntrySpec
from deeptoolsr.plotting.matrix_figure import resolve_common_legend
from tests.plotting.test_groups_as_sample_sets import _render


def _legend_count(figure):
    """Every drawn legend; external ones live on their own slot axes."""
    return sum(axis.get_legend() is not None for axis in figure.axes)


@pytest.mark.parametrize('flags,count', [
    # Every panel lists the same groups: auto draws one legend.
    ((), 1),
    (('--legendLocation', 'below'), 1),
    (('--commonLegend', 'no'), 3),
    (('--commonLegend', 'no', '--legendLocation', 'below'), 3),
    # Panels listing different samples keep their own legends.
    (('--perGroup', '--arrangeSamples', '1,2', '3'), 6),
    (('--perGroup', '--arrangeSamples', '1,2', '3', '--commonLegend'), 1),
])
def test_common_legend_modes(tmp_path, flags, count):
    figure = _render(tmp_path, '--profile', *flags)['figure']
    assert _legend_count(figure) == count


SETS = ('--perGroup', '--arrangeSamples', '1,2', '3', '--legendLocation',
        'below', '--sampleSetGroupArrangement')


@pytest.mark.parametrize('flags,owners', [
    # Rows list different samples: auto keeps one legend per row.
    ((*SETS, 'by_row'), ['legend/row/1', 'legend/row/2']),
    ((*SETS, 'by_row', '--commonLegend', 'yes'), ['legend/figure/1']),
    ((*SETS, 'by_column'), ['legend/column/1', 'legend/column/2']),
    # One sample set: every group panel's legend is identical.
    (('--perGroup', '--legendLocation', 'below',
      '--sampleSetGroupArrangement', 'by_column'), ['legend/figure/1']),
    (('--perGroup', '--legendLocation', 'below',
      '--sampleSetGroupArrangement', 'by_column', '--commonLegend', 'no'),
     ['legend/column/1']),
])
def test_scoped_legends(tmp_path, flags, owners):
    figure = _render(tmp_path, '--profile', *flags)['figure']
    assert [axis.get_gid() for axis in figure.axes
            if axis.get_legend() is not None] == owners


def test_style_difference_alone_prevents_merging():
    def panel(style):
        entry = LegendEntrySpec('Repeat', (1.0, 0.0, 0.0, 1.0), style, 1.5, '')
        return ((None, entry),)
    assert resolve_common_legend('auto', [panel('-'), panel('-')])
    assert not resolve_common_legend('auto', [panel('-'), panel('--')])
    assert resolve_common_legend('yes', [panel('-'), panel('--')])
    assert not resolve_common_legend('no', [panel('-'), panel('-')])


def test_bare_flag_means_yes():
    parser = parserCommon.plot_parser('plotProfileR')
    base = ['-m', 'm.gz', '-o', 'o.png']
    assert parser.parse_args(base).commonLegend == 'auto'
    assert parser.parse_args(base + ['--commonLegend']).commonLegend == 'yes'
    assert parser.parse_args(
        base + ['--commonLegend', 'no']).commonLegend == 'no'


def _bars(figure):
    return [axis.get_gid() for axis in figure.axes
            if (axis.get_gid() or '').startswith('colorbar')]


@pytest.mark.parametrize('flags,bars', [
    # One sample set split into groups shares one bar under its cells.
    (('--perGroup', '--zMax', '1', '2', '3'), ['colorbar/cell/1']),
    (('--perGroup', '--arrangeSamples', '1,2', '3', '--zMax', '1', '2'),
     ['colorbar/cell/1', 'colorbar/cell/4']),
    # Explicit below keeps one bar per cell.
    (('--perGroup', '--zMax', '1', '2', '3', '--colorbarLocation', 'below'),
     ['colorbar/cell/1', 'colorbar/cell/2', 'colorbar/cell/3']),
])
def test_best_colorbar_per_sample_set(tmp_path, flags, bars):
    assert _bars(_render(tmp_path, '--heatmap', *flags)['figure']) == bars
