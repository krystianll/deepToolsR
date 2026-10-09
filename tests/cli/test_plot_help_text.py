"""Plot help text is checked against its reviewed parser contract.

Each tool's reference is its reviewed help text; plotMatrixR describes the
shared cell model. argparse renders option invocations differently from Python
3.13 on, so there is one reference set per format. The installed colour map
list depends on the Matplotlib version and is pinned to a fixed list here.
"""

import sys
from pathlib import Path

import pytest

import deeptoolsr.plotting
from deeptoolsr import parserCommon


HELP = Path(__file__).parent.parent / 'contract' / 'help'
FORMAT = 'modern' if sys.version_info >= (3, 13) else 'legacy'


@pytest.mark.parametrize('variant, full_color_help',
                         [('cli', True), ('brief', False)])
@pytest.mark.parametrize('tool', ['plotHeatmapR', 'plotProfileR', 'plotMatrixR'])
def test_plot_help_text_is_unchanged(monkeypatch, tool, variant,
                                     full_color_help):
    monkeypatch.setenv('COLUMNS', '100')
    monkeypatch.setenv('NO_COLOR', '1')
    monkeypatch.delenv('FORCE_COLOR', raising=False)
    monkeypatch.setattr(deeptoolsr.plotting, 'colormap_names',
                        lambda: ['RdYlBu', 'RdYlBu_r', 'Reds', 'viridis'])
    parser = parserCommon.plot_parser(tool, full_color_help=full_color_help)
    expected = (HELP / FORMAT / f'{tool}.{variant}.txt').read_text()
    assert parser.format_help() == expected
