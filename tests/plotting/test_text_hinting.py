"""Text hinting setting: parsing, precedence, rc scoping and measurement."""

import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont, get_font
from matplotlib import _text_helpers
from matplotlib.ft2font import LoadFlags
import pytest

from deeptoolsr import config, options as run_options
from deeptoolsr.plotting import fonts
from deeptoolsr.plotting.text_layout import measure_matplotlib_text
from deeptoolsr.parserCommon import plot_parser


def _style(stored, cli=None, tmp_path=None):
    path = tmp_path / 'options.txt'
    raw = json.dumps(stored).encode()
    args = argparse.Namespace(numberOfProcessors='auto', textHinting=cli)
    return run_options.resolve_run_options_from_bytes(
        args, raw, path, plotting=True, emit=lambda *_: None).style


def test_default_is_off_and_config_value_is_used(tmp_path):
    assert _style({}, tmp_path=tmp_path).text_hinting == 'off'
    assert _style({'text_hinting': 'on'},
                  tmp_path=tmp_path).text_hinting == 'on'


def test_invalid_config_value_falls_back_with_message():
    style, messages = config.resolve_style({'text_hinting': 'maybe'})
    assert style.text_hinting == 'off'
    assert any('text_hinting' in message for message in messages)


@pytest.mark.parametrize('stored,cli', [('on', 'off'), ('off', 'on')])
def test_cli_overrides_config(tmp_path, stored, cli):
    style = _style({'text_hinting': stored}, cli=cli, tmp_path=tmp_path)
    assert style.text_hinting == cli


@pytest.mark.parametrize('tool', ['plotMatrixR', 'plotProfileR',
                                  'plotHeatmapR'])
def test_plot_tools_accept_text_hinting(tool):
    parser = plot_parser(tool, full_color_help=False)
    assert parser.parse_args(
        ['-m', 'x.gz', '-o', 'x.png', '--textHinting', 'on']).textHinting == 'on'
    with pytest.raises(SystemExit):
        parser.parse_args(['-m', 'x.gz', '-o', 'x.png', '--textHinting', 'x'])


def test_style_rc_sets_hinting_only_for_off():
    off, _ = config.resolve_style({})
    on, _ = config.resolve_style({'text_hinting': 'on'})
    assert fonts.style_rc(off)['text.hinting'] == 'none'
    assert 'text.hinting' not in fonts.style_rc(on)


@pytest.mark.parametrize('hinting', ['on', 'off'])
def test_rcparams_are_not_leaked_after_a_plot(tmp_path, hinting):
    from deeptoolsr import plotMatrix

    matrix = Path(__file__).parents[1] / 'test_heatmapper' / 'master_multi.mat.gz'
    before = dict(matplotlib.rcParams)
    plotMatrix.main(['-m', str(matrix), '--profile', '--heatmap',
                     '--textHinting', hinting,
                     '-o', str(tmp_path / 'plot.png')])
    assert (tmp_path / 'plot.png').exists()
    assert dict(matplotlib.rcParams) == before


def _advance(text, size):
    font = get_font(findfont(FontProperties(size=size)))
    font.set_size(size, 72)
    end = 0.0
    for item in _text_helpers.layout(text, font):
        glyph = item.ft_object.load_char(ord(item.char),
                                         flags=LoadFlags.NO_HINTING)
        end = item.x + glyph.linearHoriAdvance / 65536
    return end


def _measure(text, size, dpi=150):
    figure = Figure(dpi=dpi)
    canvas = FigureCanvasAgg(figure)
    return measure_matplotlib_text(text, canvas.get_renderer(),
                                   FontProperties(size=size),
                                   figure=figure).width


@pytest.mark.parametrize('text', ['B45C A123B vs. WT', 'genes_Q1 [n = 3,809]',
                                  'distance from TSS', 'WWWWWW'])
def test_off_measurement_matches_pdf_advance(text):
    # Plot figures measure on their 100-DPI layout renderer.
    with matplotlib.rc_context({'text.hinting': 'none'}):
        assert abs(_measure(text, 8.0, 100) - _advance(text, 8.0)) < 0.05


def test_on_measurement_is_unchanged_hinted_reference():
    # "on" leaves rcParams alone, so the measurement is the hinted 144-DPI
    # reference: identical across DPIs and different from the unhinted one.
    text = 'B45C A123B vs. WT'
    hinted = [_measure(text, 8.0, dpi) for dpi in (72, 150, 300)]
    assert hinted[0] == hinted[1] == hinted[2]
    with matplotlib.rc_context({'text.hinting': 'none'}):
        unhinted = _measure(text, 8.0)
    assert hinted[0] != unhinted
