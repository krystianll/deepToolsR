"""Measurement shortcuts must not change any layout result."""
import pytest
from matplotlib.font_manager import FontProperties

from deeptoolsr.plotting import legend
from deeptoolsr.plotting.legend import LegendEntrySpec, measure_legend
from deeptoolsr.plotting.scene import MatplotlibMeasureContext
from deeptoolsr.plotting.text_layout import measure_matplotlib_text


def _entries(count, word):
    return [LegendEntrySpec('{} {}'.format(word, index), 'red')
            for index in range(count)]


def _full_search(context, entries, location, maximum):
    """The descending column search, trying every column count."""
    columns = (range(len(entries), 0, -1)
               if location in ('above', 'below') else [1])
    for ncols in columns:
        size = legend._measure(context, tuple(entries), ncols, None,
                               legend.DEFAULT_GEOMETRY)
        if size.width <= maximum:
            break
    return size, ncols


def _counting(monkeypatch):
    calls = []
    original = legend._measure

    def counted(context, entries, ncols, *args):
        calls.append(ncols)
        return original(context, entries, ncols, *args)
    monkeypatch.setattr(legend, '_measure', counted)
    return calls


@pytest.mark.parametrize('location', ['above', 'below', 'right'])
def test_legend_too_wide_for_one_column_builds_one_legend(
        monkeypatch, location):
    context = MatplotlibMeasureContext()
    entries = _entries(8, 'a rather long legend key')
    expected = _full_search(context, entries, location, 40.0)
    calls = _counting(monkeypatch)
    measured = measure_legend(context, entries, location, 40.0)
    assert calls == [1]
    assert (measured.size, measured.ncols) == expected


@pytest.mark.parametrize('maximum', [60.0, 120.0, 250.0, 1000.0])
def test_legend_column_search_matches_the_full_search(maximum):
    context = MatplotlibMeasureContext()
    entries = _entries(6, 'key')
    measured = measure_legend(context, entries, 'above', maximum)
    assert (measured.size, measured.ncols) == _full_search(
        context, entries, 'above', maximum)


def test_reused_measuring_text_matches_fresh_figures_and_adds_no_artist():
    context = MatplotlibMeasureContext(dpi=120)
    font = FontProperties(size=9)
    texts = ['alpha', 'two\nlines', 'B45C A123B vs.\nWT — genes', '']
    before = (len(context.figure.texts), len(context.figure.artists))
    reused = [measure_matplotlib_text(text, context.renderer, font,
                                      context.figure, rotation)
              for text in texts for rotation in (0.0, 90.0, -90.0)]
    fresh = [measure_matplotlib_text(text, context.renderer, font,
                                     None, rotation)
             for text in texts for rotation in (0.0, 90.0, -90.0)]
    assert reused == fresh
    assert (len(context.figure.texts), len(context.figure.artists)) == before
