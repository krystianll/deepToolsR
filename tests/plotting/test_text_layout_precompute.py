"""Precomputed column starts and memoised line breaks match direct results."""

import pytest
from uniseg.linebreak import line_break_boundaries

from deeptoolsr.plotting.text_layout import (
    LabelCandidate, _column_starts, _line_break_boundaries,
    _title_reserved_encroachment, _title_span_interval)

LAYOUTS = (
    (50.0, 20.0, (15.0,)),
    ((40.0, 55.5, 31.25), 7.5, (0.0, 12.0)),
    ((10.0, 10.0, 10.0, 10.0, 10.0), 0.1, (0.3, 0.0, 2.7, 0.05)),
)


@pytest.mark.parametrize('cells, gap, private', LAYOUTS)
def test_precomputed_starts_give_identical_intervals(cells, gap, private):
    count = len(private) + 1
    starts = _column_starts(cells, gap, private)
    spans = [(i, j) for i in range(count) for j in range(i, count)]
    for width in (3.0, 47.3, 160.0, 401.9):
        candidate = LabelCandidate(('t',), width, width / 3.0, 1, 0.0, 0.0)
        for span in spans:
            for dimension in ('width', 'height'):
                assert _title_span_interval(
                    span, candidate, cells, gap, private, dimension,
                    starts) == _title_span_interval(
                    span, candidate, cells, gap, private, dimension)
                assert _title_reserved_encroachment(
                    span, candidate, cells, gap, private, dimension,
                    starts) == _title_reserved_encroachment(
                    span, candidate, cells, gap, private, dimension)


@pytest.mark.parametrize('text', (
    '', 'GENE1 knockdown versus scrambled Control',
    'log2FC [treated/control]', 'a-b/c (n = 1,234)', 'Zażółć gęślą'))
def test_line_break_cache_returns_uniseg_boundaries(text):
    expected = tuple(line_break_boundaries(text))
    assert _line_break_boundaries(text) == expected
    assert _line_break_boundaries(text) == expected
    assert _line_break_boundaries.cache_info().maxsize == 1024
