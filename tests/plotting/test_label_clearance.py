"""Minimum clearance between neighbouring labels sharing one gutter."""

import gzip
import json
from pathlib import Path

import matplotlib
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from deeptoolsr import plotMatrix
from deeptoolsr.plotting.text_layout import (
    LabelCandidate, SharedGapBandSpec, TextMeasurer, _clearance_pairs,
    measure_label_candidates, optimize_shared_gap_layout)

matplotlib.use('Agg')

DATA = Path(__file__).parents[1] / 'test_data'


def _candidate(width):
    return LabelCandidate((str(width),), float(width), 10.0, 1, 0.0, 0.0)


def _needs_clearance(left_width, right_width, gap=10.0):
    # Two 10 pt cells; each label's interval is centred on its own cell.
    return _clearance_pairs(
        ((0, 0), (1, 1)), ((_candidate(left_width),),
                           (_candidate(right_width),)),
        (10.0, 10.0), gap, (0.0,), 'width')[0][0][0]


@pytest.mark.parametrize('left, right, expected', [
    (8.0, 9.0, False),    # both inside their own cells
    (10.0, 10.0, False),  # both exactly at their cell edges
    (14.0, 9.0, True),    # only the left label enters the gutter
    (9.0, 14.0, True),    # only the right label enters the gutter
    (14.0, 14.0, True),   # both enter, neither reaches the midpoint
    (22.0, 22.0, True),   # both cross the midpoint
])
def test_clearance_applies_when_either_label_leaves_its_cell(
        left, right, expected):
    assert _needs_clearance(left, right) is expected


def test_non_adjacent_spans_never_need_clearance():
    pairs = _clearance_pairs(
        ((0, 0), (2, 2)), ((_candidate(30.0),), (_candidate(30.0),)),
        (10.0, 10.0, 10.0), 10.0, (0.0, 0.0), 'width')
    assert pairs == (((False,),),)


def _solve(left_width, right_width, base_gap, clearance=4.0):
    band = SharedGapBandSpec(
        ((_candidate(left_width),), (_candidate(right_width),)),
        ((0, 0), (1, 1)))
    return optimize_shared_gap_layout(
        (band,), cell_extent=(10.0, 10.0), base_common_gap=base_gap,
        base_private_gaps=(0.0,), original_block_extent=30.0,
        max_common_gap_points=30.0, outer_left_allowance=20.0,
        outer_right_allowance=20.0, label_min_clearance=clearance)


def _separation(layout, left_width, right_width):
    return layout.common_gap - (left_width - 10.0) / 2 - (
        right_width - 10.0) / 2


@pytest.mark.parametrize('left, right, base_gap', [
    (20.0, 17.0, 10.0), (17.0, 10.0, 6.0), (10.0, 17.0, 6.0)])
def test_one_sided_intrusion_widens_small_gap_to_clearance(
        left, right, base_gap):
    layout = _solve(left, right, base_gap=base_gap)
    assert _separation(layout, left, right) == pytest.approx(4.0)


def test_one_sided_intrusion_keeps_gap_already_wider_than_clearance():
    layout = _solve(14.0, 10.0, base_gap=10.0)
    assert layout.common_gap == 10.0
    assert _separation(layout, 14.0, 10.0) == pytest.approx(8.0)


def test_labels_inside_cells_keep_gap_smaller_than_clearance():
    layout = _solve(9.0, 9.0, base_gap=2.0)
    assert layout.common_gap == 2.0


@pytest.mark.parametrize('rotation', (0, 90, -90))
@pytest.mark.parametrize('text', ('genes_Q1 [n = 3,809]',
                                  'a longer region\nlabel on lines'))
def test_candidate_size_equals_drawn_box_at_reference_resolution(
        rotation, text):
    # Candidates are measured at the canonical 144 dpi reference; at that
    # output resolution they equal the box Matplotlib draws.
    dpi = 144
    figure = Figure(dpi=dpi)
    FigureCanvasAgg(figure)
    font = FontProperties(family='DejaVu Sans', size=9)
    renderer = figure.canvas.get_renderer()
    measurer = TextMeasurer(renderer, figure, dpi, {})
    candidates = measure_label_candidates(
        text, renderer, font, figure, 3, rotation, measurer=measurer)
    for candidate in candidates:
        artist = figure.text(.5, .5, candidate.text, fontproperties=font,
                             rotation=rotation, ha='center', va='center')
        box = artist.get_window_extent(renderer)
        artist.remove()
        assert candidate.width == pytest.approx(box.width * 72 / dpi,
                                                abs=1e-6)
        assert candidate.height == pytest.approx(box.height * 72 / dpi,
                                                 abs=1e-6)


def test_region_labels_keep_minimum_drawn_clearance(tmp_path, monkeypatch):
    lines = gzip.open(DATA / 'computeMatrixOperations.mat.gz',
                      'rt').read().splitlines()
    header = json.loads(lines[0][1:])
    header['group_labels'] = ['genes_Q1xxx', 'genes_Q2xxx']
    header['group_boundaries'] = [0, 98, 196]
    matrix = tmp_path / 'matrix.gz'
    with gzip.open(matrix, 'wt') as handle:
        handle.write('@' + json.dumps(header) + '\n' +
                     '\n'.join(lines[1:]) + '\n')
    saved = {}
    original = Figure.savefig

    def save(self, *args, **kwargs):
        saved['figure'], saved['dpi'] = self, kwargs.get('dpi')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', save)
    plotMatrix.main([
        '--heatmap', '-m', str(matrix), '--showRegionCounts',
        '--samplesLabel', *'abcdefgh', '--cellWidth', '3',
        '--heatmapHeight', '6', '-o', str(tmp_path / 'plot.png'),
        '--dpi', '144'])
    figure = saved['figure']
    figure.dpi = saved['dpi']
    renderer = figure.canvas.get_renderer()
    boxes = sorted((axis.texts[0].get_window_extent(renderer)
                    for axis in figure.axes
                    if str(axis.get_gid()).startswith('label/stack')),
                   key=lambda box: box.y0)
    assert len(boxes) == 2
    lower, upper = boxes
    assert (upper.y0 - lower.y1) * 72 / figure.dpi >= 4.0 - 0.05
