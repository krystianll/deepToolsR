"""Region labels may flow from the sample name into the group line."""
import matplotlib
import pytest

from deeptoolsr import plotMatrix
from deeptoolsr.plotting import grid
from deeptoolsr.plotting.text_layout import (
    LabelCandidate, SharedGapBandSpec, generate_label_candidates,
    optimize_shared_gap_layout)

matplotlib.use('Agg')

DATA = 'tests/test_heatmapper/master_multi.mat.gz'


def _measure(text):
    lines = text.split('\n')
    return 6.0 * max(len(line) for line in lines), 10.0 * len(lines)


def test_preferred_break_offers_flowed_layouts_after_preserved_ones():
    text = 'B45C A123B vs. WT\ngenes_Q1 [n = 3,809]'
    plain = generate_label_candidates(text, _measure, max_lines=3)
    offered = generate_label_candidates(text, _measure, max_lines=3,
                                        preferred_break=0)
    assert offered[:len(plain)] == plain
    flowed = offered[len(plain):]
    assert ('B45C A123B vs.', 'WT — genes_Q1', '[n = 3,809]') in {
        candidate.lines for candidate in flowed}
    assert all(candidate.break_penalty >= 0.1 for candidate in flowed)
    assert not any('WT genes' in line or '—' in line for candidate in plain
                   for line in candidate.lines)


def test_newlines_other_than_the_preferred_one_stay_hard():
    text = 'first\nsecond name\ngroup'
    for candidate in generate_label_candidates(
            text, _measure, max_lines=3, preferred_break=1):
        assert candidate.lines[0] == 'first'
    assert generate_label_candidates(text, _measure, max_lines=3,
                                     preferred_break=5) == \
        generate_label_candidates(text, _measure, max_lines=3)


def test_preferred_break_wins_when_the_geometry_ties():
    # Every two-line layout has the same box; the flowed one balances better.
    def box(text):
        lines = text.count('\n') + 1
        return (60.0 if lines == 2 else 200.0), 10.0 * lines

    candidates = tuple(c for c in generate_label_candidates(
        'long name here\ng', box, max_lines=2, preferred_break=0,
        line_measure=lambda line: (6.0 * len(line), 10.0))
        if c.line_count == 2)
    assert {c.lines for c in candidates} == {
        ('long name here', 'g'), ('long name', 'here — g')}
    chosen = optimize_shared_gap_layout(
        (SharedGapBandSpec(candidate_sets=(candidates,), spans=((0, 0),),
                           primary_dimension='height'),),
        cell_extent=(100.0,), base_common_gap=8.0, base_private_gaps=(),
        original_block_extent=100.0, max_common_gap_points=96.0,
        hard_outer_containment=False, label_min_clearance=4.0)
    assert chosen.bands[0].candidates[0].lines == ('long name here', 'g')


def test_long_sample_name_flows_to_shrink_the_region_gap(
        monkeypatch, tmp_path):
    solves = []
    original = grid.optimize_shared_gap_layout

    def traced(bands, **kwargs):
        result = original(bands, **kwargs)
        if (bands and bands[0].primary_dimension == 'height' and
                kwargs['hard_outer_containment'] is False):
            solves.append((kwargs, result))
        return result

    monkeypatch.setattr(grid, 'optimize_shared_gap_layout', traced)
    plotMatrix.main([
        '--heatmap', '-m', DATA, '--arrangeSamples', '1,2', '3,4',
        '--samplesLabel', 'GENE1 vs. Control', 'A123B vs. WT', 'B45C vs. WT',
        'B45C A123B vs. WT', '--sampleSetGroupArrangement', 'overlay',
        '--showRegionCounts', '--regionsLabel', 'genes_Q1', 'genes_Q2',
        '--cellWidth', '3', '--heatmapHeight', '6',
        '-o', str(tmp_path / 'wrap.png')])
    kwargs, result = [
        (kwargs, result) for kwargs, result in solves
        if any('B45C A123B' in c.text for c in result.bands[0].candidates)][-1]
    labels = result.bands[0].candidates
    assert any(c.lines[0] == 'B45C A123B vs.' and
               c.lines[1].startswith('WT — genes_Q') for c in labels)
    # Without flowing, the 77 pt label 'B45C A123B vs. WT|genes_Q…|[n = 3]'
    # forced a 44.5 pt gap.
    gap = result.common_gap
    assert gap < 44.5
    extents = kwargs['cell_extent']
    for index in range(len(labels) - 1):
        clearance = ((extents[index] + extents[index + 1]) / 2 + gap -
                     (labels[index].height + labels[index + 1].height) / 2)
        assert clearance >= 4.0 - 1e-6
    assert max(c.height for c in labels) < 77.0
    assert pytest.approx(kwargs['label_min_clearance']) == 4.0


def test_flowed_seam_uses_the_title_separator_and_is_measured():
    candidates = generate_label_candidates(
        'test\ngroup1.bed', _measure, max_lines=3, preferred_break=0)
    by_lines = {c.lines: c for c in candidates}
    assert ('test', 'group1.bed') in by_lines
    flowed = by_lines[('test \u2014 group1.bed',)]
    assert flowed.width == 6.0 * len('test \u2014 group1.bed')
    assert flowed.break_penalty == pytest.approx(0.1)
    # No flowed layout breaks at the separator itself.
    assert not any(line.startswith('\u2014') or line.endswith('\u2014')
                   for c in candidates for line in c.lines)
    assert all('\u2014' not in line for c in candidates
               if c.break_penalty == 0.0 for line in c.lines)


def _region_solves(monkeypatch, argv):
    solves = []
    original = grid.optimize_shared_gap_layout

    def traced(bands, **kwargs):
        result = original(bands, **kwargs)
        if (bands and bands[0].primary_dimension == 'height' and
                kwargs['hard_outer_containment'] is False):
            solves.append(result)
        return result

    monkeypatch.setattr(grid, 'optimize_shared_gap_layout', traced)
    plotMatrix.main(argv)
    return solves


def test_flowing_that_would_widen_the_gap_keeps_the_preferred_break(
        monkeypatch, tmp_path):
    # One-line 'test — group1.bed' labels are thinner but only fit if the
    # stacks move apart, so the name keeps its own line.
    solves = _region_solves(monkeypatch, [
        '--heatmap', '--profile', '-m', DATA, '--arrangeSamples', '1,2',
        '3,4', '--colorbarLocation', 'below',
        '-o', str(tmp_path / 'cap.png')])
    assert solves
    for result in solves:
        assert not any(c.crosses_preferred_break
                       for c in result.bands[0].candidates)
        assert all(c.lines[0] == 'test' for c in result.bands[0].candidates)


def test_flowed_candidates_are_offered_only_within_the_gap_cap():
    def make(height, flowed):
        return LabelCandidate(('x',), 10.0, height, 1, 0.0, 0.0,
                              crosses_preferred_break=flowed)

    kept = ((make(50.0, False),), (make(50.0, False),))
    short, long_ = make(44.0, True), make(80.0, True)
    offered = grid._flowed_within_gap(
        ((kept[0][0], short, long_), (kept[1][0],)), kept,
        heights=(36.0, 36.0), gap=10.0, clearance=4.0)
    assert offered == ((kept[0][0], short), kept[1])
