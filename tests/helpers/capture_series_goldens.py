"""Series golden case inputs and artist inspection.

The pinned goldens came from an earlier renderer and are not regenerated
here. The newly supported case below is captured from the current cell
planner and builder.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from matplotlib.colors import to_hex
import numpy as np

from deeptoolsr import _compute_matrix_io


HERE = Path(__file__).resolve().parents[1] / 'contract' / 'series'
LINE_COLORS = tuple('#{:02x}4060'.format(value)
                    for value in (24, 48, 72, 96, 120, 144, 168, 192, 216))
SUMMARY_COLORS = ('red', 'blue', 'green', 'purple', 'orange', 'cyan',
                  'brown', 'pink', 'olive')
# Distinctly coloured series per (--sameGroupLabels, --sameSampleLabels) in
# the placement grid: --colorsPerSample must name exactly that many colours
# (the goldens were captured with all nine; only the first N are used).
LINE_SLOTS = {('independent', 'independent'): 3,
              ('independent', 'together'): 2,
              ('together', 'independent'): 6,
              ('together', 'together'): 4,
              ('merge', 'independent'): 3,
              ('merge', 'together'): 2}
OVERLAY_BY_SAMPLE_SLOTS = {('independent', 'independent'): 9,
                           ('independent', 'together'): 6,
                           ('together', 'independent'): 6,
                           ('together', 'together'): 4,
                           ('merge', 'independent'): 6,
                           ('merge', 'together'): 4}
SUMMARY_SLOTS = 3
WRAPPING_TITLE = ' '.join(('Longword',) * 20)


@dataclass(frozen=True)
class Case:
    tool: str
    options: tuple[str, ...]
    category: str
    rejected: bool = False
    source: str = 'repeated'


def cases():
    result = {}
    for placement in ('overlay', 'adjacent', 'end', 'by_row', 'by_column'):
        for per_group in (False, True):
            for groups in ('independent', 'together', 'merge'):
                for samples in ('independent', 'together'):
                    name = ('profile_{}_{}_{}_{}'.format(
                        placement, 'per_group' if per_group else 'by_sample',
                        groups, samples))
                    slots = (OVERLAY_BY_SAMPLE_SLOTS
                             if placement == 'overlay' and not per_group
                             else LINE_SLOTS)[groups, samples]
                    options = (
                        '--arrangeSamples', '1,2', '3',
                        '--sampleSetLabels', 'Repeated samples', 'Third sample',
                        '--sampleSetGroupArrangement', placement,
                        '--sameGroupLabels', groups,
                        '--sameSampleLabels', samples,
                        '--colorsPerSample', *LINE_COLORS[:slots])
                    if per_group:
                        options += ('--perGroup',)
                    result[name] = Case('profile', options, 'placement-grid')
    result.update({
        'profile_implicit_sets': Case('profile', (), 'implicit-sets'),
        'profile_explicit_sets': Case('profile', (
            '--arrangeSamples', '1,2', '3', '--sampleSetLabels',
            WRAPPING_TITLE, 'Third', '--plotWidth', '3'),
            'explicit-sets'),
        'profile_counts': Case('profile', (
            '--showRegionCounts', '--sameGroupLabels', 'together',
            '--perGroup'), 'counts'),
        'profile_relabel_equal': Case('profile', (
            '--sameGroupLabels', 'together', '--perGroup',
            '--regionsLabel', 'Equal', 'Other', 'Equal'), 'relabel',
            source='distinct'),
        'profile_relabel_equal_counts': Case('profile', (
            '--sameGroupLabels', 'together', '--perGroup',
            '--regionsLabel', 'Equal', 'Other', 'Equal',
            '--showRegionCounts'), 'relabel', source='distinct'),
        'profile_relabel_distinct': Case('profile', (
            '--sameGroupLabels', 'together', '--perGroup',
            '--regionsLabel', 'First', 'Other', 'Last'), 'relabel'),
        'profile_relabel_distinct_counts': Case('profile', (
            '--sameGroupLabels', 'together', '--perGroup',
            '--regionsLabel', 'First', 'Other', 'Last',
            '--showRegionCounts'), 'relabel'),
        'profile_heatmap_colormaps': Case('profile', (
            '--plotType', 'heatmap', '--colorsPerSample', 'Reds', 'Blues',
            'Greens', '--startLabel', 'TSS', 'TSS', 'TSS',
            '--endLabel', 'TES', 'TES', 'TES'), 'profile-heatmap'),
        'heatmap_summary': Case('heatmap', (
            '--whatToShow', 'plot, heatmap and colorbar',
            '--colorsSummaryPlot', *SUMMARY_COLORS[:SUMMARY_SLOTS],
            '--colorMap', 'Reds', 'Blues'), 'heatmap-summary'),
        'heatmap_summary_per_group': Case('heatmap', (
            '--whatToShow', 'plot, heatmap and colorbar', '--perGroup',
            '--colorsSummaryPlot', *SUMMARY_COLORS[:SUMMARY_SLOTS],
            '--colorMap', 'Reds', 'Blues'), 'heatmap-summary'),
        'heatmap_summary_color_lists': Case('heatmap', (
            '--whatToShow', 'plot, heatmap and colorbar',
            '--colorsSummaryPlot', *SUMMARY_COLORS[:SUMMARY_SLOTS],
            '--colorList', 'white,red', 'black,blue'), 'heatmap-summary'),
        'heatmap_no_summary': Case('heatmap', (
            '--whatToShow', 'heatmap and colorbar', '--colorMap', 'Reds', 'Blues'),
            'heatmap-no-summary'),
        'reject_unknown_sample': Case('profile', (
            '--arrangeSamples', '4'), 'rejection', True),
        'reject_repeated_sample': Case('profile', (
            '--arrangeSamples', '1,1'), 'rejection', True),
        'reject_heatmap_per_group_sets': Case('heatmap', (
            '--perGroup', '--whatToShow', 'plot, heatmap and colorbar',
            '--arrangeSamples', '1,2', '3'), 'heatmap-summary'),
    })
    return result


def write_input(path, *, distinct=False):
    """Small untied values; three unequal groups and repeated labels."""
    values = (np.arange(9 * 6, dtype=np.float32).reshape(9, 6) / 10 + 1)
    regions = [['chr1', [(i * 20, i * 20 + 10)], f'region{i}', i, '+', '0']
               for i in range(9)]
    header = {
        'sample_labels': ['Repeat', 'Repeat', 'Third'],
        'sample_boundaries': [0, 2, 4, 6],
        'group_labels': (['First', 'Other', 'Last'] if distinct
                         else ['Equal', 'Other', 'Equal']),
        'group_boundaries': [0, 2, 5, 9],
        'bin size': [1, 1, 1], 'upstream': [1, 1, 1],
        'body': [0, 0, 0], 'downstream': [1, 1, 1],
        'unscaled 5 prime': [0, 0, 0],
        'unscaled 3 prime': [0, 0, 0],
        'ref point': [None, None, None],
        'min threshold': None, 'max threshold': None,
        'sort regions': 'no', 'sort using': 'mean', 'proc number': 1,
    }
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header, separators=(',', ':')),
        regions, values, 1)


def _colour(value):
    return to_hex(value, keep_alpha=True)


def _rendered(figure, axes, *, colorbars=False):
    roles = [axis.get_gid() for axis in figure.axes]
    assert all(roles) and len(set(roles)) == len(roles)
    lines = []
    for axis in axes:
        lines.append([{'label': line.get_label(), 'color': _colour(line.get_color())}
                      for line in axis.lines])
    legends = []
    for index, axis in enumerate(figure.axes):
        legend = axis.get_legend()
        if legend is not None:
            legends.append({'scope': axis.get_gid() or f'axis-{index}',
                            'entries': [item.get_text() for item in legend.get_texts()]})
    for legend in figure.legends:
        legends.append({'scope': 'figure',
                        'entries': [item.get_text() for item in legend.get_texts()]})
    slot_text = [
        {'slot': axis.get_gid(), 'text': [item.get_text() for item in axis.texts]}
        for axis in figure.axes
        if axis.texts and ((axis.get_gid() or '').startswith(
            ('label/', 'colorbar_title/')) or
            (axis.get_gid() or '').endswith('/title'))]
    record = {'lines': lines, 'legends': legends, 'slot_text': slot_text}
    if colorbars:
        record['colorbars'] = [axis.get_gid() for axis in figure.axes
                               if (axis.get_gid() or '').startswith('colorbar/')]
    return record


def regenerate_newly_supported_case(output_dir):
    """Capture the heatmap layout that the earlier renderer rejected."""
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    from deeptoolsr import plotMatrix
    from deeptoolsr.plotting.matrix_spec import project_matrix_spec
    from deeptoolsr.plotting.series import (
        cell_options, resolve_cells, series_slot)
    from tests.plotting.test_cells import _inputs

    name = 'reject_heatmap_per_group_sets'
    case = cases()[name]
    with TemporaryDirectory() as scratch:
        module, argv, args, _layout, labels, samples, sizes = _inputs(
            Path(scratch), case)
        opts = cell_options(project_matrix_spec(args, 'plotHeatmapR'))
        plan = resolve_cells(sizes, samples, labels, opts)
        by_key = {series.key: series for series in plan.series}
        panels = []
        for panel in plan.panels:
            item = {'row': panel.row, 'column': panel.column,
                    'kind': panel.kind, 'title': panel.title,
                    'export_label': panel.export_label, 'series': []}
            for key in panel.series:
                series = by_key[key]
                item['series'].append({
                    'group': series.group, 'sample': series.sample,
                    'label': series.label,
                    'color_index': series_slot(plan, 0, panel.index, key) - 1})
            panels.append(item)
        colors = [to_hex(value, keep_alpha=True) for value in args.colors or ()]
        semantic = {
            'panels': panels, 'row_labels': list(plan.row_labels),
            'color_lists': {'summary': colors,
                            'heatmap': list(args.colorList or args.colorMap)},
            'heatmap_color_option': plan.color_domains[-1].option,
            'resolved_colormap_names': list(args.colorList or args.colorMap),
        }
        captured = {}

        def save(figure, path, **_kwargs):
            axes = [axis for axis in figure.axes if
                    (axis.get_gid() or '').endswith('/profile')]
            captured['rendered'] = _rendered(figure, axes, colorbars=True)
            Path(path).write_bytes(b'')

        with patch.object(plotMatrix, 'save_figure_atomic', save):
            module.main(argv)
    output_dir = Path(output_dir)
    (output_dir / (name + '.json')).write_text(json.dumps(
        {'semantic': semantic, 'rendered': captured['rendered']},
        indent=2, sort_keys=True) + '\n', newline='\n')
    index_path = output_dir / 'index.json'
    index = json.loads(index_path.read_text())
    index[name] = {'category': case.category, 'options': list(case.options),
                   'source': case.source, 'status': 0, 'tool': case.tool}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + '\n', newline='\n')


if __name__ == '__main__':
    regenerate_newly_supported_case(HERE)
