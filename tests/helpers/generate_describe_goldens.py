"""Regenerate six byte-stable describe contracts from the series test matrices.

Run ``python -m tests.helpers.generate_describe_goldens``. The fixture
matrices are made by the series capture helper; only describe JSON is kept.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from deeptoolsr.describe import describe
from tests.helpers.capture_series_goldens import write_input


HERE = Path(__file__).resolve().parents[1] / 'contract' / 'describe'
CASES = {
    'shared_colours': ('plotProfileR', 'repeated', (
        '--sameGroupLabels', 'together', '--perGroup', '--showRegionCounts')),
    'heatmap_summary': ('plotHeatmapR', 'repeated', (
        '--whatToShow', 'plot, heatmap and colorbar', '--colorMap', 'Reds', 'Blues')),
    'profile_heatmap': ('plotProfileR', 'repeated', (
        '--plotType', 'heatmap', '--colorsPerSample', 'Reds', 'Blues')),
    'relabel_together': ('plotProfileR', 'distinct', (
        '--sameGroupLabels', 'together', '--regionsLabel',
        'Equal', 'Other', 'Equal', '--perGroup')),
    'by_row': ('plotProfileR', 'repeated', (
        '--arrangeSamples', '1,2', '3', '--sampleSetLabels', 'First', 'Second',
        '--sampleSetGroupArrangement', 'by_row')),
    'kmeans': ('plotHeatmapR', 'repeated', (
        '--kmeans', '2', '--sortRegions', 'no')),
    'matrix_both': ('plotMatrixR', 'repeated', (
        '--profile', '--heatmap', '--perGroup')),
    'matrix_profile': ('plotMatrixR', 'distinct', ('--profile',)),
}


def generate(destination=HERE):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary:
        for name, (tool, source, options) in CASES.items():
            matrix = Path(temporary) / (source + '.gz')
            if not matrix.exists():
                write_input(matrix, distinct=source == 'distinct')
            record = describe(tool, ['-m', str(matrix), *options])
            (destination / (name + '.json')).write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    generate()
