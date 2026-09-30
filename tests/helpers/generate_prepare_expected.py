"""Regenerate plot CLI ordering fixtures from the current installed build.

Run with ``python -m tests.helpers.generate_prepare_expected``.
The plot is rendered to a temporary directory; BED, matrix and diagnostics are
the retained contract. Run only against an approved reference build to
establish the reference, then use ``run_case`` in tests without rewriting these files.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from deeptoolsr import _compute_matrix_io


HERE = Path(__file__).resolve().parents[1] / 'test_data' / 'prepare_expected'
SORT_USING = ('mean', 'median', 'max', 'min', 'sum', 'region_length', 'score')
CASES = {}


def add(name, tool='heatmap', options=(), input_name='input.gz',
        rejected=False, compressed=False):
    CASES[name] = (tool, tuple(options), input_name, rejected, compressed)


for method in ('ascend', 'descend'):
    for using in SORT_USING:
        for quantiles in (1, 4):
            add(f'sort_{method}_{using}_q{quantiles}', options=(
                '--sortRegions', method, '--sortUsing', using,
                '--quantiles', str(quantiles)))
add('sort_no', options=('--sortRegions', 'no'))
add('sort_keep', options=('--sortRegions', 'keep'))
add('sort_disjoint_mean', options=(
    '--sortRegions', 'ascend', '--sortUsing', 'mean',
    '--sortUsingSamples', '1', '3'))
add('sort_disjoint_median', options=(
    '--sortRegions', 'descend', '--sortUsing', 'median',
    '--sortUsingSamples', '1', '3'))
add('gzip_matrix', options=('--sortRegions', 'ascend'), compressed=True)
for method in ('kmeans', 'hclust'):
    add(method, options=(f'--{method}', '2', '--sortRegions', 'no'))
    add(f'{method}_selected', options=(
        f'--{method}', '2', '--clusterUsingSamples', '1', '3',
        '--sortRegions', 'no'))
add('kmeans_silhouette', options=(
    '--kmeans', '2', '--clusterUsingSamples', '1', '3', '--silhouette'))
add('cluster_relabel', options=(
    '--kmeans', '2', '--regionsLabel', 'First', 'Second'))
add('show_counts', options=('--showRegionCounts',))
add('empty_override', options=(
    '--regionsLabel', 'Alpha', 'Beta', 'Gone', '--filterNans', 'any_sample'))
add('quantile_custom', options=(
    '--regionsLabel', 'Custom', '--sortRegions', 'ascend', '--quantiles', '2'),
    input_name='single.gz')
add('profile_merge_duplicate', 'profile', (
    '--regionsLabel', 'Same', 'Same', 'Gone',
    '--sameGroupLabels', 'merge', '--filterNans', 'any_sample'))
add('profile_together_duplicate', 'profile', (
    '--regionsLabel', 'Same', 'Same', 'Gone',
    '--sameGroupLabels', 'together', '--filterNans', 'any_sample'))
add('profile_independent_relabel', 'profile', (
    '--regionsLabel', 'Alpha', 'Beta', 'Gone', '--filterNans', 'any_sample'))
add('profile_show_counts', 'profile', ('--showRegionCounts',))
add('profile_cluster_relabel', 'profile', (
    '--hclust', '2', '--regionsLabel', 'First', 'Second'))
add('reject_label_count', options=('--regionsLabel', 'Only',), rejected=True)
add('reject_sort_sample', options=('--sortUsingSamples', '4'), rejected=True)
add('reject_cluster_sample', options=(
    '--kmeans', '2', '--clusterUsingSamples', '4'), rejected=True)
add('reject_quantiles', options=('--quantiles', '13'), rejected=True)


def write_input(directory):
    """Write deterministic three- and one-group inputs using the native writer."""
    directory = Path(directory)
    rows = [
        [1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3],
        [2, 3, 1, 2, 3, 4], [4, 3, 2, 1, 2, 1],
        [np.nan, 2, 1, 2, 1, 2],
        [3, 4, 2, 3, 1, 2], [4, 4, 3, 3, 2, 2],
        [1, 2, 4, 3, 2, 1], [2, 1, 3, 4, 1, 2],
        [np.nan, 3, 2, 2, 4, 4],
        # The 'Empty' group: sample one is entirely NaN, so
        # --filterNans any_sample drops exactly these rows (rows 4 and 9 keep
        # a finite bin in every sample).
        [np.nan, np.nan, -1, -1, -1, -1], [np.nan, np.nan, -2, -2, -2, -2],
    ]
    # Offset each row by index/1000 so no two rows tie on any sort key: NumPy's
    # default argsort orders ties differently on x86-64 (SIMD) and ARM, so
    # tied keys would make these fixtures platform-specific. Base keys are
    # multiples of 1/30, larger than any offset difference.
    values = (np.asarray(rows, dtype=np.float64) +
              np.arange(len(rows))[:, None] / 1000).astype(np.float32)
    scores = ('1', '1', '.', 'bad', '0', '2.5', '-0', 'nan', '3', '4', '5', '6')
    regions = []
    for index in range(len(rows)):
        start = index * 20
        blocks = [(start, start + 3), (start + 7, start + 9 + index)]
        regions.append(['chr1', blocks, f'region{index}', index,
                        '-' if index % 2 else '+', scores[index]])
    header = {
        'sample_labels': ['one', 'two', 'three'],
        'sample_boundaries': [0, 2, 4, 6],
        'group_labels': ['North', 'South', 'Empty'],
        'group_boundaries': [0, 5, 10, 12],
        'bin size': [1, 1, 1], 'upstream': [1, 1, 1],
        'body': [0, 0, 0], 'downstream': [1, 1, 1],
        'unscaled 5 prime': [0, 0, 0],
        'unscaled 3 prime': [0, 0, 0],
        'ref point': [None, None, None],
        'min threshold': 0.0, 'max threshold': None,
        'sort regions': 'keep', 'sort using': 'mean', 'proc number': 1,
    }
    for name, labels, bounds in (
            ('input.gz', ['North', 'South', 'Empty'], [0, 5, 10, 12]),
            ('single.gz', ['All'], [0, 12])):
        header['group_labels'] = labels
        header['group_boundaries'] = bounds
        _compute_matrix_io.write_matrix(
            str(directory / name), json.dumps(header, separators=(',', ':')),
            regions, values, 1)


def run_case(name, directory, input_dir):
    tool, options, input_name, rejected, compressed = CASES[name]
    executable = shutil.which('plotHeatmapR' if tool == 'heatmap'
                              else 'plotProfileR')
    if executable is None:
        raise RuntimeError('plot CLI scripts must be installed first')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    matrix_suffix = '.gz' if compressed else '.txt'
    argv = [executable, '-m', str(Path(input_dir) / input_name),
            '-o', name + '.png', '-p', '1', '--dpi', '72',
            '--outFileSortedRegions', name + '.bed',
            '--outFileNameMatrix', name + matrix_suffix, *options]
    result = subprocess.run(argv, cwd=directory, capture_output=True,
                            text=True, check=False)
    stderr = result.stderr
    if rejected and result.returncode and 'Traceback (most recent call last)' in stderr:
        stderr = '\n'.join(line for line in stderr.splitlines()
                           if line.startswith('WARNING:'))
        if stderr:
            stderr += '\n'
        stderr += result.stderr.splitlines()[-1] + '\n'
    return {'status': result.returncode, 'stdout': result.stdout,
            'stderr': stderr}


def main():
    write_input(HERE)
    manifest = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name in CASES:
            outcome = run_case(name, scratch, HERE)
            rejected = CASES[name][3]
            if rejected == (outcome['status'] == 0):
                raise RuntimeError(f'{name}: unexpected status {outcome["status"]}: '
                                   f'{outcome["stderr"]}')
            manifest[name] = outcome
            if outcome['status'] == 0:
                matrix_suffix = '.gz' if CASES[name][4] else '.txt'
                for suffix in ('.bed', matrix_suffix):
                    shutil.copyfile(Path(scratch) / (name + suffix),
                                    HERE / (name + suffix))
        (HERE / 'cases.json').write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'Wrote {len(CASES)} CLI cases')


if __name__ == '__main__':
    main()
