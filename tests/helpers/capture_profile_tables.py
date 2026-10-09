"""Capture plotProfileR --outFileNameData TSVs (run from repo root)."""
import json
import os
import subprocess
import sys
import tempfile
CMO = 'tests/test_data/computeMatrixOperations.mat.gz'
REF = 'tests/test_data/compute_matrix_expected/blacklist_skip_zeros.mat.gz'
GTF = 'tests/test_data/compute_matrix_expected/gtf.mat.gz'
CASES = {
    'lines_default': [CMO],
    'se_pergroup': [CMO, '--plotType', 'se', '--perGroup'],
    'std_kb': [CMO, '--plotType', 'std', '--distanceUnit', 'kb'],
    'ci': [CMO, '--plotType', 'ci'],
    'lines_kmeans': [CMO, '--kmeans', '2'],
    'bootstrap_median': [CMO, '--plotType', 'bootstrap', '--averageType', 'median', '--bootstrapReplicates', '50'],
    'fill_sample_sets': [CMO, '--plotType', 'fill', '--arrangeSamples', '1-4', '5-8',
                         '--sampleSetLabels', 'fwd', 'rev', '--gridColumns', '2'],
    'refpoint_groups_counts': [REF, '--showRegionCounts', '--averageType', 'sum'],
    'refpoint_escaped_labels': [REF, '--regionsLabel', 'tab\there', 'line\nbreak',
                                '--samplesLabel', 'back\\slash', '--distanceUnit', 'kb'],
    'refpoint_pergroup_quantiles': [REF, '--perGroup', '--quantileSortedRegions', '2'],
    'gtf_scaled': [GTF, '--averageType', 'std', '--startLabel', 'start', '--endLabel', 'end'],
}
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    for name, (matrix, *extra) in CASES.items():
        argv = ['plotProfileR', '-m', matrix, '-o', os.path.join(tmp, name + '.png'),
                '--outFileNameData', os.path.join(out, name + '.tsv'), *extra]
        subprocess.run(argv, check=True, capture_output=True)
with open(os.path.join(out, 'cases.json'), 'w') as handle:
    json.dump(CASES, handle, indent=1)
    handle.write('\n')
