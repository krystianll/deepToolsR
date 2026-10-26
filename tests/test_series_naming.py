"""Profile series are named by region group unless their panel mixes samples."""

import json

import numpy as np
import pytest

from deeptoolsr import _compute_matrix_io, plotProfile
from deeptoolsr.describe import describe


@pytest.fixture
def matrix(tmp_path):
    """Three samples ('Sample A'..'Sample C'), two groups ('GENE1', 'GENE2')."""
    path = tmp_path / 'matrix.gz'
    values = np.arange(4 * 9, dtype=np.float32).reshape(4, 9) / 10 + 1
    regions = [['chr1', [(i * 20, i * 20 + 10)], f'region{i}', i, '+', '0']
               for i in range(4)]
    header = {
        'sample_labels': ['Sample A', 'Sample B', 'Sample C'],
        'sample_boundaries': [0, 3, 6, 9],
        'group_labels': ['GENE1', 'GENE2'], 'group_boundaries': [0, 2, 4],
        'bin size': [1] * 3, 'upstream': [1] * 3, 'body': [1] * 3,
        'downstream': [1] * 3, 'unscaled 5 prime': [0] * 3,
        'unscaled 3 prime': [0] * 3, 'ref point': [None] * 3,
        'min threshold': None, 'max threshold': None,
        'sort regions': 'no', 'sort using': 'mean', 'proc number': 1,
    }
    _compute_matrix_io.write_matrix(
        str(path), json.dumps(header), regions, values, 1)
    return str(path)


def _naming(record):
    slots = [slot['label'] for domain in record['color_domains']
             for slot in domain['slots']]
    return record['series'], slots


def test_one_sample_per_panel_matches_the_default(matrix, tmp_path):
    arranged = ['--arrangeSamples', '1', '2', '3']
    assert _naming(describe('plotProfileR', ['-m', matrix, *arranged])) == (
        _naming(describe('plotProfileR', ['-m', matrix])))
    outputs = []
    for name, extra in (('default', []), ('arranged', arranged)):
        out = tmp_path / (name + '.png')
        plotProfile.main(['-m', matrix, '-o', str(out), *extra])
        outputs.append(out.read_bytes())
    assert outputs[0] == outputs[1]


def test_subset_of_one_sample_panels_names_groups(matrix):
    series, slots = _naming(describe(
        'plotProfileR', ['-m', matrix, '--arrangeSamples', '1', '3']))
    assert {item['label'] for item in series} == {'GENE1', 'GENE2'}
    assert slots == ['GENE1', 'GENE2']


def test_mixed_panel_names_group_and_sample(matrix):
    series, _slots = _naming(describe(
        'plotProfileR', ['-m', matrix, '--arrangeSamples', '1,2', '3']))
    labels = {item['key']: item['label'] for item in series}
    assert labels == {'g1:s1': 'GENE1: Sample A', 'g1:s2': 'GENE1: Sample B',
                      'g2:s1': 'GENE2: Sample A', 'g2:s2': 'GENE2: Sample B',
                      'g1:s3': 'GENE1', 'g2:s3': 'GENE2'}
