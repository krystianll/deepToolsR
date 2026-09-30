"""Byte pins for computeMatrix output."""

import pytest

from tests.helpers.make_compute_matrix_expected import CASES, EXPECTED, run_case


@pytest.mark.parametrize('name', CASES)
def test_compute_matrix_bytes(name, tmp_path):
    run_case(name, tmp_path)
    suffixes = ['mat.gz', 'mat']
    if name == 'reference_point':
        suffixes.append('tab')
    for suffix in suffixes:
        filename = '{}.{}'.format(name, suffix)
        assert (tmp_path / filename).read_bytes() == (EXPECTED / filename).read_bytes()
    if name == 'reference_point':
        assert (tmp_path / 'export.mat.gz').read_bytes() == \
            (EXPECTED / 'export.mat.gz').read_bytes()
