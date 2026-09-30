"""Plain matrix streams retain the compressed path's numeric tokens."""

import gzip
import math
from pathlib import Path

import pytest

from deeptoolsr import computeMatrixOperations as cmo


FIXTURES = Path(__file__).parent / 'test_data' / 'cmo_expected'
CASES = {
    'strand_plus': ['filterStrand', '--strand', '+'],
    'strand_minus': ['filterStrand', '--strand', '-'],
    'values_remove': ['filterValues', '--max', '4', '--filterUsingSamples',
                      's2', '--filterNans', 'any_bin'],
    'values_mask': ['filterValues', '--max', '4', '--filterUsingSamples',
                    's2', '--filterNans', 'any_bin', '--onFilterFail', 'maskSample'],
    'subset_reorder': ['subset', '--samples', 's2', 's1'],
}


def _same_values_before_reformatting(old, new):
    old_lines, new_lines = old.decode().splitlines(), new.decode().splitlines()
    assert len(old_lines) == len(new_lines)
    assert old_lines[0] == new_lines[0]
    for old_line, new_line in zip(old_lines[1:], new_lines[1:]):
        old_fields, new_fields = old_line.split('\t'), new_line.split('\t')
        assert old_fields[:6] == new_fields[:6]
        assert len(old_fields) == len(new_fields)
        for old_token, new_token in zip(old_fields[6:], new_fields[6:]):
            old_value, new_value = float(old_token), float(new_token)
            assert (math.isnan(old_value) and math.isnan(new_value) or
                    old_value == new_value)


@pytest.mark.parametrize('name', CASES)
def test_streamed_outputs_preserve_s4_gzip_bytes_and_tokens(tmp_path, name):
    operation, *options = CASES[name]
    source = FIXTURES / 'input.gz'
    compressed = tmp_path / f'{name}.gz'
    plain = tmp_path / f'{name}.txt'
    for output in (compressed, plain):
        cmo.main([operation, '-m', str(source), '-o', str(output),
                  '-p', '1', *options])

    expected_gzip = (FIXTURES / f'{name}.s4.gz').read_bytes()
    historical_plain = (FIXTURES / f'{name}.s4.txt').read_bytes()
    expected_plain = gzip.decompress(expected_gzip)
    assert compressed.read_bytes() == expected_gzip
    assert plain.read_bytes() == expected_plain
    assert historical_plain != expected_plain
    _same_values_before_reformatting(historical_plain, expected_plain)


def test_mask_sample_emits_nan_for_blanked_bins(tmp_path):
    output = tmp_path / 'masked.txt'
    cmo.main(['filterValues', '-m', str(FIXTURES / 'input.gz'),
              '-o', str(output), '-p', '1', *CASES['values_mask'][1:]])
    rows = {fields[3]: fields for fields in
            (line.split('\t') for line in output.read_text().splitlines()[1:])}
    assert rows['r2'][8:] == ['nan', 'nan']
    assert rows['r4'][8:] == ['nan', 'nan']
