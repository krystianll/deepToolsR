"""Regenerate the plain-stream fixtures from an unpacked reference wheel.

Run from the repository root after building and unpacking that wheel::

    .venv/bin/python tests/test_data/cmo_expected/generate_cmo_expected.py \
        --reference-package-dir <unpacked wheel>

The subprocess uses ``-S`` so the editable checkout cannot shadow the
reference package. The input and both historical output formats are stored here.
"""

import argparse
import gzip
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HEADER = {
    'group_boundaries': [0, 3, 6],
    'group_labels': ['first', 'second'],
    'sample_boundaries': [0, 2, 4],
    'sample_labels': ['s1', 's2'],
    'upstream': [0, 0], 'downstream': [0, 0], 'body': [2, 2],
    'unscaled 5 prime': [0, 0], 'unscaled 3 prime': [0, 0],
    'ref point': [None, None], 'bin size': [1, 1],
    'sort regions': 'keep', 'sort using': 'mean',
}
ROWS = [
    ('+', '1', '1e-3', '0.500000', '-0'),
    ('-', 'nan', '2', '3', '4'),
    ('+', '2', '2', 'nan', 'nan'),
    ('-', '5', '6', '1', '1'),
    ('+', '1e-3', '-0', '7', '8'),
    ('-', 'nan', 'nan', '0.500000', '1'),
]
CASES = {
    'strand_plus': ['filterStrand', '--strand', '+'],
    'strand_minus': ['filterStrand', '--strand', '-'],
    'values_remove': ['filterValues', '--max', '4', '--filterUsingSamples',
                      's2', '--filterNans', 'any_bin'],
    'values_mask': ['filterValues', '--max', '4', '--filterUsingSamples',
                    's2', '--filterNans', 'any_bin', '--onFilterFail', 'maskSample'],
    'subset_reorder': ['subset', '--samples', 's2', 's1'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-package-dir', dest='s4_package_dir',
                        type=Path, required=True)
    args = parser.parse_args()
    package_dir = args.s4_package_dir.resolve()
    if not list((package_dir / 'deeptoolsr').glob('_compute_matrix_stream*.so')):
        parser.error('the reference wheel must be unpacked at '
                     '--reference-package-dir')
    text = '@' + json.dumps(HEADER, separators=(',', ':')) + '\n'
    for i, (strand, *values) in enumerate(ROWS):
        text += '\t'.join(('chr1', str(i * 10), str(i * 10 + 1),
                           f'r{i}', '0', strand, *values)) + '\n'
    source = HERE / 'input.gz'
    source.write_bytes(gzip.compress(text.encode(), mtime=0))

    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as config_dir:
        env = os.environ.copy()
        env['DEEPTOOLSR_CONFIG_DIR'] = config_dir
        env['MPLCONFIGDIR'] = str(cache / 'matplotlib')
        env['PIP_CACHE_DIR'] = str(cache / 'pip')
        command = ('import sys; sys.path[:0] = sys.argv[1:3]; '
                   'from deeptoolsr.computeMatrixOperations import main; '
                   'main(sys.argv[3:])')
        for name, options in CASES.items():
            for suffix in ('.gz', '.txt'):
                output = HERE / f'{name}.s4{suffix}'
                subprocess.run([
                    sys.executable, '-S', '-c', command,
                    str(package_dir), site.getsitepackages()[0],
                    options[0], '-m', str(source), '-o', str(output),
                    '-p', '1', *options[1:]],
                    cwd=package_dir, env=env, check=True)


if __name__ == '__main__':
    main()
