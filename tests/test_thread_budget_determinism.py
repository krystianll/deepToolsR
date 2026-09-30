"""CLI output stays stable when independent work uses more threads."""

import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


DATA = Path(__file__).parent / 'test_heatmapper'
OPS = Path(__file__).parent / 'test_data'

PROFILE = [
    ('lines', 'mean'), ('se', 'mean'), ('std', 'mean'), ('ci', 'mean'),
    ('fill', 'mean'), ('bootstrap', 'mean'), ('lines', 'median'),
    ('lines', 'geom_mean'), ('lines', 'trim_mean'),
    ('bootstrap', 'median'), ('ci', 'geom_mean'),
]
HEATMAP = [
    ('limits', []), ('sort', ['--sortRegions', 'ascend']),
    ('quantiles', ['--sortRegions', 'ascend', '--quantiles', '2']),
    ('kmeans', ['--kmeans', '2']),
]
OPERATIONS = [
    ('filterValues', ['--min', '0', '--max', '100']),
    ('sort', ['-R', str(OPS / 'computeMatrixOperations.bed')]),
    ('transform', ['--scale', '2']), ('rbind', []),
    ('cbind', ['--blind']),
]


def _cases(group):
    if group == 'profile':
        for plot_type, average in PROFILE:
            extra = ['--plotType', plot_type, '--averageType', average]
            if plot_type == 'bootstrap':
                extra += ['--bootstrapReplicates', '80']
            yield ('{}_{}'.format(plot_type, average), 'plotProfile',
                   ['-m', str(DATA / 'master_multi.mat.gz'), *extra],
                   ('plot.png', 'data.tsv'))
    elif group == 'heatmap':
        for name, extra in HEATMAP:
            yield (name, 'plotHeatmap',
                   ['-m', str(DATA / 'master_multi.mat.gz'), '--zMin', '0',
                    '--zMax', '5', *extra],
                   ('plot.png', 'regions.bed', 'matrix.mat.gz'))
    elif group == 'compute_matrix':
        yield ('reference_point', 'computeMatrix',
               ['reference-point', '-S', str(DATA / 'test.bw'),
                '-R', str(DATA / 'test2.bed'), '-b', '100', '-a', '100',
                '-bs', '10', '--quiet'], ('matrix.mat.gz',))
    elif group == 'operations':
        for command, extra in OPERATIONS:
            sources = [str(OPS / 'computeMatrixOperations.mat.gz')]
            if command in ('rbind', 'cbind'):
                sources *= 2
            yield (command, 'computeMatrixOperations',
                   [command, '-m', *sources, *extra], ('matrix.mat.gz',))
    else:
        for name, normalization in [('bam_mean', 'coverage-mean'),
                                    ('bam_bpm', 'BPM')]:
            yield (name, 'bamCoverage',
                   ['-b', str(OPS / 'test_paired.bam'), '--binSize', '1000',
                    '--zoomLevels', '0', '--normalizeUsing', normalization],
                   ('track.bw',))
        yield ('merge', 'bigWigOperations',
               ['merge', '-b', str(DATA / 'test.bw'), str(DATA / 'test.bw'),
                '--zoomLevels', '0'], ('track.bw',))


def _normalise(name, filename, contents, threads):
    if name == 'reference_point' and filename.endswith('.gz'):
        header, body = gzip.decompress(contents).split(b'\n', 1)
        values = json.loads(header[1:])
        assert values['proc number'] == threads
        values['proc number'] = 0
        return values, body
    return contents


@pytest.mark.parametrize('group',
                         ['profile', 'heatmap', 'compute_matrix',
                          'operations', 'coverage'])
def test_cli_outputs_independent_of_thread_budget(group, tmp_path):
    for name, tool, base, filenames in _cases(group):
        output_dir = tmp_path / name
        output_dir.mkdir()
        observed = []
        for threads in (1, 3, 4):
            paths = {filename: output_dir / filename for filename in filenames}
            if tool in ('plotProfile', 'plotHeatmap'):
                argv = base + ['-o', str(paths['plot.png'])]
                if tool == 'plotProfile':
                    argv += ['--outFileNameData', str(paths['data.tsv'])]
                else:
                    argv += ['--outFileSortedRegions', str(paths['regions.bed']),
                             '--outFileNameMatrix', str(paths['matrix.mat.gz'])]
            else:
                argv = base + ['-o', str(paths[filenames[0]])]
            argv += ['-p', str(threads)]
            code = ('import sys; from deeptoolsr import {0}; '
                    '{0}.main(sys.argv[1:])').format(tool)
            result = subprocess.run([sys.executable, '-c', code, *argv],
                                    capture_output=True, text=True,
                                    env=os.environ.copy(), timeout=60)
            assert result.returncode == 0, (name, threads, result.stderr)
            observed.append({filename: _normalise(
                name, filename, path.read_bytes(), threads)
                for filename, path in paths.items()})
        assert observed[0] == observed[1] == observed[2], name


def test_heatmap_automatic_digest_limits_independent_of_thread_budget(tmp_path):
    matrix = tmp_path / 'digest.mat.gz'
    generator = Path(__file__).parents[1] / 'benchmarks/plots/make_matrix.py'
    subprocess.run([sys.executable, str(generator), '1500', '3', '250',
                    str(matrix), '--seed', '503'], check=True,
                   env=os.environ.copy())
    pngs = []
    for threads in (1, 4):
        output = tmp_path / 'p{}.png'.format(threads)
        code = ('import sys; from deeptoolsr import plotHeatmap; '
                'plotHeatmap.main(sys.argv[1:])')
        result = subprocess.run(
            [sys.executable, '-c', code, '-m', str(matrix), '-o', str(output),
             '-p', str(threads), '--heatmapHeight', '5', '--heatmapWidth', '3',
             '--dpi', '80'], capture_output=True, text=True,
            env=os.environ.copy(), timeout=120)
        assert result.returncode == 0, result.stderr
        pngs.append(output.read_bytes())
    assert pngs[0] == pngs[1]


@pytest.mark.parametrize('operation', ['filterValues', 'cbind'])
def test_cmo_gzip_header_has_no_time_or_filename(operation, tmp_path,
                                                 monkeypatch):
    from deeptoolsr import computeMatrixOperations as cmo

    original_compress = gzip.compress

    def legacy_compress_default(data, *args, **kwargs):
        # Python 3.14 defaults to mtime=0; emulate 3.11–3.13 if omitted.
        kwargs.setdefault('mtime', None)
        return original_compress(data, *args, **kwargs)

    monkeypatch.setattr(gzip, 'compress', legacy_compress_default)
    monkeypatch.setattr(time, 'time', lambda: 1_700_000_123)
    source = str(OPS / 'computeMatrixOperations.mat.gz')
    output = tmp_path / 'output.mat.gz'
    if operation == 'filterValues':
        argv = ['filterValues', '-m', source, '--min', '0', '--max', '100']
    else:
        argv = ['cbind', '-m', source, source, '--blind']
    cmo.main([*argv, '-o', str(output), '-p', '1'])
    contents = output.read_bytes()
    assert contents[:3] == b'\x1f\x8b\x08'
    assert contents[3] & 0x08 == 0  # no FNAME flag
    assert contents[4:8] == b'\0' * 4
    assert gzip.decompress(contents).startswith(b'@')
