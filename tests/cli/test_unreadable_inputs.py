"""Unreadable matrices fail with one clean line; machine output stays UTF-8."""

import gzip
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / 'tests' / 'test_data' / 'computeMatrixOperations.mat.gz'


def _text():
    return gzip.decompress(MATRIX.read_bytes()).decode('utf-8')


def _header(text, **changes):
    header, body = text.split('\n', 1)
    parameters = json.loads(header[1:])
    parameters.update(changes)
    return '@' + json.dumps(parameters) + '\n' + body


def _corrupt(path, kind):
    data = MATRIX.read_bytes()
    if kind == 'truncated':
        path.write_bytes(data[:len(data) // 2])
    elif kind == 'truncated_header':
        path.write_bytes(data[:64])
    elif kind == 'zlib':
        path.write_bytes(data[:12] + bytes(b ^ 0xff for b in data[12:40])
                         + data[40:])
    elif kind == 'header_not_json':
        path.write_text('@{not json\n' + _text().split('\n', 1)[1])
    elif kind == 'body_huge':
        path.write_text(_header(_text(), body=10 ** 15))
    elif kind == 'groups_huge':
        path.write_text(_header(_text(), group_boundaries=[0, 10 ** 12]))
    return path


def _run(*argv, env=None):
    return subprocess.run(argv, capture_output=True, cwd=ROOT, env=env,
                          timeout=120)


KINDS = ('truncated', 'truncated_header', 'zlib', 'header_not_json',
         'body_huge', 'groups_huge')


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('tool', ('plotHeatmapR', 'plotProfileR'))
def test_plot_tools_report_unreadable_matrix_in_one_line(tool, kind, tmp_path):
    matrix = _corrupt(tmp_path / 'bad.mat.gz', kind)
    result = _run(tool, '-m', str(matrix), '-o', str(tmp_path / 'plot.png'))
    stderr = result.stderr.decode()
    assert result.returncode == 1
    assert stderr.startswith(f'{tool} failed: {matrix}')
    assert stderr.count('\n') == 1 and 'Traceback' not in stderr
    assert not (tmp_path / 'plot.png').exists()


def test_smallest_possible_rows_pass_the_header_size_check(tmp_path):
    from deeptoolsr.matrix import read_header
    rows, bins = 1000, 3
    header = {'sample_labels': ['A'], 'sample_boundaries': [0, bins],
              'group_labels': ['g'], 'group_boundaries': [0, rows],
              'bin size': [1], 'upstream': [1], 'body': [0],
              'downstream': [2], 'unscaled 5 prime': [0],
              'unscaled 3 prime': [0], 'ref point': ['TSS']}
    # Empty chrom, name, score and strand; no final newline.
    row = '\t0\t0\t\t\t' + '\t0' * bins
    matrix = tmp_path / 'minimal.mat'
    matrix.write_text('@' + json.dumps(header) + '\n'
                      + '\n'.join([row] * rows))
    assert read_header(matrix).group_boundaries == (0, rows)
    short = tmp_path / 'short.mat'
    short.write_text('@' + json.dumps(header) + '\n'
                     + '\n'.join(['1\t0\t2\t\t0\t+' + '\t0' * bins] * rows))
    info = _run('computeMatrixOperationsR', 'info', '-m', str(short))
    assert info.returncode == 0, info.stderr


@pytest.mark.parametrize('kind', ('truncated_header', 'zlib', 'groups_huge'))
def test_describe_reports_unreadable_matrix_in_one_line(kind, tmp_path):
    matrix = _corrupt(tmp_path / 'bad.mat.gz', kind)
    result = _run('deeptoolsr', 'describe', 'plotHeatmapR', '-m', str(matrix),
                  '-o', str(tmp_path / 'plot.png'))
    stderr = result.stderr.decode()
    assert result.returncode == 1 and result.stdout == b''
    assert stderr.startswith(f'plotHeatmapR failed: {matrix}')
    assert stderr.count('\n') == 1


def test_plot_outputs_are_checked_before_loading(tmp_path, monkeypatch):
    from deeptoolsr import plotHeatmap
    from deeptoolsr.session import PlotSession

    def never(*_args):
        raise AssertionError('matrix loaded before the output check')

    monkeypatch.setattr(PlotSession, 'matrix_for', never)
    with pytest.raises(SystemExit) as caught:
        plotHeatmap.main(['-m', str(MATRIX), '--kmeans', '2',
                          '-o', str(tmp_path / 'missing' / 'plot.png')])
    assert caught.value.code == 2


def _labelled(tmp_path, folder='dir_é'):
    # 'é' is in the Windows ANSI code page that native paths use there; 'ł' is not.
    directory = tmp_path / folder
    directory.mkdir()
    matrix = directory / 'm.mat.gz'
    matrix.write_bytes(gzip.compress(_header(
        _text(), group_labels=['geny_ł']).encode('utf-8')))
    return matrix


def _cp1252():
    return {**os.environ, 'PYTHONIOENCODING': 'cp1252'}


def test_info_and_describe_json_are_utf8_whatever_the_locale(tmp_path):
    matrix = _labelled(tmp_path)
    info = _run('computeMatrixOperationsR', 'info', '-m', str(matrix),
                '--json', env=_cp1252())
    assert info.returncode == 0, info.stderr
    assert json.loads(info.stdout.decode('utf-8'))['groups'][0]['label'] == (
        'geny_ł')
    record = _run('deeptoolsr', 'describe', 'plotHeatmapR', '-m', str(matrix),
                  '-o', str(tmp_path / 'p.png'), env=_cp1252())
    assert record.returncode == 0, record.stderr
    assert 'geny_ł' in record.stdout.decode('utf-8')


def test_serve_speaks_utf8_whatever_the_locale(tmp_path):
    matrix = _labelled(tmp_path)
    worker = subprocess.Popen(['deeptoolsr', 'serve'], cwd=ROOT, env=_cp1252(),
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)

    def ask(request):
        worker.stdin.write((json.dumps(request, ensure_ascii=False) + '\n')
                           .encode('utf-8'))
        worker.stdin.flush()
        while 'id' not in (reply := json.loads(
                worker.stdout.readline().decode('utf-8'))):
            pass
        return reply

    try:
        info = ask({'id': 1, 'info': ['-m', str(matrix)]})
        assert info['info']['groups'][0]['label'] == 'geny_ł'
        plot = ask({'id': 2, 'command': ['plotHeatmapR', '-m', str(matrix),
                                         '-o', str(matrix.parent / 'h.png')]})
        assert plot['ok'], plot
        assert (matrix.parent / 'h.png').exists()
        if os.name == 'nt':
            outside = _labelled(tmp_path, 'dir_ł')
            plot = ask({'id': 3, 'command': ['plotHeatmapR', '-m', str(outside),
                                             '-o', str(outside.parent / 'h.png')]})
            assert 'outside the Windows ANSI code page' in plot['error']['message'], plot
    finally:
        worker.stdin.close()
        worker.wait(timeout=60)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX byte arguments')
def test_describe_undecodable_label_is_one_line(tmp_path):
    record = _run('deeptoolsr', 'describe', 'plotProfileR', '-m', str(MATRIX),
                  '--regionsLabel', b'A\xff')
    assert record.returncode == 1
    assert record.stdout == b''
    assert record.stderr.decode().splitlines() == [
        'plotProfileR failed: an argument is not valid UTF-8 text']
