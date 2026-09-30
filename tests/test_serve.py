"""Subprocess checks for the JSON-lines worker and its queue semantics."""

import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest


MATRIX = Path(__file__).parent / 'test_heatmapper' / 'master_multi.mat.gz'


class Worker:
    def __init__(self, tmp_path):
        env = os.environ.copy()
        env['DEEPTOOLSR_CONFIG_DIR'] = str(tmp_path / 'config')
        env['MPLCONFIGDIR'] = str(
            Path(__file__).parents[1] / '.cache' / 'matplotlib')
        self.process = subprocess.Popen(
            [sys.executable, '-c',
             'from deeptoolsr.deeptoolsr_list_tools import main; main()',
             'serve', '--cacheBytes', '40000000'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.lines = Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.hello = self.receive()

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def send(self, data):
        self.process.stdin.write(json.dumps(data) + '\n')
        self.process.stdin.flush()

    def send_raw(self, data):
        self.process.stdin.write(data + '\n')
        self.process.stdin.flush()

    def receive(self):
        try:
            line = self.lines.get(timeout=30)
        except Empty as error:
            raise AssertionError('worker did not write a reply') from error
        assert line is not None, self.process.stderr.read()
        return json.loads(line)

    def response(self, request_id):
        while True:
            line = self.receive()
            if line.get('id') == request_id and 'ok' in line:
                return line

    def stop(self):
        if self.process.poll() is None:
            if not self.process.stdin.closed:
                self.process.stdin.close()
            self.process.wait(timeout=30)
        self.reader.join(timeout=2)


@pytest.fixture(scope='module')
def worker(tmp_path_factory):
    value = Worker(tmp_path_factory.mktemp('serve'))
    yield value
    value.stop()


@pytest.fixture
def isolated_worker(tmp_path):
    value = Worker(tmp_path)
    yield value
    value.stop()


def command(tmp_path, name='plotProfileR', suffix='a', extra=()):
    return [name, '-m', str(MATRIX), '-o', str(tmp_path / f'{suffix}.png'),
            *extra]


def test_serve_parser_defaults_and_cache_budget():
    from deeptoolsr.deeptoolsr_list_tools import parse_arguments

    parser = parse_arguments()
    args = parser.parse_args(['serve'])
    assert args.config == 'auto'
    assert args.numberOfProcessors is None
    assert args.cacheBytes is None
    assert parser.parse_args(['serve', '--config', 'auto', '-p', '4',
                              '--cacheBytes', '0']).cacheBytes == 0
    with pytest.raises(SystemExit) as error:
        parser.parse_args(['serve', '--cacheBytes', '-1'])
    assert error.value.code == 2


def test_hello_and_warm_render(worker, tmp_path):
    from importlib.metadata import version

    assert worker.hello == {'hello': {
        'protocol': 1, 'version': version('deepToolsR'),
        'tools': ['plotMatrixR', 'plotHeatmapR', 'plotProfileR',
                  'computeMatrixOperationsR'],
        'preferred_plot_tool': 'plotMatrixR'}}
    worker.send({'id': 'first', 'command': command(tmp_path, suffix='first'),
                 'progress': True})
    stages = []
    while True:
        line = worker.receive()
        if 'stage' in line:
            stages.append(line['stage'])
        else:
            first = line
            break
    assert first['ok'] is True
    assert stages == ['parse', 'load', 'order', 'statistics', 'limits',
                      'scene', 'publish']
    worker.send({'id': 'second', 'command': command(tmp_path, suffix='second')})
    second = worker.response('second')
    assert second['ok'] is True
    assert 'statistics' in second['reused']
    assert 'scene_digest' not in second
    assert Path(second['outputs'][0]).is_file()


BAD_PROTOCOL = [
    '{', '[]', '{}',
    '{"id":"x","command":[],"describe":[]}',
    '{"command":["plotProfileR"]}',
    '{"id":[],"command":["plotProfileR"]}',
    '{"id":"x","command":3}',
    '{"id":"x","describe":["plotProfileR"],"progress":true}',
    '{"id":[],"reload":true}',
]


def test_protocol_errors_then_success(worker):
    for index, bad in enumerate(BAD_PROTOCOL):
        worker.send_raw(bad)
        failed = worker.receive()
        assert failed['ok'] is False
        assert failed['error']['kind'] == 'protocol'
        worker.send({'id': f'good-{index}', 'describe': [
            'plotProfileR', '-m', str(MATRIX)]})
        assert worker.response(f'good-{index}')['ok'] is True


def test_option_errors_then_success(worker):
    for index, bad in enumerate((['--unknown-option'], ['--help'],
                                 ['--version'])):
        worker.send({'id': f'bad-{index}',
                     'command': ['plotProfileR', *bad]})
        failed = worker.response(f'bad-{index}')
        assert failed['error']['kind'] == 'options'
        worker.send({'id': f'good-{index}', 'describe': [
            'plotProfileR', '-m', str(MATRIX)]})
        assert worker.response(f'good-{index}')['ok'] is True


def test_input_describe_info_and_reload(worker, tmp_path):
    worker.send({'id': 'missing', 'command': command(
        tmp_path, extra=['-m', str(tmp_path / 'missing.gz')])})
    assert worker.response('missing')['error']['kind'] == 'input'
    worker.send({'id': 'describe', 'describe': [
        'plotMatrixR', '-m', str(MATRIX), '--profile']})
    record = worker.response('describe')
    assert record['ok'] and record['describe']['protocol'] == 1
    worker.send({'id': 'info', 'info': ['-m', str(MATRIX)]})
    info = worker.response('info')
    assert info['ok'] and info['info']['protocol'] == 1
    cli = subprocess.run(
        [sys.executable, '-c',
         'from deeptoolsr.computeMatrixOperations import main; main()',
         'info', '--json', '-m', str(MATRIX)],
        capture_output=True, text=True, check=True)
    assert info['info'] == json.loads(cli.stdout)
    worker.send({'id': 'a', 'command': command(tmp_path, suffix='a')})
    assert worker.response('a')['ok']
    worker.send({'id': 'reload', 'reload': True})
    assert worker.response('reload')['ok']
    worker.send({'id': 'b', 'command': command(tmp_path, suffix='b')})
    assert 'matrix' in worker.response('b')['recomputed']
    worker.send({'reload': True})
    worker.send({'id': 'c', 'command': command(tmp_path, suffix='c')})
    idless_reload_reply = worker.receive()
    assert idless_reload_reply['id'] == 'c'
    assert idless_reload_reply['ok'] is True
    assert 'matrix' in idless_reload_reply['recomputed']


def test_malformed_matrix_is_input(worker, tmp_path):
    bad = tmp_path / 'bad.mat'
    bad.write_text('not a computeMatrix header\n')
    worker.send({'id': 'bad-plot', 'command': [
        'plotProfileR', '-m', str(bad), '-o', str(tmp_path / 'bad.png')]})
    assert worker.response('bad-plot')['error']['kind'] == 'input'
    worker.send({'id': 'bad-info', 'info': ['-m', str(bad)]})
    assert worker.response('bad-info')['error']['kind'] == 'input'
    worker.send({'id': 'after-bad-matrix', 'describe': [
        'plotProfileR', '-m', str(MATRIX)]})
    assert worker.response('after-bad-matrix')['ok']


@pytest.mark.skipif(os.name == 'nt', reason='POSIX worker FIFO required')
def test_running_and_queued_cancel_are_race_free(worker, tmp_path):
    fifo = tmp_path / 'config.fifo'
    os.mkfifo(fifo)
    blocked = command(tmp_path, suffix='blocked',
                      extra=['--config', str(fifo)])
    worker.send({'id': 'running', 'command': blocked, 'progress': True})
    assert worker.receive() == {'id': 'running', 'stage': 'parse'}
    worker.send({'id': 'queued', 'command': command(tmp_path, suffix='queued'),
                 'progress': True})
    worker.send({'cancel': 'queued'})
    assert worker.receive() == {'cancel_ack': 'queued', 'state': 'queued'}
    assert worker.receive()['error']['kind'] == 'cancelled'
    worker.send({'cancel': 'running'})
    assert worker.receive() == {'cancel_ack': 'running', 'state': 'running'}
    with fifo.open('w') as handle:
        handle.write('{}')
    assert worker.response('running')['error']['kind'] == 'cancelled'
    assert not (tmp_path / 'blocked.png').exists()
    assert not (tmp_path / 'queued.png').exists()
    assert not list(tmp_path.glob('*.tmp'))
    worker.send({'id': 'next', 'command': command(tmp_path, suffix='next')})
    assert worker.response('next')['ok']
    worker.send({'cancel': 'absent'})
    assert worker.receive() == {'cancel_ack': 'absent', 'state': 'unknown'}


@pytest.mark.skipif(os.name == 'nt', reason='POSIX worker FIFO required')
def test_duplicate_id_and_close_cancel_queue(isolated_worker, tmp_path):
    worker = isolated_worker
    fifo = tmp_path / 'config.fifo'
    os.mkfifo(fifo)
    worker.send({'id': 1, 'command': command(
        tmp_path, suffix='first', extra=['--config', str(fifo)]),
        'progress': True})
    assert worker.receive() == {'id': 1, 'stage': 'parse'}
    worker.send({'id': 1, 'command': command(tmp_path, suffix='duplicate')})
    assert worker.receive()['error']['kind'] == 'protocol'
    worker.send({'id': 2, 'command': command(tmp_path, suffix='queued')})
    worker.send({'id': 2, 'command': command(tmp_path, suffix='again')})
    assert worker.receive()['error']['kind'] == 'protocol'
    worker.send({'close': True})
    with fifo.open('w') as handle:
        handle.write('{}')
    assert worker.response(1)['ok']
    assert worker.response(2)['error']['kind'] == 'cancelled'
    assert not (tmp_path / 'queued.png').exists()
    assert worker.process.wait(timeout=30) == 0


@pytest.mark.skipif(os.name == 'nt', reason='POSIX worker FIFO required')
def test_eof_cancels_queued(isolated_worker, tmp_path):
    worker = isolated_worker
    fifo = tmp_path / 'config.fifo'
    os.mkfifo(fifo)
    worker.send({'id': 'one', 'command': command(
        tmp_path, suffix='one', extra=['--config', str(fifo)]),
        'progress': True})
    assert worker.receive() == {'id': 'one', 'stage': 'parse'}
    worker.send({'id': 'two', 'command': command(tmp_path, suffix='two'),
                 'progress': True})
    worker.process.stdin.close()
    with fifo.open('w') as handle:
        handle.write('{}')
    assert worker.response('one')['ok']
    assert worker.response('two')['error']['kind'] == 'cancelled'
    assert not (tmp_path / 'two.png').exists()
    assert worker.process.wait(timeout=30) == 0


def test_reader_crash_finishes_current_and_rejects_queue():
    from io import StringIO
    from deeptoolsr.serve import _Request, _Worker, _Writer

    started = threading.Event()
    release = threading.Event()

    class Session:
        def run(self, *_args, **_kwargs):
            started.set()
            assert release.wait(timeout=5)
            return SimpleNamespace(outputs=(), recomputed=(), reused=(),
                                   timings={}, warnings=())

    class BrokenInput:
        def __iter__(self):
            yield json.dumps({'id': 'queued', 'describe': [
                'plotProfileR', '-m', str(MATRIX)]}) + '\n'
            raise OSError('reader failed')

    output = StringIO()
    errors = StringIO()
    worker = _Worker(_Writer(output), errors)
    worker._enqueue(_Request('running', 'command', ['plotProfileR']))
    finished = Queue()
    consumer = threading.Thread(
        target=lambda: finished.put(worker.serve(Session())), daemon=True)
    consumer.start()
    assert started.wait(timeout=5)
    worker.read(BrokenInput())
    release.set()
    assert finished.get(timeout=5) == 1
    consumer.join(timeout=5)
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert replies[0]['id'] == 'running' and replies[0]['ok']
    assert replies[1]['id'] == 'queued'
    assert replies[1]['error']['kind'] == 'protocol'
    assert 'reader failed' in errors.getvalue()


def test_bad_colours_are_an_option_error_and_worker_survives(
        worker, tmp_path):
    worker.send({'id': 'colours', 'command': command(
        tmp_path, 'plotMatrixR', 'colours',
        ('--profile', '--colorsPerSample', 'red', '--perGroup'))})
    failed = worker.response('colours')
    assert failed['error']['kind'] == 'options'
    assert '--colorsPerSample needs exactly 4 colours' in failed['error']['message']
    worker.send({'id': 'after', 'command': command(
        tmp_path, 'plotMatrixR', 'after', ('--profile',))})
    assert worker.response('after')['ok'] is True


def test_system_exit_is_internal_and_worker_recovers():
    from io import StringIO
    from deeptoolsr.serve import _Request, _Worker, _Writer

    class Session:
        def run(self, *_args, **_kwargs):
            raise SystemExit('stray exit')

    worker = _Worker(_Writer(StringIO()), StringIO())
    reply = worker._run(Session(), _Request('one', 'command', ['plotProfileR']))
    assert reply['error'] == {'kind': 'internal', 'message': 'stray exit'}


def test_unexpected_value_error_is_internal_and_worker_recovers():
    from io import StringIO
    from deeptoolsr.serve import _Request, _Worker, _Writer

    class Session:
        def __init__(self):
            self.calls = 0

        def run(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError('unexpected bug')
            return SimpleNamespace(outputs=(), recomputed=(), reused=(),
                                   timings={}, warnings=())

    stderr = StringIO()
    worker = _Worker(_Writer(StringIO()), stderr)
    session = Session()
    first = worker._run(session, _Request('first', 'command', ['plotProfileR']))
    assert first['error'] == {'kind': 'internal', 'message': 'unexpected bug'}
    assert 'ValueError: unexpected bug' in stderr.getvalue()
    second = worker._run(session, _Request('second', 'command', ['plotProfileR']))
    assert second['ok'] is True


def test_terminal_write_makes_concurrent_cancel_unknown():
    from io import StringIO
    from deeptoolsr.serve import _Request, _Worker, _Writer

    written = threading.Event()
    release = threading.Event()
    cancelling = threading.Event()

    class BlockedWriter(_Writer):
        def write_locked(self, value):
            super().write_locked(value)
            if value.get('id') == 'done' and 'ok' in value:
                written.set()
                assert release.wait(timeout=5)

    output = StringIO()
    worker = _Worker(BlockedWriter(output), StringIO())
    with worker.condition:
        worker.running = _Request('done', 'command', ['plotProfileR'])
        worker.unanswered.add('done')
    terminal = threading.Thread(
        target=lambda: worker._reply({'id': 'done', 'ok': True}, 'done'))
    terminal.start()
    assert written.wait(timeout=5)

    def cancel():
        cancelling.set()
        worker._cancel('done')

    canceller = threading.Thread(target=cancel)
    canceller.start()
    assert cancelling.wait(timeout=5)
    release.set()
    terminal.join(timeout=5)
    canceller.join(timeout=5)
    assert not terminal.is_alive() and not canceller.is_alive()
    assert [json.loads(line) for line in output.getvalue().splitlines()] == [
        {'id': 'done', 'ok': True},
        {'cancel_ack': 'done', 'state': 'unknown'},
    ]
