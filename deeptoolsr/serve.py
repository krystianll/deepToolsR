"""One JSON-lines worker around a resident :class:`PlotSession`.

Only standard-library modules are imported before the hello line. The reader
owns framing and cancellation; the main thread is the sole session owner.
"""

from collections import deque
from dataclasses import dataclass, field
import json
import sys
import threading
import traceback


PLOT_TOOLS = ('plotMatrixR', 'plotHeatmapR', 'plotProfileR')
TOOLS = (*PLOT_TOOLS, 'computeMatrixOperationsR')
OPERATIONS = frozenset(('command', 'describe', 'info', 'cancel', 'reload',
                        'close'))


def _valid_id(value):
    return ((isinstance(value, str) and bool(value)) or
            (isinstance(value, int) and not isinstance(value, bool)))


def _error(request_id, kind, message):
    return {'id': request_id, 'ok': False,
            'error': {'kind': kind, 'message': str(message)}}


class _Writer:
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()

    def write(self, value):
        with self.lock:
            self.write_locked(value)

    def write_locked(self, value):
        """Write under ``lock``; used by cancel for ordered acknowledgement."""
        self.stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        self.stream.flush()


@dataclass
class _Request:
    request_id: object
    operation: str
    value: object
    progress: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event)


class _Worker:
    def __init__(self, writer, stderr):
        self.writer = writer
        self.stderr = stderr
        self.condition = threading.Condition()
        self.queue = deque()
        self.running = None
        self.unanswered = set()
        self.closing = False
        self.reader_failed = False

    def _reply(self, value, request_id=None):
        # Match _cancel's lock order so a terminal reply and the removal of
        # its id/running state are one observable operation to the reader.
        with self.writer.lock:
            with self.condition:
                self.writer.write_locked(value)
                if request_id is not None:
                    self.unanswered.discard(request_id)
                    if (self.running is not None and
                            self.running.request_id == request_id):
                        self.running = None

    def _protocol(self, request_id, message):
        self._reply(_error(request_id, 'protocol', message))

    def _parse(self, line):
        try:
            data = json.loads(line)
        except (ValueError, UnicodeError) as error:
            self._protocol(None, f'invalid JSON: {error}')
            return
        if not isinstance(data, dict):
            self._protocol(None, 'request must be a JSON object')
            return
        response_id = data.get('id') if _valid_id(data.get('id')) else None
        try:
            request = self._validate_request(data)
        except ValueError as error:
            self._protocol(response_id, error)
            return
        if request.operation == 'cancel':
            self._cancel(request.value)
        elif request.operation == 'close':
            with self.condition:
                duplicate = (response_id is not None and
                             response_id in self.unanswered)
            if duplicate:
                self._protocol(response_id, 'duplicate id')
                return
            self._close()
            if response_id is not None:
                self._reply({'id': response_id, 'ok': True})
        else:
            self._enqueue(request)

    @staticmethod
    def _validate_request(data):
        request_id = data.get('id')
        operations = set(data) & OPERATIONS
        if len(operations) != 1 or set(data) - OPERATIONS - {'id', 'progress'}:
            raise ValueError('expected exactly one operation key')
        operation = operations.pop()
        if 'progress' in data and operation != 'command':
            raise ValueError('progress is allowed only with command')
        if operation == 'cancel':
            if 'id' in data or not _valid_id(data['cancel']):
                raise ValueError('cancel requires a target id')
            return _Request(None, operation, data[operation])
        if operation in ('close', 'reload'):
            if data[operation] is not True:
                raise ValueError(f'{operation} must be true')
            if 'id' in data and not _valid_id(request_id):
                raise ValueError('id must be a non-empty string or integer')
            return _Request(request_id, operation, True)
        if operation in ('command', 'describe', 'info'):
            value = data[operation]
            if not (isinstance(value, list) and
                    all(isinstance(part, str) for part in value)):
                raise ValueError(f'{operation} must be a string list')
            if operation != 'info' and (not value or value[0] not in PLOT_TOOLS):
                raise ValueError('unsupported plot tool')
            if operation == 'info' and (len(value) != 2 or value[0] != '-m'):
                raise ValueError('info requires ["-m", PATH]')
        if operation == 'command' and not isinstance(
                data.get('progress', False), bool):
            raise ValueError('progress must be a boolean')
        if not _valid_id(request_id):
            raise ValueError('id must be a non-empty string or integer')
        return _Request(request_id, operation, data[operation],
                        data.get('progress', False))

    def _enqueue(self, request):
        response_id = request.request_id
        with self.condition:
            if response_id is not None and response_id in self.unanswered:
                duplicate = True
            else:
                duplicate = False
                if response_id is not None:
                    self.unanswered.add(response_id)
                self.queue.append(request)
                self.condition.notify()
        if duplicate:
            self._protocol(response_id, 'duplicate id')

    def _cancel(self, request_id):
        queued = None
        # Keep the acknowledgement ahead of a running request's terminal
        # reply even when it finishes concurrently with this reader thread.
        with self.writer.lock:
            with self.condition:
                for request in self.queue:
                    if request.request_id == request_id:
                        queued = request
                        self.queue.remove(request)
                        break
                if queued is not None:
                    state = 'queued'
                elif (self.running is not None and
                      self.running.request_id == request_id):
                    self.running.cancelled.set()
                    state = 'running'
                else:
                    state = 'unknown'
                self.writer.write_locked(
                    {'cancel_ack': request_id, 'state': state})
                if queued is not None:
                    self.writer.write_locked(
                        _error(request_id, 'cancelled', 'cancelled'))
                    self.unanswered.discard(request_id)

    def _close(self):
        with self.condition:
            self.closing = True
            self.condition.notify_all()

    def read(self, stream):
        try:
            for line in stream:
                self._parse(line)
                with self.condition:
                    if self.closing:
                        break
        except BaseException:
            traceback.print_exc(file=self.stderr)
            with self.condition:
                self.reader_failed = True
        finally:
            self._close()

    def _next(self):
        with self.condition:
            while not self.queue and not self.closing:
                self.condition.wait()
            if self.closing:
                pending = tuple(self.queue)
                self.queue.clear()
                return None, pending
            request = self.queue.popleft()
            self.running = request
            return request, ()

    def _run(self, session, request):
        from deeptoolsr.prepare import DataError, OptionError
        from deeptoolsr.matrix import MatrixFormatError
        from deeptoolsr.session import Cancelled

        operation = request.operation
        try:
            if operation == 'reload':
                session.reload()
                return ({'id': request.request_id, 'ok': True}
                        if request.request_id is not None else None)
            if operation == 'describe':
                return {'id': request.request_id, 'ok': True,
                        'describe': session.describe(request.value)}
            if operation == 'info':
                from deeptoolsr.computeMatrixOperations import info_record
                return {'id': request.request_id, 'ok': True,
                        'info': info_record(request.value[1])}

            def progress(stage):
                if request.progress:
                    self.writer.write({'id': request.request_id, 'stage': stage})

            def emit(stream, message):
                if stream == 'stdout':
                    self.stderr.write(str(message))
                    self.stderr.flush()

            result = session.run(
                request.value, mode='worker', cancelled=request.cancelled.is_set,
                progress=progress, emit=emit)
            return {'id': request.request_id, 'ok': True,
                    'outputs': list(result.outputs),
                    'recomputed': list(result.recomputed),
                    'reused': list(result.reused),
                    'timings': result.timings,
                    'warnings': [item.rstrip('\n') for item in result.warnings]}
        except OptionError as error:
            return _error(request.request_id, 'options',
                          error.prefix + str(error))
        except (DataError, OSError, MatrixFormatError) as error:
            return _error(request.request_id, 'input', error)
        except Cancelled:
            return _error(request.request_id, 'cancelled', 'cancelled')
        except (Exception, SystemExit) as error:
            # A bug must cost one request, never the resident worker.
            traceback.print_exc(file=self.stderr)
            return _error(request.request_id, 'internal', error)

    def serve(self, session):
        while True:
            request, pending = self._next()
            if request is None:
                kind = 'protocol' if self.reader_failed else 'cancelled'
                for item in pending:
                    self._reply(_error(item.request_id, kind, kind),
                                item.request_id)
                return 1 if self.reader_failed else 0
            reply = self._run(session, request)
            if reply is None:
                with self.condition:
                    self.running = None
            else:
                self._reply(reply, request.request_id)


def serve_main(*, config=None, threads=None, cache_bytes=None,
               stdin=None, stdout=None, stderr=None):
    """Write hello before loading plotting code, then serve until close/EOF."""
    from importlib.metadata import version

    input_stream = sys.stdin if stdin is None else stdin
    error_stream = sys.stderr if stderr is None else stderr
    writer = _Writer(sys.stdout if stdout is None else stdout)
    writer.write({'hello': {'protocol': 1, 'version': version('deepToolsR'),
                            'tools': list(TOOLS),
                            'preferred_plot_tool': 'plotMatrixR'}})
    from deeptoolsr.session import PlotSession

    worker = _Worker(writer, error_stream)
    reader = threading.Thread(target=worker.read, args=(input_stream,),
                              name='deeptoolsr-serve-reader', daemon=True)
    reader.start()
    with PlotSession(config=config, threads=threads,
                     cache_bytes=cache_bytes) as session:
        status = worker.serve(session)
    reader.join()
    return status
