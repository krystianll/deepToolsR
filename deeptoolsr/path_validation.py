"""Early validation for command input and output paths."""

from contextlib import contextmanager
import os
import re
import stat
import tempfile
import unicodedata

# Published outputs get the mode a plain open() would give them; mkstemp
# creates 0600 files. Read the umask once, at import, before worker threads.
_UMASK = os.umask(0)
os.umask(_UMASK)
_OUTPUT_MODE = 0o666 & ~_UMASK
_TEMP_PREFIX = '.deeptoolsr.'


def _path(value):
    if value is None:
        return None
    if not isinstance(value, (str, bytes, os.PathLike)) and hasattr(value, "name"):
        value = value.name
    try:
        value = os.fspath(value)
    except TypeError:
        return None
    if not value:
        return None
    return value


def _name_key(path):
    """Case- and normalisation-insensitive key (APFS, NTFS, HFS+)."""
    real = os.path.realpath(os.path.abspath(path))
    return unicodedata.normalize('NFC', real).casefold()


_UNPORTABLE = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_RESERVED = re.compile(r'CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]', re.IGNORECASE)


def portable_file_name(text):
    """``text`` as a file name macOS, Linux and Windows all accept.

    Characters Windows forbids and control characters become '_', trailing
    dots and spaces go, and a reserved Windows device name (the part before
    the first dot) gets '_' appended. The result may be empty.
    """
    name = _UNPORTABLE.sub('_', text).rstrip('. ')
    stem, dot, rest = name.partition('.')
    return stem + '_' + dot + rest if _RESERVED.fullmatch(stem) else name


def _aliases(left, right):
    left_key = os.path.normcase(os.path.realpath(os.path.abspath(left)))
    right_key = os.path.normcase(os.path.realpath(os.path.abspath(right)))
    if left_key == right_key:
        return True
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        return False


def validate_input_output_paths(inputs, outputs):
    """Reject output/input aliases and aliases between multiple outputs.

    Existing unrelated outputs remain valid. Outputs whose names differ only
    in case or Unicode normalisation are rejected as aliases too.
    """
    input_paths = [path for value in inputs if (path := _path(value))]
    output_paths = [path for value in outputs if (path := _path(value))]
    for output in output_paths:
        for input_path in input_paths:
            if _aliases(output, input_path):
                raise ValueError("output path {!r} aliases input path {!r}"
                                 .format(output, input_path))
    for index, output in enumerate(output_paths):
        for other in output_paths[:index]:
            if _name_key(output) == _name_key(other) or _aliases(output, other):
                raise ValueError("output paths {!r} and {!r} resolve to the same file"
                                 .format(other, output))
    # Check every destination before any expensive calculation or final output
    # begins. The later same-directory temporary remains the authoritative
    # protection against races and disk errors; this catches deterministic path
    # failures up front, including a destination that is itself a directory.
    for output in output_paths:
        destination = os.path.abspath(output)
        directory = os.path.dirname(destination) or os.curdir
        if not os.path.isdir(directory):
            raise ValueError("output directory {!r} does not exist".format(directory))
        if os.path.isdir(destination):
            raise ValueError("output path {!r} is a directory".format(output))
        if os.path.exists(destination) and not os.access(destination, os.W_OK):
            raise ValueError("output path {!r} is not writable".format(output))
        descriptor = probe = None
        try:
            descriptor, probe = tempfile.mkstemp(
                prefix='.deeptoolsr-write-check-', dir=directory)
        except OSError as error:
            raise ValueError(
                "output directory {!r} is not writable: {}".format(
                    directory, error))
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if probe is not None:
                try:
                    os.unlink(probe)
                except FileNotFoundError:
                    pass


def _reserve(directory, suffix):
    descriptor, path = tempfile.mkstemp(
        prefix=_TEMP_PREFIX, suffix=suffix, dir=directory)
    os.close(descriptor)
    return path


@contextmanager
def temporary_path_for(destination, suffix=".tmp"):
    """Yield a collision-safe scratch path beside ``destination`` and clean it."""
    destination = os.path.abspath(os.fspath(destination))
    path = _reserve(os.path.dirname(destination) or os.curdir, suffix)
    try:
        yield path
    finally:
        _discard(path)


def _discard(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _output_mode(destination):
    """Keep an existing destination's permission bits; new files get the umask default."""
    try:
        return stat.S_IMODE(os.stat(destination).st_mode)
    except OSError:
        return _OUTPUT_MODE


def publish(staged):
    """Replace every ``(temporary, destination)`` pair, or none of them.

    Existing destinations are moved to same-directory backups until every
    replacement succeeded; any failure (or interrupt) restores them. The last
    replacement is a plain atomic ``os.replace`` (a lone output is never
    missing, even briefly) and is the commit point. Whether a pair was
    installed is read from the files (its temporary is gone), so an interrupt
    between a move and its bookkeeping cannot lose or mix outputs.
    """
    staged = list(staged)
    done = []  # (temporary, destination, backup)
    try:
        for index, (temporary, destination) in enumerate(staged):
            try:
                os.chmod(temporary, _output_mode(destination))
            except OSError:
                pass  # filesystems without POSIX modes keep their default
            backup = None
            if index < len(staged) - 1 and os.path.lexists(destination):
                backup = _reserve(os.path.dirname(os.path.abspath(destination)), '.bak')
            done.append((temporary, destination, backup))
            if backup is not None:
                os.replace(destination, backup)
            os.replace(temporary, destination)
    except BaseException as error:
        committed = len(done) == len(staged) and not os.path.lexists(staged[-1][0])
        kept = () if committed else _roll_back(done, error)
        _discard_backups(done, kept)
        raise
    _discard_backups(done, ())


def _discard_backups(done, kept):
    for _, _, backup in done:
        if backup is not None and backup not in kept:
            _discard(backup)


def _roll_back(done, error):
    """Restore every pair in ``done``; name the backups that stay behind."""
    kept = []
    for temporary, destination, backup in reversed(done):
        try:
            installed = not os.path.lexists(temporary)
            if backup is None:
                if installed:
                    _discard(destination)
            elif installed or not os.path.lexists(destination):
                os.replace(backup, destination)
            else:
                _discard(backup)
        except OSError:
            if backup is not None:
                kept.append(backup)
    if kept:
        error.add_note("could not restore previous outputs; they remain at: "
                       + ", ".join(kept))
    return kept


class OutputSet:
    """Stage several outputs; publish them together when the block succeeds."""

    def __init__(self):
        self._staged = []

    def stage(self, destination, suffix=".tmp"):
        if not destination:
            return None
        destination = os.fspath(destination)
        temporary = _reserve(
            os.path.dirname(os.path.abspath(destination)) or os.curdir, suffix)
        self._staged.append((temporary, destination))
        return temporary

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                publish(self._staged)
        finally:
            for temporary, _ in self._staged:
                _discard(temporary)
        return False


@contextmanager
def atomic_output_path(destination, suffix=".tmp"):
    """Yield a same-directory temporary and atomically replace on success."""
    with OutputSet() as outputs:
        yield outputs.stage(destination, suffix)
