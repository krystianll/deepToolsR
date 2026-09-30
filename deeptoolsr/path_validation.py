"""Early validation for command input and output paths."""

from contextlib import contextmanager
import os
import tempfile


def _path(value):
    if value is None:
        return None
    if not isinstance(value, (str, bytes, os.PathLike)) and hasattr(value, "name"):
        value = value.name
    try:
        value = os.fspath(value)
    except TypeError:
        return None
    if not value or value == "-":
        return None
    return value


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

    Existing unrelated outputs remain valid.  ``-`` is treated as a stream,
    not a filesystem path.
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
            if _aliases(output, other):
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


@contextmanager
def temporary_path_for(destination, suffix=".tmp"):
    """Yield a collision-safe scratch path beside ``destination`` and clean it."""
    destination = os.path.abspath(os.fspath(destination))
    directory = os.path.dirname(destination) or os.curdir
    prefix = ".{}.".format(os.path.basename(destination))
    descriptor, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    os.close(descriptor)
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@contextmanager
def atomic_output_path(destination, suffix=".tmp"):
    """Yield a same-directory temporary and atomically replace on success."""
    with temporary_path_for(destination, suffix=suffix) as path:
        yield path
        os.replace(path, destination)
