"""Compression-transparent helpers for computeMatrix text files."""

import gzip
import io
import os
from contextlib import contextmanager

from deeptoolsr import _compute_matrix_io


GZIP_MAGIC = b'\x1f\x8b'


def matrix_input_is_compressed(path):
    """Detect gzip from file contents rather than its suffix."""
    with open(path, 'rb') as handle:
        return handle.read(2) == GZIP_MAGIC


def matrix_output_is_compressed(path):
    """Use the conventional .gz suffix to select compressed output."""
    return os.fspath(path).lower().endswith('.gz')


@contextmanager
def open_matrix_text(path, mode='rt', *, compressed=None, newline=None):
    """Open compressed or plain computeMatrix text with one interface."""
    if 'b' in mode:
        raise ValueError('open_matrix_text requires text mode')
    if compressed is None:
        compressed = (matrix_input_is_compressed(path) if 'r' in mode
                      else matrix_output_is_compressed(path))
    if compressed and 'r' not in mode:
        binary_mode = mode.replace('t', 'b')
        with open(path, binary_mode) as raw:
            with gzip.GzipFile(fileobj=raw, mode=binary_mode,
                               filename='', mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding='utf-8', newline=newline) as handle:
                    yield handle
    else:
        opener = gzip.open if compressed else open
        with opener(path, mode, encoding='utf-8', newline=newline) as handle:
            yield handle


def validated_matrix_rows(handle, parameters, path):
    """Yield body lines after shared native validation, in bounded batches.

    The handle is already positioned after its header. This consumes the body
    once; numeric parsing stays native, and the batch never becomes a resident
    matrix. Empty lines follow the native reader's convention of being skipped.
    """
    expected_rows = parameters['group_boundaries'][-1]
    bins = parameters['sample_boundaries'][-1]
    rows = 0
    while batch := handle.readlines(1024 * 1024):
        batch = [line for line in batch if line.rstrip('\r\n')]
        if rows + len(batch) > expected_rows:
            raise ValueError("{}: matrix contains more regions than its header".format(path))
        _compute_matrix_io.validate_rows(batch, bins, rows)
        rows += len(batch)
        yield from batch
    if rows != expected_rows:
        raise ValueError("{}: expected {} regions but read {}".format(path, expected_rows, rows))
