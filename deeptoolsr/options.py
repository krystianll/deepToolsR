"""Resolve per-command configuration and processor budgets."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

from deeptoolsr import config


@dataclass(frozen=True)
class RasterOptions:
    filter: str
    bit_depth: int


@dataclass(frozen=True)
class RunOptions:
    threads: int
    raster: RasterOptions
    config_path: Path
    ward_distance_budget_bytes: int
    style: object = None


def physical_memory_bytes():
    if os.name == 'nt':
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        ('total_phys', ctypes.c_ulonglong),
                        ('available_phys', ctypes.c_ulonglong),
                        ('total_page', ctypes.c_ulonglong),
                        ('available_page', ctypes.c_ulonglong),
                        ('total_virtual', ctypes.c_ulonglong),
                        ('available_virtual', ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        return status.total_phys if ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)) else None
    try:
        return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, OSError, ValueError):
        return None


def default_ward_distance_budget_bytes():
    physical = physical_memory_bytes()
    return min(2 << 30, max(1, physical // 2)) if physical else 2 << 30


def add_config_option(parser, *, default='auto'):
    parser.add_argument('--config', default=default, metavar='PATH',
                        help='Options file path, or auto for the default file.')


def processor_value(value):
    if value in ('auto', 'max'):
        return value
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('expected a positive integer, max, or auto')
    if count < 1:
        raise argparse.ArgumentTypeError('expected a positive integer, max, or auto')
    return count


def add_processor_option(parser, *, allow_default=False, default='auto'):
    def parse(value):
        if allow_default and value == 'default':
            return 'auto'
        return processor_value(value)

    parser.add_argument('-p', '--numberOfProcessors', default=default,
                        type=parse, metavar='N|max|auto',
                        help='Threads: positive integer, max, or auto.')


def config_path(value):
    return config.options_path() if value == 'auto' else Path(value).expanduser()


def read_config_bytes(path):
    """Read the effective options file once, including FIFO-backed files."""
    try:
        return Path(path).read_bytes()
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None


def config_content_hash(raw):
    return hashlib.sha256(raw).hexdigest() if raw is not None else None


def resolve_run_options_from_bytes(args, raw, path, *, plotting=False,
                                   emit=None):
    """Resolve one request without rereading its options file."""
    try:
        stored = json.loads(raw.decode('utf-8')) if raw is not None else None
    except (ValueError, TypeError, UnicodeDecodeError):
        stored = None
    stored = stored if isinstance(stored, dict) else {}
    values = config.load_options(path, stored=stored)
    available = max(1, os.cpu_count() or 1)
    fallback = max(1, available // 2)
    chosen = getattr(args, 'numberOfProcessors', 'auto') or 'auto'
    if chosen in ('auto', 'default'):
        chosen = values.get('number_of_processors', 'auto')
        if chosen == 'auto' and 'number_of_processors' in stored:
            message = 'deeptoolsr: ignoring invalid number_of_processors; using auto\n'
            (emit or _stderr_emit)('stderr', message)
    threads = (fallback if chosen == 'auto' else
               available if chosen == 'max' else int(chosen))
    raster = RasterOptions(values['raster_filter'], values['raster_bit_depth'])
    style = None
    if plotting:
        style, messages = config.resolve_style(stored)
        for message in messages:
            (emit or _stderr_emit)(
                'stderr', 'deeptoolsr: ignoring invalid option — {}\n'.format(
                    message))
    budget = values.get('ward_distance_budget_bytes')
    if budget is None:
        budget = default_ward_distance_budget_bytes()
    return RunOptions(threads, raster, path, budget, style)


def _stderr_emit(_stream, message):
    sys.stderr.write(message)


def resolve_run_options(args, *, plotting=False):
    path = config_path(getattr(args, 'config', 'auto'))
    return resolve_run_options_from_bytes(
        args, read_config_bytes(path), path, plotting=plotting)
