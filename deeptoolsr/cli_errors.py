"""Consistent, concise failures at command-line entry points."""

from functools import wraps
import os
import sys


def concise_cli_errors(tool, errors=Exception):
    """Turn ``errors`` (by default any implementation error) into one-line
    CLI failures.

    ``DTP_DEBUG=1`` retains the Python traceback for development and bug
    reports. Parser errors and explicit ``SystemExit`` messages pass through.
    """
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except KeyboardInterrupt:
                raise SystemExit("{} interrupted.".format(tool)) from None
            except SystemExit:
                raise
            except errors as error:
                if os.environ.get("DTP_DEBUG"):
                    raise
                raise SystemExit("{} failed: {}".format(tool, error)) from None
        return wrapped
    return decorate


def use_utf8_stdio():
    """Make machine-readable stdio UTF-8 whatever the locale (Windows cp1252)."""
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', newline='\n')
