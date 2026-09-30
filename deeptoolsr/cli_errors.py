"""Consistent, concise failures at command-line entry points."""

from functools import wraps
import os


def concise_cli_errors(tool):
    """Turn unexpected implementation errors into one-line CLI failures.

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
            except Exception as error:
                if os.environ.get("DTP_DEBUG"):
                    raise
                raise SystemExit("{} failed: {}".format(tool, error)) from None
        return wrapped
    return decorate
