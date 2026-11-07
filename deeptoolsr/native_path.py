"""File paths handed to the native libraries (htslib, libBigWig, zlib, stdio).

Those libraries open files through narrow C APIs. On POSIX these take the
UTF-8 bytes pybind11 passes for a ``str``; on Windows they take the ANSI code
page, so every path crosses the boundary through :func:`native_path`.
"""

from functools import lru_cache
import os
import sys

# Windows MAX_PATH (260) less the terminating NUL.
MAX_PATH_CHARACTERS = 259


@lru_cache(maxsize=None)
def long_paths_enabled():
    """Whether this Windows process may open paths longer than MAX_PATH.

    Windows allows it when long paths are enabled for the system
    (LongPathsEnabled) and the running executable's manifest is long-path
    aware, as MSYS2's and python.org's python.exe are.
    """
    import ctypes
    try:
        check = ctypes.WinDLL('ntdll').RtlAreLongPathsEnabled
    except (AttributeError, OSError):
        return False
    check.restype = ctypes.c_ubyte
    return bool(check())


def check_path_length(path):
    """Refuse, on Windows with long paths off, a path longer than MAX_PATH."""
    if sys.platform != 'win32':
        return
    text = os.path.abspath(os.fsdecode(os.fspath(path)))
    if len(text) > MAX_PATH_CHARACTERS and not long_paths_enabled():
        raise ValueError(
            "path {!r} is longer than {} characters and Windows long paths "
            "are off; enable them (LongPathsEnabled) or use a shorter "
            "path".format(text, MAX_PATH_CHARACTERS))


def native_path(path):
    """Return ``path`` as the native file-opening APIs expect it."""
    path = os.fspath(path)
    if sys.platform != 'win32':
        return path
    check_path_length(path)
    text = os.fsdecode(path)
    try:
        return text.encode('mbcs', 'strict')
    except UnicodeEncodeError:
        raise ValueError(
            "path {!r} contains characters outside the Windows ANSI code page; "
            "rename it or move it to a folder with a plain name".format(text)) from None


def from_native_path(path):
    """Decode a path returned by native code (the inverse of native_path)."""
    if sys.platform == 'win32':
        return path.decode('mbcs', 'strict')
    return os.fsdecode(path)
