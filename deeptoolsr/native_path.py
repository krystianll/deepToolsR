"""File paths handed to the native libraries (htslib, libBigWig, zlib, stdio).

Those libraries open files through narrow C APIs. On POSIX these take the
UTF-8 bytes pybind11 passes for a ``str``; on Windows they take the ANSI code
page, so every path crosses the boundary through :func:`native_path`.
"""

import os
import sys


def native_path(path):
    """Return ``path`` as the native file-opening APIs expect it."""
    path = os.fspath(path)
    if sys.platform != 'win32':
        return path
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
