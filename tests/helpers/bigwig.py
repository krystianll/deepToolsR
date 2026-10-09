"""pyBigWig for tests, skipping only the tests that actually touch it.

pyBigWig has no Windows build, so a module-level import would stop whole test
files (including their BAM/htslib tests) from being collected there.
"""

import pytest


class _PyBigWig:
    def __getattr__(self, name):
        if name.startswith('_'):  # pytest probes module globals on collection
            raise AttributeError(name)
        return getattr(pytest.importorskip("pyBigWig"), name)


pyBigWig = _PyBigWig()
