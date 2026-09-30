"""Shared test fixtures for deepToolsR."""

import importlib
import importlib.metadata

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require-reference",
        action="store_true",
        help="Fail unless original deepTools 3.5.6 and all native extensions load",
    )
    parser.addoption(
        "--rebaseline",
        action="store_true",
        help="Rewrite the rendered-output baselines in test_baselines/",
    )
    parser.addoption(
        "--require-pixels",
        action="store_true",
        help="Fail if the current renderer has no recorded PNG digests",
    )
    parser.addoption(
        "--update-contract",
        action="store_true",
        help="Rewrite the CLI parser snapshot",
    )
    parser.addoption(
        "--update-label-selections",
        action="store_true",
        help="Rewrite the shared-gap and below-common selection snapshot",
    )


def pytest_sessionstart(session):
    if not session.config.getoption("--require-reference"):
        return
    try:
        importlib.import_module("deeptools")
        if importlib.metadata.version("deepTools") != "3.5.6":
            raise RuntimeError("the reference version must be deepTools 3.5.6")
        for module in (
            "_compute_matrix_io",
            "_compute_matrix_native",
            "_compute_matrix_stream",
            "_statistics",
            "_raster",
            "_transform",
            "_coverage",
            "_bigwig",
        ):
            importlib.import_module("deeptoolsr." + module)
    except (ImportError, RuntimeError) as error:
        raise pytest.UsageError(
            f"--require-reference prerequisites are missing: {error}"
        ) from error


@pytest.fixture(autouse=True)
def _isolate_deeptoolsr_config(tmp_path_factory, monkeypatch):
    """Point every test at an empty, throwaway options directory.

    The plotting drivers now resolve their persistent style through
    ``config.load_style()``, which reads ``options.txt``. Without this fixture a
    test run would pick up the developer's real ``~/.config/deeptoolsr``
    file and become non-deterministic. Tests that need their own configuration
    simply call ``monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', ...)`` again;
    the later setenv wins.
    """
    config_dir = tmp_path_factory.mktemp("deeptoolsr_config")
    monkeypatch.setenv("DEEPTOOLSR_CONFIG_DIR", str(config_dir))
    yield
