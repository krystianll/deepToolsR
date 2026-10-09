"""Shared test fixtures for deepToolsR."""

import importlib
import importlib.metadata
import inspect
import types

import pytest

REFERENCE_SKIP = "requires original deepTools 3.5.6"


def pytest_addoption(parser):
    parser.addoption(
        "--require-reference",
        action="store_true",
        help="Fail unless original deepTools 3.5.6 and all native extensions load",
    )
    parser.addoption(
        "--reference-only",
        action="store_true",
        help="Run only the tests that compare against original deepTools 3.5.6",
    )
    parser.addoption(
        "--fuzz",
        action="store_true",
        help="Run the slow corrupt-input harness in tests/fuzz",
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


def _reaches_reference(function, seen):
    """Whether a test function can call the deepTools 3.5.6 runner.

    Follows the names its code uses through functions defined in the test
    suite, so helpers that call ``tests.helpers.parity.original`` count.
    """
    from tests.helpers import parity

    if function in seen:
        return False
    seen.add(function)
    code = getattr(function, "__code__", None)
    if code is None:
        return False
    names = set()
    stack = [code]
    while stack:
        current = stack.pop()
        names.update(current.co_names)
        stack.extend(c for c in current.co_consts if isinstance(c, types.CodeType))
    scope = dict(getattr(function, "__globals__", {}))
    for name in names:
        value = scope.get(name)
        if value is parity.original or (value is parity and "original" in names):
            return True
        if (inspect.isfunction(value) and value.__module__.startswith("tests")
                and _reaches_reference(value, seen)):
            return True
    return False


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "reference: compares against original deepTools 3.5.6")
    config.addinivalue_line(
        "markers", "fuzz: slow corrupt-input harness, run only with --fuzz")


def pytest_collection_modifyitems(config, items):
    skip_fuzz = pytest.mark.skip(reason="corrupt-input harness; run with --fuzz")
    for item in items:
        if item.get_closest_marker("fuzz") and not config.getoption("--fuzz"):
            item.add_marker(skip_fuzz)
        module = item.module.__name__.rsplit(".", 1)[-1]
        if (module == "test_upstream_compatibility"
                or _reaches_reference(getattr(item, "function", None), set())):
            item.add_marker(pytest.mark.reference)
    if config.getoption("--reference-only"):
        kept = [item for item in items if item.get_closest_marker("reference")]
        config.hook.pytest_deselected(
            items=[item for item in items if item not in kept])
        items[:] = kept


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    # Keep the marker honest: an unmarked test that needs deepTools 3.5.6
    # would be dropped by --reference-only.
    if (report.skipped and REFERENCE_SKIP in str(report.longrepr)
            and not item.get_closest_marker("reference")):
        report.outcome = "failed"
        report.longrepr = (f"{item.nodeid} skips with '{REFERENCE_SKIP}' "
                           "but is not marked reference; extend "
                           "tests/conftest.py::_reaches_reference")


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
            "_regions",
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
