"""Run the output-failure harness (slow; only with ``--fuzz``)."""

import re

import pytest

from tests.fuzz import output_failures

pytestmark = pytest.mark.fuzz

# Every failed write names its system error.
REASON = re.compile("File too large|No space left on device")


def test_failed_writes_fail_cleanly(tmp_path):
    results = output_failures.run_all(tmp_path / "outfail")
    problems = sorted({(r["class"], r["name"], r["limit"], r["sig"], *r["flags"])
                       for r in results if r["class"] != "ok-error" or r["flags"]})
    assert not problems, "\n".join(map(str, problems[:50]))
    unexplained = sorted({(r["name"], r["stderr"].strip()) for r in results
                          if r["class"] == "ok-error" and not REASON.search(r["stderr"])})
    assert not unexplained, "\n".join(map(str, unexplained[:50]))
