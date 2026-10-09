"""Run the corrupt-input harness (slow; only with ``--fuzz``)."""

import pytest

from tests.fuzz import corrupt_inputs

pytestmark = pytest.mark.fuzz

BAD = ("crash", "hang", "traceback", "bad-reply")


def test_corrupt_inputs_fail_cleanly(tmp_path):
    results = corrupt_inputs.run_all(tmp_path / "fuzz")
    problems = sorted({
        (r["class"], r["kind"], r["cmd"], r["sig"], r["variant"])
        for r in results if r["class"] in BAD or r["leftovers"]})
    assert not problems, "\n".join(map(str, problems[:50]))
