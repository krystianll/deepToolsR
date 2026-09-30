"""Pinned label-solver decisions from rendered, probe and direct cases."""

import json
from pathlib import Path

from tests.helpers.label_selection_capture import capture


SNAPSHOT = Path(__file__).parent / 'contract' / 'label_selections.json'


def test_label_selections(request, tmp_path):
    actual = capture(tmp_path)
    content = json.dumps(actual, indent=2, sort_keys=True) + '\n'
    if request.config.getoption('--update-label-selections'):
        SNAPSHOT.write_text(content)
    assert SNAPSHOT.read_text() == content
