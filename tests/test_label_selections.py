"""Pinned label-solver decisions from rendered, probe and direct cases."""
import json
from pathlib import Path

import pytest

from tests import test_plot_baselines
from tests.helpers.label_selection_capture import capture


SNAPSHOT = Path(__file__).parent / 'contract' / 'label_selections.json'
SHARDS = 4  # the baseline renders dominate; split them across xdist workers


def _dump(selections):
    return json.dumps(selections, indent=2, sort_keys=True) + '\n'


@pytest.mark.parametrize('shard', range(SHARDS))
def test_label_selections(request, tmp_path, shard):
    if request.config.getoption('--update-label-selections'):
        if shard == 0:
            SNAPSHOT.write_text(_dump(capture(tmp_path)))
        return
    expected = json.loads(SNAPSHOT.read_text())
    actual = capture(tmp_path, baselines=slice(shard, None, SHARDS),
                     solvers=shard == 0)
    for section, cases in actual.items():
        assert cases == {name: expected[section].get(name) for name in cases}, section
    if shard == 0:  # the snapshot holds exactly the captured cases
        names = {name for _, name, _ in test_plot_baselines.ALL_CASES}
        assert set(expected['baselines']) == names
        assert {key: set(value) for key, value in actual.items()
                if key != 'baselines'} == {
            key: set(value) for key, value in expected.items()
            if key != 'baselines'}
    assert SNAPSHOT.read_text() == _dump(expected)
