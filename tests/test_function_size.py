"""Keep the scene and figure builders small enough to inspect."""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    'deeptoolsr/plotting/text_layout.py',
    'deeptoolsr/plotting/grid.py',
    'deeptoolsr/plotting/scene.py',
    'deeptoolsr/plotting/heatmap.py',
    'deeptoolsr/plotting/matrix_figure.py',
    'deeptoolsr/plotting/profile.py',
    'deeptoolsr/plotting/series.py',
    'deeptoolsr/plotHeatmap.py',
    'deeptoolsr/plotMatrix.py',
    'deeptoolsr/plotProfile.py',
)


@pytest.mark.parametrize('module', MODULES)
def test_scene_functions_fit_review_limit(module):
    tree = ast.parse((ROOT / module).read_text())
    oversized = [
        (node.name, node.end_lineno - node.lineno + 1)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.end_lineno - node.lineno + 1 > 150
    ]
    assert oversized == []
