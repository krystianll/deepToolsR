"""Expected architecture rules, checked in fresh Python processes."""

import subprocess
import sys

import pytest


IMPORT_STATE = '''
import importlib
import pkgutil
import matplotlib
import numpy as np

def state():
    return {
        'backend': matplotlib.get_backend(),
        'rcParams': {key: repr(value) for key, value in matplotlib.rcParams.items()},
        'numpy_errors': np.geterr(),
        'colormaps': sorted(matplotlib.colormaps),
    }

before = state()
import deeptoolsr
for entry in pkgutil.walk_packages(deeptoolsr.__path__, 'deeptoolsr.'):
    importlib.import_module(entry.name)
after = state()
changed = [key for key in before if before[key] != after[key]]
assert not changed, 'import changed ' + ', '.join(changed)
'''

CONFIG_BOUNDARY = '''
import ast
from pathlib import Path
import deeptoolsr

allowed = {
    'plotHeatmap', 'plotProfile', 'computeMatrix', 'computeMatrixOperations',
    'bamCoverage', 'bigWigOperations', 'deeptoolsr_list_tools', 'serve',
    'describe',
    'options', 'config', 'cli_errors',
}
root = Path(deeptoolsr.__file__).parent
violations = []
for path in root.rglob('*.py'):
    module = path.stem
    if module in allowed:
        continue
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            is_config = isinstance(node.value, ast.Name) and node.value.id == 'config'
            is_environment = (isinstance(node.value, ast.Name) and
                              node.value.id == 'os' and
                              node.attr in ('environ', 'getenv'))
            if is_config or is_environment:
                violations.append(f'{path.relative_to(root)}:{node.lineno}')
        elif isinstance(node, ast.Name) and node.id in ('getenv', 'environ'):
            violations.append(f'{path.relative_to(root)}:{node.lineno}')
assert not violations, 'ambient configuration outside CLI: ' + ', '.join(violations[:12])
'''

NO_PYPLOT = '''
import ast
from pathlib import Path
import deeptoolsr

root = Path(deeptoolsr.__file__).parent
violations = []
for path in root.rglob('*.py'):
    for node in ast.walk(ast.parse(path.read_text())):
        imported = False
        if isinstance(node, ast.Import):
            imported = any(name.name == 'matplotlib.pyplot' for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported = (node.module == 'matplotlib.pyplot' or
                        (node.module == 'matplotlib' and
                         any(name.name == 'pyplot' for name in node.names)))
        if imported:
            violations.append(f'{path.relative_to(root)}:{node.lineno}')
assert not violations, 'pyplot imports: ' + ', '.join(violations[:12])
'''


GRID_BOUNDARY = '''
import ast
import sys
from pathlib import Path

path = Path('deeptoolsr/plotting/grid.py')
tree = ast.parse(path.read_text())
allowed = {'geometry', 'text_layout', 'series'}
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.level:
        assert node.module in allowed, node.module
    if isinstance(node, ast.Import):
        assert all(not item.name.startswith('matplotlib') for item in node.names)
sys.modules['matplotlib'] = None
import deeptoolsr.plotting.grid
'''


@pytest.mark.parametrize('code', [
    IMPORT_STATE,
    CONFIG_BOUNDARY,
    NO_PYPLOT,
    GRID_BOUNDARY,
], ids=['import-state', 'config-boundary', 'no-pyplot',
        'grid-boundary'])
def test_architecture(code):
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
