"""Static boundaries for the frozen matrix model and its owned exceptions."""

import ast
from pathlib import Path

import deeptoolsr


ROOT = Path(deeptoolsr.__file__).parent
# These operations own batch buffers. No plotting or loaded Matrix path may
# materialise a fancy row selection from the numeric matrix.
OWNED_COPIES = {
    ('compute.py', 'remove_empty_rows'),
    ('compute.py', 'ComputeMatrixBuilder.computeMatrix'),
    ('computeMatrixOperations.py', 'mergeSameNamedGroups'),
    ('cluster.py', 'cluster'),  # transient, zero-imputed native workspace
}
SCALAR_ROWS = {'row', 'index', 'idx', 'position', 'source_row', 'output_row'}
# These are 1-D summary vectors, not matrix rows.
SUMMARY_STACKS = {('plotting/matrix_figure.py', '_series_panel_values')}


def _modules():
    for path in sorted(ROOT.rglob('*.py')):
        yield path.relative_to(ROOT).as_posix(), ast.parse(path.read_text())


def test_no_retired_matrix_imports_or_references():
    violations = []
    for module, tree in _modules():
        for node in ast.walk(tree):
            old_import = (
                isinstance(node, ast.Import) and any(
                    alias.name in ('deeptoolsr.heatmapper', 'heatmapper')
                    for alias in node.names)
                or isinstance(node, ast.ImportFrom) and (
                    node.module in ('deeptoolsr.heatmapper', 'heatmapper')
                    or node.module in ('deeptoolsr', None) and any(
                        alias.name == 'heatmapper' for alias in node.names)))
            old_reference = (
                isinstance(node, ast.Name) and node.id in
                ('_matrix', 'read_matrix_file')
                or isinstance(node, ast.Attribute) and (
                    node.attr == 'read_matrix_file' or
                    node.attr == 'matrix' and isinstance(node.value, ast.Name)
                    and node.value.id == 'hm'))
            if old_import or old_reference:
                violations.append(f'{module}:{node.lineno}')
    assert not violations, violations


def _fancy_row_slice(node):
    row = node.elts[0] if isinstance(node, ast.Tuple) else node
    if isinstance(row, (ast.Slice, ast.Constant, ast.BinOp, ast.Subscript)):
        return False
    if isinstance(row, ast.Name):
        return row.id not in SCALAR_ROWS
    if isinstance(row, ast.Call) and isinstance(row.func, ast.Name):
        return row.func.id not in ('int',)
    return True


def test_no_unowned_matrix_value_copies():
    violations = []
    for module, tree in _modules():
        class Check(ast.NodeVisitor):
            scope = ()

            def visit_ClassDef(self, node):
                previous = self.scope
                self.scope += (node.name,)
                self.generic_visit(node)
                self.scope = previous

            def visit_FunctionDef(self, node):
                previous = self.scope
                self.scope += (node.name,)
                self.generic_visit(node)
                self.scope = previous

            def visit_Subscript(self, node):
                matrix_values = (isinstance(node.value, ast.Attribute) and
                                 node.value.attr == 'values')
                workspace = (isinstance(node.value, ast.Name) and
                             node.value.id == 'workspace')
                if ((matrix_values or workspace) and
                        _fancy_row_slice(node.slice)):
                    self.record(node)
                self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                    summary_stack = (
                        name == 'vstack' and
                        (module, '.'.join(self.scope)) in SUMMARY_STACKS and
                        len(node.args) == 1 and
                        isinstance(node.args[0], ast.Name) and
                        node.args[0].id == 'series')
                    if name in ('permute_rows_inplace', 'take') or (
                            name == 'vstack' and not summary_stack):
                        self.record(node)
                self.generic_visit(node)

            def record(self, node):
                owner = (module, '.'.join(self.scope))
                if owner not in OWNED_COPIES:
                    violations.append(f'{module}:{node.lineno} ({owner[1]})')

        Check().visit(tree)
    assert not violations, violations
