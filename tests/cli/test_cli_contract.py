"""Stable parser structure and accepted/rejected CLI invocations."""

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from deeptoolsr import plotHeatmap, plotProfile


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(__file__).parent.parent / 'contract' / 'cli_options.json'
MATRIX = Path(__file__).parent.parent / 'test_data' / 'computeMatrixOperations.mat.gz'
TOOLS = {
    'computeMatrixR': 'computeMatrix',
    'computeMatrixOperationsR': 'computeMatrixOperations',
    'plotHeatmapR': 'plotHeatmap',
    'plotMatrixR': 'plotMatrix',
    'plotProfileR': 'plotProfile',
    'bamCoverageR': 'bamCoverage',
    'bigWigOperationsR': 'bigWigOperations',
    'deeptoolsr': 'deeptoolsr_list_tools',
}


def _stable_repr(value):
    if callable(value):
        return '<{}.{}>'.format(value.__module__, value.__qualname__)
    rendered = repr(value)
    assert not re.search(r'0x[0-9a-fA-F]+', rendered), rendered
    assert str(ROOT) not in rendered, rendered
    return rendered


def _type_name(value):
    if value is None:
        return None
    if isinstance(value, argparse.FileType):
        return 'FileType'
    return getattr(value, '__name__', type(value).__name__)


def _parser_record(parser):
    groups = {id(action): 'group_{}'.format(index)
              for index, group in enumerate(parser._mutually_exclusive_groups)
              for action in group._group_actions}
    actions = []
    subcommands = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            choices = list(action.choices)
            subcommands = {name: _parser_record(action.choices[name])
                           for name in choices}
        else:
            choices = (list(action.choices) if action.choices is not None
                       else None)
        actions.append({
            'flags': list(action.option_strings),
            'dest': action.dest,
            'default': _stable_repr(action.default),
            'choices': choices,
            'nargs': action.nargs,
            'type': _type_name(action.type),
            'action': type(action).__name__,
            'required': bool(action.required),
            'mutex_group': groups.get(id(action)),
        })
    return {'actions': actions, 'subcommands': subcommands}


def _current_contract():
    return {name: _parser_record(
        importlib.import_module('deeptoolsr.' + module).parse_arguments())
        for name, module in TOOLS.items()}


def test_cli_parser_snapshot(request):
    current = _current_contract()
    if request.config.getoption('--update-contract'):
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2) + '\n')
    assert current == json.loads(SNAPSHOT.read_text())


@pytest.mark.parametrize('tool', TOOLS)
def test_version_line(tool):
    result = subprocess.run([str(Path(sys.executable).parent / tool),
                             '--version'], capture_output=True, text=True,
                            cwd=ROOT, timeout=30)
    assert result.returncode == 0, result.stderr
    assert re.fullmatch(re.escape(tool) + r' [0-9][\w.+-]*\n', result.stdout)
    if tool == 'bamCoverageR':
        seqbench = (r'(?:bamCoverageR|deepToolsR|bamCoverage)[ \t]+'
                    r'(?:version[ \t]+)?v?([0-9][\w.+-]*)')
        assert re.fullmatch(seqbench + r'\n', result.stdout)


# Each row passes through the real plot parser and process_args validator.
# Accept rows return before matrix I/O or rendering; reject rows raise there.
BEHAVIOUR = [
    ('heatmap', [], True),
    ('profile', [], True),
    ('profile', ['--plotType', 'bootstrap', '-p', 'max'], True),
    ('heatmap', ['--kmeans', '2', '--hclust', '2'], False),
    ('profile', ['--kmeans', '2', '--hclust', '2'], False),
    ('heatmap', ['--kmeans', '2', '--quantiles', '2'], False),
    ('profile', ['--hclust', '2', '--quantiles', '2'], False),
    ('heatmap', ['--silhouette'], False),
    ('heatmap', ['--heatmapHeight', '2.9'], False),
    ('heatmap', ['--heatmapHeight', '101'], False),
    ('heatmap', ['--heatmapWidth', '0.9'], False),
    ('heatmap', ['--heatmapWidth', '101'], False),
    ('heatmap', ['--aspectRatio', '0'], False),
    ('heatmap', ['--aspectRatioSummaryPlot', '0'], False),
    ('heatmap', ['--trimPercSummaryPlot', '0.5'], False),
    ('heatmap', ['--plotTypeSummaryPlot', 'se',
                 '--averageTypeSummaryPlot', 'median'], False),
    ('heatmap', ['--quantiles', '2', '--sortRegions', 'keep'], False),
    ('heatmap', ['--yMinSummaryPlot', '3', '--yMaxSummaryPlot', '2'], False),
    ('heatmap', ['--missingDataColor', 'not-a-color'], False),
    ('profile', ['--trim_perc', '0.5'], False),
    ('profile', ['--ci_level', '1'], False),
    ('profile', ['--bootstrapReplicates', '0'], False),
    ('profile', ['--plotType', 'se', '--averageType', 'median'], False),
    ('profile', ['--quantiles', '2', '--sortRegions', 'keep'], False),
    ('profile', ['--aspectRatio', '0'], False),
    ('profile', ['--profileHeight', '0.4'], False),
    ('profile', ['--profileHeight', '101'], False),
    ('profile', ['--profileWidth', '0.9'], False),
    ('profile', ['--profileWidth', '101'], False),
    ('profile', ['--yMin', '3', '--yMax', '2'], False),
    ('profile', ['--plotType', 'heatmap', '--gridRows', '2'], True),
    ('profile', ['--plotType', 'overlapped' + '_lines'], False),
]


@pytest.mark.parametrize('tool,extra,accepted', BEHAVIOUR)
def test_plot_parse_and_validation(tmp_path, tool, extra, accepted):
    process = plotHeatmap.process_args if tool == 'heatmap' else plotProfile.process_args
    output = tmp_path / 'out.png'
    argv = ['-m', str(MATRIX), '-o', str(output), *extra]
    if accepted:
        process(argv)
    else:
        with pytest.raises(SystemExit):
            process(argv)
    assert not output.exists()


@pytest.mark.parametrize('tool', ['heatmap', 'profile'])
def test_required_output_is_rejected(tool):
    process = plotHeatmap.process_args if tool == 'heatmap' else plotProfile.process_args
    with pytest.raises(SystemExit):
        process(['-m', str(MATRIX)])


WARNINGS = [
    ('heatmap', ['--pseudocountSummaryPlot', '1'], 'unused unless'),
    ('heatmap', ['--trimPercSummaryPlot', '0.1'], 'unused unless'),
    ('heatmap', ['--colorNumber', '16'], 'unused without'),
    ('heatmap', ['--whatToShow', 'heatmap and colorbar',
                 '--plotTypeSummaryPlot', 'fill'], 'does not include'),
    ('profile', ['--bootstrapReplicates', '12'], 'unused unless'),
    ('profile', ['--ci_level', '0.9'], 'unused unless'),
    ('profile', ['--pseudocount', '1'], 'unused unless'),
    ('profile', ['--trim_perc', '0.1'], 'unused unless'),
]


@pytest.mark.parametrize('tool,extra,warning', WARNINGS)
def test_plot_compatibility_warnings(tmp_path, capsys, tool, extra, warning):
    process = plotHeatmap.process_args if tool == 'heatmap' else plotProfile.process_args
    process(['-m', str(MATRIX), '-o', str(tmp_path / 'out.png'), *extra])
    assert warning in capsys.readouterr().err


def test_bad_colormap_is_rejected_before_rendering(tmp_path, monkeypatch):
    # Colormap names are resolved in the heatmap scale setup, before axes
    # exist. Figure.savefig must remain unreachable for this invalid input.
    from matplotlib.figure import Figure

    def forbidden_save(*_args, **_kwargs):
        raise AssertionError('invalid colormap reached rendering')

    monkeypatch.setattr(Figure, 'savefig', forbidden_save)
    output = tmp_path / 'out.png'
    with pytest.raises((SystemExit, ValueError)):
        plotHeatmap.main(['-m', str(MATRIX), '-o', str(output),
                          '--colorMap', 'not_a_colormap'])
    assert not output.exists()


@pytest.mark.parametrize('name', ['rocket', 'viridis'])
def test_local_and_matplotlib_colormaps_render(name, tmp_path):
    matrix = Path(__file__).parent.parent / 'test_heatmapper' / 'master.mat.gz'
    output = tmp_path / (name + '.png')
    plotHeatmap.main(['-m', str(matrix), '-o', str(output),
                      '--colorMap', name, '--zMin', '0', '--zMax', '5',
                      '-p', '1'])
    assert output.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
