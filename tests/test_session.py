"""Warm plot requests preserve cold output bytes and cache dependencies."""

import ast
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import tracemalloc
import weakref

import numpy as np
import pytest

from deeptoolsr import options, parserCommon, plotMatrix, prepare, stats
from deeptoolsr.cache import ByteBudgetCache
from deeptoolsr.matrix import Matrix, OwnedMatrix
from deeptoolsr.session import (
    Cancelled, OPTION_STAGES, PlotSession, project_color_spec,
    project_scan_spec, project_scene_digest)


MATRIX = Path(__file__).parent / 'test_heatmapper' / 'master_multi.mat.gz'
PARTIAL = Path(__file__).parent / 'test_data' / 'prepare_expected' / 'input.gz'


def test_option_ownership_covers_all_three_parsers():
    destinations = set()
    for tool in ('plotMatrixR', 'plotHeatmapR', 'plotProfileR'):
        parser = parserCommon.plot_parser(tool)
        destinations.update(action.dest for action in parser._actions)
    destinations.difference_update(('help', 'version'))
    assert set(OPTION_STAGES) == destinations
    assert set(OPTION_STAGES.values()) <= {
        'source', 'order', 'statistics', 'limits', 'color', 'labels',
        'scene', 'output', 'execution', 'ignored'}


def test_session_pipeline_has_no_direct_output_calls():
    """Only the CLI boundary owns stdout/stderr presentation."""
    from deeptoolsr import session as session_module

    for function in (session_module.PlotSession.run,
                     plotMatrix.run_session_request,
                     plotMatrix._prepare_data,
                     plotMatrix._stage_data,
                     plotMatrix._prepare_numeric,
                     prepare._common_spec):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        assert not any(
            isinstance(node, ast.Call) and
            isinstance(node.func, ast.Name) and node.func.id == 'print'
            for node in ast.walk(tree))
        assert not any(
            isinstance(node, ast.Attribute) and node.attr in ('stdout', 'stderr')
            for node in ast.walk(tree))


@pytest.mark.parametrize('tool', ('plotMatrixR', 'plotHeatmapR',
                                  'plotProfileR'))
@pytest.mark.parametrize('flag', ('--help', '--version', 'missing_output'))
def test_cli_mode_uses_original_parser_actions(tool, flag, capsys):
    raw = ['-m', str(MATRIX)] if flag == 'missing_output' else [flag]
    with pytest.raises(SystemExit) as expected:
        parserCommon.plot_parser(tool).parse_args(raw)
    parser_output = capsys.readouterr()
    with pytest.raises(SystemExit) as actual:
        plotMatrix.matrix_main(tool)(raw)
    assert actual.value.code == expected.value.code
    assert capsys.readouterr() == parser_output


def test_cli_parse_warning_uses_emit_callback(tmp_path):
    command, _paths = _command(
        tmp_path, 'warning', ('--profile', '--pseudocount', '1'))
    messages = []
    with PlotSession(cache_bytes=0) as session:
        result = session.run(command, mode='cli',
                             emit=lambda stream, text: messages.append(
                                 (stream, text)))
    assert messages and messages[0][0] == 'stderr'
    assert '--pseudocount is unused' in messages[0][1]
    assert result.warnings[0] == messages[0][1]


def _attribute_reads(function, argument):
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    direct = {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and
        isinstance(node.value, ast.Name) and node.value.id == argument}
    dynamic = {
        node.args[1].value for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == 'getattr' and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == argument
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)}
    return direct | dynamic


def test_numeric_projection_reads_are_owned_or_derived():
    order = {name for name, stage in OPTION_STAGES.items()
             if stage == 'order'}
    order_derived = {
        'show_heatmap', 'outFileSortedRegions', 'outFileNameMatrix',
        'regionsLabel', 'run_options'}
    assert _attribute_reads(prepare._common_spec, 'args') <= (
        order | order_derived)

    statistics = {name for name, stage in OPTION_STAGES.items()
                  if stage == 'statistics'}
    statistic_fields = {
        'averageType': 'average_type', 'plotType': 'plot_type',
        'pseudocount': 'pseudocount', 'trim_perc': 'trim_perc',
        'ci_level': 'ci_level',
        'bootstrapReplicates': 'bootstrap_replicates'}
    assert _attribute_reads(stats.statistics_spec, 'spec') <= {
        statistic_fields[name] for name in statistics}

    # Scan limits also depend on which panel is rendered and its band kind.
    scan_derived = {'show_heatmap', 'show_profile', 'plot_type',
                    'y_min', 'y_max'}
    assert _attribute_reads(project_scan_spec, 'figure_spec') <= (
        scan_derived | {'z_min', 'z_max'})

    # Colour alpha is a scene-owned presentation value passed into the LUT.
    color_fields = {'color_map', 'color_list', 'color_number',
                    'missing_data_color', 'z_mid',
                    'interpolation_method', 'alpha', 'show_heatmap'}
    assert _attribute_reads(project_color_spec, 'figure_spec') <= color_fields

    labels = {name for name, stage in OPTION_STAGES.items()
              if stage == 'labels'}
    assert _attribute_reads(project_scene_digest, 'args') <= labels


def _command(root, stem, flags=(), *, source=MATRIX, fmt='png', files=False,
             table=False):
    root.mkdir(parents=True, exist_ok=True)
    paths = {'plot': root / f'{stem}.{fmt}'}
    argv = ['plotMatrixR', '-m', str(source), '-p', '1', '--dpi', '70',
            '-o', str(paths['plot']), *flags]
    if files or table:
        paths['table'] = root / f'{stem}.tsv'
        argv += ['--outFileNameData', str(paths['table'])]
    if files:
        paths.update(matrix=root / f'{stem}.mat.gz',
                     bed=root / f'{stem}.bed')
        argv += ['--outFileNameMatrix', str(paths['matrix']),
                 '--outFileSortedRegions', str(paths['bed'])]
    return argv, paths


def _bytes(paths):
    return {name: path.read_bytes() for name, path in paths.items()
            if path.exists()}


def _warm_cold(session, tmp_path, before, after, *, source=MATRIX,
               fmt='png', files=False, before_files=False):
    warm = tmp_path / 'warm'
    cold = tmp_path / 'cold'
    command, _ = _command(warm, 'first', before, source=source, fmt=fmt,
                          files=before_files)
    session.run(command)
    command, warm_paths = _command(
        warm, 'last', after, source=source, fmt=fmt, files=files)
    result = session.run(command)
    command, cold_paths = _command(
        cold, 'last', after, source=source, fmt=fmt, files=files)
    plotMatrix.main(command[1:])
    assert _bytes(warm_paths) == _bytes(cold_paths)
    return result


@pytest.mark.parametrize('name,before,after,hits', [
    ('line_to_fill', ('--profile', '--plotType', 'lines'),
     ('--profile', '--plotType', 'fill'), ('layout', 'statistics')),
    ('title', ('--profile',),
     ('--profile', '--plotTitle', 'Changed'), ('layout', 'statistics')),
    ('counts', ('--profile',),
     ('--profile', '--showRegionCounts'), ('layout', 'statistics')),
    ('relabel', ('--profile',),
     ('--profile', '--regionsLabel', 'A', 'A',
      '--sameGroupLabels', 'together'), ('layout', 'statistics')),
    ('resize', ('--profile',),
     ('--profile', '--cellWidth', '9'), ('layout', 'statistics')),
    ('dpi', ('--profile',),
     ('--profile', '--dpi', '100'), ('layout', 'statistics')),
    ('aspect', ('--heatmap',),
     ('--heatmap', '--heatmapAspectRatio', '3'), ('layout', 'scan')),
    ('color', ('--profile',),
     ('--profile', '--colorsPerSample', '#ff0000', '#0000ff'),
     ('layout', 'statistics')),
    ('heatmap_lut', ('--heatmap',),
     ('--heatmap', '--colorMap', 'plasma'), ('layout', 'scan')),
    ('bad_color', ('--heatmap',),
     ('--heatmap', '--missingDataColor', 'red'), ('layout', 'scan')),
    ('z_mid', ('--heatmap',),
     ('--heatmap', '--zMid', '1.2'), ('layout', 'scan')),
    ('z_explicit', ('--heatmap',),
     ('--heatmap', '--zMin', '0'), ('layout',)),
    ('z_max', ('--heatmap',),
     ('--heatmap', '--zMax', '2'), ('layout',)),
    ('per_group', ('--heatmap', '--zMin', '-1', 'auto',
                   '--zMax', 'auto', '4'),
     ('--heatmap', '--perGroup', '--zMin', '-1', 'auto',
      '--zMax', 'auto', '4'), ('layout', 'scan')),
    ('band', ('--profile', '--plotType', 'lines'),
     ('--profile', '--plotType', 'ci'), ('layout',)),
    ('average', ('--profile', '--averageType', 'mean'),
     ('--profile', '--averageType', 'median'), ('layout',)),
    ('ci_level', ('--profile', '--plotType', 'ci', '--ci_level', '0.5'),
     ('--profile', '--plotType', 'ci', '--ci_level', '0.99'), ('layout',)),
    ('legend', ('--profile',),
     ('--profile', '--legendLocation', 'right'),
     ('layout', 'statistics')),
    ('merge', ('--profile', '--sameGroupLabels', 'merge'),
     ('--profile', '--sameGroupLabels', 'merge', '--regionsLabel', 'A', 'A'),
     ('matrix',)),
    ('first_profile', ('--heatmap', '--sortRegions', 'keep'),
     ('--heatmap', '--profile', '--sortRegions', 'keep'),
     ('layout', 'scan', 'bitmaps')),
    ('enable_heatmap_keep', ('--profile', '--sortRegions', 'keep'),
     ('--profile', '--heatmap', '--sortRegions', 'keep'),
     ('layout', 'statistics')),
    ('enable_heatmap_sorted', ('--profile', '--sortRegions', 'descend'),
     ('--profile', '--heatmap', '--sortRegions', 'descend'),
     ('layout', 'statistics')),
    ('enable_heatmap_unconsumed', ('--profile', '--sortRegions', 'descend'),
     ('--profile', '--heatmap', '--sortRegions', 'descend'), ('matrix',)),
    ('arrangement', ('--heatmap', '--profile'),
     ('--heatmap', '--profile', '--sampleSetGroupArrangement', 'by_row'),
     ('layout', 'scan')),
    ('grid_columns', ('--heatmap', '--profile'),
     ('--heatmap', '--profile', '--gridColumns', '2'),
     ('layout', 'scan')),
    ('filter', ('--heatmap',),
     ('--heatmap', '--filterNans', 'any_bin'), ('matrix',)),
    ('quantiles', ('--heatmap', '--sortRegions', 'descend'),
     ('--heatmap', '--sortRegions', 'descend',
      '--quantileSortedRegions', '2'), ('matrix',)),
])
def test_warm_transition_matches_cold(name, before, after, hits, tmp_path):
    recomputes = {
        'band': ('statistics',), 'average': ('statistics',),
        'ci_level': ('statistics',),
        'heatmap_lut': ('bitmaps',), 'bad_color': ('bitmaps',),
        'z_mid': ('bitmaps',), 'per_group': ('bitmaps',),
        'filter': ('layout',), 'quantiles': ('layout',),
        'merge': ('layout',),
        'enable_heatmap_unconsumed': ('layout', 'statistics'),
    }
    with PlotSession(cache_bytes=64 << 20) as session:
        result = _warm_cold(
            session, tmp_path, before, after,
            before_files=name == 'enable_heatmap_sorted')
    assert set(hits) <= set(result.reused), name
    assert set(recomputes.get(name, ())) <= set(result.recomputed), name


@pytest.mark.parametrize('fmt', ('png', 'pdf', 'svg'))
def test_output_path_and_format_match_cold(fmt, tmp_path):
    with PlotSession(cache_bytes=64 << 20) as session:
        result = _warm_cold(
            session, tmp_path, ('--profile', '--sortRegions', 'keep'),
            ('--profile', '--sortRegions', 'keep'),
            fmt=fmt, files=True)
    assert {'layout', 'statistics'} <= set(result.reused)


@pytest.mark.parametrize('fmt', ('pdf', 'svg'))
def test_heatmap_export_formats_match_cold(fmt, tmp_path):
    with PlotSession(cache_bytes=64 << 20) as session:
        result = _warm_cold(
            session, tmp_path, ('--heatmap', '--sortRegions', 'keep'),
            ('--heatmap', '--sortRegions', 'keep'), fmt=fmt)
    assert {'layout', 'scan', 'bitmaps'} <= set(result.reused)


def test_inert_heatmap_options(tmp_path):
    with PlotSession(cache_bytes=64 << 20) as session:
        first, _ = _command(tmp_path, 'plain', ('--profile',))
        left = session.run(first)
        changed, paths = _command(
            tmp_path, 'inert',
            ('--profile', '--zMin', '0', '--colorMap', 'viridis'))
        right = session.run(changed)
    assert (tmp_path / 'plain.png').read_bytes() == _bytes(paths)['plot']
    assert left.scene_digest == right.scene_digest
    assert 'scan' not in right.recomputed
    assert {'layout', 'statistics'} <= set(right.reused)


def test_equivalent_size_spellings_have_one_scene_digest(tmp_path):
    with PlotSession(cache_bytes=64 << 20) as session:
        first = session.run(_command(
            tmp_path, 'height', ('--profile', '--cellWidth', '6',
                                 '--profileHeight', '3'))[0])
        second = session.run(_command(
            tmp_path, 'ratio', ('--profile', '--cellWidth', '6',
                                '--profileAspectRatio', '2'))[0])
    assert first.scene_digest == second.scene_digest
    assert (tmp_path / 'height.png').read_bytes() == (
        tmp_path / 'ratio.png').read_bytes()


def test_heatmap_only_both_hide_show_round_trip(tmp_path):
    flags = ('--heatmap', '--sortRegions', 'keep')
    with PlotSession(cache_bytes=64 << 20) as session:
        session.run(_command(tmp_path, 'heatmap', flags)[0])
        session.run(_command(tmp_path, 'both1', (*flags, '--profile'))[0])
        hidden = session.run(_command(tmp_path, 'hidden', flags)[0])
        final = session.run(_command(tmp_path, 'both2',
                                     (*flags, '--profile'))[0])
    fresh, path = _command(tmp_path / 'fresh', 'both2',
                           (*flags, '--profile'))
    plotMatrix.main(fresh[1:])
    assert (tmp_path / 'both2.png').read_bytes() == _bytes(path)['plot']
    assert {'layout', 'statistics', 'scan', 'bitmaps'} <= set(final.reused)
    assert {'layout', 'scan', 'bitmaps'} <= set(hidden.reused)


def test_statistics_edited_while_profile_hidden(tmp_path):
    heatmap = ('--heatmap', '--sortRegions', 'keep')
    both = (*heatmap, '--profile')
    edited = (*both, '--plotType', 'ci', '--ci_level', '0.5')
    with PlotSession(cache_bytes=64 << 20) as session:
        session.run(_command(tmp_path, 'both', both)[0])
        session.run(_command(tmp_path, 'hidden', heatmap)[0])
        result = session.run(_command(tmp_path, 'edited', edited)[0])
    fresh, paths = _command(tmp_path / 'fresh', 'edited', edited)
    plotMatrix.main(fresh[1:])
    assert (tmp_path / 'edited.png').read_bytes() == paths['plot'].read_bytes()
    assert {'layout', 'scan', 'bitmaps'} <= set(result.reused)
    assert 'statistics' in result.recomputed


def test_ci_level_changes_heatmap_summary(tmp_path):
    base = ('--profile', '--heatmap', '--plotType', 'ci',
            '--averageType', 'mean', '--sortRegions', 'keep')
    with PlotSession(cache_bytes=64 << 20) as session:
        session.run(_command(tmp_path, 'low',
                             (*base, '--ci_level', '0.5'))[0])
        high = session.run(_command(tmp_path, 'high',
                                    (*base, '--ci_level', '0.99'))[0])
    fresh, paths = _command(tmp_path / 'fresh', 'high',
                            (*base, '--ci_level', '0.99'))
    plotMatrix.main(fresh[1:])
    assert (tmp_path / 'low.png').read_bytes() != (
        tmp_path / 'high.png').read_bytes()
    assert (tmp_path / 'high.png').read_bytes() == paths['plot'].read_bytes()
    assert 'statistics' in high.recomputed
    assert {'layout', 'scan', 'bitmaps'} <= set(high.reused)


def test_identical_heatmap_request_reuses_all_numeric_work(tmp_path):
    flags = ('--heatmap', '--sortRegions', 'keep')
    with PlotSession(cache_bytes=64 << 20) as session:
        first = session.run(_command(tmp_path, 'first', flags)[0])
        result = session.run(_command(tmp_path, 'second', flags)[0])
    assert first.reused == ()
    assert 'text' in first.recomputed
    assert {'matrix', 'layout', 'scan', 'bitmaps', 'text'} <= set(result.reused)
    assert (tmp_path / 'first.png').read_bytes() == (
        tmp_path / 'second.png').read_bytes()


def test_wrapper_and_matrix_spelling_share_scene_and_caches(tmp_path):
    flags = ('--sortRegions', 'keep', '--yAxisLimits', 'per_sample_set')
    with PlotSession(cache_bytes=64 << 20) as session:
        wrapper, _ = _command(tmp_path, 'wrapper', flags)
        wrapper[0] = 'plotProfileR'
        first = session.run(wrapper)
        matrix, _ = _command(tmp_path, 'matrix', (*flags, '--profile'))
        second = session.run(matrix)
    assert (tmp_path / 'wrapper.png').read_bytes() == (
        tmp_path / 'matrix.png').read_bytes()
    assert first.scene_digest == second.scene_digest
    assert {'matrix', 'layout', 'statistics', 'text'} <= set(second.reused)


def test_series_heatmap_scan_round_trip(tmp_path):
    auto = ('--profile', '--plotType', 'heatmap')
    explicit = (*auto, '--yMin', '0', '--yMax', '2')
    with PlotSession(cache_bytes=64 << 20) as session:
        session.run(_command(tmp_path, 'auto1', auto)[0])
        middle = session.run(_command(tmp_path, 'fixed', explicit)[0])
        final = session.run(_command(tmp_path, 'auto2', auto)[0])
    fresh, paths = _command(tmp_path / 'fresh', 'auto2', auto)
    plotMatrix.main(fresh[1:])
    assert (tmp_path / 'auto2.png').read_bytes() == _bytes(paths)['plot']
    assert 'scan' not in middle.recomputed
    assert 'scan' in final.reused


def test_partial_nans_toggle_heatmap(tmp_path):
    base = ('--profile', '--filterNans', 'any_bin', '--sortRegions', 'keep')
    with PlotSession(cache_bytes=64 << 20) as session:
        session.run(_command(tmp_path, 'profile1', base,
                             source=PARTIAL)[0])
        both = session.run(_command(tmp_path, 'both', (*base, '--heatmap'),
                                    source=PARTIAL)[0])
        final = session.run(_command(tmp_path, 'profile2', base,
                                     source=PARTIAL)[0])
    fresh, paths = _command(tmp_path / 'fresh', 'profile2', base,
                            source=PARTIAL)
    plotMatrix.main(fresh[1:])
    assert (tmp_path / 'profile2.png').read_bytes() == _bytes(paths)['plot']
    assert 'layout' in both.reused
    assert 'layout' in final.reused


def _order_sensitive_matrix(path):
    source = Matrix.load(str(MATRIX), 1)
    values = np.random.default_rng(17).lognormal(
        size=(24, source.values.shape[1])).astype(np.float32)
    values *= np.linspace(0.5, 2.0, 24, dtype=np.float32)[:, None]
    parameters = dict(source.header.parameters)
    parameters['group_boundaries'] = [0, 12, 24]
    regions = [source.regions[row % 3 + (row // 12) * 3]
               for row in range(24)]
    OwnedMatrix.from_compute(parameters, values, regions).save(
        path, compressed=True, threads=1)


def _unequal_matrix(path):
    source = Matrix.load(str(MATRIX), 1)
    parameters = dict(source.header.parameters)
    parameters['group_boundaries'] = [0, 2, 5]
    OwnedMatrix.from_compute(parameters, source.values[:5].copy(),
                             source.regions[:5]).save(
        path, compressed=True, threads=1)


def test_per_group_unequal_groups_matches_cold(tmp_path):
    source = tmp_path / 'unequal.gz'
    _unequal_matrix(source)
    with PlotSession(cache_bytes=64 << 20) as session:
        result = _warm_cold(session, tmp_path,
                            ('--heatmap',), ('--heatmap', '--perGroup'),
                            source=source)
    assert {'layout', 'scan'} <= set(result.reused)


def test_together_relabel_reuses_layout_but_changes_scene(tmp_path):
    base = ('--profile', '--sameGroupLabels', 'together')
    changed = (*base, '--regionsLabel', 'Equal', 'Equal')
    with PlotSession(cache_bytes=64 << 20) as session:
        first, _ = _command(tmp_path, 'first', base)
        old = session.run(first)
        result = _warm_cold(session, tmp_path, base, changed)
    assert {'layout', 'statistics'} <= set(result.reused)
    assert old.scene_digest != result.scene_digest


def test_adding_matrix_output_consumes_sort(tmp_path):
    base = ('--profile', '--sortRegions', 'descend')
    with PlotSession(cache_bytes=64 << 20) as session:
        session.run(_command(tmp_path, 'no_matrix', base)[0])
        command, warm_paths = _command(
            tmp_path, 'with_matrix', base, files=True)
        result = session.run(command)
    fresh, fresh_paths = _command(
        tmp_path / 'fresh', 'with_matrix', base, files=True)
    plotMatrix.main(fresh[1:])
    assert _bytes(warm_paths) == _bytes(fresh_paths)
    assert 'matrix' in result.reused
    assert 'layout' in result.recomputed


def test_kmeans_quantiles_kmeans_no_compounding(tmp_path):
    base = ('--profile', '--kmeans', '2', '--sortRegions', 'keep')
    other = ('--profile', '--numberOfProcessors', '1',
             '--quantileSortedRegions', '2', '--sortRegions', 'descend')
    with PlotSession(cache_bytes=64 << 20) as session:
        session.run(_command(tmp_path, 'kmeans1', base)[0])
        session.run(_command(tmp_path, 'quantiles', other)[0])
        result = session.run(_command(tmp_path, 'kmeans2', base)[0])
    fresh, paths = _command(tmp_path / 'fresh', 'kmeans2', base)
    plotMatrix.main(fresh[1:])
    assert (tmp_path / 'kmeans2.png').read_bytes() == paths['plot'].read_bytes()
    assert 'layout' in result.reused


def test_clustering_workspace_does_not_survive_request(tmp_path, monkeypatch):
    from deeptoolsr import cluster

    observed = []
    original = cluster._statistics.copy_values

    def capture(*args, **kwargs):
        workspace = original(*args, **kwargs)
        observed.append(weakref.ref(workspace))
        return workspace

    monkeypatch.setattr(cluster._statistics, 'copy_values', capture)
    sequence = (('--kmeans', '2'), ('--hclust', '2'),
                ('--kmeans', '3'), ('--hclust', '3'),
                ('--kmeans', '2'))
    with PlotSession(cache_bytes=64 << 20) as session:
        for index, clustering in enumerate(sequence):
            session.run(_command(tmp_path, f'cluster{index}',
                                 ('--profile', *clustering,
                                  '--sortRegions', 'keep'))[0])
            gc.collect()
            assert all(reference() is None for reference in observed)
    assert len(observed) >= 2


def test_bootstrap_order_round_trip(tmp_path):
    source = tmp_path / 'order_sensitive.gz'
    _order_sensitive_matrix(source)
    base = ('--profile', '--plotType', 'bootstrap', '--averageType', 'mean',
            '--bootstrapReplicates', '30', '--sortRegions', 'descend')
    with PlotSession(cache_bytes=64 << 20) as session:
        session.run(_command(tmp_path, 'profile1', base, source=source,
                             table=True)[0])
        session.run(_command(tmp_path, 'sorted', (*base, '--heatmap'),
                             source=source)[0])
        final = session.run(_command(tmp_path, 'profile2', base,
                                     source=source, table=True)[0])
    fresh, paths = _command(tmp_path / 'fresh', 'profile2', base,
                            source=source, table=True)
    plotMatrix.main(fresh[1:])
    warm = {'plot': tmp_path / 'profile2.png',
            'table': tmp_path / 'profile2.tsv'}
    assert _bytes(warm) == _bytes(paths)
    sorted_command, sorted_paths = _command(
        tmp_path / 'sorted_profile', 'sorted', base, source=source,
        files=True)
    plotMatrix.main(sorted_command[1:])
    assert sorted_paths['table'].read_bytes() != paths['table'].read_bytes()
    assert 'layout' in final.reused


def test_zero_budget_and_publish_boundary(tmp_path):
    seen = []
    with PlotSession(cache_bytes=0) as session:
        command, paths = _command(tmp_path, 'zero', ('--profile',), files=True)

        def progress(stage):
            seen.append(stage)

        session.run(
            command, progress=progress,
            cancelled=lambda: bool(seen and seen[-1] == 'publish'))
        again = session.run(_command(tmp_path, 'zero2',
                                     ('--profile',))[0])
    assert _bytes(paths)
    assert 'publish' in seen
    assert not set(again.reused) & {'layout', 'statistics', 'scan',
                                    'bitmaps', 'text'}


def test_source_replacement_and_reload(tmp_path):
    source = tmp_path / 'source.gz'
    source.write_bytes(MATRIX.read_bytes())
    with PlotSession(cache_bytes=64 << 20) as session:
        def args(stem):
            return _command(tmp_path, stem, ('--profile',), source=source)[0]
        session.run(args('first'))
        assert 'matrix' in session.run(args('second')).reused
        replacement = tmp_path / 'replacement.gz'
        replacement.write_bytes(MATRIX.read_bytes())
        os.replace(replacement, source)
        assert 'matrix' in session.run(args('replaced')).recomputed
        session.reload()
        assert 'matrix' in session.run(args('reloaded')).recomputed


def test_config_content_hash_with_restored_mtime(tmp_path):
    config = tmp_path / 'options.json'
    left = {'font_multiplier': 1.0}
    right = {'font_multiplier': 1.2}
    config.write_text(json.dumps(left))
    before = config.stat()
    flags = ('--profile', '--config', str(config))
    with PlotSession(cache_bytes=64 << 20) as session:
        initial = session.run(_command(tmp_path, 'initial', flags)[0])
        config.write_text(json.dumps(right))
        os.utime(config, ns=(before.st_atime_ns, before.st_mtime_ns))
        changed = session.run(_command(tmp_path, 'changed', flags)[0])
    fresh, paths = _command(tmp_path / 'fresh', 'changed', flags)
    plotMatrix.main(fresh[1:])
    assert initial.scene_digest != changed.scene_digest
    assert {'layout', 'statistics'} <= set(changed.reused)
    assert (tmp_path / 'initial.png').read_bytes() != (
        tmp_path / 'changed.png').read_bytes()
    assert (tmp_path / 'changed.png').read_bytes() == paths['plot'].read_bytes()


def test_request_config_and_auto_override_session_defaults(
        tmp_path, monkeypatch):
    inherited = tmp_path / 'inherited.json'
    explicit = tmp_path / 'explicit.json'
    automatic_dir = tmp_path / 'automatic'
    automatic_dir.mkdir()
    automatic = automatic_dir / 'options.txt'
    inherited.write_text(json.dumps({'font_multiplier': 1.0}))
    explicit.write_text(json.dumps({'font_multiplier': 1.2}))
    automatic.write_text(json.dumps({'font_multiplier': 1.4,
                                     'number_of_processors': 2}))
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(automatic_dir))
    seen = []
    original = plotMatrix.run_session_request

    def capture(session, args, run_options, **kwargs):
        seen.append((run_options.config_path, run_options.threads))
        return original(session, args, run_options, **kwargs)

    monkeypatch.setattr(plotMatrix, 'run_session_request', capture)
    with PlotSession(config=str(inherited), threads=1,
                     cache_bytes=64 << 20) as session:
        session.run(_command(tmp_path, 'inherited', ('--profile',))[0])
        assert session.config_digest == hashlib.sha256(
            inherited.read_bytes()).hexdigest()
        session.run(_command(tmp_path, 'explicit',
                             ('--profile', '--config', str(explicit),
                              '-p', '2'))[0])
        assert session.config_digest == hashlib.sha256(
            explicit.read_bytes()).hexdigest()
        session.run(_command(tmp_path, 'automatic',
                             ('--profile', '--config', 'auto',
                              '-p', 'auto'))[0])
        assert session.config_digest == hashlib.sha256(
            automatic.read_bytes()).hexdigest()
    assert seen == [(inherited, 1), (explicit, 2), (automatic, 2)]


@pytest.mark.parametrize('config_flag,processor_flag,threads', [
    (('--config',), ('-p', '2'), 2),
    (('--config=',), ('-p2',), 2),
    (('--config=',), ('-p4',), 4),
    (('--conf',), ('--numberOfProcessors', '2'), 2),
    (('--conf=',), ('--numberOfProcessors=2',), 2),
    (('--conf',), ('--numberOf', '4'), 4),
    (('--config',), ('--numberOfProc=2',), 2),
])
def test_explicit_run_option_spellings_override_session_defaults(
        tmp_path, monkeypatch, config_flag, processor_flag, threads):
    """Running and describing share one config/thread precedence."""
    assert parserCommon.plot_parser('plotMatrixR').allow_abbrev
    inherited = tmp_path / 'inherited.json'
    explicit = tmp_path / 'explicit.json'
    inherited.write_text('{}')
    explicit.write_text('{}')
    seen = []
    original = options.resolve_run_options_from_bytes

    def capture(*args, **kwargs):
        resolved = original(*args, **kwargs)
        seen.append((resolved.config_path, resolved.threads))
        return resolved

    monkeypatch.setattr(options, 'resolve_run_options_from_bytes', capture)
    config_args = (config_flag[0] + str(explicit),) if config_flag[0].endswith('=') \
        else (*config_flag, str(explicit))
    command = _command(tmp_path, 'spelling',
                       ('--profile', *config_args, *processor_flag))[0]
    with PlotSession(config=str(inherited), threads=1, cache_bytes=0) as session:
        session.run(command)
        session.describe(command)
    assert seen == [(explicit, threads)] * 2


def test_emit_order_and_no_figure_retention(tmp_path, monkeypatch):
    import matplotlib.figure

    weak = []
    original = matplotlib.figure.Figure.__init__

    def capture(figure, *args, **kwargs):
        original(figure, *args, **kwargs)
        weak.append(weakref.ref(figure))

    monkeypatch.setattr(matplotlib.figure.Figure, '__init__', capture)
    notices = []
    stages = []
    with PlotSession(cache_bytes=64 << 20) as session:
        session.run(_command(tmp_path, 'figure', ('--profile',))[0],
                    progress=stages.append,
                    emit=lambda stream, text: notices.append((stream, text)))
        gc.collect()
        assert all(reference() is None for reference in weak)
    assert stages[:3] == ['parse', 'load', 'order']
    assert stages[-1] == 'publish'
    assert all(stream in ('stdout', 'stderr') for stream, _ in notices)


def test_measurer_ownership_across_styles_and_failure(tmp_path, monkeypatch):
    import matplotlib.figure

    weak = []
    original = matplotlib.figure.Figure.__init__

    def capture(figure, *args, **kwargs):
        original(figure, *args, **kwargs)
        weak.append(weakref.ref(figure))

    monkeypatch.setattr(matplotlib.figure.Figure, '__init__', capture)
    configs = []
    for index, multiplier in enumerate((1.0, 1.2, 1.0)):
        path = tmp_path / f'style{index}.json'
        path.write_text(json.dumps({'font_multiplier': multiplier}))
        configs.append(path)
    with PlotSession(cache_bytes=64 << 20) as session:
        for index, config in enumerate(configs):
            flags = ('--profile', '--dpi', str(80 + index * 20),
                     '--cellWidth', str(7 + index),
                     '--config', str(config))
            session.run(_command(tmp_path, f'style{index}', flags)[0])
            gc.collect()
            assert all(reference() is None for reference in weak)

        def fail_save(*_args, **_kwargs):
            raise RuntimeError('injected save failure')

        with monkeypatch.context() as patch:
            patch.setattr(plotMatrix, 'save_figure_atomic', fail_save)
            with pytest.raises(RuntimeError, match='injected save failure'):
                session.run(_command(tmp_path, 'fail',
                                     ('--profile',))[0])
        gc.collect()
        assert all(reference() is None for reference in weak)


def test_cancel_before_build_does_not_publish(tmp_path):
    stages = []
    command, paths = _command(tmp_path, 'cancelled', ('--heatmap',))
    with PlotSession(cache_bytes=64 << 20) as session:
        with pytest.raises(Cancelled):
            session.run(command, progress=stages.append,
                        cancelled=lambda: 'scene' in stages)
    assert not any(path.exists() for path in paths.values())


@pytest.mark.parametrize('kind', ['--profile', '--heatmap'])
def test_cancel_pending_at_publish_does_not_publish(tmp_path, monkeypatch,
                                                    kind):
    # A cancel that arrives before publication begins wins, even when the
    # last check before it was an earlier stage.
    stages, pending = [], []
    command, paths = _command(tmp_path, 'cancel_publish', (kind,),
                              files=kind == '--profile')
    with PlotSession(cache_bytes=64 << 20) as session:
        original_stage = session.stage

        def stage(name, cancelled, progress):
            if name == 'publish':
                pending.append(True)
            return original_stage(name, cancelled, progress)

        monkeypatch.setattr(session, 'stage', stage)
        with pytest.raises(Cancelled):
            session.run(command, progress=stages.append,
                        cancelled=lambda: bool(pending))
    assert 'publish' not in stages
    assert not any(path.exists() for path in paths.values())


def test_bitmap_cancellation_releases_figure(tmp_path, monkeypatch):
    import matplotlib.figure

    weak = []
    original = matplotlib.figure.Figure.__init__

    def capture(figure, *args, **kwargs):
        original(figure, *args, **kwargs)
        weak.append(weakref.ref(figure))

    monkeypatch.setattr(matplotlib.figure.Figure, '__init__', capture)
    command, paths = _command(tmp_path, 'cancel_bitmaps', ('--heatmap',))
    with PlotSession(cache_bytes=64 << 20) as session:
        original_stage = session.stage

        def stage(name, cancelled, progress):
            if name == 'bitmaps':
                raise Cancelled()
            return original_stage(name, cancelled, progress)

        monkeypatch.setattr(session, 'stage', stage)
        with pytest.raises(Cancelled):
            session.run(command)
        gc.collect()
        assert weak and all(reference() is None for reference in weak)
    assert not any(path.exists() for path in paths.values())


def test_cache_counts_shared_backing_and_each_view_header():
    base = np.arange(1024, dtype=np.float64)
    cache = ByteBudgetCache(1 << 20)
    assert cache.put('statistics', 'first', base[:10])
    assert cache.put('statistics', 'second', base[10:20])
    standalone = (cache.entry_cost('statistics', 'first') +
                  cache.entry_cost('statistics', 'second'))
    assert cache.bytes_used == standalone - base.nbytes
    assert cache.bytes_used > base.nbytes + 2 * 256

    mask = np.zeros(base.size, dtype=bool)
    masked = np.ma.array(base[:10], mask=mask[:10], copy=False)
    assert cache.put('statistics', 'masked', masked)
    assert cache.bytes_used > base.nbytes + mask.nbytes
    assert cache.bytes_used < standalone + cache.entry_cost(
        'statistics', 'masked')


def test_cache_oversized_and_eviction_class_order():
    cache = ByteBudgetCache(1 << 20)
    huge = np.zeros(2 << 20, dtype=np.uint8)
    assert not cache.put('bitmaps', 'huge', huge)
    assert cache.bytes_used == 0
    for kind in ('bitmaps', 'text', 'statistics', 'layout'):
        assert cache.put(kind, kind, bytearray(1024))
    before = cache.bytes_used
    cache.budget = before
    assert cache.put('layout', 'new', bytearray(1024))
    assert not cache.peek('bitmaps', 'bitmaps')[1]
    assert cache.peek('text', 'text')[1]


def test_many_masked_views_respect_heap_budget():
    base = np.arange(1024, dtype=np.float64)
    mask = np.zeros(1024, dtype=bool)
    cache = ByteBudgetCache(2 << 20)
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for index in range(5000):
        offset = index % 1000
        value = np.ma.array(base[offset:offset + 4],
                            mask=mask[offset:offset + 4], copy=False)
        cache.put('statistics', index, value)
    value = None
    gc.collect()
    growth = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    assert cache.bytes_used <= cache.budget
    assert growth <= cache.budget * 1.1


def test_many_text_entries_respect_heap_budget():
    cache = ByteBudgetCache(4 << 20)
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for index in range(20000):
        cache.put('text', f'label {index}', (3.0, 4.0))
    for index in range(200):
        cache.put('layout', index, ('row', index))
    gc.collect()
    growth = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    assert cache.bytes_used <= cache.budget
    assert growth <= cache.budget * 1.1
