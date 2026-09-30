from tests.helpers.bigwig import pyBigWig
from deeptoolsr import plotProfile
import argparse
import gzip
import json
import os
from pathlib import Path

import pytest

from deeptoolsr import config
from deeptoolsr import computeMatrix
from deeptoolsr import deeptoolsr_list_tools
from deeptoolsr import plotHeatmap
from deeptoolsr import options as run_options
from deeptoolsr.plotting import heatmap as heatmap_plotting


def test_options_are_persisted(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    deeptoolsr_list_tools.process_args([
        'options', '--numberOfProcessors', '2'])

    assert config.load_options() == {
        'raster_filter': 'cubic',
        'raster_bit_depth': 8,
        'number_of_processors': 2,
    }
    with (tmp_path / 'options.txt').open() as handle:
        assert json.load(handle)['number_of_processors'] == 2
    output = capsys.readouterr().out
    assert 'Native I/O:' not in output
    assert 'Native matrix computation:' not in output
    assert 'Number of processors: 2' in output
    assert 'Native pool workers:' in output
    assert 'Ward distance budget:' in output


def test_ward_distance_budget_from_options_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run_options, 'physical_memory_bytes',
                        lambda: 3 << 30)
    args = argparse.Namespace(config=str(tmp_path / 'options.txt'),
                              numberOfProcessors='auto')
    assert run_options.resolve_run_options(args).ward_distance_budget_bytes == 3 << 29
    config.save_options({'ward_distance_budget_bytes': 12345},
                        tmp_path / 'options.txt')
    assert run_options.resolve_run_options(args).ward_distance_budget_bytes == 12345
    config.save_options({'ward_distance_budget_bytes': True},
                        tmp_path / 'options.txt')
    assert run_options.resolve_run_options(args).ward_distance_budget_bytes == 3 << 29


@pytest.mark.parametrize('flag', ['--enableNativeIO', '--enableNativeCompute'])
def test_obsolete_native_flags_are_rejected(flag):
    with pytest.raises(SystemExit):
        deeptoolsr_list_tools.parse_arguments().parse_args(['options', flag])


def test_old_native_disable_settings_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    config.save_options({'native_compute': False, 'native_io': False})
    monkeypatch.setenv('DEEPTOOLSR_DISABLE_NATIVE_COMPUTE', '1')
    monkeypatch.setenv('DEEPTOOLSR_DISABLE_NATIVE_IO', '1')
    assert 'native_compute' not in config.load_options()
    assert 'native_io' not in config.load_options()
    deeptoolsr_list_tools.process_args(['options'])
    stored = json.loads((tmp_path / 'options.txt').read_text())
    assert 'native_compute' not in stored
    assert 'native_io' not in stored


def test_legacy_raster_setting_and_environment_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    monkeypatch.setenv('DEEPTOOLSR_DISABLE_NATIVE_RASTER', '1')
    config.save_options({'native_raster': False})
    assert 'native_raster' not in config.load_options()

    rendered = []
    original = heatmap_plotting.render_heatmap_rgba

    def record_native_render(request):
        rendered.append(request['data'].values.dtype)
        return original(request)

    monkeypatch.setattr(heatmap_plotting, 'render_heatmap_rgba', record_native_render)
    matrix = Path(__file__).parent.parent / 'test_heatmapper' / 'master.mat.gz'
    output = tmp_path / 'heatmap.png'
    plotHeatmap.main(['-m', str(matrix), '-o', str(output),
                      '--whatToShow', 'heatmap and colorbar'])
    assert output.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert rendered


def test_removed_raster_flag_is_rejected():
    with pytest.raises(SystemExit):
        deeptoolsr_list_tools.parse_arguments().parse_args(
            ['options', '--enableNativeRaster'])


def test_compute_matrix_uses_configured_processors(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    config.save_options({'number_of_processors': 3})
    args = computeMatrix.process_args([
        'reference-point', '-S', 'signal.bw', '-R', 'regions.bed',
        '-o', str(tmp_path / 'matrix.gz')])
    assert run_options.resolve_run_options(args).threads == 3


def test_compute_matrix_explicit_processors_override_config(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    config.save_options({'number_of_processors': 3})
    args = computeMatrix.process_args([
        'reference-point', '-S', 'signal.bw', '-R', 'regions.bed',
        '-o', str(tmp_path / 'matrix.gz'), '-p', '1'])
    assert run_options.resolve_run_options(args).threads == 1


def test_default_processor_value_is_available_to_other_commands(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    config.save_options({'number_of_processors': 3})
    assert run_options.resolve_run_options(
        computeMatrix.process_args([
            'reference-point', '-S', 'signal.bw', '-R', 'regions.bed',
            '-o', str(tmp_path / 'matrix.gz')])).threads == 3


def test_auto_threads_use_stored_value_or_half_the_machine(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'cpu_count', lambda: 8)
    args = argparse.Namespace(config=str(tmp_path / 'options.txt'),
                              numberOfProcessors='auto')
    assert run_options.resolve_run_options(args).threads == 4
    config.save_options({'number_of_processors': 3}, tmp_path / 'options.txt')
    assert run_options.resolve_run_options(args).threads == 3
    args.numberOfProcessors = 'max'
    assert run_options.resolve_run_options(args).threads == 8


def test_processor_edits_preserve_or_remove_only_the_requested_key(tmp_path):
    path = tmp_path / 'chosen' / 'options.txt'
    base = ['options', '--config', str(path)]
    deeptoolsr_list_tools.process_args(base + ['--fontMultiplier', '1.2'])
    assert 'number_of_processors' not in json.loads(path.read_text())
    for value, expected in [('3', 3), ('max', 'max')]:
        deeptoolsr_list_tools.process_args(base + ['-p', value])
        assert json.loads(path.read_text())['number_of_processors'] == expected
        deeptoolsr_list_tools.process_args(base + ['--fontMultiplier', '1.3'])
        assert json.loads(path.read_text())['number_of_processors'] == expected
    deeptoolsr_list_tools.process_args(base + ['-p', 'auto'])
    assert 'number_of_processors' not in json.loads(path.read_text())


def test_invalid_stored_processor_warns_and_uses_auto(tmp_path, monkeypatch,
                                                      capsys):
    monkeypatch.setattr(os, 'cpu_count', lambda: 6)
    path = tmp_path / 'options.txt'
    path.write_text(json.dumps({'number_of_processors': 'unreadable'}))
    args = argparse.Namespace(config=str(path), numberOfProcessors='auto')
    assert run_options.resolve_run_options(args).threads == 3
    assert 'invalid number_of_processors' in capsys.readouterr().err


def test_config_file_changes_plot_fonts_and_auto_uses_config_dir(tmp_path,
                                                                 monkeypatch):
    matrix = Path(__file__).parent.parent / 'test_heatmapper' / 'master.mat.gz'
    custom = tmp_path / 'custom.txt'
    custom.write_text(json.dumps({'font_multiplier': 1.5}))
    first, second = tmp_path / 'default.png', tmp_path / 'custom.png'
    base = ['-m', str(matrix), '--whatToShow', 'heatmap and colorbar', '-p', '1']
    plotHeatmap.main(base + ['-o', str(first)])
    plotHeatmap.main(base + ['-o', str(second), '--config', str(custom)])
    assert first.read_bytes() != second.read_bytes()
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    (tmp_path / 'options.txt').write_bytes(custom.read_bytes())
    auto = tmp_path / 'auto.png'
    plotHeatmap.main(base + ['-o', str(auto), '--config', 'auto'])
    assert auto.read_bytes() == second.read_bytes()
    resolved = run_options.resolve_run_options(
        argparse.Namespace(config='auto', numberOfProcessors='auto'),
        plotting=True)
    assert resolved.style.font_multiplier == 1.5


def test_compute_matrix_header_records_resolved_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'cpu_count', lambda: 8)
    path = tmp_path / 'options.txt'
    config.save_options({'number_of_processors': 2}, path)
    source = Path(__file__).parent.parent / 'test_heatmapper'
    output = tmp_path / 'matrix.gz'
    computeMatrix.main([
        'reference-point', '-S', str(source / 'test.bw'),
        '-R', str(source / 'test2.bed'), '-b', '100', '-a', '100',
        '-bs', '10', '-o', str(output), '--config', str(path), '-p', 'auto'])
    with gzip.open(output, 'rt') as handle:
        header = json.loads(handle.readline()[1:])
    assert header['proc number'] == 2


# --- persistent typography / geometry style ---------------------------------


def _stored(tmp_path):
    with (tmp_path / 'options.txt').open() as handle:
        return json.load(handle)


def test_options_command_materialises_full_expanded_schema(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    # Any options invocation, even one that changes nothing, writes the file.
    deeptoolsr_list_tools.process_args(['options'])
    stored = _stored(tmp_path)
    assert stored['font_family'] == ['sans-serif']
    assert stored['font_multiplier'] == 1.0
    assert set(stored['typography']) == set(config.default_style_dict()['typography'])
    assert 'figure_edge_padding' in stored['geometry']
    assert stored['drawing'] == {'profile_line_width': 1.5}
    assert stored['label_layout'] == {
        'auto_panel_title_column_gap': True,
        'max_column_gap_points': 96.0,
        'panel_title_max_lines': 3,
        'panel_title_extra_row_penalty': 0.08,
        'panel_title_balance_weight': 0.02,
        'panel_title_orphan_weight': 0.05,
        'axis_label_max_lines': 3,
        'axis_label_extra_row_penalty': 0.08,
        'axis_label_balance_weight': 0.02,
        'axis_label_orphan_weight': 0.05,
        'auto_axis_label_layout': True,
        'auto_horizontal_colorbar_label_layout': True,
        'horizontal_colorbar_label_max_lines': 3,
        'horizontal_colorbar_label_extra_row_penalty': 0.08,
        'horizontal_colorbar_label_balance_weight': 0.02,
        'horizontal_colorbar_label_orphan_weight': 0.05,
        'auto_facet_label_layout': True,
        'auto_heatmap_region_label_layout': True,
        'heatmap_region_label_gap_growth': True,
        'heatmap_region_label_max_lines': 3,
        'heatmap_region_label_extra_row_penalty': 0.08,
        'heatmap_region_label_balance_weight': 0.02,
        'heatmap_region_label_orphan_weight': 0.05,
        'max_row_gap_points': 96.0,
        'max_heatmap_region_gap_points': 96.0,
        'auto_legend_label_layout': True,
        'legend_label_max_lines': 3,
    }
    assert '_invalid' not in stored


def test_region_label_gap_growth_is_read_from_options_file(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    config.save_options({'label_layout': {
        'heatmap_region_label_gap_growth': False}})
    args = argparse.Namespace(config='auto', numberOfProcessors='auto')
    run = run_options.resolve_run_options(args, plotting=True)
    assert run.style.label_layout.heatmap_region_label_gap_growth is False


def test_font_family_and_multiplier_are_persisted_and_reset(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    deeptoolsr_list_tools.process_args(
        ['options', '--fontFamily', 'Arial', 'Helvetica', 'sans-serif',
         '--fontMultiplier', '1.25'])
    stored = _stored(tmp_path)
    assert stored['font_family'] == ['Arial', 'Helvetica', 'sans-serif']
    assert stored['font_multiplier'] == 1.25

    deeptoolsr_list_tools.process_args(['options', '--fontFamily', 'default'])
    assert _stored(tmp_path)['font_family'] == ['sans-serif']
    deeptoolsr_list_tools.process_args(['options', '--fontMultiplier', 'default'])
    assert _stored(tmp_path)['font_multiplier'] == 1.0


def test_invalid_font_multiplier_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    for bad in ('nonsense', '-1', '0'):
        with pytest.raises(SystemExit, match='multiplier'):
            deeptoolsr_list_tools.process_args(
                ['options', '--fontMultiplier', bad])
    assert not (tmp_path / 'options.txt').exists()


def test_multiplier_never_compounds_base_sizes(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    deeptoolsr_list_tools.process_args(['options', '--fontMultiplier', '1.5'])
    first = _stored(tmp_path)['typography']['figure_title_text']['size']
    deeptoolsr_list_tools.process_args(['options', '--fontMultiplier', '1.5'])
    second = _stored(tmp_path)['typography']['figure_title_text']['size']
    assert first == second == 12.0


def test_options_writer_preserves_typos_and_unknown_keys(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    (tmp_path / 'options.txt').write_text(json.dumps({
        'typography': {'x_tick_label_text': {'size': 'huge'}},
        'geometry': {'figure_edge_padding': -4},
        'my_experiment_note': 'keep me',
    }))
    deeptoolsr_list_tools.process_args(['options'])
    stored = _stored(tmp_path)
    # The user's original (invalid) values and unknown keys survive verbatim.
    assert stored['typography']['x_tick_label_text']['size'] == 'huge'
    assert stored['geometry']['figure_edge_padding'] == -4
    assert stored['my_experiment_note'] == 'keep me'
    # ... and the reasons are recorded for the user.
    assert any('x_tick_label_text.size' in msg for msg in stored['_invalid'])
    assert any('figure_edge_padding' in msg for msg in stored['_invalid'])


def test_options_writer_preserves_invalid_whole_style_section(
        tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    (tmp_path / 'options.txt').write_text(json.dumps({
        'geometry': 'not an object'}))
    deeptoolsr_list_tools.process_args(['options'])
    stored = _stored(tmp_path)
    assert stored['geometry'] == 'not an object'
    assert any('geometry: expected an object' in msg
               for msg in stored['_invalid'])


def test_load_style_falls_back_without_writing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    (tmp_path / 'options.txt').write_text(json.dumps({
        'font_multiplier': 3.0,
        'typography': {'y_tick_label_text': {'size': 0}},
    }))
    before = (tmp_path / 'options.txt').read_text()
    style = run_options.resolve_run_options(
        argparse.Namespace(config='auto', numberOfProcessors='auto'),
        plotting=True).style
    # Valid override applied, invalid one replaced by its default in memory.
    assert style.font_multiplier == 3.0
    assert style.typography.y_tick_label_text.size == 8.0
    assert 'y_tick_label_text.size' in capsys.readouterr().err
    # The plotting path must never rewrite the user's file.
    assert (tmp_path / 'options.txt').read_text() == before


def test_load_style_missing_file_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    style = run_options.resolve_run_options(
        argparse.Namespace(config='auto', numberOfProcessors='auto'),
        plotting=True).style
    assert style.font_family == ('sans-serif',)
    assert style.font_multiplier == 1.0
    assert not (tmp_path / 'options.txt').exists()


def test_load_options_stays_native_only(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path))
    deeptoolsr_list_tools.process_args(['options', '--fontMultiplier', '1.5'])
    # The native fast path must not gain style keys (and must stay light).
    assert set(config.load_options()) == set(config.DEFAULTS)


# From audit regressions.

@pytest.fixture()
def io_native_backends(monkeypatch):
    monkeypatch.delenv('DTP_BAM_BACKEND', raising=False)


def sparse_inputs(root):
    source = root / 'source.bw'
    with pyBigWig.open(str(source), 'w') as bw:
        bw.addHeader([('chr1', 100)])
        bw.addEntries(['chr1', 'chr1'], [10, 50], ends=[20, 70], values=[100.0, 1.0])
    bed = root / 'regions.bed'
    bed.write_text('chr1\t10\t30\tbad\t0\t+\nchr1\t50\t70\tgood\t0\t+\n')
    return ['reference-point', '-S', str(source), '-R', str(bed), '-a', '20', '-b', '0', '--binSize', '10', '--sortRegions', 'keep']


@pytest.mark.usefixtures('io_native_backends')
def test_per_bin_std_is_no_longer_advertised(tmp_path):
    output = tmp_path / 'std.gz'
    with pytest.raises(SystemExit) as error:
        computeMatrix.main(sparse_inputs(tmp_path) + ['--averageTypeBins', 'std', '-o', str(output)])
    assert error.value.code == 2
    assert not output.exists()


# Argument contract.

TEST_DATA = Path(__file__).parent.parent / 'test_data'


MATRIX = TEST_DATA / 'computeMatrixOperations.mat.gz'


@pytest.mark.parametrize('command,extra,message', [('heatmap', ['--kmeans', '2', '--hclust', '2'], 'not allowed with argument'), ('heatmap', ['--kmeans', '2', '--quantiles', '2'], 'cannot be combined'), ('heatmap', ['--silhouette'], 'requires --kmeans or --hclust'), ('profile', ['--kmeans', '2', '--hclust', '2'], 'not allowed with argument'), ('profile', ['--hclust', '2', '--quantiles', '2'], 'cannot be combined'), ('profile', ['--silhouette'], 'requires --kmeans or --hclust')])
def test_clustering_compatibility_table(tmp_path, capsys, command, extra, message):
    process = plotHeatmap.process_args if command == 'heatmap' else plotProfile.process_args
    with pytest.raises(SystemExit):
        process(['-m', str(MATRIX), '-o', str(tmp_path / 'out.pdf')] + extra)
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('command,option,value,message', [('heatmap', '--heatmapHeight', '2.9', 'between 3 and 100'), ('heatmap', '--heatmapHeight', '101', 'between 3 and 100'), ('heatmap', '--heatmapWidth', '0.9', 'between 1 and 100'), ('heatmap', '--heatmapWidth', '101', 'between 1 and 100'), ('profile', '--profileHeight', '0.4', 'between 0.5 and 100'), ('profile', '--profileHeight', '101', 'between 0.5 and 100'), ('profile', '--profileWidth', '0.9', 'between 1 and 100'), ('profile', '--profileWidth', '101', 'between 1 and 100')])
def test_plot_dimension_contract_table(tmp_path, capsys, command, option, value, message):
    process = plotHeatmap.process_args if command == 'heatmap' else plotProfile.process_args
    with pytest.raises(SystemExit):
        process(['-m', str(MATRIX), '-o', str(tmp_path / 'out.pdf'), option, value])
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('extra,message', [(['--bootstrapReplicates', '20'], 'unused unless --plotType bootstrap'), (['--ci_level', '0.8'], 'unused unless --plotType ci or bootstrap'), (['--pseudocount', '1'], 'unused unless --averageType geom_mean'), (['--trim_perc', '0.1'], 'unused unless --averageType trim_mean')])
def test_explicit_inert_profile_bootstrap_controls_warn(tmp_path, capsys, extra, message):
    plotProfile.process_args(['-m', str(MATRIX), '-o', str(tmp_path / 'out.pdf')] + extra)
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('extra,message', [(['--pseudocountSummaryPlot', '1'], 'unused unless --averageTypeSummaryPlot geom_mean'), (['--trimPercSummaryPlot', '0.1'], 'unused unless --averageTypeSummaryPlot trim_mean'), (['--colorNumber', '16'], 'unused without --colorList'), (['--bootstrapReplicates', '20'], 'unused unless --plotTypeSummaryPlot bootstrap'), (['--ci_level', '0.8'], 'unused unless --plotTypeSummaryPlot ci or bootstrap'), (['--whatToShow', 'heatmap and colorbar', '--averageTypeSummaryPlot', 'median'], 'summary-plot statistic options are unused'), (['--whatToShow', 'heatmap and colorbar', '--ci_level', '0.8'], 'summary-plot statistic options are unused')])
def test_explicit_inert_heatmap_controls_warn(tmp_path, capsys, extra, message):
    plotHeatmap.process_args(['-m', str(MATRIX), '-o', str(tmp_path / 'out.pdf')] + extra)
    assert message in capsys.readouterr().err


def test_incompatible_hidden_heatmap_summary_is_only_warned(tmp_path, capsys):
    plotHeatmap.process_args(['-m', str(MATRIX), '-o', str(tmp_path / 'out.pdf'), '--whatToShow', 'heatmap and colorbar', '--plotTypeSummaryPlot', 'se', '--averageTypeSummaryPlot', 'median'])
    assert 'summary-plot statistic options are unused' in capsys.readouterr().err
