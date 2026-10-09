"""Plot CLI path errors retain their parser-facing form."""

import subprocess
from pathlib import Path

import pytest

from deeptoolsr import plotHeatmap, plotProfile


@pytest.mark.parametrize('tool', ('plotHeatmapR', 'plotProfileR'))
def test_missing_matrix_path_reports_parser_error(tool, tmp_path):
    missing = tmp_path / 'missing.gz'
    output = tmp_path / 'plot.png'
    result = subprocess.run([tool, '-m', str(missing), '-o', str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert result.stdout == ''
    assert result.stderr == (
        f'usage: {tool} -m matrix.gz\n'
        f'help: {tool} -h / {tool} --help\n'
        f"{tool}: error: argument --matrixFile/-m: can't open '{missing}': "
        f"[Errno 2] No such file or directory: '{missing}'\n")


@pytest.mark.parametrize('module', (plotHeatmap, plotProfile))
def test_plot_output_writability_is_checked_when_staging(module, tmp_path):
    matrix = tmp_path / 'matrix.gz'
    matrix.touch()
    output = tmp_path / 'missing-directory' / 'plot.png'
    parsed = module.parse_arguments().parse_args(
        ['-m', str(matrix), '-o', str(output)])
    assert parsed.matrixFile == str(matrix)
    assert parsed.outFileName == str(output)
    # An unreadable destination is accepted by argparse and diagnosed by
    # the staging check with the same option-specific message as before.
    from deeptoolsr.plotting.cli_io import check_plot_outputs
    from deeptoolsr.prepare import DataError

    with pytest.raises(DataError, match="file can't be opened for writing"):
        check_plot_outputs(parsed.matrixFile, (
            (parsed.outFileName, '--outFileName/-out/-o'),))


@pytest.mark.parametrize('tool', ('plotHeatmapR', 'plotProfileR'))
def test_unwritable_plot_output_keeps_option_message(tool, tmp_path):
    matrix = Path(__file__).parent / 'test_heatmapper' / 'master.mat.gz'
    output = tmp_path / 'missing-directory' / 'plot.png'
    result = subprocess.run([tool, '-m', str(matrix), '-o', str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert result.stderr.endswith(
        f"{tool}: error: argument --outFileName/-out/-o: "
        f"{output} file can't be opened for writing\n")
