"""Header-only JSON metadata for computeMatrixOperationsR info."""

import json
from pathlib import Path

from deeptoolsr import computeMatrixOperations as operations
from deeptoolsr.matrix import Matrix, read_header


MATRIX = Path(__file__).parent / 'test_data' / 'computeMatrixOperations.mat.gz'


def test_info_json_reads_header_only(monkeypatch, capsys):
    def fail_load(*_args, **_kwargs):
        raise AssertionError('info --json loaded matrix values')

    monkeypatch.setattr(Matrix, 'load', fail_load)
    operations.main(['info', '-m', str(MATRIX), '--json'])
    record = json.loads(capsys.readouterr().out)
    header = read_header(MATRIX, normalize=True)
    assert record == {
        'protocol': 1,
        'samples': list(header.sample_labels),
        'groups': [
            {'label': label,
             'regions': header.group_boundaries[i + 1] -
             header.group_boundaries[i]}
            for i, label in enumerate(header.group_labels)
        ],
        'bins_per_sample': [
            header.sample_boundaries[i + 1] - header.sample_boundaries[i]
            for i in range(len(header.sample_labels))
        ],
        'parameters': dict(header.parameters),
    }


def test_info_json_reports_the_sort_of_an_exported_plot_matrix(
        tmp_path, capsys):
    from deeptoolsr import plotHeatmap

    source = (Path(__file__).parent / 'test_data' / 'prepare_expected' /
              'input.gz')
    cases = {
        'sorted': (['--sortRegions', 'descend', '--sortUsing', 'max'],
                   ('descend', 'max')),
        # Clustering reorders rows without a sort.
        'clustered': (['--kmeans', '2', '--sortRegions', 'no'],
                      ('no', 'mean')),
    }
    for name, (options, expected) in cases.items():
        exported = tmp_path / f'{name}.gz'
        plotHeatmap.main(['-m', str(source), '-o', str(tmp_path / 'p.png'),
                          '--outFileNameMatrix', str(exported), *options])
        capsys.readouterr()
        operations.main(['info', '-m', str(exported), '--json'])
        parameters = json.loads(capsys.readouterr().out)['parameters']
        assert (parameters['sort regions'],
                parameters['sort using']) == expected, name
