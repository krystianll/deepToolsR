"""A 3-sample matrix whose third sample is meant to be hidden by
--arrangeSamples, and a runner that returns a plot's region outputs."""

import numpy as np

from deeptoolsr.matrix import OwnedMatrix
from deeptoolsr.session import PlotSession

LABELS = ['Sample A', 'Sample B', 'Sample C']


def write_hidden_sample_matrix(path, nans=(), b_mirrors_c=False):
    """8 rows (r0..r7), one group, samples of 2 bins each.

    Sample A ascends (row * 0.5), sample B is A + 0.25 (or 100 - C when
    ``b_mirrors_c``), sample C is 100 + row on even rows (no ties, so a
    sort by C alone has one order on every platform) and 0 on odd rows.
    ``nans`` holds (row, column) cells set to NaN.
    """
    rows = np.arange(8, dtype=np.float32)
    low = np.repeat(rows[:, None] * 0.5, 2, axis=1)
    hidden = np.repeat(np.where(rows % 2 == 0, 100 + rows, 0)[:, None], 2, axis=1)
    middle = 100 - hidden if b_mirrors_c else low + 0.25
    values = np.hstack([low, middle, hidden]).astype(np.float32)
    for cell in nans:
        values[cell] = np.nan
    regions = [['chr1', [(i * 10, i * 10 + 5)], f'r{i}', 0, '+', '0']
               for i in range(8)]
    parameters = {
        'upstream': [0] * 3, 'downstream': [0] * 3, 'body': [2] * 3,
        'unscaled 5 prime': [0] * 3, 'unscaled 3 prime': [0] * 3,
        'ref point': [None] * 3, 'bin size': [1] * 3,
        'sort regions': 'keep', 'sort using': 'mean',
        'min threshold': None, 'max threshold': None,
        'sample_labels': LABELS, 'sample_boundaries': [0, 2, 4, 6],
        'group_labels': ['Group A'], 'group_boundaries': [0, 8]}
    OwnedMatrix.from_compute(parameters, values, regions).save(
        str(path), compressed=True, threads=1)
    return str(path)


def plot_outputs(out, tool, source, *flags):
    """Run ``tool`` once in a fresh session writing into the new directory
    ``out``; return the sorted-regions (name, group) rows and the
    --outFileNameMatrix path."""
    out.mkdir()
    bed, matrix = out / 'sorted.bed', out / 'matrix.gz'
    extra = ['--heatmap'] if tool == 'plotMatrixR' else []
    with PlotSession(cache_bytes=0) as session:
        session.run([tool, '-m', str(source), '-o', str(out / 'plot.png'),
                     *extra, '--outFileSortedRegions', str(bed),
                     '--outFileNameMatrix', str(matrix), *flags])
    rows = [tuple(line.split('\t')[i] for i in (3, 12))
            for line in bed.read_text().splitlines()
            if not line.startswith('#')]
    return rows, matrix
