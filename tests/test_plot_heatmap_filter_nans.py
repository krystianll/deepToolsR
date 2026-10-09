"""End-to-end coverage for plotHeatmapR --filterNans.

The row-policy semantics are covered by matrix and cMO tests; this checks
that the plot CLI applies the selected policy to the final BED output.
"""

from deeptoolsr.plotHeatmap import main
from deeptoolsr.matrix import OwnedMatrix
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')


def _nan_pattern_matrix(path):
    """4 regions x 2 samples (2 bins each) with assorted NaN patterns.

    Mirrors test_computeMatrixOperations.py's _nan_pattern_matrix fixture,
    whose per-mode expected survivors are already verified there.
    """
    nan = np.nan
    data = np.array([
        [nan, nan, nan, nan],     # r0: whole row NaN
        [10.0, nan, 50.0, 50.0],  # r1: s1 partially NaN, s2 fine
        [nan, nan, 50.0, 50.0],   # r2: s1 entirely NaN, s2 fine
        [10.0, 20.0, 30.0, 40.0],  # r3: all finite
    ], dtype=np.float32)
    n, cols = data.shape
    half = cols // 2
    regions = [['chr1', [(i, i + 1)], 'r%d' % i, 0, '+', '0']
               for i in range(n)]
    parameters = {
        'upstream': [0, 0], 'downstream': [0, 0], 'body': [half, half],
        'unscaled 5 prime': [0, 0], 'unscaled 3 prime': [0, 0],
        'ref point': [None, None], 'bin size': [1, 1],
        'sort regions': 'keep', 'sort using': 'mean',
        'sample_labels': ['s1', 's2'], 'group_labels': ['A'],
        'sample_boundaries': [0, half, cols], 'group_boundaries': [0, n],
        # plotHeatmap re-applies these unconditionally; a matrix built
        # in-memory (rather than through computeMatrix) needs them present.
        'min threshold': None, 'max threshold': None,
    }
    OwnedMatrix.from_compute(parameters, data, regions).save(
        str(path), compressed=True, threads=1)


@pytest.mark.parametrize('nan_mode,expected', [
    ('keep', ['r0', 'r1', 'r2', 'r3']),        # nothing dropped
    ('all_bins', ['r1', 'r2', 'r3']),          # only the wholly-NaN row
    ('any_sample', ['r1', 'r3']),              # rows with an all-NaN sample
    ('any_bin', ['r3']),                       # rows with any NaN bin
])
def test_filter_nans_argument_reaches_plot_heatmap(
        tmp_path, monkeypatch, nan_mode, expected):
    src = str(tmp_path / 'nan.gz')
    _nan_pattern_matrix(src)
    out_png = str(tmp_path / 'out.png')
    out_bed = str(tmp_path / 'sorted.bed')
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig',
                        lambda *args, **kwargs: None)
    main(['-m', src, '-o', out_png, '--filterNans', nan_mode,
          '--sortRegions', 'keep', '--outFileSortedRegions', out_bed])
    with open(out_bed) as handle:
        names = [line.split('\t')[3] for line in handle
                 if not line.startswith('#')]
    assert names == expected
