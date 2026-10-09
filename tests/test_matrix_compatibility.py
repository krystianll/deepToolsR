import gzip
import os
import shutil
import sys

import deeptoolsr.computeMatrix
import deeptoolsr.plotHeatmap
import deeptoolsr.plotProfile
import json
import numpy as np
import pytest
from deeptoolsr import compute
from deeptoolsr.matrix import (Matrix, MatrixHeader, RowLayout, sort)
from deeptoolsr.prepare import resolve_labels

__author__ = 'Fidel'

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/test_heatmapper/"


def test_sort_groups_by_bed_score_keeps_invalid_values_last():
    scores = ['2.5', 'NA', '10', '.', '-1', 'nan']
    regions = [['chr1', [(i, i + 1)], str(i), 0, '+', score]
               for i, score in enumerate(scores)]
    matrix = Matrix(MatrixHeader.from_parameters({
        'group_boundaries': [0, 6], 'sample_boundaries': [0, 2],
        'group_labels': ['group'], 'sample_labels': ['sample']}),
        np.arange(12).reshape(6, 2), regions, None)
    layout = sort(matrix, RowLayout.identity(matrix), using='score',
                  method='ascend', threads=1)
    assert [matrix.regions[index][5] for index in layout.row_indices()] == [
        '-1', '2.5', '10', 'NA', '.', 'nan']
    layout = sort(matrix, RowLayout.identity(matrix), using='score',
                  method='descend', threads=1)
    assert [matrix.regions[index][5] for index in layout.row_indices()] == [
        '10', '2.5', '-1', 'NA', '.', 'nan']


def _quantile_test_matrix():
    regions = [['chr1', [(index, index + 1)], str(index), 0, '+', '0']
               for index in range(8)]
    values = np.asarray([[8], [1], [7], [2], [6], [3], [5], [4]],
                        dtype=np.float32)
    return Matrix(MatrixHeader.from_parameters({
        'group_boundaries': [0, 4, 8], 'sample_boundaries': [0, 1],
        'group_labels': ['first', 'second'], 'sample_labels': ['sample']}),
        values, regions, None)


def test_quantile_sorting_combines_groups_and_splits_global_order():
    matrix = _quantile_test_matrix()
    layout = sort(matrix, RowLayout.identity(matrix), using='mean',
                  method='ascend', quantiles=4, threads=1)
    np.testing.assert_array_equal(matrix.values[layout.row_indices(), 0],
                                  np.arange(1, 9))
    assert layout.group_bounds == (0, 2, 4, 6, 8)
    assert resolve_labels(layout, header=matrix.header).groups == (
        'first_Q1', 'first_Q2', 'first_Q3', 'first_Q4')


def test_quantiles_require_actual_sorting():
    matrix = _quantile_test_matrix()
    with pytest.raises(ValueError, match='ascending or descending'):
        sort(matrix, RowLayout.identity(matrix), method='keep',
             quantiles=2, threads=1)


def test_region_counts_after_quantiles_describe_final_groups():
    matrix = _quantile_test_matrix()
    layout = sort(matrix, RowLayout.identity(matrix), method='ascend',
                  quantiles=4, threads=1)
    assert resolve_labels(layout, show_counts=True, header=matrix.header).groups == (
        'first_Q1 [n = 2]', 'first_Q2 [n = 2]',
        'first_Q3 [n = 2]', 'first_Q4 [n = 2]')


def cmpMatrices(f1, f2):
    """
    The header produced by computeMatrix will be different every time a command is run in python3!
    """
    rv = True
    file1 = open(f1)
    file2 = open(f2)
    for l1, l2 in zip(file1, file2):
        if isinstance(l1, bytes):
            l1 = l1.decode()
            l2 = l2.decode()
        l1 = l1.strip()
        l2 = l2.strip()
        if l1.startswith("@"):
            p1 = json.loads(l1[1:])
            p2 = json.loads(l2[1:])
            for k, v in p1.items():
                if k not in p2.keys():
                    sys.stderr.write("key in {} missing: {} not in {}\n".format(f1, k, p2.keys()))
                    rv = False
                if p1[k] != p2[k]:
                    sys.stderr.write("values of '{}' is different: {} not in {}\n".format(k, p1[k], p2[k]))
                    rv = False
            for k in p2.keys():
                if k not in p1.keys():
                    sys.stderr.write("key in {} missing: {} not in {}\n".format(f2, k, p1.keys()))
                    rv = False
        else:
            if l1 != l2:
                # Data rows are compared with a float32 tolerance: matrices are
                # stored as float32, so the last significant digits differ from
                # the float64-era master files. The first six BED columns are
                # still compared exactly.
                f1f = l1.split("\t")
                f2f = l2.split("\t")
                mismatch = len(f1f) != len(f2f) or f1f[:6] != f2f[:6]
                if not mismatch:
                    for a, b in zip(f1f[6:], f2f[6:]):
                        if a == b:
                            continue
                        an = a.lower() in ("nan", "-nan")
                        bn = b.lower() in ("nan", "-nan")
                        if an or bn:
                            if an != bn:
                                mismatch = True
                                break
                            continue
                        if not np.isclose(float(a), float(b),
                                          rtol=1e-4, atol=1e-3):
                            mismatch = True
                            break
                if mismatch:
                    sys.stderr.write("lines differ:\n{}\n    vs\n{}\n".format(l1, l2))
                    rv = False
    file1.close()
    file2.close()
    return rv


def _run_matrix(tmp_path, fixed_args, master):
    """Run computeMatrix to a temp .mat.gz, decompress it in-process (no external
    ``gunzip``, no hardcoded ``/tmp``), and compare against the golden matrix.

    ``fixed_args`` is the whitespace-splittable part of the command line; the
    output path is appended as discrete tokens so a temp dir containing spaces or
    backslashes (Windows) can never break argument parsing.

    The historical golden files remain independent reference results. Current
    computation and bigWig input always use the required native extensions.
    """
    out_gz = str(tmp_path / "_test.mat.gz")
    args = fixed_args.format(ROOT).split() + ["--outFileName", out_gz]
    deeptoolsr.computeMatrix.main(args)
    out_mat = str(tmp_path / "_test.mat")
    with gzip.open(out_gz, "rb") as fin, open(out_mat, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    assert cmpMatrices(master, out_mat) is True


def test_computeMatrix_reference_point(tmp_path):
    _run_matrix(
        tmp_path,
        "reference-point -R {0}/test2.bed -S {0}/test.bw -b 100 -a 100 -bs 1 -p 1",
        ROOT + '/master.mat')


def test_computeMatrix_reference_point_center(tmp_path):
    _run_matrix(
        tmp_path,
        "reference-point -R {0}/test2.bed -S {0}/test.bw -b 100 -a 100 "
        "--referencePoint center -bs 1 -p 1",
        ROOT + '/master_center.mat')


def test_computeMatrix_reference_point_tes(tmp_path):
    _run_matrix(
        tmp_path,
        "reference-point -R {0}/test2.bed -S {0}/test.bw -b 100 -a 100 "
        "--referencePoint TES -bs 1 -p 1",
        ROOT + '/master_TES.mat')


def test_computeMatrix_reference_point_missing_data_as_zero(tmp_path):
    _run_matrix(
        tmp_path,
        "reference-point -R {0}/test2.bed -S {0}/test.bw -b 100 -a 100 "
        "-bs 1 -p 1 --missingDataAsZero",
        ROOT + '/master_nan_to_zero.mat')


def test_computeMatrix_scale_regions(tmp_path):
    _run_matrix(
        tmp_path,
        "scale-regions -R {0}/test2.bed -S {0}/test.bw -b 100 -a 100 -m 100 "
        "-bs 1 -p 1",
        ROOT + '/master_scale_reg.mat')


def test_computeMatrix_multiple_bed(tmp_path):
    _run_matrix(
        tmp_path,
        "reference-point -R {0}/group1.bed {0}/group2.bed -S {0}/test.bw "
        "-b 100 -a 100 -bs 1 -p 1",
        ROOT + '/master_multibed.mat')


def test_computeMatrix_region_extend_over_chr_end(tmp_path):
    _run_matrix(
        tmp_path,
        "reference-point -R {0}/group1.bed {0}/group2.bed -S {0}/test.bw "
        "-b 100 -a 500 -bs 1 -p 1",
        ROOT + '/master_extend_beyond_chr_size.mat')


def test_computeMatrix_unscaled(tmp_path):
    _run_matrix(
        tmp_path,
        "scale-regions -S {0}/unscaled.bigWig -R {0}/unscaled.bed -a 300 -b 500 "
        "--unscaled5prime 100 --unscaled3prime 50 -bs 10 -p 1",
        ROOT + '/master_unscaled.mat')


def test_computeMatrix_gtf(tmp_path):
    _run_matrix(
        tmp_path,
        "scale-regions -S {0}../test_data/test1.bw.bw -R {0}../test_data/test.gtf "
        "-a 300 -b 500 --unscaled5prime 20 --unscaled3prime 50 -bs 10 -p 1",
        ROOT + '/master_gtf.mat')


def test_computeMatrix_metagene(tmp_path):
    _run_matrix(
        tmp_path,
        "scale-regions -S {0}../test_data/test1.bw.bw -R {0}../test_data/test.gtf "
        "-a 300 -b 500 --unscaled5prime 20 --unscaled3prime 50 -bs 10 -p 1 --metagene",
        ROOT + '/master_metagene.mat')


def test_chopRegions_body():
    region = [(0, 200), (300, 400), (800, 900)]
    lbins, bodybins, rbins, padLeft, padRight = compute.chopRegions(region, left=0, right=0)
    e_lbins = []
    e_rbins = []
    e_padLeft = 0
    e_padRight = 0
    assert f"{lbins}" == f"{e_lbins}"
    assert f"{rbins}" == f"{e_rbins}"
    assert f"{bodybins}" == f"{region}"
    assert f"{padLeft}" == f"{e_padLeft}"
    assert f"{padRight}" == f"{e_padRight}"
    # Unscaled 5', 3'
    lbins, bodybins, rbins, padLeft, padRight = compute.chopRegions(region, left=150, right=150)
    e_lbins = [(0, 150)]
    e_rbins = [(350, 400), (800, 900)]
    e_bodybins = [(150, 200), (300, 350)]
    e_padLeft = 0
    e_padRight = 0
    assert f"{lbins}" == f"{e_lbins}"
    assert f"{rbins}" == f"{e_rbins}"
    assert f"{bodybins}" == f"{e_bodybins}"
    assert f"{padLeft}" == f"{e_padLeft}"
    assert f"{padRight}" == f"{e_padRight}"


def test_chopRegions_TSS():
    region = [(0, 200), (300, 400), (800, 900)]
    # + strand, 250 downstream
    downstream, body, unscaled3prime, padRight, _ = compute.chopRegions(region, left=250)
    e_downstream = [(0, 200), (300, 350)]
    e_body = [(350, 400), (800, 900)]
    e_unscaled3prime = []
    e_padRight = 0
    e_ = 0
    assert f"{downstream}" == f"{e_downstream}"
    assert f"{body}" == f"{e_body}"
    assert f"{unscaled3prime}" == f"{e_unscaled3prime}"
    assert f"{padRight}" == f"{e_padRight}"
    assert f"{_}" == f"{e_}"
    # + strand, 500 downstream
    downstream, body, unscaled3prime, padRight, _ = compute.chopRegions(region, left=500)
    e_body = []
    e_unscaled3prime = []
    e_padRight = 100
    e_ = 0
    assert f"{downstream}" == f"{region}"
    assert f"{body}" == f"{e_body}"
    assert f"{unscaled3prime}" == f"{e_unscaled3prime}"
    assert f"{padRight}" == f"{e_padRight}"
    assert f"{_}" == f"{e_}"
    # - strand, 250 downstream (labeled "upstream" due to being on the - strand)
    unscaled5prime, body, upstream, _, padLeft = compute.chopRegions(region, right=250)
    e_upstream = [(150, 200), (300, 400), (800, 900)]
    e_body = [(0, 150)]
    e_unscaled5prime = []
    e_padLeft = 0
    e_ = 0
    assert f"{upstream}" == f"{e_upstream}"
    assert f"{body}" == f"{e_body}"
    assert f"{unscaled5prime}" == f"{e_unscaled5prime}"
    assert f"{padLeft}" == f"{e_padLeft}"
    assert f"{_}" == f"{e_}"
    # - strand, 500 downstream (labeled "upstream" due to being on the - strand)
    unscaled5prime, body, upstream, _, padLeft = compute.chopRegions(region, right=500)
    e_body = []
    e_unscaled5prime = []
    e_padLeft = 100
    e_ = 0
    assert f"{upstream}" == f"{region}"
    assert f"{body}" == f"{e_body}"
    assert f"{unscaled5prime}" == f"{e_unscaled5prime}"
    assert f"{padLeft}" == f"{e_padLeft}"
    assert f"{_}" == f"{e_}"


def test_chopRegions_TES():
    region = [(0, 200), (300, 400), (800, 900)]
    # + strand, 250 upstream
    unscaled5prime, body, upstream, _, padLeft = compute.chopRegions(region, right=250)
    e_unscaled5prime = []
    e_body = [(0, 150)]
    e_upstream = [(150, 200), (300, 400), (800, 900)]
    e_ = 0
    e_padLeft = 0
    assert f"{unscaled5prime}" == f"{e_unscaled5prime}"
    assert f"{body}" == f"{e_body}"
    assert f"{upstream}" == f"{e_upstream}"
    assert f"{_}" == f"{e_}"
    assert f"{padLeft}" == f"{e_padLeft}"
    # + strand, 500 upstream
    unscaled5prime, body, upstream, _, padLeft = compute.chopRegions(region, right=500)
    e_unscaled5prime = []
    e_body = []
    e_ = 0
    e_padLeft = 100
    assert f"{unscaled5prime}" == f"{e_unscaled5prime}"
    assert f"{body}" == f"{e_body}"
    assert f"{upstream}" == f"{region}"
    assert f"{_}" == f"{e_}"
    assert f"{padLeft}" == f"{e_padLeft}"
    # + strand, 250 downstream (labeled "upstream" due to being on the - strand)
    downstream, body, unscaled3prime, padRight, _ = compute.chopRegions(region, left=250)
    e_downstream = [(0, 200), (300, 350)]
    e_body = [(350, 400), (800, 900)]
    e_unscaled3prime = []
    e_padRight = 0
    e_ = 0
    assert f"{downstream}" == f"{e_downstream}"
    assert f"{body}" == f"{e_body}"
    assert f"{unscaled3prime}" == f"{e_unscaled3prime}"
    assert f"{padRight}" == f"{e_padRight}"
    assert f"{_}" == f"{e_}"
    # + strand, 500 downstream (labeled "upstream" due to being on the - strand)
    downstream, body, unscaled3prime, padRight, _ = compute.chopRegions(region, left=500)
    e_body = []
    e_unscaled3prime = []
    e_padRight = 100
    e_ = 0
    assert f"{downstream}" == f"{region}"
    assert f"{body}" == f"{e_body}"
    assert f"{unscaled3prime}" == f"{e_unscaled3prime}"
    assert f"{padRight}" == f"{e_padRight}"
    assert f"{_}" == f"{e_}"


def test_chopRegionsFromMiddle():
    region = [(0, 200), (300, 400), (800, 900)]
    # + strand, 100 upstream/200 downstream
    upstream, downstream, padLeft, padRight = compute.chopRegionsFromMiddle(region, left=100, right=200)
    e_upstream = [(100, 200)]
    e_downstream = [(300, 400), (800, 900)]
    e_padLeft = 0
    e_padRight = 0
    assert f"{upstream}" == f"{e_upstream}"
    assert f"{downstream}" == f"{e_downstream}"
    assert f"{padLeft}" == f"{e_padLeft}"
    assert f"{padRight}" == f"{e_padRight}"
    # + strand, 250 upstream/300 downstream
    upstream, downstream, padLeft, padRight = compute.chopRegionsFromMiddle(region, left=250, right=300)
    e_upstream = [(0, 200)]
    e_downstream = [(300, 400), (800, 900)]
    e_padLeft = 50
    e_padRight = 100
    assert f"{upstream}" == f"{e_upstream}"
    assert f"{downstream}" == f"{e_downstream}"
    assert f"{padLeft}" == f"{e_padLeft}"
    assert f"{padRight}" == f"{e_padRight}"
    # - strand, 100 upstream/200 downstream
    upstream, downstream, padLeft, padRight = compute.chopRegionsFromMiddle(region, left=200, right=100)
    e_upstream = [(0, 200)]
    e_downstream = [(300, 400)]
    e_padLeft = 0
    e_padRight = 0
    assert f"{upstream}" == f"{e_upstream}"
    assert f"{downstream}" == f"{e_downstream}"
    assert f"{padLeft}" == f"{e_padLeft}"
    assert f"{padRight}" == f"{e_padRight}"
    # - strand, 250 upstream/300 downstream
    upstream, downstream, padLeft, padRight = compute.chopRegionsFromMiddle(region, left=300, right=250)
    e_upstream = [(0, 200)]
    e_downstream = [(300, 400), (800, 900)]
    e_padLeft = 100
    e_padRight = 50
    assert f"{upstream}" == f"{e_upstream}"
    assert f"{downstream}" == f"{e_downstream}"
    assert f"{padLeft}" == f"{e_padLeft}"
    assert f"{padRight}" == f"{e_padRight}"


def test_append_group_counts_uses_current_boundaries():
    matrix = Matrix(MatrixHeader.from_parameters({
        'group_labels': ['first', 'second'],
        'group_boundaries': [0, 3, 1237],
        'sample_labels': ['sample'], 'sample_boundaries': [0, 1]}),
        np.empty((1237, 1)), [], None)
    labels = resolve_labels(RowLayout.identity(matrix), show_counts=True,
                            header=matrix.header)
    assert labels.groups == ('first [n = 3]', 'second [n = 1,234]')


def test_append_group_counts_leaves_hidden_labels_empty():
    """A group label of '' is an explicit request to hide it (--regionsLabel
    ""); --showRegionCounts must not turn that back into a non-empty
    ' [n = ...]' string, which would defeat the hiding and make plotHeatmap
    reserve real layout space for a label the user asked to suppress."""
    matrix = Matrix(MatrixHeader.from_parameters({
        'group_labels': ['first', ''],
        'group_boundaries': [0, 3, 1237],
        'sample_labels': ['sample'], 'sample_boundaries': [0, 1]}),
        np.empty((1237, 1)), [], None)
    labels = resolve_labels(RowLayout.identity(matrix), show_counts=True,
                            header=matrix.header)
    assert labels.groups == ('first [n = 3]', '')
