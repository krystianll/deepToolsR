from deeptoolsr import computeMatrixOperations as coverage_cmo
from deeptoolsr import computeMatrix
from tests.helpers.bigwig import pyBigWig
from deeptoolsr import computeMatrixOperations as io_cmo
import bz2
from deeptoolsr import computeMatrixOperations as general_cmo
from tests.helpers.parity import read_matrix
from deeptoolsr.matrix import RowLayout
from deeptoolsr import stats as kernels
import math
from deeptoolsr.matrix import merge_groups_by_key
from deeptoolsr import stats as profile
from deeptoolsr import computeMatrixOperations
from tests.helpers.parity import original
from tests.test_matrix_validation import _header
from deeptoolsr import computeMatrixOperations as validation_cmo
from pathlib import Path
# from unittest import TestCase

import deeptoolsr.computeMatrixOperations as cmo
import os
import hashlib
import gzip
import json
import numpy as np
import pytest
from types import SimpleNamespace

from deeptoolsr.matrix import Matrix, MatrixHeader, OwnedMatrix
from deeptoolsr import matrix as matrix_module

__author__ = 'Devon'


def getHeader(fp):
    s = fp.readline()
    if isinstance(s, bytes):
        s = s.decode()
    s = s[1:]
    return json.loads(s)


class TestComputeMatrixOperations(object):
    root = os.path.dirname(os.path.abspath(__file__)) + "/test_data/"
    matrix = root + "computeMatrixOperations.mat.gz"
    bed = root + "computeMatrixOperations.bed"
    rbindMatrix1 = root + "somegenes.txt.gz"
    rbindMatrix2 = root + "othergenes.txt.gz"

    def _transform_matrix(self, path, values):
        matrix = OwnedMatrix.load(self.rbindMatrix1, 1)
        matrix.values = np.asarray(values, dtype=np.float32)
        matrix.save(str(path), compressed=True, threads=1)

    def _update_header(self, path, **changes):
        with gzip.open(path, 'rt') as handle:
            lines = handle.readlines()
        header = json.loads(lines[0][1:])
        header.update(changes)
        lines[0] = '@' + json.dumps(header, separators=(',', ':')) + '\n'
        with gzip.open(path, 'wt') as handle:
            handle.writelines(lines)

    def testTransformOrderedPipeline(self, tmp_path):
        first = tmp_path / 'first.gz'
        second = tmp_path / 'second.gz'
        output = tmp_path / 'output.gz'
        self._transform_matrix(first, [[1, 2, 3, 4]] * 3)
        self._transform_matrix(second, [[3, 4, 5, 6]] * 3)

        cmo.main(['transform', '-m', str(first), str(second),
                  '--add', '5', '--scale', '2', '1', '--log2FC',
                  '--scale', '-1', '-o', str(output)])
        result = Matrix.load(str(output), 1)
        expected = -np.log2((np.array([1, 2, 3, 4]) + 5) * 2 /
                            (np.array([3, 4, 5, 6]) + 5))
        np.testing.assert_allclose(result.values,
                                   np.tile(expected, (3, 1)), atol=1e-6)

    def testTransformCanProduceMultipleOutputs(self, tmp_path):
        first = tmp_path / 'first.gz'
        second = tmp_path / 'second.gz'
        outputs = [tmp_path / 'out1.gz', tmp_path / 'out2.gz']
        self._transform_matrix(first, [[1, 2, 3, 4]] * 3)
        self._transform_matrix(second, [[10, 20, 30, 40]] * 3)
        cmo.main(['transform', '-m', str(first), str(second),
                  '--add', '5', '-o', *(str(path) for path in outputs)])
        for output, expected in zip(outputs, ([6, 7, 8, 9],
                                              [15, 25, 35, 45])):
            result = Matrix.load(str(output), 1)
            np.testing.assert_allclose(result.values,
                                       np.tile(expected, (3, 1)))

    def testTransformStages_all_outputs_before_committing(self, tmp_path,
                                                          monkeypatch):
        first = tmp_path / 'first.gz'
        second = tmp_path / 'second.gz'
        outputs = [tmp_path / 'out1.gz', tmp_path / 'out2.gz']
        self._transform_matrix(first, [[1, 2, 3, 4]] * 3)
        self._transform_matrix(second, [[10, 20, 30, 40]] * 3)
        for index, output in enumerate(outputs):
            output.write_bytes(('old-{}'.format(index)).encode())

        original = matrix_module._compute_matrix_io.write_matrix
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                with open(args[0], 'wb') as handle:
                    handle.write(b'partial')
                raise RuntimeError('injected second-output failure')
            return original(*args, **kwargs)

        monkeypatch.setattr(matrix_module._compute_matrix_io,
                            'write_matrix', fail_second)
        with pytest.raises(SystemExit, match='second-output failure'):
            cmo.main(['transform', '-m', str(first), str(second),
                      '--add', '1', '-o', *(str(path) for path in outputs)])
        assert outputs[0].read_bytes() == b'old-0'
        assert outputs[1].read_bytes() == b'old-1'

    def testTransformRejectsOperationAfterReduction(self, tmp_path):
        first = tmp_path / 'first.gz'
        second = tmp_path / 'second.gz'
        self._transform_matrix(first, [[1, 2, 3, 4]] * 3)
        self._transform_matrix(second, [[2, 3, 4, 5]] * 3)
        with pytest.raises(SystemExit, match='requires exactly two.*1 remain'):
            cmo.main(['transform', '-m', str(first), str(second),
                      '--log2FC', '--ratio', '-o', str(tmp_path / 'out.gz')])

    def testTransformPseudocountExcludesZeroAndZeroHandlingPrecedesIt(self, tmp_path):
        first = tmp_path / 'first.gz'
        second = tmp_path / 'second.gz'
        keep = tmp_path / 'keep.gz'
        masked = tmp_path / 'masked.gz'
        self._transform_matrix(first, [[2, 4, 6, 8]] * 3)
        self._transform_matrix(second, [[0, 2, 4, 6]] * 3)
        base = ['transform', '-m', str(first), str(second), '--ratio',
                '--pseudocount', 'min-denominator']
        cmo.main(base + ['-o', str(keep)])
        cmo.main(base + ['--zeroHandling', 'any', '-o', str(masked)])
        keep_hm = Matrix.load(str(keep), 1)
        masked_hm = Matrix.load(str(masked), 1)
        assert keep_hm.values[0, 0] == 2  # (2 + 2) / (0 + 2)
        missing = masked_hm.values[0, 0]
        assert np.ma.is_masked(missing) or np.isnan(missing)

    def testResolvePseudocountsUsesEachSampleIndependently(self):
        numerator = np.ma.array([[2, 4, 20, 40], [0, 8, 0, 80]])
        denominator = np.ma.array([[1, 3, 10, 30], [0, 6, 0, 60]])
        boundaries = [0, 2, 4]

        assert cmo.resolvePseudocounts(
            'min-denominator', numerator, denominator, boundaries) == [1, 10]
        assert cmo.resolvePseudocounts(
            'min-numerator/2', numerator, denominator, boundaries) == [1, 10]
        assert cmo.resolvePseudocounts(
            '2.5', numerator, denominator, boundaries) == [2.5, 2.5]

    def testTransformMissingDataAsZeroAndBothZeroPolicy(self, tmp_path):
        first = tmp_path / 'first.gz'
        second = tmp_path / 'second.gz'
        output = tmp_path / 'output.gz'
        self._transform_matrix(first, [[np.nan, np.nan, 2, 4]] * 3)
        self._transform_matrix(second, [[np.nan, 2, 0, 4]] * 3)
        cmo.main(['transform', '-m', str(first), str(second), '--ratio',
                  '--pseudocount', '1', '--missingDataAsZero',
                  '--zeroHandling', 'both', '-o', str(output)])
        result = Matrix.load(str(output), 1)
        values = result.values[0]
        assert np.ma.is_masked(values[0]) or np.isnan(values[0])
        assert not np.isnan(values[1:]).any()
        np.testing.assert_allclose(values[1:], [1 / 3, 3, 1], atol=1e-6)

    def testTransformRowScaleUsingUpstream(self, tmp_path):
        source = tmp_path / 'source.gz'
        output = tmp_path / 'output.gz'
        self._transform_matrix(source, [[2, 4, 6, 8],
                                        [4, 8, 12, 16],
                                        [1, 2, 3, 4]])
        cmo.main(['transform', '-m', str(source), '--scale', 'row-sum',
                  '--scaleUsing', 'upstream', '-o', str(output)])
        result = Matrix.load(str(output), 1)
        np.testing.assert_allclose(result.values,
                                   [[1, 2, 3, 4],
                                    [1, 2, 3, 4],
                                    [1, 2, 3, 4]])

    def testTransformUnaryOperationsAllowDifferentLocusLayouts(self, tmp_path):
        first = tmp_path / 'first.gz'
        second = tmp_path / 'second.gz'
        outputs = [tmp_path / 'out1.gz', tmp_path / 'out2.gz']
        self._transform_matrix(first, [[1, 2, 3, 4]] * 3)
        self._transform_matrix(second, [[5, 6, 7, 8]] * 3)
        self._update_header(second, upstream=[2], body=[1], downstream=[1])
        cmo.main(['transform', '-m', str(first), str(second), '--add', '1',
                  '-o', *(str(path) for path in outputs)])
        assert all(path.exists() for path in outputs)

    def testTransformCombinationChecksHeadersBeforeReadingRows(self,
                                                               tmp_path,
                                                               monkeypatch):
        first = tmp_path / 'first.gz'
        second = tmp_path / 'second.gz'
        self._transform_matrix(first, [[1, 2, 3, 4]] * 3)
        self._transform_matrix(second, [[5, 6, 7, 8]] * 3)
        self._update_header(second, upstream=[2], body=[1], downstream=[1])

        def rows_must_not_be_read(*args, **kwargs):
            raise AssertionError('matrix rows were read before preflight')

        monkeypatch.setattr(OwnedMatrix, 'load', rows_must_not_be_read)
        monkeypatch.setattr(Matrix, 'load', rows_must_not_be_read)
        with pytest.raises(SystemExit, match='requires compatible.*upstream'):
            cmo.main(['transform', '-m', str(first), str(second), '--ratio',
                      '-o', str(tmp_path / 'out.gz')])

    def testTransformCombinationIgnoresSampleNames(self, tmp_path):
        first = tmp_path / 'first.gz'
        second = tmp_path / 'second.gz'
        output = tmp_path / 'output.gz'
        self._transform_matrix(first, [[2, 4, 6, 8]] * 3)
        self._transform_matrix(second, [[1, 2, 3, 4]] * 3)
        self._update_header(second, sample_labels=['different sample'])
        cmo.main(['transform', '-m', str(first), str(second), '--ratio',
                  '-o', str(output)])
        result = Matrix.load(str(output), 1)
        np.testing.assert_allclose(result.values, 2)

    def testTransformCombinationDoesNotRelabelDifferentGroups(self, tmp_path):
        first = tmp_path / 'first.gz'
        second = tmp_path / 'second.gz'
        output = tmp_path / 'output.gz'
        self._transform_matrix(first, [[2, 4, 6, 8]] * 3)
        self._transform_matrix(second, [[1, 2, 3, 4]] * 3)
        self._update_header(second, group_labels=['different group'])
        cmo.main(['transform', '-m', str(first), str(second), '--ratio',
                  '-o', str(output)])
        result = Matrix.load(str(output), 1)
        assert np.isnan(np.ma.filled(result.values, np.nan)).all()

    def testTransformScaleUsingRequiresSegmentButNotEqualSize(self, tmp_path):
        first = tmp_path / 'first.gz'
        second = tmp_path / 'second.gz'
        outputs = [tmp_path / 'out1.gz', tmp_path / 'out2.gz']
        self._transform_matrix(first, [[1, 2, 3, 4]] * 3)
        self._transform_matrix(second, [[5, 6, 7, 8]] * 3)
        self._update_header(second, upstream=[2], body=[1], downstream=[1])
        cmo.main(['transform', '-m', str(first), str(second),
                  '--scale', 'row-mean', '--scaleUsing', 'upstream',
                  '-o', *(str(path) for path in outputs)])
        assert all(path.exists() for path in outputs)

        self._update_header(second, upstream=[0], body=[3], downstream=[1])
        with pytest.raises(SystemExit, match='requires that segment'):
            cmo.main(['transform', '-m', str(first), str(second),
                      '--scale', 'row-mean', '--scaleUsing', 'upstream',
                      '-o', *(str(path) for path in outputs)])

    def testSubsetSupportsOneBasedIndicesAndReorderAlias(self, tmp_path):
        output = tmp_path / 'reordered.mat.gz'
        cmo.main(['reorder', '-m', self.matrix, '-o', str(output),
                  '--groups', '1', '--samples', '2', '1'])
        hm = Matrix.load(str(output), 1)
        assert hm.header.group_labels == ('genes',)
        assert hm.header.sample_labels == ('SRR648668.forward',
                                           'SRR648667.forward')

    def testMergeSameNamedGroupsBeforeSubset(self):
        matrix = OwnedMatrix.load(self.rbindMatrix1, 1)
        original_rows = matrix.values.shape[0]
        matrix.values = np.concatenate([matrix.values, matrix.values], axis=0)
        matrix.regions.extend(list(matrix.regions))
        parameters = dict(matrix.header.parameters)
        parameters['group_labels'] = ['same', 'same']
        parameters['group_boundaries'] = [0, original_rows, 2 * original_rows]
        matrix.header = MatrixHeader.from_parameters(parameters)

        cmo.mergeSameNamedGroups(matrix)

        assert matrix.header.group_labels == ('same',)
        assert matrix.header.group_boundaries == (0, 2 * original_rows)
        assert matrix.values.shape[0] == 2 * original_rows

    def testRbindGroupPolicies(self, tmp_path):
        first = tmp_path / 'first.gz'
        second = tmp_path / 'second.gz'
        for path in (first, second):
            matrix = OwnedMatrix.load(self.rbindMatrix1, 1)
            rows = matrix.values.shape[0]
            matrix.values = np.concatenate([matrix.values, matrix.values], axis=0)
            matrix.regions.extend(list(matrix.regions))
            parameters = dict(matrix.header.parameters)
            parameters['group_labels'] = ['cluster_1', 'cluster_2']
            parameters['group_boundaries'] = [0, rows, 2 * rows]
            matrix.header = MatrixHeader.from_parameters(parameters)
            matrix.save(str(path), compressed=True, threads=1)

        expected = {
            'separate': ['cluster_1', 'cluster_2', 'cluster_1', 'cluster_2'],
            'merge': ['cluster_1', 'cluster_2'],
        }
        for policy, labels in expected.items():
            output = tmp_path / f'{policy}.gz'
            cmo.main(['rbind', '-m', str(first), str(second),
                      '--sameGroupLabels', policy, '-o', str(output)])
            result = Matrix.load(output, 1)
            assert list(result.header.group_labels) == labels

    def testRbindGroupPolicyDefaultsToMerge(self):
        args = cmo.parse_arguments().parse_args([
            'rbind', '-m', 'one.gz', 'two.gz', '-o', 'output.gz'])
        assert args.sameGroupLabels == 'merge'

    def testSubset(self, tmp_path):
        """
        computeMatrixOperations subset
        """

        dCorrect = {"verbose": True, "scale": 1, "skip zeros": False, "nan after end": False, "sort using": "mean", "unscaled 5 prime": [0, 0, 0, 0], "body": [1000, 1000, 1000, 1000], "sample_labels": ["SRR648667.forward", "SRR648668.forward", "SRR648669.forward", "SRR648670.forward"], "downstream": [0, 0, 0, 0], "unscaled 3 prime": [0, 0, 0, 0], "group_labels": ["genes"], "bin size": [10, 10, 10, 10], "upstream": [0, 0, 0, 0], "group_boundaries": [0, 196], "sample_boundaries": [0, 100, 200, 300, 400], "max threshold": None, "ref point": [None, None, None, None], "min threshold": None, "sort regions": "no", "proc number": 20, "bin avg type": "mean", "missing data as zero": False}
        oname = str(tmp_path / "subset.mat.gz")
        args = "subset -m {} --sample SRR648667.forward SRR648668.forward SRR648669.forward SRR648670.forward -o {}".format(self.matrix, oname)
        args = args.split()
        cmo.main(args)
        f = gzip.GzipFile(oname)
        d = getHeader(f)  # Skip the header, which can be in a different order
        h = hashlib.md5(f.read()).hexdigest()
        f.close()
        assert d == dCorrect
        expectedh = 'edb3c8506c3f27ebb8c7ddf94d5ba594'
        assert f'{h}' == f'{expectedh}'
        os.remove(oname)

    def testRelabel(self, tmp_path):
        """
        computeMatrixOperations relabel
        """
        dCorrect = {"verbose": True, "scale": 1, "skip zeros": False, "nan after end": False, "sort using": "mean", "unscaled 5 prime": [0, 0, 0, 0, 0, 0, 0, 0], "body": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], "sample_labels": ["first", "sec ond", "3rd", "4th", "5th", "6th", "7th", "8th"], "downstream": [0, 0, 0, 0, 0, 0, 0, 0], "unscaled 3 prime": [0, 0, 0, 0, 0, 0, 0, 0], "group_labels": ["foo bar"], "bin size": [10, 10, 10, 10, 10, 10, 10, 10], "upstream": [0, 0, 0, 0, 0, 0, 0, 0], "group_boundaries": [0, 196], "sample_boundaries": [0, 100, 200, 300, 400, 500, 600, 700, 800], "max threshold": None, "ref point": [None, None, None, None, None, None, None, None], "min threshold": None, "sort regions": "no", "proc number": 20, "bin avg type": "mean", "missing data as zero": False}
        oname = str(tmp_path / "relabeled.mat.gz")
        args = "relabel -m {} -o {} --sampleLabels first sec_ond 3rd 4th 5th 6th 7th 8th --groupLabels foo_bar".format(self.matrix, oname)
        args = args.split()
        args[7] = 'sec ond'  # split mucks up spaces
        args[-1] = 'foo bar'
        cmo.main(args)
        f = gzip.GzipFile(oname)
        d = getHeader(f)
        assert d == dCorrect
        f.close()
        os.remove(oname)

    def testfilterStrand(self, tmp_path):
        """
        computeMatrixOperations filterStrand
        """
        dCorrect = {"verbose": True, "scale": 1, "skip zeros": False, "nan after end": False, "sort using": "mean", "unscaled 5 prime": [0, 0, 0, 0, 0, 0, 0, 0], "body": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], "sample_labels": ["SRR648667.forward", "SRR648668.forward", "SRR648669.forward", "SRR648670.forward", "SRR648667.reverse", "SRR648668.reverse", "SRR648669.reverse", "SRR648670.reverse"], "downstream": [0, 0, 0, 0, 0, 0, 0, 0], "unscaled 3 prime": [0, 0, 0, 0, 0, 0, 0, 0], "group_labels": ["genes"], "bin size": [10, 10, 10, 10, 10, 10, 10, 10], "upstream": [0, 0, 0, 0, 0, 0, 0, 0], "group_boundaries": [0, 107], "sample_boundaries": [0, 100, 200, 300, 400, 500, 600, 700, 800], "max threshold": None, "ref point": [None, None, None, None, None, None, None, None], "min threshold": None, "sort regions": "no", "proc number": 20, "bin avg type": "mean", "missing data as zero": False}
        oname = str(tmp_path / "filterStrand1.mat.gz")
        args = "filterStrand -m {} -o {} --strand +".format(self.matrix, oname)
        args = args.split(' ')
        cmo.main(args)
        f = gzip.GzipFile(oname)
        d = getHeader(f)  # Skip the header, which can be in a different order
        h = hashlib.md5(f.read()).hexdigest()
        f.close()
        assert d == dCorrect
        expectedh = '300f8000be5b5f51e803b57ef08f1c9e'
        assert f'{h}' == f'{expectedh}'
        os.remove(oname)

        dCorrect = {u'verbose': True, u'scale': 1, u'skip zeros': False, u'nan after end': False, u'sort using': u'mean', u'unscaled 5 prime': [0, 0, 0, 0, 0, 0, 0, 0], u'body': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], u'sample_labels': [u'SRR648667.forward', u'SRR648668.forward', u'SRR648669.forward', u'SRR648670.forward', u'SRR648667.reverse', u'SRR648668.reverse', u'SRR648669.reverse', u'SRR648670.reverse'], u'downstream': [0, 0, 0, 0, 0, 0, 0, 0], u'unscaled 3 prime': [0, 0, 0, 0, 0, 0, 0, 0], u'group_labels': [u'genes'], u'bin size': [10, 10, 10, 10, 10, 10, 10, 10], u'upstream': [0, 0, 0, 0, 0, 0, 0, 0], u'group_boundaries': [0, 89], u'sample_boundaries': [0, 100, 200, 300, 400, 500, 600, 700, 800], u'missing data as zero': False, u'ref point': [None, None, None, None, None, None, None, None], u'min threshold': None, u'sort regions': u'no', u'proc number': 20, u'bin avg type': u'mean', u'max threshold': None}
        oname = str(tmp_path / "filterStrand2.mat.gz")
        args = "filterStrand -m {} -o {} --strand -".format(self.matrix, oname)
        args = args.split()
        cmo.main(args)
        f = gzip.GzipFile(oname)
        d = getHeader(f)  # Skip the header, which can be in a different order
        h = hashlib.md5(f.read()).hexdigest()
        f.close()
        assert d == dCorrect
        expectedh = '0a6ca070a5ba4564f1ab950ac3b7c8f1'
        assert f'{h}' == f'{expectedh}'
        os.remove(oname)

    def testrbind(self, tmp_path):
        """
        computeMatrixOperations rbind
        """
        dCorrect = {"verbose": True, "scale": 1, "skip zeros": False, "nan after end": False, "sort using": "mean", "unscaled 5 prime": [0, 0, 0, 0, 0, 0, 0, 0], "body": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], "sample_labels": ["SRR648667.forward", "SRR648668.forward", "SRR648669.forward", "SRR648670.forward", "SRR648667.reverse", "SRR648668.reverse", "SRR648669.reverse", "SRR648670.reverse"], "downstream": [0, 0, 0, 0, 0, 0, 0, 0], "unscaled 3 prime": [0, 0, 0, 0, 0, 0, 0, 0], "group_labels": ["genes"], "bin size": [10, 10, 10, 10, 10, 10, 10, 10], "upstream": [0, 0, 0, 0, 0, 0, 0, 0], "group_boundaries": [0, 392], "sample_boundaries": [0, 100, 200, 300, 400, 500, 600, 700, 800], "max threshold": None, "ref point": [None, None, None, None, None, None, None, None], "min threshold": None, "sort regions": "no", "proc number": 20, "bin avg type": "mean", "missing data as zero": False}
        oname = str(tmp_path / "rbind.mat.gz")
        args = "rbind -m {0} {0} -o {1}".format(self.matrix, oname)
        args = args.split()
        cmo.main(args)
        f = gzip.GzipFile(oname)
        d = getHeader(f)  # Skip the header, which can be in a different order
        h = hashlib.md5(f.read()).hexdigest()
        f.close()
        assert d == dCorrect
        expectedh = '3dd96c7b05e0ca5ada21212defe57fba'
        assert f'{h}' == f'{expectedh}'
        os.remove(oname)

    def testrbind2(self, tmp_path):
        """
        computeMatrixOperations rbind with different groups
        """
        dCorrect = {"verbose": False, "scale": 1, "skip zeros": False, "nan after end": False, "sort using": "mean", "unscaled 5 prime": [0], "body": [2], "sample_labels": ["signal"], "downstream": [1], "unscaled 3 prime": [0], "group_labels": ["somegenes", "othergenes"], "bin size": [1], "upstream": [1], "group_boundaries": [0, 3, 7], "sample_boundaries": [0, 4], "max threshold": None, "ref point": [None], "min threshold": None, "sort regions": "keep", "proc number": 1, "bin avg type": "mean", "missing data as zero": True}
        oname = str(tmp_path / "rbind2.mat.gz")
        args = "rbind -m {0} {1} -o {2}".format(self.rbindMatrix1, self.rbindMatrix2, oname)
        args = args.split()
        cmo.main(args)
        f = gzip.GzipFile(oname)
        d = getHeader(f)  # Skip the header, which can be in a different order
        h = hashlib.md5(f.read()).hexdigest()
        f.close()
        assert d == dCorrect
        # Distinct groups stream (no merge), copying data rows verbatim rather
        # than re-formatting them through the writer, so the body bytes differ
        # from the old in-memory path while the matrix is identical.
        expectedh = '2b6467153e5cbb1778de17893c3c9bb4'
        assert f'{h}' == f'{expectedh}'
        os.remove(oname)

    def testcbind(self, tmp_path):
        """
        computeMatrixOperations cbind
        """
        dCorrect = {"verbose": True, "scale": 1, "skip zeros": False, "nan after end": False, "sort using": "mean", "unscaled 5 prime": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "body": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], "sample_labels": ["SRR648667.forward", "SRR648668.forward", "SRR648669.forward", "SRR648670.forward", "SRR648667.reverse", "SRR648668.reverse", "SRR648669.reverse", "SRR648670.reverse", "SRR648667.forward", "SRR648668.forward", "SRR648669.forward", "SRR648670.forward", "SRR648667.reverse", "SRR648668.reverse", "SRR648669.reverse", "SRR648670.reverse"], "downstream": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "unscaled 3 prime": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "group_labels": ["genes"], "bin size": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "upstream": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "group_boundaries": [0, 196], "sample_boundaries": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600], "max threshold": None, "ref point": [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], "min threshold": None, "sort regions": "no", "proc number": 20, "bin avg type": "mean", "missing data as zero": False}
        oname = str(tmp_path / "filterStrand.mat.gz")
        args = "cbind -m {0} {0} -o {1}".format(self.matrix, oname)
        args = args.split()
        cmo.main(args)
        f = gzip.GzipFile(oname)
        d = getHeader(f)  # Skip the header, which can be in a different order
        h = hashlib.md5(f.read()).hexdigest()
        f.close()
        assert d == dCorrect
        expectedh = 'e55d89704bb16a11f366663a8fd90a47'
        assert f'{h}' == f'{expectedh}'
        os.remove(oname)

    def testsort(self, tmp_path):
        """
        computeMatrixOperations sort
        """
        dCorrect = {"verbose": True, "scale": 1, "skip zeros": False, "nan after end": False, "sort using": "mean", "unscaled 5 prime": [0, 0, 0, 0, 0, 0, 0, 0], "body": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], "sample_labels": ["SRR648667.forward", "SRR648668.forward", "SRR648669.forward", "SRR648670.forward", "SRR648667.reverse", "SRR648668.reverse", "SRR648669.reverse", "SRR648670.reverse"], "downstream": [0, 0, 0, 0, 0, 0, 0, 0], "unscaled 3 prime": [0, 0, 0, 0, 0, 0, 0, 0], "group_labels": ["genes"], "bin size": [10, 10, 10, 10, 10, 10, 10, 10], "upstream": [0, 0, 0, 0, 0, 0, 0, 0], "group_boundaries": [0, 196], "sample_boundaries": [0, 100, 200, 300, 400, 500, 600, 700, 800], "max threshold": None, "ref point": [None, None, None, None, None, None, None, None], "min threshold": None, "sort regions": "no", "proc number": 20, "bin avg type": "mean", "missing data as zero": False}
        oname = str(tmp_path / "sorted.mat.gz")
        args = "sort -m {} -o {} -R {}".format(self.matrix, oname, self.bed)
        args = args.split()
        cmo.main(args)
        f = gzip.GzipFile(oname)
        d = getHeader(f)  # Skip the header, which can be in a different order
        h = hashlib.md5(f.read()).hexdigest()
        f.close()
        assert d == dCorrect
        expectedh = '10ea07d1aa58f44625abe2142ef76094'
        assert f'{h}' == f'{expectedh}'
        os.remove(oname)


def test_printinfo_reports_enriched_summary(capsys):
    data = np.arange(24, dtype=np.float32).reshape(4, 6)
    regions = [['c', [(i, i + 1)], 'r%d' % i, 0, '+', '0'] for i in range(4)]
    parameters = {
        'upstream': [1000, 1000], 'body': [0, 0], 'downstream': [2000, 2000],
        'unscaled 5 prime': [0, 0], 'unscaled 3 prime': [500, 500],
        'ref point': ['TSS', 'TSS'], 'bin size': [10, 10],
        'sort regions': 'descend', 'sort using': 'mean', 'scale': 1,
        'scale minus': -1.0, 'transform operations': [['log2', None]],
        'group_labels': ['A', 'B'], 'group_boundaries': [0, 2, 4],
        'sample_labels': ['s1', 's2'], 'sample_boundaries': [0, 3, 6],
    }
    cmo.printInfo(OwnedMatrix.from_compute(parameters, data, regions))
    out = capsys.readouterr().out
    assert 'Samples: 2' in out and 'Regions: 4' in out
    assert 'A: 2' in out and 'B: 2' in out            # per-group counts
    assert 'bins: 3' in out                            # per-sample bins
    assert 'upstream 1000' in out and 'downstream 2000' in out
    assert 'body' not in out                           # zero body omitted
    assert "unscaled 3' 500" in out                    # non-zero unscaled kept
    assert 'reference point: TSS' in out
    assert 'Sorted: descend by mean' in out
    assert 'scale minus: -1.0' in out
    assert 'Transformations:' in out and 'log2' in out


def _make_small_matrix(path):
    data = np.arange(24, dtype=np.float32).reshape(4, 6)
    regions = [['chr1', [(i, i + 1)], 'r%d' % i, 0, '+', '0'] for i in range(4)]
    parameters = {
        'upstream': [10, 10], 'downstream': [10, 10], 'body': [0, 0],
        'unscaled 5 prime': [0, 0], 'unscaled 3 prime': [0, 0],
        'ref point': ['TSS', 'TSS'], 'bin size': [1, 1],
        'sort regions': 'keep', 'sort using': 'mean',
        'sample_labels': ['s1', 's2'], 'group_labels': ['A', 'B'],
        'sample_boundaries': [0, 3, 6], 'group_boundaries': [0, 2, 4],
    }
    OwnedMatrix.from_compute(parameters, data, regions).save(
        path, compressed=True, threads=1)


def test_stream_rbind_matches_in_memory(tmp_path):
    src = str(tmp_path / 'small.gz')
    _make_small_matrix(src)

    streamed = str(tmp_path / 'stream.gz')
    assert cmo.stream_rbind(streamed, [src, src], 'separate') is True

    matrix = OwnedMatrix.load(src, 1)
    cmo.rbindMatrices(matrix, SimpleNamespace(matrixFile=[src, src],
                                              sameGroupLabels='separate',
                                              threads=1))
    resident = str(tmp_path / 'mem.gz')
    matrix.save(resident, compressed=True, threads=1)

    a = Matrix.load(streamed, 1)
    b = Matrix.load(resident, 1)
    assert a.header.group_boundaries == b.header.group_boundaries == (0, 2, 4, 6, 8)
    assert a.header.group_labels == b.header.group_labels == ('A', 'B', 'A', 'B')
    assert np.array_equal(np.ma.filled(a.values, np.nan),
                          np.ma.filled(b.values, np.nan), equal_nan=True)
    assert [r[2] for r in a.regions] == [r[2] for r in b.regions]


def test_stream_rbind_falls_back_on_merge(tmp_path):
    src = str(tmp_path / 'small.gz')
    _make_small_matrix(src)
    # Overlapping group labels under merge need interleaving -> not streamable.
    assert cmo.stream_rbind(str(tmp_path / 'out.gz'), [src, src], 'merge') is False


def test_stream_relabel_targeted_and_full(tmp_path):
    src = str(tmp_path / 'small.gz')
    _make_small_matrix(src)
    ref = Matrix.load(src, 1)
    ref_matrix = np.ma.filled(ref.values, np.nan)

    # Targeted: group by name, sample by 1-based index; the rest unchanged.
    out = str(tmp_path / 'rl.gz')
    cmo.main(['relabel', '-m', src, '-o', out,
              '--setGroupLabel', 'A', 'A2', '--setSampleLabel', '2', 's2new'])
    hm = Matrix.load(out, 1)
    assert hm.header.group_labels == ('A2', 'B')
    assert hm.header.sample_labels == ('s1', 's2new')
    # Body is untouched.
    assert np.array_equal(np.ma.filled(hm.values, np.nan), ref_matrix,
                          equal_nan=True)

    # Full replacement still works.
    out2 = str(tmp_path / 'rl2.gz')
    cmo.main(['relabel', '-m', src, '-o', out2,
              '--groupLabels', 'X', 'Y', '--sampleLabels', 'p', 'q'])
    hm2 = Matrix.load(out2, 1)
    assert hm2.header.group_labels == ('X', 'Y')
    assert hm2.header.sample_labels == ('p', 'q')


def test_datarange_native_matches_numpy(tmp_path, capsys):
    src = str(tmp_path / 'small.gz')
    _make_small_matrix(src)
    hm = Matrix.load(src, 1)
    cmo.printDataRange(Matrix.load(src, 1), threads=1)
    lines = capsys.readouterr().out.strip().splitlines()
    m = hm.values
    for row in lines[1:]:                       # skip header
        name, *vals = row.split('\t')
        i = hm.header.sample_labels.index(name)
        a, b = hm.header.sample_boundaries[i], hm.header.sample_boundaries[i + 1]
        f = np.ma.filled(m[:, a:b], np.nan)
        expected = [np.nanmin(f), np.nanmax(f), np.nanmedian(f),
                    np.nanpercentile(f, 10), np.nanpercentile(f, 90)]
        assert np.allclose([float(v) for v in vals], expected, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Chunked-streaming filters (filterStrand / filterValues / subset)
# ---------------------------------------------------------------------------

_OPS_MATRIX = os.path.dirname(os.path.abspath(__file__)) + \
    "/test_data/computeMatrixOperations.mat.gz"


def _load(path):
    return Matrix.load(str(path), 1)


def _same_matrix(a, b):
    return (a.header.group_boundaries == b.header.group_boundaries
            and a.header.sample_boundaries == b.header.sample_boundaries
            and a.header.group_labels == b.header.group_labels
            and a.header.sample_labels == b.header.sample_labels
            and [r[2] for r in a.regions] == [r[2] for r in b.regions]
            and np.allclose(np.ma.filled(a.values, np.nan),
                            np.ma.filled(b.values, np.nan), equal_nan=True))


def _run_inmemory(argv):
    """Compare streaming filters with the native resident row policy."""
    if argv[0] == 'filterValues':
        args = cmo.parse_arguments().parse_args(argv)
        from deeptoolsr import stats
        matrix = OwnedMatrix.load(args.matrixFile, 1)
        indices = cmo.resolveLabels(args.filterUsingSamples,
                                    matrix.header.sample_labels, 'sample')
        keep, masked = stats.filter_matrix(
            matrix.values, matrix.header.sample_boundaries, indices,
            args.filterUsingStatistic,
            -np.inf if args.min is None else args.min,
            np.inf if args.max is None else args.max,
            nan_mode=cmo._FILTER_NAN_MODES[args.filterNans],
            on_fail=args.onFilterFail, threads=1)
        if masked is not None:
            matrix.values = masked
        else:
            bounds = [0]
            for start, end in zip(matrix.header.group_boundaries,
                                  matrix.header.group_boundaries[1:]):
                bounds.append(bounds[-1] + int(np.count_nonzero(keep[start:end])))
            matrix.regions = [region for region, take in zip(matrix.regions, keep)
                              if take]
            matrix.values = matrix.values[keep, :]
            parameters = dict(matrix.header.parameters)
            parameters['group_boundaries'] = bounds
            matrix.header = MatrixHeader.from_parameters(parameters)
        matrix.save(args.outFileName, compressed=True, threads=1)
        return
    if '--groups' not in argv:
        source = argv[argv.index('-m') + 1]
        argv = [*argv, '--groups', *_load(source).header.group_labels]
    cmo.main(argv)


@pytest.mark.parametrize("argv", [
    ["filterValues", "--min", "5", "--max", "500"],
    ["filterValues", "--min", "5"],
    ["filterValues", "--max", "500"],
    ["filterValues"],
    ["subset", "--samples", "SRR648667.forward", "SRR648669.forward"],
    ["subset", "--samples", "SRR648669.reverse", "SRR648667.forward"],  # reorder
    ["reorder", "--samples", "4", "3", "2", "1"],                       # by index
    ["subset"],                                                          # identity
])
def test_stream_ops_match_in_memory(tmp_path, argv):
    streamed = str(tmp_path / "stream.gz")
    resident = str(tmp_path / "mem.gz")
    base = ["-m", _OPS_MATRIX]
    cmo.main([argv[0]] + base + ["-o", streamed] + argv[1:])
    _run_inmemory([argv[0]] + base + ["-o", resident] + argv[1:])
    assert _same_matrix(_load(streamed), _load(resident))


@pytest.mark.parametrize("argv", [
    ["filterStrand", "--strand", "+"],
    ["filterValues", "--min", "5", "--max", "500"],
    ["subset", "--samples", "SRR648669.reverse", "SRR648667.forward"],
])
@pytest.mark.parametrize('suffix', ['.gz', '.txt'])
def test_stream_ops_never_load_the_matrix(tmp_path, monkeypatch, argv, suffix):
    def forbidden(*a, **k):
        raise AssertionError("streaming path must not load the matrix body")

    monkeypatch.setattr(OwnedMatrix, "load", forbidden)
    monkeypatch.setattr(Matrix, "load", forbidden)
    out = str(tmp_path / ('out' + suffix))
    cmo.main([argv[0], "-m", _OPS_MATRIX, "-o", out] + argv[1:])
    assert os.path.exists(out)


def _make_grouped_matrix(path, data, group_boundaries, sample_boundaries,
                         strands=None):
    n = data.shape[0]
    strands = strands or ['+'] * n
    regions = [['chr1', [(i, i + 1)], 'r%d' % i, 0, strands[i], '0']
               for i in range(n)]
    parameters = {
        'upstream': [0], 'downstream': [0], 'body': [data.shape[1]],
        'unscaled 5 prime': [0], 'unscaled 3 prime': [0],
        'ref point': [None], 'bin size': [1],
        'sort regions': 'keep', 'sort using': 'mean',
        'sample_labels': ['s1'], 'group_labels': ['A', 'B'],
        'sample_boundaries': sample_boundaries,
        'group_boundaries': group_boundaries,
    }
    OwnedMatrix.from_compute(parameters, data.astype(np.float32), regions).save(
        str(path), compressed=True, threads=1)


def test_stream_filter_values_boundaries_and_all_nan(tmp_path):
    # Two groups of three rows, one sample of four bins. Row 2 is entirely NaN
    # (must be kept); row 1 exceeds max; row 4 falls below min.
    nan = np.nan
    data = np.array([
        [10, 12, 14, 16],     # A row0: in range -> keep
        [10, 12, 14, 999],    # A row1: > max     -> drop
        [nan, nan, nan, nan],  # A row2: all NaN   -> keep
        [20, 22, 24, 26],     # B row3: in range  -> keep
        [1, 22, 24, 26],     # B row4: < min     -> drop
        [30, 32, 34, 36],     # B row5: in range  -> keep
    ], dtype=float)
    src = str(tmp_path / "grouped.gz")
    _make_grouped_matrix(src, data, [0, 3, 6], [0, 4])

    out = str(tmp_path / "filtered.gz")
    cmo.main(["filterValues", "-m", src, "-o", out, "--min", "5", "--max", "100"])
    hm = _load(out)
    # Each group drops exactly one row.
    assert list(hm.header.group_boundaries) == [0, 2, 4]
    assert [r[2] for r in hm.regions] == ['r0', 'r2', 'r3', 'r5']
    # The all-NaN row survived.
    assert bool(np.all(np.isnan(hm.values[1])))

    resident = str(tmp_path / "mem.gz")
    _run_inmemory(["filterValues", "-m", src, "-o", resident,
                   "--min", "5", "--max", "100"])
    assert _same_matrix(hm, _load(resident))


def test_stream_filter_strand_boundaries(tmp_path):
    data = np.arange(24, dtype=float).reshape(6, 4)
    src = str(tmp_path / "grouped.gz")
    _make_grouped_matrix(src, data, [0, 3, 6], [0, 4],
                         strands=['+', '-', '+', '-', '-', '+'])
    out = str(tmp_path / "plus.gz")
    cmo.main(["filterStrand", "-m", src, "-o", out, "--strand", "+"])
    hm = _load(out)
    assert list(hm.header.group_boundaries) == [0, 2, 3]           # A: r0,r2 ; B: r5
    assert [r[2] for r in hm.regions] == ['r0', 'r2', 'r5']


def test_stream_filter_scratch_path_cannot_alias_input(tmp_path):
    source = tmp_path / "result.gz.dtpbody"
    with open(_OPS_MATRIX, "rb") as handle:
        source.write_bytes(handle.read())
    original = source.read_bytes()
    output = tmp_path / "result.gz"

    cmo.main([
        "filterStrand", "-m", str(source), "-o", str(output), "--strand", "+"])

    assert source.read_bytes() == original
    assert output.exists()
    _load(str(output))
    assert not list(tmp_path.glob(".result.gz.*"))


# ---- filterValues enrichment: per-sample statistic / scoping / mask mode ----

def _make_two_sample_matrix(path, data, group_boundaries):
    """data is (rows, 2*bins): first half sample s1, second half s2."""
    n, cols = data.shape
    half = cols // 2
    regions = [['chr1', [(i, i + 1)], 'r%d' % i, 0, '+', '0'] for i in range(n)]
    parameters = {
        'upstream': [0, 0], 'downstream': [0, 0], 'body': [half, half],
        'unscaled 5 prime': [0, 0], 'unscaled 3 prime': [0, 0],
        'ref point': [None, None], 'bin size': [1, 1],
        'sort regions': 'keep', 'sort using': 'mean',
        'sample_labels': ['s1', 's2'], 'group_labels': ['A'],
        'sample_boundaries': [0, half, cols],
        'group_boundaries': group_boundaries,
    }
    OwnedMatrix.from_compute(parameters, data.astype(np.float32), regions).save(
        str(path), compressed=True, threads=1)


@pytest.mark.parametrize("statistic", ['perBin', 'mean', 'median', 'sum', 'min', 'max'])
@pytest.mark.parametrize("on_fail", ['removeRegion', 'maskSample'])
@pytest.mark.parametrize("samples", [None, ['s1'], ['2']])
def test_stream_filter_values_enrichment_matches_in_memory(
        tmp_path, statistic, on_fail, samples):
    rng = np.random.default_rng(1)
    data = (rng.random((30, 8)) * 100)
    data[3, :4] = np.nan          # an all-NaN sample block
    data[7, :] = np.nan           # an all-NaN row
    src = str(tmp_path / "two.gz")
    _make_two_sample_matrix(src, data, [0, 30])

    argv = ["filterValues", "-m", src, "--min", "20", "--max", "80",
            "--filterUsingStatistic", statistic, "--onFilterFail", on_fail]
    if samples:
        argv += ["--filterUsingSamples"] + samples

    streamed = str(tmp_path / "s.gz")
    resident = str(tmp_path / "m.gz")
    cmo.main(argv + ["-o", streamed])
    _run_inmemory(argv + ["-o", resident])
    assert _same_matrix(_load(streamed), _load(resident))


def test_stream_filter_values_mask_blanks_only_failing_sample(tmp_path):
    # One region, two samples of two bins. s1 mean 50 (pass), s2 mean 5 (fail).
    data = np.array([[50.0, 50.0, 5.0, 5.0]])
    src = str(tmp_path / "two.gz")
    _make_two_sample_matrix(src, data, [0, 1])
    out = str(tmp_path / "mask.gz")
    cmo.main(["filterValues", "-m", src, "-o", out, "--min", "20", "--max", "80",
              "--filterUsingStatistic", "mean", "--onFilterFail", "maskSample"])
    hm = _load(out)
    m = hm.values
    assert m.shape == (1, 4)                       # region kept
    assert not np.isnan(m)[0, 0:2].any()  # s1 intact
    assert bool(np.all(np.isnan(m)[0, 2:4]))  # s2 blanked


def test_stream_filter_values_using_samples_scopes_decision(tmp_path):
    # Row0: s1 out of range, s2 in range. Gating on s2 only must keep it whole.
    data = np.array([
        [999.0, 999.0, 50.0, 50.0],   # only s1 offends
        [50.0, 50.0, 50.0, 50.0],     # both fine
    ])
    src = str(tmp_path / "two.gz")
    _make_two_sample_matrix(src, data, [0, 2])
    out = str(tmp_path / "o.gz")
    cmo.main(["filterValues", "-m", src, "-o", out, "--min", "0", "--max", "100",
              "--filterUsingStatistic", "mean", "--filterUsingSamples", "s2"])
    hm = _load(out)
    assert list(hm.header.group_boundaries) == [0, 2]    # nothing dropped
    assert not np.isnan(hm.values).any()


@pytest.mark.parametrize('suffix', ['.gz', '.txt'])
def test_stream_filter_values_mask_never_loads_matrix(tmp_path, monkeypatch,
                                                      suffix):
    monkeypatch.setattr(OwnedMatrix, "load",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not load the matrix body")))
    monkeypatch.setattr(Matrix, "load",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not load the matrix body")))
    out = str(tmp_path / ('o' + suffix))
    cmo.main(["filterValues", "-m", _OPS_MATRIX, "-o", out, "--min", "5",
              "--onFilterFail", "maskSample", "--filterUsingStatistic", "mean"])
    assert os.path.exists(out)


# ---- filterValues --filterNans (missing-data filter) ------------------------

def _nan_pattern_matrix(path):
    """4 regions x 2 samples (2 bins each) with assorted NaN patterns."""
    nan = np.nan
    data = np.array([
        [nan, nan, nan, nan],   # r0: whole row NaN
        [10.0, nan, 50.0, 50.0],  # r1: s1 partially NaN, s2 fine
        [nan, nan, 50.0, 50.0],   # r2: s1 entirely NaN, s2 fine
        [50.0, 50.0, 50.0, 50.0],  # r3: all finite
    ])
    _make_two_sample_matrix(path, data, [0, 4])


@pytest.mark.parametrize("nan_mode", ['keep', 'any_bin', 'any_sample', 'all_bins'])
@pytest.mark.parametrize("on_fail", ['removeRegion', 'maskSample'])
@pytest.mark.parametrize("statistic", ['perBin', 'mean'])
def test_stream_filter_nans_match_in_memory(tmp_path, nan_mode, on_fail, statistic):
    src = str(tmp_path / "nan.gz")
    _nan_pattern_matrix(src)
    argv = ["filterValues", "-m", src, "--filterNans", nan_mode,
            "--onFilterFail", on_fail, "--filterUsingStatistic", statistic]
    streamed, resident = str(tmp_path / "s.gz"), str(tmp_path / "m.gz")
    cmo.main(argv + ["-o", streamed])
    _run_inmemory(argv + ["-o", resident])
    assert _same_matrix(_load(streamed), _load(resident))


@pytest.mark.parametrize("nan_mode,expected", [
    ('keep', ['r0', 'r1', 'r2', 'r3']),        # nothing dropped
    ('all_bins', ['r1', 'r2', 'r3']),          # only the wholly-NaN row
    ('any_sample', ['r1', 'r3']),              # rows with an all-NaN sample
    ('any_bin', ['r3']),                       # rows with any NaN bin
])
def test_filter_nans_semantics_remove(tmp_path, nan_mode, expected):
    # No value bounds given -> the NaN filter runs on its own.
    src = str(tmp_path / "nan.gz")
    _nan_pattern_matrix(src)
    out = str(tmp_path / "o.gz")
    cmo.main(["filterValues", "-m", src, "-o", out, "--filterNans", nan_mode])
    assert [r[2] for r in _load(out).regions] == expected


def test_filter_nans_respects_filter_using_samples(tmp_path):
    # any_bin scoped to s2: only r0 has a NaN bin in s2, so only r0 is dropped.
    src = str(tmp_path / "nan.gz")
    _nan_pattern_matrix(src)
    out = str(tmp_path / "o.gz")
    cmo.main(["filterValues", "-m", src, "-o", out, "--filterNans", "any_bin",
              "--filterUsingSamples", "s2"])
    assert [r[2] for r in _load(out).regions] == ['r1', 'r2', 'r3']


def test_filter_nans_mask_blanks_partially_missing_sample(tmp_path):
    # any_bin + maskSample: r1's s1 has a NaN bin -> blank all of s1 for r1;
    # s2 (fine) is untouched and no region is removed.
    src = str(tmp_path / "nan.gz")
    _nan_pattern_matrix(src)
    out = str(tmp_path / "o.gz")
    cmo.main(["filterValues", "-m", src, "-o", out, "--filterNans", "any_bin",
              "--onFilterFail", "maskSample"])
    hm = _load(out)
    assert list(hm.header.group_boundaries) == [0, 4]        # nothing removed
    m = hm.values
    assert bool(np.all(np.isnan(m)[1, 0:2]))  # r1 s1 fully blanked
    assert not np.isnan(m)[1, 2:4].any()      # r1 s2 intact


# ---- 7b: write-time row_order / column_order (reorder / sort / subset) -------

def _make_multigroup_matrix(path, data, group_boundaries, sample_boundaries,
                            group_labels, sample_labels):
    n = data.shape[0]
    regions = [['chr1', [(i, i + 1)], 'r%d' % i, 0, '+', '0'] for i in range(n)]
    ns = len(sample_labels)
    parameters = {
        'upstream': [0] * ns, 'downstream': [0] * ns,
        'body': [sample_boundaries[i + 1] - sample_boundaries[i] for i in range(ns)],
        'unscaled 5 prime': [0] * ns, 'unscaled 3 prime': [0] * ns,
        'ref point': [None] * ns, 'bin size': [1] * ns,
        'sort regions': 'keep', 'sort using': 'mean',
        'sample_labels': list(sample_labels), 'group_labels': list(group_labels),
        'sample_boundaries': list(sample_boundaries),
        'group_boundaries': list(group_boundaries),
    }
    OwnedMatrix.from_compute(parameters, data.astype(np.float32), regions).save(
        str(path), compressed=str(path).endswith('.gz'), threads=1)


@pytest.mark.parametrize("groups,samples", [
    (["C", "A"], ["s2", "s1"]),        # reorder groups and columns
    (["B"], None),                     # drop to one group, all columns
    (None, ["s2"]),                    # all groups, one column (still 7b here)
    (["A", "B", "C"], ["s1", "s2"]),   # identity
])
def test_writetime_reorder_matches_physical(tmp_path, groups, samples):
    data = np.arange(6 * 4, dtype=float).reshape(6, 4)
    src = str(tmp_path / "g3.gz")
    _make_multigroup_matrix(src, data, [0, 2, 4, 6], [0, 2, 4],
                            ['A', 'B', 'C'], ['s1', 's2'])
    argv = ["reorder", "-m", src]
    if groups:
        argv += ["--groups"] + groups
    if samples:
        argv += ["--samples"] + samples
    native, fallback = str(tmp_path / "n.gz"), str(tmp_path / "f.gz")
    cmo.main(argv + ["-o", native])
    _run_inmemory(argv + ["-o", fallback])
    assert _same_matrix(_load(native), _load(fallback))


def test_writetime_reorder_content(tmp_path):
    data = np.arange(6 * 4, dtype=float).reshape(6, 4)
    src = str(tmp_path / "g3.gz")
    _make_multigroup_matrix(src, data, [0, 2, 4, 6], [0, 2, 4],
                            ['A', 'B', 'C'], ['s1', 's2'])
    out = str(tmp_path / "o.gz")
    cmo.main(["reorder", "-m", src, "-o", out,
              "--groups", "C", "A", "--samples", "s2", "s1"])
    hm = _load(out)
    assert list(hm.header.group_labels) == ['C', 'A']
    assert list(hm.header.group_boundaries) == [0, 2, 4]
    assert list(hm.header.sample_labels) == ['s2', 's1']
    assert [r[2] for r in hm.regions] == ['r4', 'r5', 'r0', 'r1']
    expected = data[[4, 5, 0, 1]][:, [2, 3, 0, 1]]
    assert np.allclose(np.ma.filled(hm.values, np.nan), expected)


def test_native_writer_row_and_column_order_selection():
    # Direct writer test: row_order selects/reorders rows, column_order columns.
    import tempfile
    from deeptoolsr import _compute_matrix_io as io
    regions = [['c', [(i, i + 1)], 'r%d' % i, 0, '+', '0'] for i in range(4)]
    values = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    # Output has 3 rows (row_order) and 2 columns (column_order).
    header = '{"sample_boundaries":[0,2],"group_boundaries":[0,3],' \
             '"sample_labels":["s"],"group_labels":["g"]}'
    with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as fh:
        out = fh.name
        io.write_matrix(out, header, regions, values, 1, 6, 0,
                        np.array([3, 1, 0], dtype=np.int64),
                        np.array([2, 0], dtype=np.int64))  # rows 3,1,0; cols 2,0
    hm = Matrix.load(out, 1)
    os.remove(out)
    assert [r[2] for r in hm.regions] == ['r3', 'r1', 'r0']
    assert np.allclose(np.ma.filled(hm.values, np.nan),
                       values[[3, 1, 0]][:, [2, 0]])


# ---- 7c: bind --blind (rbind geometry-skip, cbind position-match streaming) --

def _make_bind_matrix(path, data, sample_labels, sample_boundaries,
                      group_boundaries, group_labels, names=None):
    n = data.shape[0]
    names = names or ['r%d' % i for i in range(n)]
    regions = [['chr1', [(i, i + 1)], names[i], 0, '+', '0'] for i in range(n)]
    ns = len(sample_labels)
    parameters = {
        'upstream': [0] * ns, 'downstream': [0] * ns,
        'body': [sample_boundaries[i + 1] - sample_boundaries[i] for i in range(ns)],
        'unscaled 5 prime': [0] * ns, 'unscaled 3 prime': [0] * ns,
        'ref point': [None] * ns, 'bin size': [1] * ns,
        'sort regions': 'keep', 'sort using': 'mean',
        'sample_labels': list(sample_labels), 'group_labels': list(group_labels),
        'sample_boundaries': list(sample_boundaries),
        'group_boundaries': list(group_boundaries),
    }
    OwnedMatrix.from_compute(parameters, data.astype(np.float32), regions).save(
        str(path), compressed=str(path).endswith('.gz'), threads=1)


def test_cbind_blind_matches_by_position(tmp_path):
    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    b = np.array([[7, 8], [9, 10], [11, 12]], dtype=float)
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_bind_matrix(pa, a, ['a1'], [0, 2], [0, 3], ['g'], names=['x', 'y', 'z'])
    _make_bind_matrix(pb, b, ['b1'], [0, 2], [0, 3], ['g'], names=['p', 'q', 'r'])
    out = str(tmp_path / "o.gz")
    cmo.main(["cbind", "-m", pa, pb, "-o", out, "--blind"])
    hm = _load(out)
    assert list(hm.header.sample_labels) == ['a1', 'b1']
    assert list(hm.header.sample_boundaries) == [0, 2, 4]
    assert [r[2] for r in hm.regions] == ['x', 'y', 'z']  # first file's names
    assert np.allclose(np.ma.filled(hm.values, np.nan), np.hstack([a, b]))


def test_cbind_blind_equals_name_join_when_aligned(tmp_path):
    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    b = np.array([[7, 8], [9, 10], [11, 12]], dtype=float)
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_bind_matrix(pa, a, ['a1'], [0, 2], [0, 3], ['g'])
    _make_bind_matrix(pb, b, ['b1'], [0, 2], [0, 3], ['g'])
    blind, name = str(tmp_path / "bl.gz"), str(tmp_path / "nm.gz")
    cmo.main(["cbind", "-m", pa, pb, "-o", blind, "--blind"])
    cmo.main(["cbind", "-m", pa, pb, "-o", name])
    assert _same_matrix(_load(blind), _load(name))


def test_cbind_blind_row_count_mismatch_errors(tmp_path):
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_bind_matrix(pa, np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
                      ['a1'], [0, 2], [0, 3], ['g'])
    _make_bind_matrix(pb, np.array([[7, 8], [9, 10]], dtype=float),
                      ['b1'], [0, 2], [0, 2], ['g'])
    with pytest.raises(SystemExit, match="same number of rows"):
        cmo.main(["cbind", "-m", pa, pb, "-o", str(tmp_path / "o.gz"), "--blind"])


def test_cbind_validated_uses_first_row_universe(tmp_path):
    first = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    later = np.array([[7, 8], [9, 10]], dtype=float)
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_bind_matrix(pa, first, ['a'], [0, 2], [0, 3], ['g'],
                      names=['x', 'y', 'z'])
    _make_bind_matrix(pb, later, ['b'], [0, 2], [0, 2], ['g'],
                      names=['y', 'later-only'])
    # Full BED identity, not the column-4 name alone, determines a match.
    first_holder = _load(pa)
    later_holder = OwnedMatrix.load(pb, 1)
    later_holder.regions[0] = first_holder.regions[1]
    later_holder.save(pb, compressed=True, threads=1)
    out = str(tmp_path / "joined.gz")

    cmo.main(["cbind", "-m", pa, pb, "-o", out])

    joined = _load(out)
    observed = np.ma.filled(joined.values, np.nan)
    expected = np.array([
        [1, 2, np.nan, np.nan],
        [3, 4, 7, 8],
        [5, 6, np.nan, np.nan],
    ])
    assert [region[2] for region in joined.regions] == ['x', 'y', 'z']
    np.testing.assert_allclose(observed, expected, equal_nan=True)


def test_cbind_validated_rejects_empty_region_names(tmp_path):
    data = np.array([[1, 2], [3, 4]], dtype=float)
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_bind_matrix(pa, data, ['a'], [0, 2], [0, 2], ['g'],
                      names=['', 'y'])
    _make_bind_matrix(pb, data, ['b'], [0, 2], [0, 2], ['g'],
                      names=['x', 'y'])
    with pytest.raises(SystemExit, match='non-empty region names'):
        cmo.main(["cbind", "-m", pa, pb, "-o", str(tmp_path / "out.gz")])


def test_cbind_validated_allows_repeated_names_at_distinct_bed_rows(tmp_path):
    data = np.array([[1, 2], [3, 4]], dtype=float)
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_bind_matrix(pa, data, ['a'], [0, 2], [0, 2], ['g'],
                      names=['x', 'x'])
    _make_bind_matrix(pb, data, ['b'], [0, 2], [0, 2], ['g'],
                      names=['x', 'x'])
    out = str(tmp_path / "out.gz")
    cmo.main(["cbind", "-m", pa, pb, "-o", out])
    np.testing.assert_allclose(
        np.ma.filled(_load(out).values, np.nan),
        np.hstack([data, data]))


def test_cbind_validated_rejects_duplicate_complete_bed_rows(tmp_path):
    data = np.array([[1, 2], [3, 4]], dtype=float)
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_bind_matrix(pa, data, ['a'], [0, 2], [0, 2], ['g'],
                      names=['x', 'x'])
    first = OwnedMatrix.load(pa, 1)
    first.regions[1] = first.regions[0]
    first.save(pa, compressed=True, threads=1)
    _make_bind_matrix(pb, data, ['b'], [0, 2], [0, 2], ['g'],
                      names=['x', 'y'])
    with pytest.raises(SystemExit, match='unique complete BED rows'):
        cmo.main(["cbind", "-m", pa, pb, "-o", str(tmp_path / "out.gz")])


def test_transform_validated_aligns_complete_bed_rows_but_blind_is_positional(
        tmp_path):
    first_values = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
    later_values = np.array([[1, 2], [5, 10], [7, 14]], dtype=float)
    pa, pb = str(tmp_path / 'a.gz'), str(tmp_path / 'b.gz')
    _make_bind_matrix(pa, first_values, ['a'], [0, 2], [0, 3], ['g'],
                      names=['x', 'y', 'z'])
    _make_bind_matrix(pb, later_values, ['b'], [0, 2], [0, 3], ['g'],
                      names=['y', 'x', 'later-only'])
    first = _load(pa)
    later = OwnedMatrix.load(pb, 1)
    later.regions[0] = first.regions[1]
    later.regions[1] = first.regions[0]
    later.save(pb, compressed=True, threads=1)

    validated = str(tmp_path / 'validated.gz')
    blind = str(tmp_path / 'blind.gz')
    cmo.main(['transform', '-m', pa, pb, '--ratio', '-o', validated])
    cmo.main(['transform', '-m', pa, pb, '--ratio', '--blind', '-o', blind])

    expected_validated = np.array([[2, 2], [30, 20], [np.nan, np.nan]])
    expected_blind = first_values / later_values
    np.testing.assert_allclose(
        np.ma.filled(_load(validated).values, np.nan),
        expected_validated, equal_nan=True)
    np.testing.assert_allclose(
        np.ma.filled(_load(blind).values, np.nan), expected_blind)


@pytest.mark.parametrize('command,extra', [
    ('relabel', ['--sampleLabels', 'renamed']),
    ('subset', ['--samples', '1']),
    ('filterStrand', ['--strand', '+']),
    ('filterValues', ['--min', '-1000', '--max', '1000']),
])
def test_matrix_operations_accept_plain_input_and_write_plain_output(
        tmp_path, command, extra):
    source = str(tmp_path / 'source.mat')
    output = str(tmp_path / 'output.mat')
    _make_bind_matrix(
        source, np.array([[1, 2], [3, 4]], dtype=float),
        ['sample'], [0, 2], [0, 2], ['group'])

    cmo.main([command, '-m', source, '-o', output, *extra])

    assert open(output, 'rb').read(1) == b'@'
    restored = _load(output)
    assert restored.values.shape == (2, 2)


def test_transform_plain_matrix_io(tmp_path):
    first = str(tmp_path / 'first.mat')
    second = str(tmp_path / 'second.mat')
    output = str(tmp_path / 'output.mat')
    _make_bind_matrix(first, np.array([[2, 4]], dtype=float),
                      ['sample'], [0, 2], [0, 1], ['group'])
    _make_bind_matrix(second, np.array([[1, 2]], dtype=float),
                      ['sample'], [0, 2], [0, 1], ['group'])

    cmo.main(['transform', '-m', first, second, '--ratio', '-o', output])

    assert open(output, 'rb').read(1) == b'@'
    np.testing.assert_allclose(_load(output).values, 2)


@pytest.mark.parametrize('labels,match', [
    (['', 'g2'], 'non-empty group labels'),
    (['g', 'g'], 'unique group labels'),
])
def test_cbind_validated_rejects_ambiguous_group_labels(tmp_path, labels, match):
    data = np.array([[1, 2], [3, 4]], dtype=float)
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_bind_matrix(pa, data, ['a'], [0, 2], [0, 1, 2], labels,
                      names=['x', 'y'])
    _make_bind_matrix(pb, data, ['b'], [0, 2], [0, 2], ['g'],
                      names=['x', 'y'])
    with pytest.raises(SystemExit, match=match):
        cmo.main(["cbind", "-m", pa, pb, "-o", str(tmp_path / "out.gz")])


def test_cbind_blind_never_loads_matrix(tmp_path, monkeypatch):
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_bind_matrix(pa, np.array([[1, 2], [3, 4]], dtype=float),
                      ['a1'], [0, 2], [0, 2], ['g'])
    _make_bind_matrix(pb, np.array([[7, 8], [9, 10]], dtype=float),
                      ['b1'], [0, 2], [0, 2], ['g'])
    monkeypatch.setattr(OwnedMatrix, "load",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("cbind --blind must not load the body")))
    monkeypatch.setattr(Matrix, "load",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("cbind --blind must not load the body")))
    out = str(tmp_path / "o.gz")
    cmo.main(["cbind", "-m", pa, pb, "-o", out, "--blind"])
    assert os.path.exists(out)


def test_rbind_blind_allows_differing_sample_metadata(tmp_path):
    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    b = np.array([[7, 8], [9, 10], [11, 12]], dtype=float)
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_bind_matrix(pa, a, ['s1'], [0, 2], [0, 3], ['g1'])
    _make_bind_matrix(pb, b, ['DIFFERENT'], [0, 2], [0, 3], ['g2'])
    # Without --blind the differing sample labels are rejected.
    with pytest.raises(SystemExit, match="identical samples"):
        cmo.main(["rbind", "-m", pa, pb, "-o", str(tmp_path / "x.gz")])
    out = str(tmp_path / "o.gz")
    cmo.main(["rbind", "-m", pa, pb, "-o", out, "--blind"])
    hm = _load(out)
    assert list(hm.header.group_labels) == ['g1', 'g2']
    assert list(hm.header.group_boundaries) == [0, 3, 6]
    assert np.allclose(np.ma.filled(hm.values, np.nan), np.vstack([a, b]))


def test_rbind_validated_rejects_differing_sample_geometry(tmp_path):
    data = np.array([[1, 2], [3, 4]], dtype=float)
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_bind_matrix(pa, data, ['s1'], [0, 2], [0, 2], ['g1'])
    _make_bind_matrix(pb, data, ['s1'], [0, 2], [0, 2], ['g2'])
    with gzip.open(pb, 'rt') as handle:
        lines = handle.readlines()
    header = json.loads(lines[0][1:])
    header['upstream'] = [1]
    header['body'] = [1]
    lines[0] = '@' + json.dumps(header, separators=(',', ':')) + '\n'
    with gzip.open(pb, 'wt') as handle:
        handle.writelines(lines)

    with pytest.raises(SystemExit, match='sample geometry.*upstream'):
        cmo.main(["rbind", "-m", pa, pb, "-o", str(tmp_path / "bad.gz")])

    # The explicitly blind mode retains its shape-only contract.
    cmo.main(["rbind", "-m", pa, pb, "-o", str(tmp_path / "blind.gz"),
              "--blind"])


def test_rbind_validates_all_inputs_before_group_merge_fallback(tmp_path):
    data = np.array([[1, 2], [3, 4]], dtype=float)
    pa = str(tmp_path / "a.gz")
    pb = str(tmp_path / "b.gz")
    pc = str(tmp_path / "c.gz")
    _make_bind_matrix(pa, data, ['s1'], [0, 2], [0, 2], ['shared'])
    _make_bind_matrix(pb, data, ['s1'], [0, 2], [0, 2], ['shared'])
    _make_bind_matrix(pc, data, ['s1'], [0, 2], [0, 2], ['later'])

    with gzip.open(pc, 'rt') as handle:
        lines = handle.readlines()
    header = json.loads(lines[0][1:])
    header['upstream'] = [1]
    header['body'] = [1]
    lines[0] = '@' + json.dumps(header, separators=(',', ':')) + '\n'
    with gzip.open(pc, 'wt') as handle:
        handle.writelines(lines)

    # File b forces the merge/interleave fallback, but file c must still be
    # validated before that fallback starts loading and combining bodies.
    with pytest.raises(SystemExit, match='sample geometry.*upstream'):
        cmo.main(["rbind", "-m", pa, pb, pc, "-o", str(tmp_path / "bad.gz")])


def test_rbind_blind_column_count_mismatch_errors(tmp_path):
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_bind_matrix(pa, np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
                      ['s1'], [0, 2], [0, 3], ['g1'])
    _make_bind_matrix(pb, np.array([[1, 2, 3]] * 3, dtype=float),
                      ['s1'], [0, 3], [0, 3], ['g2'])
    with pytest.raises(SystemExit, match="same number of columns"):
        cmo.main(["rbind", "-m", pa, pb, "-o", str(tmp_path / "o.gz"), "--blind"])


# ---- Native in-place transform kernels -------------------------------------

def _make_transform_pair(tmp_path):
    # 5 rows, 2 samples of 3 bins (upstream 1 + body 2), assorted NaN/zero cells.
    rng = np.random.default_rng(0)
    a = (rng.random((5, 6)) * 10).astype(np.float32)
    b = (rng.random((5, 6)) * 10).astype(np.float32)
    a[0, :] = np.nan
    a[1, 2] = np.nan
    a[2, 0] = 0.0
    b[3, :] = np.nan
    b[1, 4] = 0.0
    b[2, 0] = 0.0
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    for path, data in ((pa, a), (pb, b)):
        _make_multigroup_matrix(path, data, [0, 5], [0, 3, 6], ['g'], ['s1', 's2'])
        # give each sample a 1-bin upstream so --scaleUsing upstream is valid
        matrix = OwnedMatrix.load(path, 1)
        parameters = dict(matrix.header.parameters)
        parameters['upstream'] = [1, 1]
        parameters['body'] = [2, 2]
        matrix.header = MatrixHeader.from_parameters(parameters)
        matrix.save(path, compressed=True, threads=1)
    return pa, pb


@pytest.mark.parametrize("op", [
    ["--add", "5"], ["--subtract", "2.5"], ["--scale", "3"],
    ["--scale", "row-sum"], ["--scale", "row-min", "--scaleUsing", "upstream"],
    ["--scale", "row-mean", "--scaleUsing", "body"], ["--log2"],
])
def test_native_transform_unary_matches_numpy(tmp_path, op):
    pa, _ = _make_transform_pair(tmp_path)
    native = str(tmp_path / "n.gz")
    data = np.ma.filled(_load(pa).values, np.nan).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        if op[0] == '--add':
            expected = data + float(op[1])
        elif op[0] == '--subtract':
            expected = data - float(op[1])
        elif op[0] == '--log2':
            expected = np.log2(data)
        elif not op[1].startswith('row-'):
            expected = data * float(op[1])
        else:
            expected = data.copy()
            using = op[-1] if '--scaleUsing' in op else 'all'
            for start in (0, 3):
                selected = data[:, start:start + 3]
                if using == 'upstream':
                    selected = selected[:, :1]
                elif using == 'body':
                    selected = selected[:, 1:]
                for row in range(len(data)):
                    finite = selected[row, np.isfinite(selected[row])]
                    divisor = getattr(np, op[1][4:])(finite) if len(finite) else np.nan
                    expected[row, start:start + 3] /= divisor
    expected[~np.isfinite(expected)] = np.nan
    cmo.main(["transform", "-m", pa] + op + ["-o", native])
    np.testing.assert_allclose(np.ma.filled(_load(native).values, np.nan),
                               expected, rtol=3e-7, atol=1.1e-6, equal_nan=True)


@pytest.mark.parametrize("op", [
    ["--sum"], ["--mean"], ["--difference"],
    ["--ratio", "--pseudocount", "1"],
    ["--log2FC", "--pseudocount", "min-denominator"],
    ["--ratio", "--missingDataAsZero", "--zeroHandling", "any", "--pseudocount", "1"],
    ["--difference", "--missingDataAsZero", "--zeroHandling", "both"],
])
def test_native_transform_binary_matches_numpy(tmp_path, op):
    pa, pb = _make_transform_pair(tmp_path)
    native = str(tmp_path / "n.gz")
    a, b = [np.ma.filled(_load(path).values, np.nan).astype(np.float64)
            for path in (pa, pb)]
    if '--missingDataAsZero' in op:
        a, b = np.nan_to_num(a), np.nan_to_num(b)
    with np.errstate(divide='ignore', invalid='ignore'):
        if op[0] in ('--sum', '--mean'):
            count = np.isfinite(a).astype(int) + np.isfinite(b)
            expected = np.nan_to_num(a) + np.nan_to_num(b)
            if op[0] == '--mean':
                expected /= count
            expected[count == 0] = np.nan
        elif op[0] == '--difference':
            expected = a - b
        else:
            pc = np.ones(6)
            if 'min-denominator' in op:
                for start in (0, 3):
                    sample = b[:, start:start + 3]
                    pc[start:start + 3] = np.min(sample[np.isfinite(sample) & (sample != 0)])
            expected = (a + pc) / (b + pc)
            if op[0] == '--log2FC':
                expected = np.log2(expected)
        if '--zeroHandling' in op:
            zero = (a == 0) | (b == 0) if 'any' in op else (a == 0) & (b == 0)
            expected[zero] = np.nan
    expected[~np.isfinite(expected)] = np.nan
    cmo.main(["transform", "-m", pa, pb] + op + ["-o", native])
    np.testing.assert_allclose(np.ma.filled(_load(native).values, np.nan),
                               expected, rtol=3e-7, atol=1.1e-6, equal_nan=True)


def test_native_transform_sum_mean_semantics(tmp_path):
    # sum/mean skip NaN contributors; an all-NaN cell stays NaN.
    a = np.array([[1.0, np.nan, 4.0]], dtype=np.float32)
    b = np.array([[3.0, np.nan, np.nan]], dtype=np.float32)
    pa, pb = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    _make_multigroup_matrix(pa, a, [0, 1], [0, 3], ['g'], ['s'])
    _make_multigroup_matrix(pb, b, [0, 1], [0, 3], ['g'], ['s'])
    out = str(tmp_path / "o.gz")
    cmo.main(["transform", "-m", pa, pb, "--sum", "-o", out])
    s = np.ma.filled(_load(out).values, np.nan)
    assert s[0, 0] == 4.0                 # 1 + 3
    assert np.isnan(s[0, 1])              # both NaN -> NaN
    assert s[0, 2] == 4.0                 # only a covers it
    cmo.main(["transform", "-m", pa, pb, "--mean", "-o", out])
    m = np.ma.filled(_load(out).values, np.nan)
    assert m[0, 0] == 2.0                 # (1 + 3) / 2
    assert np.isnan(m[0, 1])
    assert m[0, 2] == 4.0                 # mean of the single finite value


# From audit followup 2026 09 12.

@pytest.fixture()
def coverage_isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path / 'config'))


def write_track(path, values):
    with pyBigWig.open(str(path), 'w') as bw:
        bw.addHeader([('chr1', len(values))], maxZooms=0)
        starts = [i for i, value in enumerate(values) if np.isfinite(value)]
        bw.addEntries(['chr1'] * len(starts), starts, ends=[i + 1 for i in starts], values=[float(values[i]) for i in starts])


@pytest.mark.usefixtures('coverage_isolated_config')
@pytest.mark.parametrize('other_ref', ['TES', 'center'])
@pytest.mark.parametrize('blind', [False, True])
def test_combining_matrices_must_validate_reference_point(tmp_path, other_ref, blind):
    track, bed = (tmp_path / 'ramp.bw', tmp_path / 'regions.bed')
    write_track(track, np.arange(1000, dtype=float))
    bed.write_text('chr1\t100\t200\tr\t0\t+\n')
    inputs = []
    for ref in ('TSS', other_ref):
        output = tmp_path / f'{ref}.gz'
        computeMatrix.main(['reference-point', '--referencePoint', ref, '-S', str(track), '-R', str(bed), '-o', str(output), '-b', '20', '-a', '20', '-bs', '10', '--quiet'])
        inputs.append(str(output))
    with pytest.raises((ValueError, SystemExit), match='ref|locus|geometr'):
        coverage_cmo.main(['transform', '-m', *inputs, '--difference', '-o', str(tmp_path / 'diff.gz'), *(['--blind'] if blind else [])])


# From audit regressions.

@pytest.fixture()
def io_native_backends(monkeypatch):
    monkeypatch.delenv('DTP_BAM_BACKEND', raising=False)


@pytest.mark.usefixtures('io_native_backends')
def test_blind_cbind_preserves_scalar_sample_geometry(tmp_path):
    header = {'group_boundaries': [0, 1], 'group_labels': ['g'], 'sample_boundaries': [0, 4], 'sample_labels': ['s1'], 'upstream': 1, 'downstream': 3, 'body': 0, 'bin size': 1, 'unscaled 5 prime': 0, 'unscaled 3 prime': 0, 'ref point': 'TSS'}
    paths = [tmp_path / 'a.gz', tmp_path / 'b.gz']
    changes = [{}, {'sample_labels': ['s2'], 'upstream': 2, 'downstream': 2}]
    for path, change in zip(paths, changes):
        with gzip.open(path, 'wt') as handle:
            handle.write('@' + json.dumps({**header, **change}) + '\n')
            handle.write('chr1\t5\t10\tx\t0\t+\t1\t2\t3\t4\n')
    output = tmp_path / 'joined.gz'
    io_cmo.main(['cbind', '-m', *map(str, paths), '-o', str(output), '--blind'])
    with gzip.open(output, 'rt') as handle:
        observed = json.loads(handle.readline()[1:])
    assert observed['upstream'] == [1, 2] and observed['downstream'] == [3, 2]


# From general second audit regressions.

def signal(tmp_path):
    path = tmp_path / 'signal.bw'
    with pyBigWig.open(str(path), 'w') as bw:
        bw.addHeader([('chr1', 100)])
        bw.addEntries(['chr1', 'chr1'], [0, 50], ends=[50, 100], values=[2.0, 7.0])
    return path


def matrix_args(bw, regions, output, extra=()):
    return ['reference-point', '-S', str(bw), '-R', str(regions), '-o', str(output), '-b', '0', '-a', '10', '-bs', '10', *extra]


def assert_rows(path, names=('one', 'two'), values=(2.0, 7.0)):
    header, rows, observed = read_matrix(path)
    assert [row[3] for row in rows] == list(names)
    np.testing.assert_array_equal(observed, np.asarray(values).reshape(-1, 1))
    return header


def compressed_bed(tmp_path, compression):
    text = b'chr1\t60\t70\ttwo\t0\t+\nchr1\t10\t20\tone\t0\t+\n'
    path = tmp_path / ('regions.bed' + ('.' + compression if compression else ''))
    path.write_bytes(gzip.compress(text) if compression == 'gz' else bz2.compress(text) if compression == 'bz2' else text)
    return path


@pytest.mark.parametrize('operation', ['relabel', 'rbind_keep', 'rbind_merge', 'subset'])
def test_saved_generated_provenance_survives_matrix_operations(tmp_path, operation):
    bw, bed = (signal(tmp_path), compressed_bed(tmp_path, None))
    source, modified, out = [tmp_path / name for name in ('source.gz', 'modified.gz', 'out.gz')]
    computeMatrix.main(matrix_args(bw, bed, source, ['--scoreFileNameMinus', str(bw), '--antisense', 'as_groups']))
    if operation == 'relabel':
        general_cmo.main(['relabel', '-m', str(source), '-o', str(modified), '--groupLabels', 'genes', 'opposite'])
        expected_names, expected_values = (['two', 'one', 'two_antisense', 'one_antisense'], [7, 2, 7, 2])
    elif operation == 'subset':
        general_cmo.main(['subset', '-m', str(source), '-o', str(modified), '--groups', 'genes_antisense'])
        expected_names, expected_values = (['two_antisense', 'one_antisense'], [7, 2])
    else:
        general_cmo.main(['rbind', '-m', str(source), str(source), '-o', str(modified), '--sameGroupLabels', 'merge' if operation == 'rbind_merge' else 'separate'])
        if operation == 'rbind_merge':
            expected_names = ['two', 'two', 'one', 'one', 'two_antisense', 'two_antisense', 'one_antisense', 'one_antisense']
            expected_values = [7, 7, 2, 2, 7, 7, 2, 2]
        else:
            expected_names = ['two', 'one', 'two_antisense', 'one_antisense'] * 2
            expected_values = [7, 2, 7, 2] * 2
    general_cmo.main(['sort', '-m', str(modified), '-R', str(bed), '-o', str(out)])
    assert_rows(out, expected_names, expected_values)


def test_combining_generated_and_literal_groups_retains_ambiguity():
    from deeptoolsr.region_provenance import combined_antisense_sources
    literal = {'group_labels': ['genes_antisense']}
    generated = {'group_labels': ['genes_antisense'], 'antisense_group_sources': {'genes_antisense': 'genes'}}
    assert combined_antisense_sources([generated, literal, generated]) == {'genes_antisense': None}
    assert combined_antisense_sources([generated], ['renamed']) == {'renamed': 'genes'}


# From numerical audit regressions.

def make_matrix(data, groups=None, labels=None):
    rows, cols = data.shape
    regions = [['chr1', [(i, i + 1)], f'r{i}', 0, '+'] for i in range(rows)]
    return Matrix(MatrixHeader.from_parameters({'group_boundaries': groups or [0, rows], 'sample_boundaries': list(range(cols + 1)), 'group_labels': labels or ['g'], 'sample_labels': [f's{i}' for i in range(cols)]}), data, regions, None)


@pytest.mark.parametrize('operation,expected', [('mean', 2), ('sum', 2), ('median', 2), ('min', 2), ('max', 2), ('std', 0)])
def test_n05_group_merge_keeps_missing_value(operation, expected):
    data = np.array([[np.nan], [2.0]], dtype=np.float32)
    matrix = make_matrix(data, [0, 1, 2], ['g', 'g'])
    assert profile.calc_avg(matrix.values, operation, threads=1)[0] == expected
    layout = merge_groups_by_key(RowLayout.identity(matrix), ['g', 'g'])
    assert layout.group_bounds == (0, 2)
    assert profile.calc_avg(layout.block(matrix, 0, 0), operation, threads=1)[0] == expected
    assert np.isnan(matrix.values[0, 0])


@pytest.mark.parametrize('fortran', [False, True])
def test_n05_group_merge_preserves_order_and_finite_filtering(fortran):
    data = np.array([[np.nan, 2], [20, 30], [np.nan, 4]], np.float32)
    if fortran:
        data = np.asfortranarray(data)
    matrix = make_matrix(data, [0, 1, 2, 3], ['a', 'b', 'a'])
    layout = merge_groups_by_key(RowLayout.identity(matrix), ['a', 'b', 'a'])
    assert [layout.base_names[origin.source] for origin in layout.origins] == ['a', 'b']
    assert layout.group_bounds == (0, 2, 3)
    assert [matrix.regions[index][2] for index in layout.row_indices()] == ['r0', 'r2', 'r1']
    assert not isinstance(matrix.values, np.ma.MaskedArray)
    first = matrix.values[layout.row_indices()[:2]]
    np.testing.assert_array_equal(kernels.filter_rows(first, 'mean', 0, 10, threads=1)[0], [False, False])
    np.testing.assert_array_equal(kernels.reduce(first, 'count', threads=1), [0, 2])
    assert np.ma.is_masked(profile.calc_avg(first, 'sum', threads=1)[0])


@pytest.mark.parametrize('operation', ['sum', 'mean', 'trim_mean'])
def test_n07_rescaling_preserves_finite_cancellation_residual(operation):
    maximum = np.finfo(np.float64).max
    values = np.array([[maximum, 1e-20, -maximum]])
    expected = 1e-20 if operation == 'sum' else 1e-20 / 3
    actual = kernels.reduce(values, operation, axis=1, threads=1)[0]
    assert math.isclose(actual, expected, rel_tol=2e-15, abs_tol=0)


# From parity audit regressions.

@pytest.fixture
def parity_matrix_inputs(tmp_path):
    paths = []
    for sample in range(2):
        path = tmp_path / f's{sample}.bw'
        with pyBigWig.open(str(path), 'w') as bw:
            bw.addHeader([('chr1', 200)])
            starts = [i for i in range(200) if not 85 <= i < 100]
            bw.addEntries(['chr1'] * len(starts), starts, ends=[i + 1 for i in starts], values=[0.0 if 40 <= i < 55 else float((i % 11 + 1) * (sample + 1)) for i in starts])
        paths.append(str(path))
    bed = tmp_path / 'regions.bed'
    bed.write_text(''.join((f'chr1\t{s}\t{e}\tr{i}\t{i}\t{strand}\n' for i, (s, e, strand) in enumerate([(2, 23, '+'), (35, 53, '-'), (80, 95, '.'), (130, 151, '-'), (183, 198, '+')]))))
    return (paths, str(bed))


@pytest.fixture
def parity_input_matrix(tmp_path, parity_matrix_inputs):
    paths, bed = parity_matrix_inputs
    output = tmp_path / 'input.gz'
    computeMatrix.main(['reference-point', '-S', *paths, '-R', bed, '-b', '10', '-a', '30', '-bs', '5', '--quiet', '-p', '1', '-o', str(output)])
    return str(output)


@pytest.mark.parametrize('operation', ['subset', 'relabel', 'filterStrand', 'filterValues', 'rbind', 'cbind'])
def test_matrix_operations_parity(tmp_path, parity_input_matrix, operation):
    extra = {'subset': ['--samples', 's1'], 'relabel': ['--sampleLabels', 'X', 'Y'], 'filterStrand': ['--strand', '-'], 'filterValues': ['--min', '0', '--max', '20'], 'rbind': [], 'cbind': []}[operation]
    args = [operation, '-m', parity_input_matrix]
    if operation in ('rbind', 'cbind'):
        args += [parity_input_matrix]
    args += extra
    upstream, plus = (tmp_path / 'original-operation.gz', tmp_path / 'plus-operation.gz')
    original('computeMatrixOperations', args + ['-o', str(upstream)])
    computeMatrixOperations.main(args + ['-o', str(plus)])
    oh, orows, ov = read_matrix(upstream)
    ph, prows, pv = read_matrix(plus)
    assert orows == prows
    for field in ('sample_labels', 'group_labels', 'sample_boundaries', 'group_boundaries'):
        assert oh[field] == ph[field]
    np.testing.assert_allclose(pv, ov, rtol=3e-07, atol=1.01e-06, equal_nan=True)


# From read validation regressions.

ROW = 'chr1\t0\t2\tr\t0\t+\t1\t2\n'


OLD = b'previous successful output'


@pytest.fixture()
def validation_isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path / 'config'))


def write_matrix(path, body=ROW, header=None):
    text = '@' + json.dumps(_header() if header is None else header) + '\n' + body
    path.write_bytes(gzip.compress(text.encode()))


def operate(operation, source, output, threads=1):
    args = {'filterStrand': ['--strand', '+'], 'subset': ['--samples', 'sample'], 'reorder': ['--samples', 'sample'], 'mask': ['--onFilterFail', 'maskSample'], 'drop': [], 'relabel': ['--sampleLabels', 'new'], 'rbind': ['--sameGroupLabels', 'separate'], 'cbind': ['--blind']}
    if operation == 'read':
        return Matrix.load(source, threads=threads)
    command = 'filterValues' if operation in ('mask', 'drop') else operation
    validation_cmo.main([command, '-m', str(source), *([str(source)] if operation in ('rbind', 'cbind') else []), '-o', str(output), *args[operation], *(['-p', str(threads)] if operation not in ('relabel',) else [])])


def expect_rejection(operation, source, output):
    output.write_bytes(OLD)
    try:
        operate(operation, source, output)
    except (ValueError, RuntimeError, OverflowError, EOFError, OSError, SystemExit):
        assert output.read_bytes() == OLD
        return
    pytest.fail(f'{operation} accepted malformed input; old output preserved: {output.read_bytes() == OLD}')


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('statistic', ['perBin', 'mean', 'median', 'min', 'max', 'sum'])
@pytest.mark.parametrize('action', ['removeRegion', 'maskSample'])
def test_filter_validates_ungated_values(tmp_path, statistic, action):
    source, output = (tmp_path / 'bad.gz', tmp_path / 'out.gz')
    header = {**_header(), 'sample_labels': ['a', 'b'], 'sample_boundaries': [0, 2, 4]}
    write_matrix(source, ROW.rstrip() + '\t3\toops\n', header)
    output.write_bytes(OLD)
    with pytest.raises(SystemExit, match='invalid numeric value'):
        validation_cmo.main(['filterValues', '-m', str(source), '-o', str(output), '--filterUsingSamples', 'a', '--filterUsingStatistic', statistic, '--onFilterFail', action])
    assert output.read_bytes() == OLD


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('operation', ['filterStrand', 'subset', 'mask', 'relabel', 'rbind', 'cbind'])
@pytest.mark.parametrize('damage', ['late_bad_value', 'missing_rows', 'missing_footer'])
def test_late_stream_failure_preserves_old_output(tmp_path, operation, damage):
    source, output = (tmp_path / 'late.gz', tmp_path / 'out.gz')
    count = 80000
    body = ROW * count
    header = _header(rows=count)
    if damage == 'late_bad_value':
        body = body[:-len(ROW)] + ROW.replace('\t2\n', '\toops\n')
    elif damage == 'missing_rows':
        header = _header(rows=count + 1)
    write_matrix(source, body, header)
    if damage == 'missing_footer':
        source.write_bytes(source.read_bytes()[:-8])
    expect_rejection(operation, source, output)


# Argument contract.

TEST_DATA = Path(__file__).parent / 'test_data'


MATRIX = TEST_DATA / 'computeMatrixOperations.mat.gz'


def test_cbind_rejects_rbind_only_group_policy(capsys):
    from deeptoolsr import computeMatrixOperations as operations
    with pytest.raises(SystemExit):
        operations.parse_arguments().parse_args(['cbind', '-m', 'one.gz', 'two.gz', '-o', 'output.gz', '--sameGroupLabels', 'separate'])
    assert 'unrecognized arguments' in capsys.readouterr().err


@pytest.mark.parametrize('extra,message', [(['--add', '1', '--pseudocount', '2'], 'unused without --ratio or --log2FC'), (['--add', '1', '--zeroHandling', 'any'], 'unused without --difference, --ratio, or --log2FC'), (['--scale', '2', '--scaleUsing', 'body'], 'unused without a row-sum/row-max/row-min/row-mean --scale')])
def test_explicit_inert_transform_controls_warn(tmp_path, monkeypatch, capsys, extra, message):
    from deeptoolsr import computeMatrixOperations as operations
    monkeypatch.setattr(operations, 'transformMatrices', lambda _args: None)
    operations.main(['transform', '-m', str(MATRIX), '-o', str(tmp_path / 'out.gz')] + extra)
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('specification', ['min-denominator/0', 'max-numerator/nan', 'min-denominator/1e-9999'])
def test_transform_pseudocount_divisor_is_validated_before_input_scan(tmp_path, specification):
    from deeptoolsr import computeMatrixOperations as operations
    with pytest.raises(SystemExit):
        operations.main(['transform', '-m', str(tmp_path / 'missing.gz'), '-o', str(tmp_path / 'out.gz'), '--ratio', '--pseudocount', specification])
    assert not (tmp_path / 'out.gz').exists()
