from tests.test_matrix_validation import _header as validation__header
from deeptoolsr import computeMatrixOperations as cmo
import gzip
import json
import os

import numpy as np
import pytest

import deeptoolsr.matrix as matrix_module
from deeptoolsr.matrix import Matrix, Labels, RowLayout, save


native_io = pytest.importorskip('deeptoolsr._compute_matrix_io')
native_compute = pytest.importorskip('deeptoolsr._compute_matrix_native')
TEST_DATA = os.path.join(os.path.dirname(__file__), 'test_data')


def test_native_writer_serializes_nan_value(tmp_path):
    path = tmp_path / 'missing.gz'
    values = np.arange(5, dtype=np.float32).reshape(1, 5)
    values[0, 2] = np.nan
    data = values
    regions = [['chr1', [(0, 5)], 'region', 0, '+', '0']]

    native_io.write_matrix(
        str(path), json.dumps(_header(1, 5), separators=(',', ':')),
        regions, data, 1, 6, 256)

    with gzip.open(path, 'rt') as handle:
        fields = handle.readlines()[1].rstrip().split('\t')
    assert fields[8] == 'nan'


def test_writer_borrows_cpp_owned_matrix_buffer(tmp_path):
    path = tmp_path / 'cpp-owned.gz'
    matrix = native_compute.NativeMatrix(2, 3)
    values = np.asarray(matrix)
    values[:] = [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]
    regions = [
        ['chr1', [(0, 3)], 'first', 0, '+', '0'],
        ['chr1', [(3, 6)], 'second', 0, '+', '0'],
    ]

    native_io.write_matrix(
        str(path), json.dumps(_header(2, 3), separators=(',', ':')),
        regions, matrix, 1)

    with gzip.open(path, 'rt') as handle:
        lines = handle.readlines()[1:]
    assert lines[0].rstrip().split('\t')[6:] == ['1.000000', 'nan', '3.000000']
    assert lines[1].rstrip().split('\t')[6:] == ['4.000000', '5.000000', '6.000000']


def test_writer_applies_row_permutation_without_reordering_matrix(tmp_path):
    path = tmp_path / 'permuted.gz'
    values = np.asarray([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float32)
    regions = [
        ['chr1', [(0, 2)], 'first', 0, '+', '0'],
        ['chr1', [(2, 4)], 'second', 0, '+', '0'],
        ['chr1', [(4, 6)], 'third', 0, '+', '0'],
    ]

    native_io.write_matrix(
        str(path), json.dumps(_header(3, 2), separators=(',', ':')),
        regions, values, 1, 6, 256,
        np.array([2, 0, 1], dtype=np.int64))

    with gzip.open(path, 'rt') as handle:
        lines = [line.rstrip().split('\t') for line in handle.readlines()[1:]]
    assert [line[3] for line in lines] == ['third', 'first', 'second']
    assert [[float(value) for value in line[6:]] for line in lines] == [
        [30.0, 31.0], [10.0, 11.0], [20.0, 21.0]]


def _header(rows, bins, threads=4):
    return {
        'sample_labels': ['sample'],
        'sample_boundaries': [0, bins],
        'group_labels': ['group'],
        'group_boundaries': [0, rows],
        'bin size': [1],
        'upstream': [1],
        'body': [max(0, bins - 2)],
        'downstream': [1],
        'unscaled 5 prime': [0],
        'unscaled 3 prime': [0],
        'ref point': [None],
        'proc number': threads,
    }


def test_parallel_reader_error_propagates_and_pool_recovers(tmp_path):
    rows = 40
    good = tmp_path / 'valid.gz'
    broken = tmp_path / 'invalid.gz'
    regions = [
        ['chr1', [(i, i + 1)], str(i), 0, '+', '0'] for i in range(rows)
    ]
    native_io.write_matrix(
        str(good), json.dumps(_header(rows, 2), separators=(',', ':')),
        regions, np.ones((rows, 2), dtype=np.float32), 1)
    lines = gzip.decompress(good.read_bytes()).splitlines(keepends=True)
    lines[-1] = lines[-1].replace(b'1.000000', b'bad', 1)
    broken.write_bytes(gzip.compress(b''.join(lines), mtime=0))

    with pytest.raises(RuntimeError, match=r'row 40: .*'):
        native_io.read_matrix(str(broken), rows, 2, [0, rows], 4, 4096)
    _, _, matrix = native_io.read_matrix(str(good), rows, 2, [0, rows], 4, 4096)
    np.testing.assert_array_equal(matrix, np.ones((rows, 2), dtype=np.float32))


def test_native_reader_matches_independently_parsed_file():
    path = os.path.join(TEST_DATA, 'somegenes.txt.gz')
    native = Matrix.load(path, threads=1)
    with gzip.open(path, 'rt') as handle:
        header = json.loads(handle.readline()[1:])
        fields = [line.rstrip().split('\t') for line in handle if line.strip()]
    expected = np.array([[float(x) for x in row[6:]] for row in fields])
    assert [r[2] for r in native.regions] == [r[3] for r in fields]
    assert native.header.parameters['sample_boundaries'] == header['sample_boundaries']
    np.testing.assert_allclose(native.values, expected, equal_nan=True)


def test_parallel_native_round_trip_with_multiblock_regions(tmp_path):
    rows, bins = 17, 5
    path = tmp_path / 'parallel.gz'
    parameters = _header(rows, bins)
    regions = []
    for row in range(rows):
        regions.append([
            'chr1', [(row * 10, row * 10 + 3),
                     (row * 10 + 7, row * 10 + 10)],
            'region{}'.format(row), rows,
            '-' if row % 2 else '+', str(row / 10),
        ])
    values = np.arange(rows * bins, dtype=np.float32).reshape(rows, bins)
    values[3, 2] = np.nan

    native_io.write_matrix(
        str(path), json.dumps(parameters, separators=(',', ':')),
        regions, values, 4, 6, 2)

    with gzip.open(path, 'rt') as handle:
        lines = handle.readlines()
    assert len(lines) == rows + 1
    assert lines[0].startswith('@')

    reader = Matrix.load(path, threads=1)
    assert reader.regions == regions
    assert np.isnan(reader.values[3, 2])
    np.testing.assert_allclose(
        np.ma.filled(reader.values, np.nan), values, equal_nan=True)


def test_native_output_contains_expected_numeric_text(tmp_path):
    source = Matrix.load(os.path.join(TEST_DATA, 'somegenes.txt.gz'), threads=1)
    output = tmp_path / 'native.gz'
    save(source, RowLayout.identity(source),
         Labels(source.header.group_labels, source.header.group_labels,
                source.header.sample_labels), output,
         compressed=str(output).endswith('.gz'), threads=1)
    with gzip.open(output, 'rt') as handle:
        json.loads(handle.readline()[1:])
        fields = [line.rstrip().split('\t') for line in handle if line.strip()]
    assert [r[3] for r in fields] == [r[2] for r in source.regions]
    values = np.array([[float(x) for x in r[6:]] for r in fields])
    np.testing.assert_allclose(values, source.values, atol=1e-6, equal_nan=True)


def test_plain_matrix_round_trip_is_selected_by_output_suffix(tmp_path):
    source = Matrix.load(os.path.join(TEST_DATA, 'somegenes.txt.gz'), threads=1)
    output = tmp_path / 'matrix.mat'

    save(source, RowLayout.identity(source),
         Labels(source.header.group_labels, source.header.group_labels,
                source.header.sample_labels), output, compressed=False, threads=1)

    assert output.read_bytes().startswith(b'@')
    restored = Matrix.load(output, threads=1)
    assert restored.regions == source.regions
    assert restored.header.sample_labels == source.header.sample_labels
    np.testing.assert_allclose(
        np.ma.filled(restored.values, np.nan),
        np.ma.filled(source.values, np.nan),
        atol=1e-6, equal_nan=True)


@pytest.mark.parametrize('compressed', [False, True])
def test_matrix_header_escapes_control_characters_in_labels(
        tmp_path, compressed):
    source = Matrix.load(os.path.join(TEST_DATA, 'somegenes.txt.gz'), threads=1)
    sample_labels = [
        'sample{}\nline\tcolumn\\slash\rreturn'.format(index)
        for index in range(len(source.header.sample_labels))]
    group_labels = [
        'group{}\nline\tcolumn\\slash\rreturn'.format(index)
        for index in range(len(source.header.group_labels))]
    labels = Labels(tuple(group_labels), tuple(group_labels),
                    tuple(sample_labels))
    output = tmp_path / ('matrix.mat.gz' if compressed else 'matrix.mat')

    save(source, RowLayout.identity(source), labels, output,
         compressed=compressed, threads=1)

    opener = gzip.open if compressed else open
    with opener(output, 'rt', newline='') as handle:
        header_line = handle.readline()
        assert handle.readline()  # the first body row was not absorbed
    assert header_line.startswith('@')
    assert header_line.endswith('\n')
    assert header_line.count('\n') == 1
    assert '\r' not in header_line
    parsed = json.loads(header_line[1:])
    assert parsed['sample_labels'] == sample_labels
    assert parsed['group_labels'] == group_labels


def test_matrix_input_compression_is_detected_from_magic_not_suffix(tmp_path):
    source = os.path.join(TEST_DATA, 'somegenes.txt.gz')
    disguised = tmp_path / 'compressed-without-gz.mat'
    with open(source, 'rb') as input_handle, open(disguised, 'wb') as output_handle:
        output_handle.write(input_handle.read())

    restored = Matrix.load(disguised, threads=1)

    expected = Matrix.load(source, threads=1)
    np.testing.assert_allclose(
        np.ma.filled(restored.values, np.nan),
        np.ma.filled(expected.values, np.nan), equal_nan=True)


def test_save_matrix_preserves_existing_output_on_native_failure(
        tmp_path, monkeypatch):
    source = Matrix.load(os.path.join(TEST_DATA, 'somegenes.txt.gz'), threads=1)
    output = tmp_path / 'matrix.gz'
    output.write_bytes(b'previous matrix')

    def fail_after_partial_write(path, *_args, **_kwargs):
        with open(path, 'wb') as handle:
            handle.write(b'partial matrix')
        raise RuntimeError('injected matrix failure')

    monkeypatch.setattr(matrix_module._compute_matrix_io, 'write_matrix',
                        fail_after_partial_write)
    with pytest.raises(RuntimeError, match='injected matrix failure'):
        save(source, RowLayout.identity(source),
             Labels(source.header.group_labels, source.header.group_labels,
                    source.header.sample_labels), output,
             compressed=True, threads=1)
    assert output.read_bytes() == b'previous matrix'


# From read validation regressions.

ROW = 'chr1\t0\t2\tr\t0\t+\t1\t2\n'


OLD = b'previous successful output'


@pytest.fixture()
def validation_isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPTOOLSR_CONFIG_DIR', str(tmp_path / 'config'))


def write_matrix(path, body=ROW, header=None):
    text = '@' + json.dumps(validation__header() if header is None else header) + '\n' + body
    path.write_bytes(gzip.compress(text.encode()))


def operate(operation, source, output, threads=1):
    args = {'filterStrand': ['--strand', '+'], 'subset': ['--samples', 'sample'], 'reorder': ['--samples', 'sample'], 'mask': ['--onFilterFail', 'maskSample'], 'drop': [], 'relabel': ['--sampleLabels', 'new'], 'rbind': ['--sameGroupLabels', 'separate'], 'cbind': ['--blind']}
    if operation == 'read':
        return Matrix.load(source, threads=threads)
    command = 'filterValues' if operation in ('mask', 'drop') else operation
    cmo.main([command, '-m', str(source), *([str(source)] if operation in ('rbind', 'cbind') else []), '-o', str(output), *args[operation], *(['-p', str(threads)] if operation not in ('relabel',) else [])])


def expect_rejection(operation, source, output):
    output.write_bytes(OLD)
    try:
        operate(operation, source, output)
    except (ValueError, RuntimeError, OverflowError, EOFError, OSError, SystemExit):
        assert output.read_bytes() == OLD
        return
    pytest.fail(f'{operation} accepted malformed input; old output preserved: {output.read_bytes() == OLD}')


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('processors', [1, 3])
@pytest.mark.parametrize('encoding', ['plain', 'gzip', 'members'])
@pytest.mark.parametrize('ending', ['lf', 'crlf', 'no_final_newline'])
@pytest.mark.parametrize('operation', ['read', 'subset', 'mask'])
def test_valid_matrix_encodings_and_line_endings(tmp_path, monkeypatch, processors, encoding, ending, operation):
    header = '@' + json.dumps(validation__header()) + '\n'
    body = ROW
    if ending == 'crlf':
        header, body = (header.replace('\n', '\r\n'), body.replace('\n', '\r\n'))
    elif ending == 'no_final_newline':
        body = body.rstrip('\n')
    data = (header + body).encode()
    if encoding == 'gzip':
        data = gzip.compress(data)
    elif encoding == 'members':
        data = gzip.compress(header.encode() + body[:7].encode()) + gzip.compress(body[7:].encode())
    source, output = (tmp_path / 'input.dat', tmp_path / 'out.gz')
    source.write_bytes(data)
    result = operate(operation, source, output, processors)
    if operation != 'read':
        result = Matrix.load(output, threads=1)
    np.testing.assert_array_equal(result.values, [[1.0, 2.0]])


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('operation', ['read', 'filterStrand', 'subset', 'mask', 'drop'])
def test_bad_gzip_crc_is_rejected_and_output_preserved(tmp_path, operation):
    source, output = (tmp_path / 'bad.gz', tmp_path / 'out.gz')
    write_matrix(source)
    data = bytearray(source.read_bytes())
    data[-8] ^= 1
    source.write_bytes(data)
    expect_rejection(operation, source, output)


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('removed_bytes', [4, 8])
@pytest.mark.parametrize('operation', ['read', 'filterStrand', 'subset', 'mask', 'drop'])
def test_incomplete_gzip_footer_must_abort(tmp_path, operation, removed_bytes):
    source, output = (tmp_path / 'truncated.gz', tmp_path / 'out.gz')
    write_matrix(source)
    source.write_bytes(source.read_bytes()[:-removed_bytes])
    with pytest.raises(EOFError):
        gzip.decompress(source.read_bytes())
    expect_rejection(operation, source, output)


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('body', ['', ROW * 2], ids=['too_few', 'too_many'])
@pytest.mark.parametrize('operation', ['subset', 'reorder', 'mask', 'relabel', 'rbind', 'cbind'])
def test_streaming_row_count_must_match_header(tmp_path, operation, body):
    source, output = (tmp_path / 'bad.gz', tmp_path / 'out.gz')
    write_matrix(source, body)
    expect_rejection(operation, source, output)


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('body', ['', ROW * 2], ids=['too_few', 'too_many'])
@pytest.mark.parametrize('operation', ['read', 'filterStrand', 'drop'])
def test_count_checking_routes_preserve_output(tmp_path, operation, body):
    source, output = (tmp_path / 'bad.gz', tmp_path / 'out.gz')
    write_matrix(source, body)
    expect_rejection(operation, source, output)


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('body', [ROW.rstrip() + '\t3\n', ROW.replace('\t0\t2\t', '\t5\t2\t'), ROW.replace('\t2\n', '\toops\n')], ids=['extra_bin', 'reversed_block', 'invalid_number'])
@pytest.mark.parametrize('operation', ['filterStrand', 'subset', 'relabel', 'rbind', 'cbind'])
def test_streaming_rows_must_be_valid(tmp_path, operation, body):
    source, output = (tmp_path / 'bad.gz', tmp_path / 'out.gz')
    write_matrix(source, body)
    expect_rejection(operation, source, output)


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('body', [ROW.rstrip() + '\t3\n', ROW.replace('\t0\t2\t', '\t5\t2\t'), ROW.replace('\t2\n', '\toops\n')])
def test_resident_row_validation_rejects_same_inputs(tmp_path, body):
    source, output = (tmp_path / 'bad.gz', tmp_path / 'out.gz')
    write_matrix(source, body)
    expect_rejection('read', source, output)


@pytest.mark.usefixtures('validation_isolated_config')
@pytest.mark.parametrize('operation', ['subset', 'reorder'])
def test_shared_sample_geometry_works_for_gzip_output(tmp_path, operation):
    source = tmp_path / 'two_samples.gz'
    header = {**validation__header(), 'sample_labels': ['a', 'b'], 'sample_boundaries': [0, 2, 4]}
    write_matrix(source, ROW.rstrip() + '\t3\t4\n', header)
    plain = tmp_path / 'plain.mat'
    cmo.main([operation, '-m', str(source), '-o', str(plain), '--samples', 'b'])
    matrix = Matrix.load(plain, threads=1)
    np.testing.assert_array_equal(matrix.values, [[3.0, 4.0]])
    compressed = tmp_path / 'compressed.gz'
    cmo.main([operation, '-m', str(source), '-o', str(compressed), '--samples', 'b'])
    matrix = Matrix.load(compressed, threads=1)
    np.testing.assert_array_equal(matrix.values, [[3.0, 4.0]])


@pytest.mark.usefixtures('validation_isolated_config')
def test_native_matrix_read_rejects_oversized_zlib_buffer(tmp_path):
    from deeptoolsr import _compute_matrix_io
    source = tmp_path / 'input.gz'
    write_matrix(source)
    with pytest.raises(RuntimeError, match='buffer'):
        _compute_matrix_io.read_matrix(str(source), 1, 2, [0, 1], 1, 1 << 32)
