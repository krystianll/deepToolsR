"""Compare pinned CLI fixtures, allowing only libm silhouette rounding."""

from math import isclose


def assert_bed_equal(actual, expected, *, silhouette=False):
    actual_bytes = actual.read_bytes()
    expected_bytes = expected.read_bytes()
    if not silhouette:
        assert actual_bytes == expected_bytes
        return
    actual_lines = actual_bytes.splitlines(keepends=True)
    expected_lines = expected_bytes.splitlines(keepends=True)
    assert len(actual_lines) == len(expected_lines)
    for actual_line, expected_line in zip(actual_lines, expected_lines):
        actual_fields = actual_line.rsplit(b'\t', 1)
        expected_fields = expected_line.rsplit(b'\t', 1)
        assert actual_fields[0] == expected_fields[0]
        if expected_fields[1] == b'silhouette\n':
            assert actual_fields[1] == expected_fields[1]
        else:
            assert actual_fields[1].endswith(b'\n')
            assert expected_fields[1].endswith(b'\n')
            assert actual_fields[1].endswith(b'\r\n') == (
                expected_fields[1].endswith(b'\r\n'))
            assert isclose(float(actual_fields[1]),
                           float(expected_fields[1]), rel_tol=1e-12)


def assert_diagnostics_equal(actual, expected, *, silhouette=False):
    if not silhouette:
        assert actual == expected
        return
    assert actual['status'] == expected['status']
    for stream in ('stdout', 'stderr'):
        actual_lines = actual[stream].splitlines(keepends=True)
        expected_lines = expected[stream].splitlines(keepends=True)
        assert len(actual_lines) == len(expected_lines)
        for observed, reference in zip(actual_lines, expected_lines):
            prefix = 'The average silhouette score is: '
            if reference.startswith(prefix):
                assert observed.startswith(prefix)
                assert observed.endswith('\n') == reference.endswith('\n')
                assert isclose(float(observed[len(prefix):]),
                               float(reference[len(prefix):]), rel_tol=1e-12)
            else:
                assert observed == reference
