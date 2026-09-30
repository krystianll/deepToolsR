"""Validation of computeMatrix header and in-memory geometry invariants."""

import numbers
import sys


SPECIAL_SAMPLE_PARAMETERS = frozenset({
    'unscaled 5 prime', 'unscaled 3 prime', 'body', 'downstream',
    'upstream', 'ref point', 'bin size',
})


def _integer(value, description):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("{} must contain integers".format(description))
    return int(value)


def validate_boundaries(values, description, *, expected_end=None,
                        require_positive_width=False):
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError("{} must contain at least [0, end]".format(description))
    result = [_integer(value, description) for value in values]
    if result[0] != 0:
        raise ValueError("{} must start at zero".format(description))
    for previous, current in zip(result, result[1:]):
        if current < previous or (require_positive_width and current == previous):
            relation = "strictly increasing" if require_positive_width else "monotonic"
            raise ValueError("{} must be non-negative and {}".format(
                description, relation))
    if expected_end is not None and result[-1] != expected_end:
        raise ValueError("{} must end at {}, not {}".format(
            description, expected_end, result[-1]))
    return result


def _sample_values(parameters, name, sample_count):
    if name not in parameters:
        return None
    return sample_parameter_values(parameters[name], name, sample_count)


def sample_parameter_values(value, name, sample_count):
    """Expand a shared scalar/singleton using the header validation contract."""
    values = value if isinstance(value, list) else [value]
    if len(values) not in (1, sample_count):
        raise ValueError(
            "header {!r} must contain one value or one per sample ({})".format(
                name, sample_count))
    return values * sample_count if len(values) == 1 else values


def normalize_sample_parameters(parameters):
    result = dict(parameters)
    sample_count = len(parameters['sample_labels'])
    for name in SPECIAL_SAMPLE_PARAMETERS:
        if name in parameters:
            result[name] = sample_parameter_values(parameters[name], name, sample_count)
    return result


def validate_matrix_header(parameters, *, rows=None, bins=None):
    if not isinstance(parameters, dict):
        raise ValueError("matrix header must be a JSON object")
    for required in ('group_boundaries', 'sample_boundaries',
                     'group_labels', 'sample_labels'):
        if required not in parameters:
            raise ValueError("matrix header is missing {!r}".format(required))

    group_boundaries = validate_boundaries(
        parameters['group_boundaries'], 'group_boundaries', expected_end=rows)
    sample_boundaries = validate_boundaries(
        parameters['sample_boundaries'], 'sample_boundaries', expected_end=bins,
        require_positive_width=True)
    declared_rows = group_boundaries[-1]
    declared_bins = sample_boundaries[-1]
    if declared_bins <= 0:
        raise ValueError("sample_boundaries must describe at least one bin")
    if declared_rows and declared_bins > sys.maxsize // declared_rows:
        raise ValueError("matrix dimensions overflow the platform index range")

    group_labels = parameters['group_labels']
    sample_labels = parameters['sample_labels']
    if not isinstance(group_labels, list) or \
            len(group_labels) != len(group_boundaries) - 1:
        raise ValueError("group label count must match group boundaries")
    if not isinstance(sample_labels, list) or \
            len(sample_labels) != len(sample_boundaries) - 1:
        raise ValueError("sample label count must match sample boundaries")

    sample_count = len(sample_labels)
    for name in SPECIAL_SAMPLE_PARAMETERS:
        _sample_values(parameters, name, sample_count)

    numeric = {}
    for name in ('unscaled 5 prime', 'unscaled 3 prime', 'body',
                 'downstream', 'upstream', 'bin size'):
        values = _sample_values(parameters, name, sample_count)
        if values is None:
            continue
        numeric[name] = [_integer(value, "header {!r}".format(name))
                         for value in values]
        minimum = 1 if name == 'bin size' else 0
        if any(value < minimum for value in numeric[name]):
            raise ValueError("header {!r} values must be >= {}".format(
                name, minimum))

    return declared_rows, declared_bins


def validate_matrix_geometry(matrix, group_boundaries, sample_boundaries,
                             group_labels=None, sample_labels=None,
                             regions=None):
    if getattr(matrix, 'ndim', None) != 2:
        raise ValueError("matrix must be two-dimensional")
    rows, bins = matrix.shape
    groups = validate_boundaries(
        group_boundaries, 'group_boundaries', expected_end=rows)
    samples = validate_boundaries(
        sample_boundaries, 'sample_boundaries', expected_end=bins,
        require_positive_width=True)
    if regions is not None and len(regions) != rows:
        raise ValueError("region metadata count must match matrix rows")
    if group_labels is not None and len(group_labels) != len(groups) - 1:
        raise ValueError("number of group labels does not match number of groups")
    if sample_labels is not None and len(sample_labels) != len(samples) - 1:
        raise ValueError("number of sample labels does not match number of samples")
    return groups, samples
