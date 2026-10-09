"""Strict, reusable numerical argument validators.

Python accepts arbitrarily large integers and converts strings such as ``1e999``
to infinity.  Native extensions and plotting libraries generally do neither
safely, so command-line values are normalized here before inputs or outputs are
opened.  Public native entry points repeat their own essential checks.
"""

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import re
import sys


INT32_MAX = (1 << 31) - 1
UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
FLOAT32_MAX = float.fromhex('0x1.fffffep+127')
FLOAT32_MIN_SUBNORMAL = float.fromhex('0x1p-149')


def _type_error(value, requirement):
    raise argparse.ArgumentTypeError(
        "{} is invalid; expected {}".format(repr(value), requirement))


def bounded_int(minimum=None, maximum=None, requirement=None):
    """Return an argparse converter for a strict, bounded base-10 integer."""
    if requirement is None:
        if minimum is not None and maximum is not None:
            requirement = "an integer from {} to {}".format(minimum, maximum)
        elif minimum is not None:
            requirement = "an integer >= {}".format(minimum)
        elif maximum is not None:
            requirement = "an integer <= {}".format(maximum)
        else:
            requirement = "an integer"

    def convert(value):
        try:
            # int() is deliberately strict about decimal/exponent notation:
            # values such as 5.0 and 5e0 are not silently truncated.
            result = int(value)
        except (TypeError, ValueError, OverflowError):
            _type_error(value, requirement)
        if isinstance(value, float) and not value.is_integer():
            _type_error(value, requirement)
        if minimum is not None and result < minimum:
            _type_error(value, requirement)
        if maximum is not None and result > maximum:
            _type_error(value, requirement)
        return result

    return convert


positive_int = bounded_int(1, INT32_MAX, "a positive integer")
nonnegative_int = bounded_int(0, INT32_MAX, "a non-negative integer")
positive_uint32 = bounded_int(1, UINT32_MAX, "an integer from 1 to 4294967295")
uint16_int = bounded_int(0, UINT16_MAX, "an integer from 0 to 65535")
uint8_int = bounded_int(0, 255, "an integer from 0 to 255")


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        _type_error(value, "a finite floating-point number")
    if not math.isfinite(result):
        _type_error(value, "a finite floating-point number")
    # float() silently turns sufficiently small non-zero decimal input into
    # signed zero. Treat that as numeric underflow rather than user-requested 0.
    if result == 0.0:
        try:
            if Decimal(str(value)) != 0:
                _type_error(value, "a finite floating-point number without underflow")
        except InvalidOperation:
            _type_error(value, "a finite floating-point number")
    return result


def float32_finite(value):
    """Finite scalar representable by the float32 matrix/output buffers."""
    result = finite_float(value)
    magnitude = abs(result)
    if magnitude > FLOAT32_MAX or (0 < magnitude < FLOAT32_MIN_SUBNORMAL):
        _type_error(value, "a finite floating-point number within float32 range")
    return result


def positive_finite_float(value):
    result = finite_float(value)
    if result <= 0:
        _type_error(value, "a finite floating-point number > 0")
    return result


def bounded_finite_float(minimum=None, maximum=None, *,
                         minimum_inclusive=True, maximum_inclusive=True,
                         requirement=None):
    """Return a strict finite-float converter with configurable endpoints."""
    if requirement is None:
        left = "[" if minimum_inclusive else "("
        right = "]" if maximum_inclusive else ")"
        requirement = "a finite floating-point number in {}{}, {}{}".format(
            left, "-inf" if minimum is None else minimum,
            "+inf" if maximum is None else maximum, right)

    def convert(value):
        result = finite_float(value)
        if minimum is not None:
            invalid = (result < minimum if minimum_inclusive
                       else result <= minimum)
            if invalid:
                _type_error(value, requirement)
        if maximum is not None:
            invalid = (result > maximum if maximum_inclusive
                       else result >= maximum)
            if invalid:
                _type_error(value, requirement)
        return result

    return convert


def lower_bound_float(value):
    """A finite lower bound, or -inf to express no lower bound."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        _type_error(value, "a finite lower bound or -inf")
    if math.isnan(result) or result == math.inf:
        _type_error(value, "a finite lower bound or -inf")
    return result


def upper_bound_float(value):
    """A finite upper bound, or +inf to express no upper bound."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        _type_error(value, "a finite upper bound or +inf")
    if math.isnan(result) or result == -math.inf:
        _type_error(value, "a finite upper bound or +inf")
    return result


def fraction(value):
    result = finite_float(value)
    if result < 0 or result > 1:
        _type_error(value, "a finite number from 0 to 1")
    return result


def compression_level(value):
    result = bounded_int(-1, 12, "-1 or an integer from 1 to 12")(value)
    if result == 0:
        _type_error(value, "-1 or an integer from 1 to 12")
    return result


def extend_reads(value):
    """Explicit extension length; argparse's const=-1 bypasses this converter."""
    return positive_int(value)


def positive_int_or_max(value):
    if str(value).lower() == "max":
        return "max"
    return positive_int(value)


_ASSIGNMENT_INDEX = r'\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*'
_ASSIGNMENT = re.compile(r'^\s*(' + _ASSIGNMENT_INDEX + r')\s*=(.*)$', re.S)


def assignment_parts(value):
    """Return the index specification and value of one ``N=value`` entry."""
    match = _ASSIGNMENT.match(value) if isinstance(value, str) else None
    return match.groups() if match else None


def decode_text_escapes(value):
    """Decode only the three documented escapes in a label or title."""
    if not isinstance(value, str):
        return value
    result = []
    index = 0
    translations = {'=': '=', '\\': '\\', 'n': '\n'}
    while index < len(value):
        if value[index] == '\\' and index + 1 < len(value):
            following = value[index + 1]
            if following in translations:
                result.append(translations[following])
                index += 2
                continue
        result.append(value[index])
        index += 1
    return ''.join(result)


def assigned_finite_float(*, keywords=(), allow_empty=False):
    """Validate a numeric token while preserving optional ``N,M=value`` syntax."""
    allowed = frozenset(str(keyword).lower() for keyword in keywords)

    def convert(value):
        text = str(value)
        payload = text
        assignment = assignment_parts(text)
        if assignment:
            payload = assignment[1]
        if allow_empty and payload == '':
            return text
        if payload.lower() in allowed:
            return text
        finite_float(payload)
        return text

    return convert


def finite_float_or_empty(value):
    if value == '':
        return value
    return finite_float(value)


def min_max_error(minimum, maximum, minimum_name, maximum_name, *, strict=False):
    """Return an error string for an invalid pair, otherwise ``None``."""
    if minimum is None or maximum is None:
        return None
    invalid = minimum >= maximum if strict else minimum > maximum
    if invalid:
        relation = "less than" if strict else "less than or equal to"
        return "{} must be {} {}".format(minimum_name, relation, maximum_name)
    return None


def recycled_min_max_error(minima, maxima, count, minimum_name,
                           maximum_name, *, strict=False):
    """Validate every pair that independently recycled lists give `count` uses.

    Position ``i`` pairs ``minima[i % len(minima)]`` with
    ``maxima[i % len(maxima)]``; None (or 'auto') entries are data-derived and
    are not checked here.
    """
    if not minima or not maxima:
        return None
    for index in range(count):
        minimum = minima[index % len(minima)]
        maximum = maxima[index % len(maxima)]
        if minimum in (None, '', 'auto') or maximum in (None, '', 'auto'):
            continue
        error = min_max_error(
            float(minimum), float(maximum), minimum_name, maximum_name,
            strict=strict)
        if error:
            return "{} for recycled pair {} ({:g}, {:g})".format(
                error, index + 1, float(minimum), float(maximum))
    return None


def specified_options(argv):
    """Return option spellings explicitly present in an argv sequence."""
    options = set()
    for token in argv or ():
        text = str(token)
        if text.startswith('-') and text != '-':
            options.add(text.split('=', 1)[0])
    return frozenset(options)


def option_was_supplied(options, *spellings):
    return any(spelling in options for spelling in spellings)


@dataclass(frozen=True)
class CompatibilityRule:
    """One declarative post-parse error or explicit-option warning."""

    predicate: object
    message: str
    severity: str = 'error'


class CompatibilityError(ValueError):
    """A rule failure and the warnings that preceded it."""

    def __init__(self, message, warnings):
        super().__init__(message)
        self.warnings = tuple(warnings)


def compatibility_warnings(namespace, options, rules):
    """Evaluate rules without a parser or output stream."""
    warnings = []
    for rule in rules:
        if not rule.predicate(namespace, options):
            continue
        if rule.severity == 'warning':
            warnings.append("Warning: {}\n".format(rule.message))
        elif rule.severity == 'error':
            raise CompatibilityError(rule.message, warnings)
        else:
            raise ValueError("unknown compatibility-rule severity: {}".format(
                rule.severity))
    return tuple(warnings)


def validate_compatibility_rules(parser, namespace, options, rules,
                                 warning_stream=None):
    """Apply cross-option rules before any input is opened.

    Predicates receive ``(namespace, explicitly_supplied_options)``. Errors use
    argparse's normal concise failure path; warnings are emitted only when the
    rule predicate (normally including an explicit-option test) requests one.
    """
    stream = warning_stream if warning_stream is not None else sys.stderr
    try:
        warnings = compatibility_warnings(namespace, options, rules)
    except CompatibilityError as error:
        for warning in error.warnings:
            stream.write(warning)
        parser.error(str(error))
    else:
        for warning in warnings:
            stream.write(warning)
