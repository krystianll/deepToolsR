"""Plot CLI file operations performed after argument parsing."""

import argparse

from deeptoolsr import parserCommon
from deeptoolsr.matrix import Matrix, MatrixFormatError
from deeptoolsr.path_validation import validate_input_output_paths
from deeptoolsr.prepare import DataError


def load_plot_matrix(path, threads):
    try:
        return Matrix.load(path, threads)
    except OSError as error:
        raise DataError(
            "argument --matrixFile/-m: can't open {!r}: {}".format(path, error),
            parser_error=True) from error
    except ValueError as error:
        raise MatrixFormatError(str(error)) from error


def check_plot_outputs(matrix_path, outputs):
    """Validate output destinations immediately before staging begins."""
    for path, option in outputs:
        if path is not None:
            try:
                parserCommon.writableFile(path)
            except argparse.ArgumentTypeError as error:
                raise DataError(f'argument {option}: {error}',
                                parser_error=True) from error
    validate_input_output_paths([matrix_path], [path for path, _ in outputs])
