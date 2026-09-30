"""plotProfileR command entry point."""

from deeptoolsr import parserCommon
from deeptoolsr.plotMatrix import matrix_main
from deeptoolsr.prepare import parse_command


def parse_arguments(args=None):
    return parserCommon.plot_parser('plotProfileR')


def process_args(args=None):
    return parse_command('plotProfileR', args, 'cli')


main = matrix_main('plotProfileR')
