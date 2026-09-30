"""Describe plot commands without rendering them."""

import json
import sys

from deeptoolsr import options as run_options
from deeptoolsr.matrix import (
    MatrixFormatError, RowLayout, SourceKey, read_header, remove_empty_groups)
from deeptoolsr.plotting.matrix_spec import project_matrix_spec
from deeptoolsr.plotting.matrix_plan import resolve_matrix_plan
from deeptoolsr.prepare import (
    DataError, OptionError, _common_spec, _sample_columns, command_labels,
    option_error, parse_command, warning_text)


TOOLS = ('plotHeatmapR', 'plotProfileR', 'plotMatrixR')


def _record(tool, layout, labels, plan):
    bounds = layout.group_bounds
    return {
        'protocol': 1,
        'tool': tool,
        'groups': [
            {'index': index + 1, 'key': key, 'label': label,
             'regions': bounds[index + 1] - bounds[index]}
            for index, (key, label) in enumerate(
                zip(labels.group_keys, labels.groups))],
        'samples': [
            {'index': index + 1, 'label': label}
            for index, label in enumerate(labels.samples)],
        'panels': [
            {'index': panel.index, 'kind': panel.kind, 'title': panel.title,
             'row': panel.row + 1, 'column': panel.column + 1,
             'series': list(panel.series)}
            for panel in plan.panels],
        'row_labels': list(plan.row_labels),
        'cells': [
            {'index': cell.index, 'row': cell.row + 1,
             'column': cell.column + 1,
             'samples': [sample + 1 for sample in cell.samples],
             'groups': [group + 1 for group in cell.groups],
             'title': cell.title,
             'profile': cell.profile.index if cell.profile else None,
             'blocks': [
                 {'group': block.group + 1, 'sample': block.sample + 1}
                 for block in cell.blocks],
             'stack_labels': list(cell.stack_labels)}
            for cell in plan.cells],
        'series': [
            {'key': item.key, 'group': item.group + 1,
             'sample': item.sample + 1, 'label': item.label}
            for item in plan.series],
        'color_domains': [
            {'option': domain.option, 'kind': domain.kind,
             'slots': [
                 {'position': slot.position, 'series': list(slot.series),
                  'panels': list(slot.panels), 'cells': list(slot.cells)}
                 for slot in domain.slots]}
            for domain in plan.color_domains],
    }


def describe(tool, argv, *, force_load=False):
    """Describe a plot command without rendering or reading values unless needed."""
    from deeptoolsr.session import PlotSession

    with PlotSession(cache_bytes=0) as session:
        return describe_session(session, [tool, *argv], force_load=force_load)


def describe_session(session, command, *, force_load=False):
    """Describe using a resident matrix only when membership needs values."""
    tool, *argv = command
    if tool not in TOOLS:
        raise OptionError(f'unsupported tool: {tool}')
    args = parse_command(tool, argv, 'describe')
    if args.matrixFile is None:
        raise OptionError('the following arguments are required: --matrixFile/-m')
    session.apply_run_defaults(args, argv)
    path = run_options.config_path(args.config)
    raw = run_options.read_config_bytes(path)
    args.run_options = run_options.resolve_run_options_from_bytes(
        args, raw, path)
    threads = args.run_options.threads
    try:
        header = (session.matrix.header if session.matrix is not None and
                  session.source_key is not None and
                  session.source_key == SourceKey.of(args.matrixFile)
                  else read_header(args.matrixFile, normalize=True))
    except ValueError as error:
        raise MatrixFormatError(str(error)) from error
    spec = _common_spec(args, header)
    if force_load or spec.needs_values_for_membership():
        matrix = session.matrix_for(args.matrixFile, threads)
        header = matrix.header
        layout, _warnings = session.layout(spec, threads)
    else:
        if spec.sort_samples is not None:
            _sample_columns(spec.sort_samples, header, '--sortUsingSamples')
        layout = RowLayout.from_header(header)
        if layout.nrows == 0:
            raise DataError('No regions remain after filtering; cannot plot an '
                            'empty matrix.')
        layout, _removed = remove_empty_groups(layout)
    labels = command_labels(args, layout, spec, header)
    figure_spec = project_matrix_spec(args, tool)
    try:
        plan, _samples, _scales = resolve_matrix_plan(
            header, layout, labels, figure_spec)
    except OptionError:
        raise
    except ValueError as error:
        raise option_error(error) from error
    return _record(tool, layout, labels, plan)


def describe_cli(tool, argv):
    """Print exactly one JSON document, or one diagnostic and an exit code."""
    try:
        record = describe(tool, argv)
    except OptionError as error:
        print(error.prefix + str(error), file=sys.stderr)
        raise SystemExit(error.exit_code) from error
    except DataError as error:
        for warning in error.warnings:
            message = (warning_text(warning, error.layout)
                       if error.layout is not None else str(warning))
            print(message.rstrip(), file=sys.stderr)
        print(error, file=sys.stderr)
        raise SystemExit(2 if error.parser_error else 1) from error
    except OSError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
    except ValueError as error:
        # The CLI shows a raw ValueError as its traceback's last line.
        print(f'ValueError: {error}', file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(record, indent=2, ensure_ascii=False))
