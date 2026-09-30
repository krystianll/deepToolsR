"""The shared plot command front end over one matrix and cell plan."""

from contextlib import ExitStack
import sys

from deeptoolsr import parserCommon
from deeptoolsr.matrix import save, save_bed
from deeptoolsr.matrix_file import matrix_output_is_compressed
from deeptoolsr.path_validation import atomic_output_path
from deeptoolsr.prepare import (
    DataError, OptionError, PlotData, _common_spec, command_labels,
    option_error, parse_command, warning_text)
from deeptoolsr.plotting.cli_io import check_plot_outputs
from deeptoolsr.plotting.matrix_plan import resolve_matrix_plan
from deeptoolsr.plotting.matrix_spec import project_matrix_spec
from deeptoolsr.plotting.matrix_figure import build_matrix_figure
from deeptoolsr.plotting.prepared import prepare_matrix
from deeptoolsr.plotting.profile import write_profile_table
from deeptoolsr.plotting.rendering import (
    resolve_figure_format, save_figure_atomic)
from deeptoolsr.stats import statistics_spec


class _NumericProvider:
    """Batch projected statistics and scan misses through session caches."""

    def __init__(self, session, cancelled, progress):
        self.session = session
        self.cancelled = cancelled
        self.progress = progress
        self.limits_staged = False

    def statistics_batch(self, data, pairs, spec, threads):
        return self.session.statistics_batch(data, pairs, spec, threads)

    def scan(self, data, probabilities, exact, threads):
        if not self.limits_staged:
            self.session.stage('limits', self.cancelled, self.progress)
            self.limits_staged = True
        return self.session.scan(data, probabilities, exact, threads)

    def render(self, requests, threads):
        if not requests:
            return
        self.session.stage('bitmaps', self.cancelled, self.progress)
        self.session.render_bitmaps(requests, threads)


def _prepare_data(session, args, emit):
    threads = args.run_options.threads
    matrix = session.matrix
    spec = _common_spec(args, matrix.header)
    if args.hclust is not None:
        emit('stdout', 'Performing hierarchical clustering.'
             'Please note that it might be very slow for large datasets.\n\n')
    layout, warnings = session.layout(spec, threads)
    labels = command_labels(args, layout, spec, matrix.header)
    for warning in warnings:
        if not args.show_heatmap or not warning.kind.startswith('silhouette_'):
            emit('stderr', warning_text(
                warning, layout, regions_label=args.regionsLabel))
    if args.show_heatmap:
        if args.sortRegions != 'no' and args.sortUsingSamples is not None:
            emit('stdout', 'Samples used for ordering within each group:  '
                 '{}\n'.format([index - 1 for index in args.sortUsingSamples]))
        for warning in warnings:
            if warning.kind.startswith('silhouette_'):
                emit('stderr', warning_text(warning, layout))
    elif (spec.row_order_consumed and spec.sort_method not in ('no', 'keep')
          and spec.cluster_k is None and args.sortUsingSamples is not None):
        emit('stdout', 'Samples used for ordering within each group:  '
             '{}\n'.format([index - 1 for index in args.sortUsingSamples]))
    return (PlotData(matrix, layout, labels, threads,
                     args.distanceUnit, args.distanceUnitLocation), spec)


# How each tool asks for a profile panel, for errors that need one.
_PROFILE_PANEL_OPTION = {
    'plotMatrixR': '--profile',
    'plotHeatmapR': "--whatToShow 'plot, heatmap and colorbar'",
}


def _check_outputs(args):
    if args.outFileNameData and not args.show_profile:
        raise OptionError(
            '--outFileNameData writes the profile values and needs a '
            'profile panel ({})'.format(_PROFILE_PANEL_OPTION[args.tool]),
            prefix='Error: ', exit_code=1)
    outputs = [
        (args.outFileName, '--outFileName/-out/-o'),
        (args.outFileNameData, '--outFileNameData'),
        (args.outFileNameMatrix, '--outFileNameMatrix'),
        (args.outFileSortedRegions, '--outFileSortedRegions')]
    try:
        check_plot_outputs(args.matrixFile, outputs)
    except ValueError as error:
        if isinstance(error, DataError):
            raise
        raise option_error(error) from error


def _stage_outputs(outputs, args):
    """Stage the auxiliary outputs; each is published atomically."""
    def staged(path, suffix):
        return (outputs.enter_context(atomic_output_path(path, suffix=suffix))
                if path else None)

    return (staged(args.outFileNameData, '.profile.tsv.tmp'),
            staged(args.outFileNameMatrix, '.matrix.tmp'),
            staged(args.outFileSortedRegions, '.regions.tmp'))


def _write_outputs(paths, args, data, plan, spec, prepared):
    table, matrix, regions = paths
    if table:
        write_profile_table(table, plan, prepared, spec)
    if matrix:
        save(data.matrix, data.layout, data.labels, matrix,
             compressed=matrix_output_is_compressed(args.outFileNameMatrix),
             threads=args.run_options.threads)
    if regions:
        with open(regions, 'w', newline='', encoding='utf-8') as handle:
            save_bed(data.matrix, data.layout, data.labels, handle)


def _stage_data(session, args, cancelled, progress, emit):
    session.stage('load', cancelled, progress)
    session.matrix_for(args.matrixFile, args.run_options.threads)
    session.stage('order', cancelled, progress)
    try:
        return _prepare_data(session, args, emit)
    except DataError as error:
        for warning in error.warnings:
            emit('stderr', warning_text(
                warning, error.layout, regions_label=args.regionsLabel)
                if error.layout is not None else str(warning))
        raise


def _plan_for_request(args, tool, data):
    _check_outputs(args)
    figure_spec = project_matrix_spec(args, tool)
    try:
        plan, sample_plans, scales = resolve_matrix_plan(
            data.matrix.header, data.layout, data.labels, figure_spec)
    except OptionError:
        raise
    except ValueError as error:
        raise option_error(error) from error
    return figure_spec, plan, sample_plans, scales


def _prepare_numeric(session, args, data, figure_spec, plan, sample_plans,
                     scales, cancelled, progress, emit):
    numeric = _NumericProvider(session, cancelled, progress)
    session.stage('statistics', cancelled, progress)
    try:
        prepared = prepare_matrix(
            data, plan, sample_plans, scales, figure_spec, args.run_options,
            numeric=numeric)
    except ValueError as error:
        if isinstance(error, (DataError, OptionError)):
            raise
        raise option_error(error) from error
    for message in prepared.warnings:
        emit('stderr', f'WARNING: {message}\n')
    if not numeric.limits_staged:
        session.stage('limits', cancelled, progress)
    return prepared, numeric


def run_session_request(session, args, run_options, *, cancelled, progress,
                        emit):
    """Run one parsed plot request; the CLI and worker share this path."""
    from deeptoolsr.session import (project_color_spec,
                                    project_scan_spec, project_scene_digest)

    args.run_options = run_options
    data, data_spec = _stage_data(session, args, cancelled, progress, emit)
    figure_spec, plan, sample_plans, scales = _plan_for_request(
        args, args.tool, data)
    prepared, numeric = _prepare_numeric(
        session, args, data, figure_spec, plan, sample_plans, scales,
        cancelled, progress, emit)
    digest = project_scene_digest(
        args, figure_spec, data_spec,
        statistics_spec(figure_spec) if figure_spec.show_profile else None,
        project_scan_spec(figure_spec), project_color_spec(figure_spec),
        data.layout, config_digest=session.config_digest)
    with ExitStack() as outputs:
        plot_path = outputs.enter_context(
            atomic_output_path(args.outFileName, suffix='.plot.tmp'))
        paths = _stage_outputs(outputs, args)
        session.stage('scene', cancelled, progress)
        try:
            figure, _ = build_matrix_figure(
                data.matrix, data.layout, data.labels, plan, figure_spec,
                prepared, run_options, session.extents,
                raster_renderer=numeric.render)
            session.stage('publish', cancelled, progress)
            save_figure_atomic(
                figure, plot_path, dpi=figure_spec.dpi,
                image_format=resolve_figure_format(
                    args.outFileName, args.plotFileFormat))
            _write_outputs(paths, args, data, plan, figure_spec, prepared)
        except ValueError as error:
            if isinstance(error, (DataError, OptionError)):
                raise
            raise option_error(error) from error
    return tuple(path for path in (
        args.outFileName, args.outFileNameData, args.outFileNameMatrix,
        args.outFileSortedRegions) if path), digest


def matrix_main(tool):
    """Return a CLI entry point with the tool's pinned parser and messages."""
    def main(argv=None):
        from deeptoolsr.session import PlotSession

        raw = list(sys.argv[1:] if argv is None else argv)

        def emit(stream, message):
            (sys.stdout if stream == 'stdout' else sys.stderr).write(message)

        try:
            PlotSession(cache_bytes=0).run([tool, *raw], mode='cli', emit=emit)
        except OptionError as error:
            if error.parser_error:
                parserCommon.plot_parser(tool).error(str(error))
            if error.exit_code == 1:
                raise SystemExit(error.prefix + str(error)) from error
            sys.stderr.write(error.prefix + str(error) + '\n')
            raise SystemExit(error.exit_code) from error
        except DataError as error:
            if error.parser_error:
                parserCommon.plot_parser(tool).error(str(error))
            raise SystemExit(str(error)) from error
    return main


def parse_arguments(args=None):
    return parserCommon.plot_parser('plotMatrixR')


def process_args(args=None):
    return parse_command('plotMatrixR', args, 'cli')


main = matrix_main('plotMatrixR')
