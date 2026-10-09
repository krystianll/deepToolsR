"""Build heatmaps in tests through the production projection and preparation."""

from dataclasses import replace

import matplotlib

from deeptoolsr import options as run_options
from deeptoolsr.prepare import parse_command
from deeptoolsr.plotting import fonts
from deeptoolsr.plotting.matrix_spec import project_matrix_spec
from deeptoolsr.plotting.matrix_plan import resolve_matrix_plan
from deeptoolsr.plotting.prepared import prepare_matrix
from deeptoolsr.plotting.matrix_figure import build_matrix_figure
from deeptoolsr.plotting.rendering import save_figure_atomic
from deeptoolsr.plotting.text_layout import ExtentCache


# These are the former direct-rendering defaults, which differ from the
# command's defaults. Keeping them here preserves the test cases' exact inputs.
DIRECT_DEFAULTS = {
    'colorMap': ['binary'], 'colorList': None, 'colorNumber': 256,
    'missingDataColor': 'black', 'alpha': 1.0,
    'plotTitle': '', 'xAxisLabel': '', 'heatmapYAxisLabel': '',
    'zMin': None, 'zMax': None, 'yMin': None,
    'yMax': None, 'averageType': 'median',
    'refPointLabel': None, 'startLabel': 'TSS', 'endLabel': 'TES',
    'heatmapHeight': 25, 'cellWidth': 7.5, 'heatmapAspectRatio': None,
    'profileAspectRatio': None, 'yAxisLabel': '',
    'pseudocount': -1, 'trim_perc': 0.05,
    'perGroup': False, 'whatToShow': 'plot, heatmap and colorbar',
    'linesAtTickMarks': False, 'minorTickMarks': False,
    'plotType': 'lines', 'legendLocation': 'upper-left',
    'boxAroundHeatmaps': True, 'label_rotation': 0.0, 'dpi': 200,
    'interpolationMethod': 'auto', 'zMid': None, 'sortIndicator': 'auto',
    'colors': None, 'arrangeSamples': None,
    'sampleSetLabels': None, 'xAxisVisibility': 'outer',
    'yAxisVisibility': 'outer',
    'yAxisLimits': 'per_sample_set',
    'colorbarLocation': 'best', 'colorbarLabels': None,
    'regionLabelLocation': 'right',
}


def render_heatmap(data, output_path=None, *, image_format='png', threads=1,
                   style=None, raster=None, colorMapDict=None,
                   series_options=None, **settings):
    """Return figure and solution, saving only when a test requests a file."""
    args = parse_command('plotHeatmapR', ['-m', 'unused'], 'worker')
    for name, value in DIRECT_DEFAULTS.items():
        setattr(args, name, value)
    if colorMapDict:
        for name, value in colorMapDict.items():
            setattr(args, name, value)
    for name, value in settings.items():
        if name in ('regionsLabel', 'outFileName'):
            continue
        if name == 'heatmapWidth':
            name = 'cellWidth'
        if not hasattr(args, name):
            raise TypeError('unknown heatmap setting: {}'.format(name))
        setattr(args, name, value)
    if (settings.get('heatmapAspectRatio') is not None and
            'heatmapWidth' not in settings and 'cellWidth' not in settings):
        # The former direct-render default width was overridden by a ratio.
        args.cellWidth = None
    args.distanceUnit = data.distance_unit
    args.distanceUnitLocation = data.distance_unit_location
    args.numberOfProcessors = threads
    run = run_options.resolve_run_options(args, plotting=True)
    if style is not None or raster is not None:
        run = replace(run, style=style or run.style,
                      raster=raster or run.raster)
    matrix_spec = project_matrix_spec(args, 'plotHeatmapR')
    spec = replace(matrix_spec, series_options=(
        series_options if series_options is not None else
        matrix_spec.series_options))
    plan, sample_plans, scales = resolve_matrix_plan(
        data.matrix.header, data.layout, data.labels, spec)
    prepared = prepare_matrix(data, plan, sample_plans, scales, spec, run)
    with matplotlib.rc_context(fonts.style_rc(run.style)):
        figure, solution = build_matrix_figure(
            data.matrix, data.layout, data.labels, plan, spec, prepared,
            run, ExtentCache())
        if output_path is not None:
            save_figure_atomic(
                figure, str(output_path), dpi=spec.dpi,
                image_format=image_format)
    return figure, solution
