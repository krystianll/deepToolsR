"""Build profile test cases through the production projection and planner."""

from dataclasses import dataclass, replace

import matplotlib

from deeptoolsr import options as run_options
from deeptoolsr.prepare import parse_command
from deeptoolsr.plotting import fonts
from deeptoolsr.plotting.matrix_spec import project_matrix_spec
from deeptoolsr.plotting.matrix_plan import resolve_matrix_plan
from deeptoolsr.plotting.matrix_figure import build_matrix_figure
from deeptoolsr.plotting.series import cell_options
from deeptoolsr.plotting.sizes import solve_sizes
from deeptoolsr.plotting.prepared import prepare_matrix
from deeptoolsr.plotting.rendering import save_figure_atomic
from deeptoolsr.plotting.text_layout import ExtentCache


# Former Profile.__init__ values that differed from CLI defaults.
DIRECT_DEFAULTS = {
    'y_axis_label': '', 'x_axis_label': None,
    'y_min': None, 'y_max': None, 'average_type': 'median',
    'start_label': 'TSS', 'end_label': 'TES',
    'series_options': None,
}

CASE_NAMES = {
    'averagetype': 'average_type',
    'real_profile_height': 'profile_height',
    'real_profile_width': 'cell_width',
    'real_height': 'profile_height',
    'real_width': 'cell_width',
    'plot_height': 'profile_height',
    'plot_width': 'cell_width',
    'color_list': 'colors',
    'subplot_samples': 'arrange_samples',
    'subplot_columns': 'grid_columns',
    'subplot_rows': 'grid_rows',
    'subplot_group_placement': 'placement',
    'subplot_labels': 'sample_set_labels',
    'subplot_common_legend': 'common_legend',
    'subplot_x_axes': 'x_axis_visibility',
    'subplot_y_axes': 'y_axis_visibility',
    'subplot_y_limits': 'y_axis_limits',
}


@dataclass(frozen=True)
class ProfileBuild:
    data: object
    figure: object
    solution: object
    plan: object
    sample_set_plans: tuple
    prepared: object
    spec: object
    run: object


def build_profile_case(data, output_path=None, *, style=None, threads=1,
                       render=True, **settings):
    """Resolve a direct-render test case with its original effective defaults."""
    args = parse_command('plotProfileR', ['-m', 'unused'], 'worker')
    args.numberOfProcessors = threads
    run = run_options.resolve_run_options(args, plotting=True)
    if style is not None:
        run = replace(run, style=style)
    matrix_spec = project_matrix_spec(args, 'plotProfileR')
    updates = dict(DIRECT_DEFAULTS)
    real_width = settings.get('real_profile_width', settings.get('real_width'))
    real_height = settings.get('real_profile_height', settings.get('real_height'))
    for name, value in settings.items():
        if name in ('image_format', 'plot_width', 'plot_height',
                    'real_profile_width', 'real_profile_height',
                    'real_width', 'real_height'):
            continue
        target = CASE_NAMES.get(name, name)
        if target not in matrix_spec.__dataclass_fields__:
            raise TypeError('unknown profile setting: {}'.format(name))
        updates[target] = value
    spec = replace(matrix_spec, **updates)
    plan_spec = replace(
        matrix_spec, per_group=spec.per_group, plot_type=spec.plot_type,
        same_group_labels=spec.same_group_labels,
        same_sample_labels=spec.same_sample_labels,
        arrange_samples=spec.arrange_samples,
        grid_columns=spec.grid_columns, grid_rows=spec.grid_rows,
        x_axis_label=spec.x_axis_label, y_axis_label=spec.y_axis_label,
        reference_point_label=spec.reference_point_label,
        start_label=spec.start_label, end_label=spec.end_label,
        placement=spec.placement, sample_set_labels=spec.sample_set_labels)
    plan_spec = replace(plan_spec, series_options=(
        spec.series_options if spec.series_options is not None else
        cell_options(plan_spec)))
    plan, sample_plans, scales = resolve_matrix_plan(
        data.matrix.header, data.layout, data.labels, plan_spec)
    sizes = solve_sizes(
        cell_width=(real_width if real_width is not None else
                    settings.get('plot_width', updates.get('cell_width'))),
        profile_height=(real_height if real_height is not None else
                        settings.get('plot_height',
                                     updates.get('profile_height'))),
        profile_aspect_ratio=updates.get('profile_aspect_ratio'),
        show_profile=True)
    spec = replace(spec,
                   cell_width=sizes.cell_width,
                   profile_height=sizes.profile_height,
                   profile_aspect_ratio=sizes.profile_aspect_ratio)
    if not render:
        return ProfileBuild(data, None, None, plan, sample_plans, None,
                            spec, run)
    prepared = prepare_matrix(data, plan, sample_plans, scales, spec, run)
    with matplotlib.rc_context(fonts.style_rc(run.style)):
        figure, solution = build_matrix_figure(
            data.matrix, data.layout, data.labels, plan, spec, prepared,
            run, ExtentCache())
        if output_path is not None:
            save_figure_atomic(figure, str(output_path), dpi=spec.dpi,
                               image_format=settings.get('image_format'))
    return ProfileBuild(data, figure, solution, plan, sample_plans, prepared,
                        spec, run)
