"""One explicit option projection for every matrix figure front end."""

from dataclasses import dataclass, field, fields, replace

from functools import cache

from deeptoolsr.parserCommon import plot_option_strings_by_dest, plot_parser

from .series import cell_options
from .sizes import solve_sizes


def _panel(kind):
    return field(metadata={'panel': kind})


def panel_fields(kind):
    """Names of the MatrixFigureSpec fields that only affect ``kind``."""
    return frozenset(item.name for item in fields(MatrixFigureSpec)
                     if item.metadata.get('panel') == kind)


@dataclass(frozen=True)
class MatrixFigureSpec:
    """Every figure option, projected once; a field's ``panel`` metadata
    names the panel it only affects (inert when that panel is not drawn)."""

    tool: str
    show_profile: bool
    show_heatmap: bool
    sort_regions: str
    plot_type: str = _panel('profile')
    average_type: str = _panel('profile')
    pseudocount: float = _panel('profile')
    trim_perc: float = _panel('profile')
    ci_level: float = _panel('profile')
    bootstrap_replicates: int = _panel('profile')
    cell_width: float
    profile_height: float | None = _panel('profile')
    profile_aspect_ratio: float | None = _panel('profile')
    heatmap_height: float | None = _panel('heatmap')
    heatmap_aspect_ratio: float | None = _panel('heatmap')
    x_axis_label: object
    y_axis_label: object
    heatmap_y_axis_label: object = _panel('heatmap')
    y_min: object = _panel('profile')
    y_max: object = _panel('profile')
    y_axis_limits: str = _panel('profile')
    x_axis_visibility: str
    y_axis_visibility: str
    axis_visibility: object
    color_map: object = _panel('heatmap')
    color_list: object = _panel('heatmap')
    color_number: int = _panel('heatmap')
    colors: object = _panel('profile')
    missing_data_color: str = _panel('heatmap')
    alpha: float = _panel('heatmap')
    z_min: object = _panel('heatmap')
    z_max: object = _panel('heatmap')
    z_mid: object = _panel('heatmap')
    plot_title: str
    reference_point_label: object
    start_label: object
    end_label: object
    per_group: bool
    same_group_labels: str
    same_sample_labels: str
    arrange_samples: object
    sample_set_labels: object
    placement: str
    grid_columns: object
    grid_rows: object
    common_legend: bool = _panel('profile')
    legend_location: str = _panel('profile')
    lines_at_tick_marks: bool
    minor_tick_marks: object
    box_around_heatmaps: bool = _panel('heatmap')
    label_rotation: float
    dpi: int
    interpolation_method: str = _panel('heatmap')
    sort_indicator: str = _panel('heatmap')
    colorbar_location: str = _panel('heatmap')
    colorbar_labels: object = _panel('heatmap')
    region_label_location: str = _panel('heatmap')
    distance_unit: str
    distance_unit_location: str
    series_options: object
    invoked_spellings: tuple


@cache
def _matrix_defaults():
    parser = plot_parser('plotMatrixR', full_color_help=False)
    return {action.dest: parser.get_default(action.dest)
            for action in parser._actions}


def _option(args, dest):
    """A parsed destination, or plotMatrixR's default when the invoked tool
    does not offer it (the option then belongs to a panel it never draws)."""
    return (getattr(args, dest) if hasattr(args, dest)
            else _matrix_defaults()[dest])


def project_matrix_spec(args, tool):
    """Project parsed destinations once, including tool-dependent defaults."""
    if tool not in ('plotHeatmapR', 'plotProfileR', 'plotMatrixR'):
        raise ValueError(f'unsupported plot tool: {tool}')
    if tool == 'plotMatrixR' and args.show_heatmap and args.plotType == 'heatmap':
        from deeptoolsr.prepare import OptionError
        raise OptionError('--plotType heatmap cannot be combined with --heatmap')
    sizes = solve_sizes(
        cell_width=args.cellWidth,
        profile_height=_option(args, 'profileHeight'),
        profile_aspect_ratio=args.profileAspectRatio,
        heatmap_height=_option(args, 'heatmapHeight'),
        heatmap_aspect_ratio=_option(args, 'heatmapAspectRatio'),
        show_profile=args.show_profile, show_heatmap=args.show_heatmap)
    spec = MatrixFigureSpec(
        tool=tool, show_profile=args.show_profile,
        show_heatmap=args.show_heatmap,
        sort_regions=args.sortRegions,
        plot_type=args.plotType, average_type=args.averageType,
        pseudocount=args.pseudocount, trim_perc=args.trim_perc,
        ci_level=args.ci_level, bootstrap_replicates=args.bootstrapReplicates,
        cell_width=sizes.cell_width, profile_height=sizes.profile_height,
        profile_aspect_ratio=sizes.profile_aspect_ratio,
        heatmap_height=sizes.heatmap_height,
        heatmap_aspect_ratio=sizes.heatmap_aspect_ratio,
        x_axis_label=args.xAxisLabel, y_axis_label=args.yAxisLabel,
        heatmap_y_axis_label=_option(args, 'heatmapYAxisLabel'),
        y_min=args.yMin, y_max=args.yMax, y_axis_limits=args.yAxisLimits,
        x_axis_visibility=args.xAxisVisibility,
        y_axis_visibility=args.yAxisVisibility,
        axis_visibility=_option(args, 'axisVisibility'),
        color_map=_option(args, 'colorMap'),
        color_list=_option(args, 'colorList'),
        color_number=_option(args, 'colorNumber'),
        colors=args.colors,
        missing_data_color=_option(args, 'missingDataColor'),
        alpha=_option(args, 'alpha'),
        z_min=_option(args, 'zMin'), z_max=_option(args, 'zMax'),
        z_mid=_option(args, 'zMid'), plot_title=args.plotTitle,
        reference_point_label=args.refPointLabel,
        start_label=args.startLabel, end_label=args.endLabel,
        per_group=args.perGroup,
        same_group_labels=args.sameGroupLabels,
        same_sample_labels=args.sameSampleLabels,
        arrange_samples=args.arrangeSamples,
        sample_set_labels=args.sampleSetLabels,
        placement=args.sampleSetGroupArrangement,
        grid_columns=args.gridColumns, grid_rows=args.gridRows,
        common_legend=args.commonLegend,
        legend_location=args.legendLocation,
        lines_at_tick_marks=args.linesAtTickMarks,
        minor_tick_marks=_option(args, 'minorTickMarks'),
        box_around_heatmaps=_option(args, 'boxAroundHeatmaps'),
        label_rotation=args.label_rotation, dpi=args.dpi,
        interpolation_method=_option(args, 'interpolationMethod'),
        sort_indicator=_option(args, 'sortIndicator'),
        colorbar_location=_option(args, 'colorbarLocation'),
        colorbar_labels=_option(args, 'colorbarLabels'),
        region_label_location=_option(args, 'regionLabelLocation'),
        distance_unit=args.distanceUnit,
        distance_unit_location=args.distanceUnitLocation,
        series_options=None,
        invoked_spellings=getattr(args, '_invoked_spellings', ()))
    colors_option = plot_option_strings_by_dest(tool)['colors'][0]
    return replace(spec, series_options=cell_options(
        spec, colors_option=colors_option))
