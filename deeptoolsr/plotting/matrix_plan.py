"""Pure cell planning shared by plotting commands and describe."""

from deeptoolsr.parserCommon import plot_option_strings_by_dest

from .series import (build_sample_set_plans, parse_sample_sets,
                     promote_group_panels, resolve_cells)


def tool_option_spellings(tool, invoked=()):
    spellings = {dest: next((name for name in names if name.startswith('--')),
                            names[0])
                 for dest, names in plot_option_strings_by_dest(tool).items()}
    spellings.update(invoked)
    return spellings


def resolve_matrix_plan(header, layout, labels, spec):
    """Resolve sample sets, cells and their per-scale heatmap assignments.

    The assignments are ``None`` without heatmaps. Resolving them here
    rejects an assignment naming a missing scale before anything is drawn,
    for ``describe`` as well as for plotting. With --groupsAsSampleSets,
    every group panel becomes a sample set here, so every per-set option
    counts panels.
    """
    sample_sets = parse_sample_sets(
        labels.samples, spec.arrange_samples, per_group=spec.per_group)
    bounds = layout.group_bounds
    sizes = tuple(end - start for start, end in zip(bounds, bounds[1:]))
    groups = default_labels = None
    if spec.series_options.groups_as_sample_sets:
        sample_sets, groups, default_labels = promote_group_panels(
            sample_sets, labels, sizes, spec.series_options)
    sample_plans = build_sample_set_plans(
        labels.samples, header.sample_boundaries, header.parameters,
        sample_sets, subplot_labels=spec.sample_set_labels,
        x_axis_labels=spec.x_axis_label, y_axis_labels=spec.y_axis_label,
        reference_labels=spec.reference_point_label, start_labels=spec.start_label,
        end_labels=spec.end_label, groups=groups,
        default_labels=default_labels,
        option_spellings=tool_option_spellings(spec.tool, spec.invoked_spellings),
        context=('--arrangeSamples ' + ' '.join(spec.arrange_samples)
                 if spec.arrange_samples else None))
    plan = resolve_cells(sizes, sample_plans, labels, spec.series_options)
    scales = None
    if spec.show_heatmap:
        from .heatmap import scale_assignments
        scales = scale_assignments(spec, plan)
    return plan, tuple(sample_plans), scales
