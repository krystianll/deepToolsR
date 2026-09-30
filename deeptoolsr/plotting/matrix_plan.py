"""Pure cell planning shared by plotting commands and describe."""

from deeptoolsr.parserCommon import plot_option_strings_by_dest

from .series import build_sample_set_plans, parse_sample_sets, resolve_cells


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
    for ``describe`` as well as for plotting.
    """
    sample_sets = parse_sample_sets(
        labels.samples, spec.arrange_samples, per_group=spec.per_group)
    references = (header.parameters['ref point']
                  if spec.reference_point_label is None
                  else spec.reference_point_label)
    sample_plans = build_sample_set_plans(
        labels.samples, header.sample_boundaries, header.parameters,
        sample_sets, subplot_labels=spec.sample_set_labels,
        x_axis_labels=spec.x_axis_label, y_axis_labels=spec.y_axis_label,
        reference_labels=references, start_labels=spec.start_label,
        end_labels=spec.end_label,
        option_spellings=tool_option_spellings(spec.tool, spec.invoked_spellings),
        context=('--arrangeSamples ' + ' '.join(spec.arrange_samples)
                 if spec.arrange_samples else None))
    bounds = layout.group_bounds
    sizes = tuple(end - start for start, end in zip(bounds, bounds[1:]))
    plan = resolve_cells(sizes, sample_plans, labels, spec.series_options)
    scales = None
    if spec.show_heatmap:
        from .heatmap import scale_assignments
        scales = scale_assignments(spec, plan)
    return plan, tuple(sample_plans), scales
