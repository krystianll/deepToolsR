"""Build a figure input from a real frozen matrix in rendering tests."""

from dataclasses import replace

from deeptoolsr.matrix import Matrix, RowLayout, SortState, remove_empty_groups
from deeptoolsr.prepare import PlotData, resolve_labels


def plot_data(matrix: Matrix, *, group_labels=None, sample_labels=None,
              sort_method=None, show_counts=False, threads=1):
    layout, _ = remove_empty_groups(RowLayout.identity(matrix))
    if sort_method is not None:
        layout = replace(layout, sort=SortState(sort_method, 'mean'))
    labels = resolve_labels(layout, regions_label=group_labels,
                            samples_label=sample_labels,
                            show_counts=show_counts, header=matrix.header)
    return PlotData(matrix, layout, labels, threads)
