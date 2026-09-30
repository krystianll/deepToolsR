"""A profile looks the same whether or not a heatmap stack lies under it."""

from matplotlib.colors import to_hex
import pytest

from tests.test_matrix_cells import _render


def _profiles(result):
    """Series, colours, limits and legend text of every profile panel."""
    figure = result['figure']
    panels = {}
    for axis in figure.axes:
        gid = axis.get_gid() or ''
        if not gid.endswith('/profile'):
            continue
        legend = axis.get_legend()
        panels[gid] = {
            'series': [(line.get_label(), to_hex(line.get_color()),
                        tuple(line.get_ydata()))
                       for line in axis.get_lines()
                       if not line.get_label().startswith('_')],
            'ylim': tuple(round(value, 6) for value in axis.get_ylim()),
            'legend': (None if legend is None else
                       [text.get_text() for text in legend.get_texts()]),
        }
    slots = sorted(
        (axis.get_gid(), [text.get_text() for text in
                          axis.get_legend().get_texts()])
        for axis in figure.axes
        if (axis.get_gid() or '').startswith('legend/') and
        axis.get_legend() is not None)
    return panels, slots


@pytest.mark.parametrize('flags', [
    (),
    ('--arrangeSamples', '1,2', '3,4'),
    ('--arrangeSamples', '1,2', '3,4', '--sampleSetGroupArrangement',
     'by_row'),
    ('--arrangeSamples', '1,2', '3,4', '--sampleSetGroupArrangement',
     'by_column'),
    ('--sampleSetGroupArrangement', 'adjacent'),
    ('--perGroup',),
    ('--legendLocation', 'below'),
    ('--legendLocation', 'right', '--arrangeSamples', '1,2', '3,4'),
    ('--commonLegend', '--legendLocation', 'below'),
    ('--colorsPerSample', 'red', 'blue',
     '--plotType', 'se'),
], ids=lambda flags: ' '.join(flags) or 'default')
def test_heatmap_stack_does_not_change_profiles(tmp_path, monkeypatch, flags):
    samples = ('--samplesLabel', 'A', 'B', 'C', 'D')
    (tmp_path / 'alone').mkdir()
    (tmp_path / 'stacked').mkdir()
    alone = _render(tmp_path / 'alone', monkeypatch, '--profile',
                    *samples, *flags)
    stacked = _render(tmp_path / 'stacked', monkeypatch, '--profile',
                      '--heatmap', *samples, *flags)
    assert _profiles(stacked) == _profiles(alone)


def _profile_decorations(result):
    """Vertical offsets of titles and legends from the profile they hang on.

    Below legends are left out: their offset includes the X decorations,
    which hang under the stack when there is one.  Row legends are centred
    on the row, whose column gaps may widen for stack labels, so only their
    height is compared.
    """
    rects = result['solution'].rects
    profiles = {role: rect for role, rect in rects.items()
                if role.endswith('/profile')}

    def anchor(role, rect):
        parts = role.split('/')
        if role.startswith('cell/'):
            return f'cell/{parts[1]}/{parts[2]}/profile'
        if role.startswith('legend/row/'):
            return f'cell/{parts[2]}/1/profile'
        return min(profiles, key=lambda name: (
            abs(profiles[name].x + profiles[name].width / 2 -
                rect.x - rect.width / 2) +
            abs(profiles[name].y + profiles[name].height / 2 -
                rect.y - rect.height / 2)))

    decorations = {}
    for role, rect in rects.items():
        if not ((role.startswith('cell/') and role.endswith('/title')) or
                role.startswith(('legend/cell/', 'legend/row/'))):
            continue
        panel = profiles[anchor(role, rect)]
        offset = (round(rect.y - panel.y, 6),)
        if not role.startswith('legend/row/'):
            offset += (round(rect.x - panel.x, 6),)
        decorations[role] = (anchor(role, rect), *offset)
    return decorations


@pytest.mark.parametrize('flags', [
    ('--legendLocation', 'above'),
    ('--legendLocation', 'right', '--arrangeSamples', '1,2', '3,4'),
    ('--legendLocation', 'above', '--arrangeSamples', '1,2', '3,4',
     '--sampleSetGroupArrangement', 'by_row'),
    ('--legendLocation', 'right', '--arrangeSamples', '1,2', '3,4',
     '--sampleSetGroupArrangement', 'by_row'),
], ids=' '.join)
def test_heatmap_stack_does_not_move_profile_decorations(
        tmp_path, monkeypatch, flags):
    # Titles and legends hang from the profile by the same rules whether or
    # not a stack lies under it (one row model in the grid solver).
    (tmp_path / 'alone').mkdir()
    (tmp_path / 'stacked').mkdir()
    alone = _render(tmp_path / 'alone', monkeypatch, '--profile', *flags)
    stacked = _render(tmp_path / 'stacked', monkeypatch, '--profile',
                      '--heatmap', *flags)
    decorations = _profile_decorations(alone)
    assert any(role.startswith('legend/') for role in decorations)
    assert _profile_decorations(stacked) == decorations
