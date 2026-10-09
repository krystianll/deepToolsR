"""Renderer-state and role captures used by the plot baselines."""


def role_rects(figure):
    """Return every placed role in points, at full float precision."""
    width = figure.get_figwidth() * 72.0
    height = figure.get_figheight() * 72.0
    roles = {}
    for axis in figure.axes:
        gid = axis.get_gid()
        if not gid or gid in roles:
            raise AssertionError(f'missing or duplicate plot role: {gid!r}')
        x, y, w, h = axis.get_position().bounds
        roles[gid] = [x * width, y * height, w * width, h * height]
    return roles


def artist_state(axis):
    """Capture visible major ticks and labels after all placement passes."""
    state = {}
    for name in ('x', 'y'):
        direction = getattr(axis, f'{name}axis')
        ticks = direction.get_major_ticks()
        labels = direction.get_ticklabels()
        if not direction.get_visible() or not any(
                tick.tick1line.get_visible() or tick.tick2line.get_visible() or
                tick.label1.get_visible() or tick.label2.get_visible()
                for tick in ticks):
            continue
        sides = ('bottom', 'top') if name == 'x' else ('left', 'right')
        state[name] = {
            'locations': [float(value) for value in direction.get_majorticklocs()],
            'labels': [{'text': label.get_text(), 'ha': label.get_ha(),
                        'va': label.get_va(), 'rotation': float(label.get_rotation()),
                        'visible': bool(label.get_visible())}
                       for label in labels],
            'label_sides': [side for side, selected in zip(sides, (
                any(tick.label1.get_visible() for tick in ticks),
                any(tick.label2.get_visible() for tick in ticks))) if selected],
        }
    return state
