"""One resident plot matrix and bounded reusable numeric work per worker."""

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from time import perf_counter

import numpy as np

from .cache import ByteBudgetCache, TextExtentCache
from .matrix import SourceKey


# Each parser destination has exactly one owner. Other stages may receive
# derived values through explicit projection functions below.
OPTION_STAGES = {
    'matrixFile': 'source',
    'kmeans': 'order', 'hclust': 'order',
    'clusterUsingSamples': 'order', 'sortRegions': 'order',
    'sortUsing': 'order', 'sortUsingSamples': 'order',
    'quantileSortedRegions': 'order', 'filterNans': 'order',
    'silhouette': 'order', 'sameGroupLabels': 'order',
    'averageType': 'statistics', 'plotType': 'statistics',
    'pseudocount': 'statistics', 'trim_perc': 'statistics',
    'ci_level': 'statistics', 'bootstrapReplicates': 'statistics',
    'zMin': 'limits', 'zMax': 'limits',
    'colorMap': 'color', 'colorList': 'color',
    'colorNumber': 'color', 'zMid': 'color',
    'missingDataColor': 'color', 'interpolationMethod': 'color',
    'regionsLabel': 'labels', 'samplesLabel': 'labels',
    'showRegionCounts': 'labels', 'sampleSetLabels': 'labels',
    'plotTitle': 'labels', 'xAxisLabel': 'labels',
    'yAxisLabel': 'labels', 'heatmapYAxisLabel': 'labels',
    'startLabel': 'labels', 'endLabel': 'labels',
    'refPointLabel': 'labels', 'arrangeSamples': 'scene',
    'axisVisibility': 'scene', 'xAxisVisibility': 'scene',
    'yAxisVisibility': 'scene', 'distanceUnit': 'scene',
    'distanceUnitLocation': 'scene', 'dpi': 'scene',
    'label_rotation': 'scene', 'linesAtTickMarks': 'scene',
    'minorTickMarks': 'scene', 'perGroup': 'scene',
    'alpha': 'scene', 'boxAroundHeatmaps': 'scene',
    'colorbarLabels': 'scene', 'colorbarLocation': 'scene',
    'colors': 'scene', 'commonLegend': 'scene',
    'gridColumns': 'scene', 'gridRows': 'scene',
    'legendLocation': 'scene', 'sameSampleLabels': 'scene',
    'sampleSetGroupArrangement': 'scene',
    'yAxisLimits': 'scene', 'yMax': 'scene', 'yMin': 'scene',
    'show_profile': 'scene', 'show_heatmap': 'scene',
    'cellWidth': 'scene', 'profileHeight': 'scene',
    'profileAspectRatio': 'scene',
    'heatmapAspectRatio': 'scene', 'heatmapHeight': 'scene',
    'regionLabelLocation': 'scene',
    'sortIndicator': 'scene',
    # plotHeatmapR still exposes this historical destination while its
    # spelling is projected to show_profile at parse time.
    'whatToShow': 'scene',
    'outFileName': 'output', 'outFileNameMatrix': 'output',
    'outFileSortedRegions': 'output', 'outFileNameData': 'output',
    'plotFileFormat': 'output',
    'numberOfProcessors': 'execution', 'verbose': 'execution',
    'config': 'execution', 'nanAfterEnd': 'ignored',
}


STAGES = ('parse', 'load', 'order', 'statistics', 'limits', 'scene',
          'bitmaps', 'publish')


class Cancelled(Exception):
    """A request stopped before publication began."""


@dataclass(frozen=True)
class Result:
    outputs: tuple[str, ...]
    recomputed: tuple[str, ...]
    reused: tuple[str, ...]
    timings: dict[str, float]
    warnings: tuple[str, ...]
    scene_digest: str | None = None


@dataclass(frozen=True)
class ScanSpec:
    panel: str
    probabilities: tuple[float, ...]
    exact: bool = False


@dataclass(frozen=True)
class ColorSpec:
    color_map: object
    color_list: object
    color_number: int
    missing_data_color: str
    alpha: float
    z_mid: object
    interpolation_method: str


def _automatic(values):
    return values is None or values == 'auto' or (
        isinstance(values, (tuple, list)) and any(
            value is None or value == 'auto' for value in values))


def project_scan_spec(figure_spec, *, exact=False):
    """Request a scan only for a rendered panel with automatic limits."""
    if figure_spec.show_heatmap and (
            _automatic(figure_spec.z_min) or
            _automatic(figure_spec.z_max)):
        return ScanSpec('heatmap', (0.0, 1.0, 98.0, 100.0), exact)
    if (figure_spec.show_profile and figure_spec.plot_type == 'heatmap'
            and (_automatic(figure_spec.y_min) or
                 _automatic(figure_spec.y_max))):
        return ScanSpec('series_heatmap', (1.0, 98.0), exact)
    return None


def project_color_spec(figure_spec):
    """Heatmap colour settings; hidden heatmaps have no colour key."""
    if not figure_spec.show_heatmap:
        return None
    return ColorSpec(
        figure_spec.color_map, figure_spec.color_list,
        figure_spec.color_number, figure_spec.missing_data_color,
        figure_spec.alpha, figure_spec.z_mid,
        figure_spec.interpolation_method)


def statistics_key(source, group_digest, sample, spec):
    return source, group_digest, sample, spec


def scan_key(source, layout, scan_spec):
    return (source, layout.digest(), scan_spec.probabilities,
            scan_spec.exact)


def bitmap_key(source, group_digest, sample, lut_bytes, bad_color,
               vmin, vmax, out_h, out_w, raster_filter, bit_depth):
    lut = hashlib.sha256(memoryview(lut_bytes)).digest()
    return (source, group_digest, sample, lut, bad_color, float(vmin),
            float(vmax), int(out_h), int(out_w), raster_filter,
            int(bit_depth))


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {'dtype': str(value.dtype), 'shape': value.shape,
                'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name))
                for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return str(value)


def project_scene_digest(args, figure_spec, data_spec, statistics_spec,
                         scan_spec, color_spec, layout, *,
                         config_digest=None):
    """Hash effective scene inputs, excluding absent panels and spellings."""
    from deeptoolsr.plotting.matrix_spec import panel_fields

    inert = (panel_fields('heatmap') if not figure_spec.show_heatmap
             else frozenset()) | (panel_fields('profile')
                                  if not figure_spec.show_profile
                                  else frozenset())
    active = {}
    for field in fields(figure_spec):
        name = field.name
        if name in ('tool', 'sort_regions', 'invoked_spellings') or \
                name in inert:
            continue
        active[name] = _jsonable(getattr(figure_spec, name))
    payload = {
        'figure': active, 'data': _jsonable(data_spec),
        'layout': layout.digest().hex(),
        'statistics': _jsonable(statistics_spec)
        if figure_spec.show_profile else None,
        'scan': _jsonable(scan_spec), 'color': _jsonable(color_spec),
        'regions_label': _jsonable(getattr(args, 'regionsLabel', None)),
        'samples_label': _jsonable(getattr(args, 'samplesLabel', None)),
        'show_region_counts': getattr(args, 'showRegionCounts', False),
        'config': config_digest,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=True).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _default_cache_bytes():
    from .options import physical_memory_bytes
    physical = physical_memory_bytes()
    return min(2 * 1024 ** 3, physical // 4) if physical else 2 * 1024 ** 3


def _default_emit(_stream, _message):
    """Leave presentation to the caller when no callback was supplied."""


def explicit_run_options(argv):
    """Return explicitly supplied run settings, including argparse prefixes."""
    supplied = set()
    for value in argv:
        token = str(value)
        if token == '-p' or (token.startswith('-p') and
                             not token.startswith('--')):
            supplied.add('numberOfProcessors')
        elif token.startswith('--'):
            spelling = token.split('=', 1)[0]
            for option, dest in (('--numberOfProcessors', 'numberOfProcessors'),
                                 ('--config', 'config')):
                if len(spelling) > 2 and option.startswith(spelling):
                    supplied.add(dest)
    return frozenset(supplied)


class PlotSession:
    """Run independent plot requests with one matrix and numeric LRUs."""

    def __init__(self, *, config=None, threads=None, cache_bytes=None):
        self.config = config
        self.threads = threads
        self.cache = ByteBudgetCache(
            _default_cache_bytes() if cache_bytes is None else cache_bytes)
        self.extents = TextExtentCache(
            self.cache, on_hit=lambda: self._mark('reused', 'text'),
            on_store=lambda: self._mark('recomputed', 'text'))
        self.matrix = None
        self.source_key = None
        self.config_digest = None
        self.recomputed = []
        self.reused = []
        self.timings = {}
        self.warnings = []
        self._stage = None
        self._stage_started = None
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        self.close()

    def _finish_stage(self):
        if self._stage is not None:
            elapsed = perf_counter() - self._stage_started
            self.timings[self._stage] = self.timings.get(self._stage, 0) + elapsed
            self._stage = None

    def _mark(self, collection, kind):
        values = getattr(self, collection)
        if kind not in values:
            values.append(kind)

    def stage(self, name, cancelled, progress):
        if name not in STAGES:
            raise ValueError(f'unknown plot stage: {name}')
        self._finish_stage()
        if cancelled():
            raise Cancelled()
        if progress is not None:
            progress(name)
        self._stage = name
        self._stage_started = perf_counter()

    def get_or_compute(self, kind, key, compute):
        value, hit = self.lookup(kind, key)
        if hit:
            return value, True
        value = compute()
        self.store(kind, key, value)
        return value, False

    def layout(self, spec, threads):
        """The resident matrix's filtered, ordered rows and their warnings."""
        from .prepare import prepare

        return self.get_or_compute(
            'layout', (self.source_key, spec),
            lambda: prepare(self.matrix, spec, threads))[0]

    def lookup(self, kind, key):
        value, hit = self.cache.get(kind, key)
        if hit:
            self._mark('reused', kind)
        return value, hit

    def store(self, kind, key, value):
        self.cache.put(kind, key, value)
        self._mark('recomputed', kind)

    def statistics_batch(self, data, pairs, spec, threads):
        """Compute all missing series in one fixed-order native batch."""
        from .stats import calculate_group_statistics_batch

        pairs = tuple(pairs)
        keys = tuple(statistics_key(
            self.source_key, data.layout.group_digest(group), sample, spec)
            for group, sample in pairs)
        values = {}
        missing = {}
        for key, pair in zip(keys, pairs):
            if key in values or key in missing:
                continue
            value, hit = self.lookup('statistics', key)
            if hit:
                values[key] = value
            else:
                missing[key] = pair
        if missing:
            computed = calculate_group_statistics_batch(
                data, tuple(missing.values()), spec, threads=threads)
            for key, value in zip(missing, computed):
                self.store('statistics', key, value)
                values[key] = value
        return tuple(values[key] for key in keys)

    def scan(self, data, probabilities, exact, threads):
        """Cache a global percentile scan separately from limit assignment."""
        from .stats import scan_prepared_matrix

        key = (self.source_key, data.layout.digest(),
               tuple(probabilities), bool(exact))
        value, _hit = self.get_or_compute(
            'scan', key,
            lambda: scan_prepared_matrix(
                data, probabilities, exact=exact, threads=threads))
        return value

    def render_bitmaps(self, requests, threads):
        """Reuse RGBA buffers and place them in their original draw order."""
        from .parallel import parallel_map
        from .plotting.heatmap import place_heatmap_rgba, render_heatmap_rgba

        requests = tuple(requests)
        values = {}
        keys = []
        missing = {}
        for request in requests:
            raster = request['raster']
            key = bitmap_key(
                self.source_key, request['group_digest'], request['sample'],
                request['lut'].tobytes(), tuple(request['bad']),
                request['vmin'], request['vmax'], request['out_h'],
                request['out_w'], raster.filter, raster.bit_depth)
            keys.append(key)
            if key in values or key in missing:
                continue
            value, hit = self.lookup('bitmaps', key)
            if hit:
                values[key] = value
            else:
                missing[key] = request
        if missing:
            def render(request, inner_threads):
                return render_heatmap_rgba(
                    {**request, 'threads': inner_threads})

            computed = parallel_map(
                render, tuple(missing.values()), threads=threads)
            for key, value in zip(missing, computed):
                self.store('bitmaps', key, value)
                values[key] = value
        for request, key in zip(requests, keys):
            place_heatmap_rgba(request, values[key])

    def matrix_for(self, path, threads):
        from .plotting.cli_io import load_plot_matrix

        matrix = None
        try:
            key = SourceKey.of(path)
        except OSError:
            # Keep the plot CLI's argparse-style input error on a cold miss.
            matrix = load_plot_matrix(path, threads)
            key = matrix.source
        if key == self.source_key and self.matrix is not None:
            if 'matrix' not in self.reused:
                self.reused.append('matrix')
            return self.matrix
        if matrix is None:
            matrix = load_plot_matrix(path, threads)
        self.cache.clear()
        self.matrix = matrix
        self.source_key = key
        if 'matrix' not in self.recomputed:
            self.recomputed.append('matrix')
        return matrix

    def warn(self, message):
        self.warnings.append(str(message))

    def run(self, command, *, mode='worker', cancelled=lambda: False,
            progress=None, emit=None):
        """Parse one request, then let the shared plot pipeline publish it."""
        from . import options
        from .plotMatrix import run_session_request
        from .prepare import OptionError, parse_command

        if self._closed:
            raise RuntimeError('PlotSession is closed')
        if mode not in ('worker', 'cli'):
            raise ValueError(f'unknown plot session mode: {mode}')
        self.recomputed = []
        self.reused = []
        self.timings = {}
        self.warnings = []
        self._stage = None
        self._stage_started = None
        self.extents.clear_transient()

        def forward(stream, message):
            if stream == 'stderr':
                self.warn(message)
            (emit or _default_emit)(stream, message)

        try:
            self.stage('parse', cancelled, progress)
            tool, *argv = tuple(command)
            args = parse_command(tool, argv, mode, emit=forward)
            args.tool = tool
            if mode == 'worker' and args.outFileName is None:
                raise OptionError(
                    'the following arguments are required: '
                    '--outFileName/-out/-o', parser_error=True)
            self.apply_run_defaults(args, argv)
            path = options.config_path(getattr(args, 'config', 'auto'))
            raw = options.read_config_bytes(path)
            self.config_digest = options.config_content_hash(raw)
            run_options = options.resolve_run_options_from_bytes(
                args, raw, path, plotting=True, emit=forward)
            args.run_options = run_options
            from matplotlib import rc_context
            from .plotting.fonts import style_rc

            with rc_context(style_rc(run_options.style)):
                outputs, scene_digest = run_session_request(
                    self, args, run_options, cancelled=cancelled,
                    progress=progress, emit=forward)
            self._finish_stage()
            return Result(tuple(outputs), tuple(self.recomputed),
                          tuple(self.reused), dict(self.timings),
                          tuple(self.warnings), scene_digest)
        finally:
            self._finish_stage()
            self.extents.clear_transient()

    def apply_run_defaults(self, args, argv):
        """Use the session's config and threads unless ``argv`` sets them."""
        explicit = explicit_run_options(argv)
        if self.config is not None and 'config' not in explicit:
            args.config = self.config
        if self.threads is not None and 'numberOfProcessors' not in explicit:
            args.numberOfProcessors = self.threads

    def describe(self, command):
        from .describe import describe_session
        if self._closed:
            raise RuntimeError('PlotSession is closed')
        return describe_session(self, command)

    def reload(self):
        self.cache.clear()
        self.extents.clear_transient()
        self.matrix = None
        self.source_key = None

    def close(self):
        self.reload()
        self._closed = True
