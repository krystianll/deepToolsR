#!/usr/bin/env python
"""Time deepToolsR plotting scenarios; count full-figure Agg draws.

Usage:
    python benchmarks/plots/run.py MATRIX [--scenario NAME ...] [--repeat 3]
        [--profile] [--threads N]

CLI runs execute the tool in a fresh interpreter. Session scenarios execute
several requests in one child process and report the warm variant requests.
Reported per run:
  wall_s        parent-measured wall time of the whole child process
                (interpreter start, imports, run, exit) -- the headline number;
  import_s      time to import the tool module (includes Matplotlib) inside the child;
  run_s         time spent in main() inside the child;
  draws         FigureCanvasAgg.draw calls;
  peak_rss_mib  child peak RSS (ru_maxrss: KiB on Linux, bytes on macOS).
--profile adds the 25 most expensive cumulative functions for the first run.
Results print as JSON lines, prefixed by one environment line (commit, CPU count,
platform, Python/Matplotlib versions, threads). Export a writable MPLCONFIGDIR and a
throwaway DEEPTOOLSR_CONFIG_DIR first.
"""

import argparse
import gzip
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

SCENARIOS = {
    "profile_ci_grid": ["plotProfile", "--averageType", "geom_mean", "--plotType", "ci",
                        "--quantiles", "4", "--sortUsing", "region_length",
                        "--arrangeSamples", "1,2", "3,4", "--sampleSetGroupArrangement",
                        "by_row", "--commonLegend", "--gridColumns", "4"],
    "profile_default": ["plotProfile"],
    "heatmap_quantiles": ["plotHeatmap", "--quantiles", "4", "--sortUsing",
                          "region_length", "--dpi", "200"],
    "heatmap_explicit_limits": ["plotHeatmap", "--quantiles", "4", "--sortUsing",
                                "region_length", "--dpi", "200", "--zMin", "0", "--zMax", "5"],
    # Clustering references (a baseline for the clustering memory checks).
    "heatmap_kmeans": ["plotHeatmap", "--kmeans", "4", "--dpi", "200"],
    "heatmap_hclust": ["plotHeatmap", "--hclust", "4", "--dpi", "200"],
    "matrix_both": ["plotMatrix", "--heatmap", "--profile", "--dpi", "200"],
    "matrix_by_row": ["plotMatrix", "--heatmap", "--profile",
                      "--arrangeSamples", "1,2", "3,4",
                      "--sampleSetGroupArrangement", "by_row", "--dpi", "200"],
}

# The first request warms the session. Every subsequent request is a variant;
# median_wall_s for these scenarios excludes the warm-up and process startup.
SESSION_SCENARIOS = {
    "warm_label_profile": [
        ("--profile",),
        ("--profile", "--plotTitle", "Warm title")],
    "warm_label_heatmap": [
        ("--heatmap",),
        ("--heatmap", "--plotTitle", "Warm title")],
    "warm_colormap_heatmap": [
        ("--heatmap",),
        ("--heatmap", "--colorMap", "viridis")],
    "warm_resize_heatmap": [
        ("--heatmap",),
        ("--heatmap", "--heatmapWidth", "6")],
    "toggle_profile": [
        ("--heatmap",),
        ("--heatmap", "--profile"),
        ("--heatmap",),
        ("--heatmap", "--profile")],
    "toggle_heatmap": [
        ("--profile", "--sortRegions", "descend"),
        ("--profile", "--heatmap", "--sortRegions", "descend")],
}

_MIXED_CYCLE = [
    ("--profile",),
    ("--heatmap",),
    ("--profile", "--heatmap"),
    ("--profile", "--kmeans", "4"),
    ("--profile", "--hclust", "4"),
    ("--profile", "--regionsLabel", "renamed"),
    ("--heatmap", "--colorMap", "viridis"),
    ("--heatmap", "--heatmapWidth", "6"),
    ("--heatmap", "--profile", "--plotTitle", "Both panels"),
    ("--profile", "--plotTitle", "Profile again"),
]
SESSION_SCENARIOS["mixed20"] = _MIXED_CYCLE * 2

CHILD = r"""
import cProfile, io, json, pstats, resource, sys, time
start = time.perf_counter()
tool, profile, argv = sys.argv[1], sys.argv[2] == "1", sys.argv[3:]
module = __import__("deeptoolsr." + tool, fromlist=["main"])
from matplotlib.backends.backend_agg import FigureCanvasAgg
draws = [0]
original = FigureCanvasAgg.draw
def counted(self, *args, **kwargs):
    draws[0] += 1
    return original(self, *args, **kwargs)
FigureCanvasAgg.draw = counted
imported = time.perf_counter()
profiler = cProfile.Profile() if profile else None
if profiler: profiler.enable()
module.main(argv)
if profiler: profiler.disable()
end = time.perf_counter()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mib = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
result = {"import_s": imported - start, "run_s": end - imported, "draws": draws[0],
          "peak_rss_mib": rss_mib}
if profiler:
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(25)
    result["profile"] = stream.getvalue()
print("RESULT " + json.dumps(result))
"""

SESSION_CHILD = r"""
import cProfile, io, json, pstats, resource, sys, time
start = time.perf_counter()
profile = sys.argv[1] == "1"
commands = json.loads(sys.argv[2])
from deeptoolsr.session import PlotSession
from matplotlib.backends.backend_agg import FigureCanvasAgg
draws = [0]
original = FigureCanvasAgg.draw
def counted(self, *args, **kwargs):
    draws[0] += 1
    return original(self, *args, **kwargs)
FigureCanvasAgg.draw = counted
imported = time.perf_counter()
profiler = cProfile.Profile() if profile else None
requests = []
with PlotSession() as session:
    if profiler: profiler.enable()
    for command in commands:
        before = time.perf_counter()
        result = session.run(command)
        requests.append({"wall_s": time.perf_counter() - before,
                         "recomputed": result.recomputed,
                         "reused": result.reused})
    if profiler: profiler.disable()
end = time.perf_counter()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mib = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
result = {"import_s": imported - start, "run_s": end - imported,
          "requests": requests, "draws": draws[0], "peak_rss_mib": rss_mib}
if profiler:
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(25)
    result["profile"] = stream.getvalue()
print("RESULT " + json.dumps(result))
"""


def _group_relabels(matrix):
    opener = gzip.open if matrix.endswith('.gz') else open
    with opener(matrix, 'rt') as handle:
        header = json.loads(handle.readline()[1:])
    return tuple(f'renamed {index + 1}'
                 for index in range(len(header['group_labels'])))


def session_commands(name, matrix, scratch, repeat):
    """Give every request its own paths, including sort-consuming matrices."""
    commands = []
    relabels = _group_relabels(matrix) if name == 'mixed20' else ()
    for index, options in enumerate(SESSION_SCENARIOS[name]):
        prefix = os.path.join(scratch, f'{name}-{repeat}-{index}')
        if '--regionsLabel' in options:
            options = (*options[:-1], *relabels)
        argv = ['plotMatrixR', '-m', matrix, '-o', prefix + '.pdf',
                '-p', '4', *options]
        if name == 'toggle_heatmap':
            argv.extend(['--outFileNameMatrix', prefix + '.matrix.gz'])
        commands.append(argv)
    return commands


def run_session_scenario(name, matrix, scratch, repeat, profile):
    commands = session_commands(name, matrix, scratch, repeat)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, '-c', SESSION_CHILD, '1' if profile else '0',
         json.dumps(commands)], capture_output=True, text=True, check=True,
        cwd=scratch)
    wall = time.perf_counter() - started
    line = next(line for line in completed.stdout.splitlines()
                if line.startswith('RESULT '))
    result = json.loads(line[7:])
    result['wall_s'] = wall
    result['warm_wall_s'] = statistics.median(
        request['wall_s'] for request in result['requests'][1:])
    return commands, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix")
    parser.add_argument("--scenario", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--threads", help="value for -p, when the tool accepts it")
    args = parser.parse_args()
    metadata = environment(args.threads)
    if any(name in SESSION_SCENARIOS for name in args.scenario):
        metadata['session_threads'] = 4
    print(json.dumps({"environment": metadata}))
    with tempfile.TemporaryDirectory() as scratch:
        for name in args.scenario:
            if name in SESSION_SCENARIOS:
                runs = []
                for index in range(args.repeat):
                    commands, result = run_session_scenario(
                        name, os.path.abspath(args.matrix), scratch, index,
                        args.profile and index == 0)
                    runs.append(result)
                print(json.dumps({"scenario": name, "commands": commands,
                                  "median_wall_s": statistics.median(
                                      run['warm_wall_s'] for run in runs),
                                  "median_process_wall_s": statistics.median(
                                      run['wall_s'] for run in runs),
                                  "runs": runs}))
                continue
            tool, *options = SCENARIOS[name]
            argv = ["-m", os.path.abspath(args.matrix), "-o", os.path.join(scratch, name + ".pdf"), *options]
            if args.threads:
                argv += ["-p", args.threads]
            runs = []
            for index in range(args.repeat):
                profile = "1" if args.profile and index == 0 else "0"
                started = time.perf_counter()
                completed = subprocess.run([sys.executable, "-c", CHILD, tool, profile, *argv],
                                           capture_output=True, text=True, check=True,
                                           cwd=scratch)  # never import the uncompiled source tree
                wall = time.perf_counter() - started
                line = next(line for line in completed.stdout.splitlines()
                            if line.startswith("RESULT "))
                runs.append({"wall_s": wall, **json.loads(line[7:])})
            summary = {"scenario": name, "argv": argv,
                       "median_wall_s": statistics.median(r["wall_s"] for r in runs),
                       "runs": runs}
            print(json.dumps(summary))


def environment(threads):
    def version(module):
        try:
            return __import__(module).__version__
        except Exception:
            return None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    commit = commit or os.environ.get("DEEPTOOLSR_BENCH_COMMIT") or "unknown"
    return {"commit": commit, "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version(), "matplotlib": version("matplotlib"),
            "numpy": version("numpy"), "threads": threads or "tool default"}


if __name__ == "__main__":
    main()
