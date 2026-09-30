"""Line figures of the release benchmark: mean of the reps per tool with a 95% t-interval band."""

import math
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

TOOLS = ("deepToolsR", "3.5.6", "4.0.0")
LABEL = {"deepToolsR": "deepToolsR 0.9.0", "3.5.6": "deepTools 3.5.6",
         "4.0.0": "deepTools 4.0.0 (Rust)"}
COLOR = {"deepToolsR": "#2a78d6", "3.5.6": "#eb6834", "4.0.0": "#1baf7a"}  # fixed slots 1-3
# The plot group draws deepToolsR twice (explicit -p 1 and -p 8); deepTools plots single-threaded.
PLOT_LINES = [("deepToolsR", 1, "deepToolsR 0.9.0 (-p 1)", "-"),
              ("deepToolsR", 8, "deepToolsR 0.9.0 (-p 8)", "--"),
              ("3.5.6", 1, LABEL["3.5.6"], "-"), ("4.0.0", 1, LABEL["4.0.0"], "-")]
MARKER = {"deepToolsR": "o", "3.5.6": "s", "4.0.0": "D"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUPS = {"bamcov_chip": "bamCoverage, ChIP-seq (28.4 M reads, CPM)",
          "rna": "bamCoverage, stranded RNA, both strands (70.8 M alignments)",
          "matrix": "computeMatrix, POINT-seq bigWigs, binSize 10",
          "matrix_stranded": "computeMatrix, stranded (one run vs + / - runs and rbind)",
          "plot": "plotHeatmap / plotProfile"}
PANEL = {"bamCoverage ChIP bs50": "binSize 50", "bamCoverage ChIP bs1": "binSize 1",
         "RNA stranded bs10": "binSize 10", "RNA stranded bs1": "binSize 1",
         "small": "small: 3k genes, 1 sample", "medium": "medium: 10k genes, 1 sample",
         "big": "big: 20k genes, 4 samples"}
SIZES = ("small", "medium", "big")
METRICS = {"wall": ("wall_s", "Wall time (s)"), "rss": ("peak_rss_mb", "Peak RSS, process tree (MB)")}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}  # two-sided 95% t quantiles
CAPTION = "Line: mean of n={n} runs; band: 95% t-interval (mean ± t(0.975, n-1)·sd/√n)."
NOTES = {"rna": " deepTools 3.5.6 at binSize 1 was run at -p 8 only (single point).",
         "matrix_stranded": " Stranded big was run at -p 8 only (single points).",
         "plot": "\ndeepTools plots single-threaded; deepToolsR at -p 1 and -p 8;"
                 " same matrix and panel pixel size for every tool."}


def mean_ci(values):
    m = statistics.fmean(values)
    if len(values) < 2:
        return m, m, m
    h = T975.get(len(values) - 1, 1.96) * statistics.stdev(values) / math.sqrt(len(values))
    return m, m - h, m + h


def _series(recs, group, case, tool, key, threads=None):
    """x -> (ok values, timed-out value or None) for one panel line."""
    out = {}
    for r in recs:
        if r["group"] != group or r["tool"] != tool:
            continue
        if threads is not None and r["threads"] != threads:
            continue
        if group == "plot":
            name, size = r["case"].split(" ")
            if name != case:
                continue
            x = SIZES.index(size)
        elif r["case"] != case:
            continue
        else:
            x = r["threads"]
        ok, to = out.setdefault(x, ([], None))
        if r["status"] == 0:
            ok.append(r[key])
        elif r["status"] == "timeout":
            out[x] = (ok, r[key])
    return out


def _panel(ax, recs, group, case, metric):
    key, ylabel = METRICS[metric]
    lo_all, hi_all = [], []
    lines = PLOT_LINES if group == "plot" else [(t, None, LABEL[t], "-") for t in TOOLS]
    for tool, threads, label, ls in lines:
        ser = _series(recs, group, case, tool, key, threads)
        xs = sorted(x for x, (ok, _) in ser.items() if ok)
        if xs:
            stats = [mean_ci(ser[x][0]) for x in xs]
            m = [s[0] for s in stats]
            ax.plot(xs, m, color=COLOR[tool], lw=2, ls=ls, marker=MARKER[tool], ms=6,
                    mfc=SURFACE if ls != "-" else COLOR[tool], label=label, zorder=3)
            ax.fill_between(xs, [max(s[1], s[0] * 0.05) for s in stats], [s[2] for s in stats],
                            color=COLOR[tool], alpha=0.18, lw=0, zorder=2)
            lo_all += m
            hi_all += [s[2] for s in stats]
        for x, (ok, to) in ser.items():
            if to is not None and not ok:
                ax.plot([x], [to], marker=MARKER[tool], ms=8, mfc=SURFACE, mec=COLOR[tool], mew=2,
                        ls="none", zorder=4)
                what = "timed out" if metric == "wall" else "timed out\n(peak before kill)"
                right = x == max(ser) and group != "plot" or group == "plot" and x == len(SIZES) - 1
                ax.annotate(what, (x, to), xytext=(-8 if right else 8, -4),
                            textcoords="offset points", ha="right" if right else "left", fontsize=7,
                            color=INK2, va="top")
                lo_all.append(to)
                hi_all.append(to)
    if lo_all and max(hi_all) / max(min(lo_all), 1e-9) > 10:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ylabel += " (log)"
    else:
        ax.set_ylim(bottom=0)
    if group == "plot":
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels([PANEL[s].split(":")[0] for s in SIZES])
        ax.set_xlabel("matrix size", color=INK2, fontsize=8)
        ax.set_title(case, loc="left", color=INK, fontsize=9)
    else:
        ax.set_xticks([1, 4, 8])
        ax.set_xlim(0.5, 8.5)
        ax.set_xlabel("processors (-p)", color=INK2, fontsize=8)
        ax.set_title(PANEL.get(case.rsplit(" ", 1)[-1], PANEL.get(case, case)), loc="left",
                     color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=8)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_group(recs, group, metric, out):
    # Owner decision: stranded big is shown at -p 8 only (its -p 1 records, if any, are kept
    # in runs.json and summary.csv but not plotted).
    recs = [r for r in recs if not (r["case"] == "computeMatrix stranded big" and r["threads"] != 8)]
    if group == "plot":
        cases = ["plotHeatmap", "plotProfile"]
    else:
        cases = []
        for r in recs:
            if r["group"] == group and r["case"] not in cases:
                cases.append(r["case"])
    n = max((len([x for x in recs if x["group"] == group and x["case"] == r["case"]
                  and x["tool"] == r["tool"] and x["threads"] == r["threads"] and x["status"] == 0])
             for r in recs if r["group"] == group), default=0)
    fig, axes = plt.subplots(1, len(cases), figsize=(3.6 * len(cases) + 0.6, 3.9), dpi=150,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, case in zip(axes[0], cases):
        _panel(ax, recs, group, case, metric)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.suptitle(GROUPS[group], x=0.01, ha="left", color=INK, fontsize=10)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.95), ncol=3, frameon=False, fontsize=8,
               labelcolor=INK)
    fig.text(0.01, 0.01, CAPTION.format(n=n) + NOTES.get(group, ""), color=INK2, fontsize=7)
    fig.tight_layout(rect=(0, 0.07 if "\n" in NOTES.get(group, "") else 0.04, 1, 0.88))
    for ext in ("png", "svg"):
        fig.savefig(out / f"{group}_{metric}.{ext}", facecolor=SURFACE)
    plt.close(fig)


def plot_all(recs, out):
    out.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        if any(r["group"] == group for r in recs):
            for metric in METRICS:
                plot_group(recs, group, metric, out)
