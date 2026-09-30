#!/usr/bin/env python
"""Release benchmark: deepToolsR vs deepTools 3.5.6 vs deepTools 4.0.0.

Runs every benchmark case for each tool, measuring wall time and peak RSS of
the whole process tree (psutil sampling every 50 ms, RSS summed over the
process and all its descendants; forked workers' shared pages are counted in
each worker, so multi-process tools are slightly over-reported).

Data is never bundled: pass paths. Example:

    python benchmarks/release/bench.py run \
        --chip-bam DATA/ENCFF772IUE.bam --rna-bam RNA.bam \
        --regions DATA/protein_coding_genes.bed --workdir DATA/runs \
        --matrix-bigwigs S1.plus.bw S2.plus.bw S3.plus.bw S4.plus.bw \
        --matrix-bigwigs-minus S1.minus.bw S2.minus.bw S3.minus.bw S4.minus.bw \
        --r-bin R_VENV/bin --v356-bin DT356_VENV/bin --v400-bin DT400_ENV/bin
    python benchmarks/release/bench.py parity --workdir DATA/runs
    python benchmarks/release/bench.py report --workdir DATA/runs \
        --out benchmarks/release

`run` appends one JSON line per run to WORKDIR/runs.jsonl (resumable: runs
already recorded are skipped). `--only` selects benchmark groups
(bamcov_chip, rna, matrix, matrix_stranded, plot); `--tools` a subset of deepToolsR, 3.5.6,
4.0.0 (only the selected tools' --*-bin options are needed). Portable across
macOS, Linux and Windows. Before every run the harness waits until the machine
is idle (system CPU below --idle-cpu percent, no pytest/pip/cmake process) and
discards and repeats a run during which such a process or a busy VM
(qemu/UTM/VirtualBox/Hyper-V) appeared. Needs psutil, numpy, pyBigWig,
matplotlib.
"""

import argparse
import gzip
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import psutil

TOOLS = ("deepToolsR", "3.5.6", "4.0.0")
EXE = {  # tool -> command name map
    "deepToolsR": {"bamCoverage": "bamCoverageR", "computeMatrix": "computeMatrixR",
                   "plotHeatmap": "plotHeatmapR", "plotProfile": "plotProfileR",
                   "computeMatrixOperations": "computeMatrixOperationsR"},
    "3.5.6": {}, "4.0.0": {},
}


BUSY_CMD = re.compile(r"pytest|pip install|cmake", re.I)
VM_NAME = re.compile(r"qemu|virtualbox|vmmem|vmware", re.I)
IDLE_CPU = 15.0
NO_WAIT = False


def busy():
    """True while a test suite, pip/cmake build or busy VM runs on this machine."""
    if NO_WAIT:
        return False
    me = os.getpid()
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] == me:
                continue
            if BUSY_CMD.search(" ".join(p.info["cmdline"] or [])):
                return True
            if VM_NAME.search(p.info["name"] or "") and p.cpu_percent(0.2) > 20:
                return True
        except psutil.Error:
            pass
    return False


def wait_idle():
    while not NO_WAIT and (busy() or psutil.cpu_percent(interval=2) > IDLE_CPU):
        time.sleep(15)


def measure(cmd, timeout):
    """Run cmd; return (wall_s, peak_rss_bytes, returncode); repeat if the machine got busy."""
    while True:
        wait_idle()
        res = _measure(cmd, timeout)
        if res is not None:
            return res
        print("contended run discarded; retrying", flush=True)


def _measure(cmd, timeout):
    errlog = Path(cmd[cmd.index("-o") + 1] + ".stderr")
    t0 = time.perf_counter()
    with errlog.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=fh)
    ps = psutil.Process(proc.pid)
    peak, last_check, contended = 0, t0, False
    while proc.poll() is None:
        if time.perf_counter() - last_check > 2:
            contended = contended or busy()
            last_check = time.perf_counter()
        total = 0
        try:
            for p in [ps] + ps.children(recursive=True):
                try:
                    total += p.memory_info().rss
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
        peak = max(peak, total)
        if time.perf_counter() - t0 > timeout:
            for p in ps.children(recursive=True):
                p.kill()
            proc.kill()
            proc.wait()
            return time.perf_counter() - t0, peak, "timeout"
        time.sleep(0.05)
    wall = time.perf_counter() - t0
    if contended:
        return None
    if proc.returncode:
        print(errlog.read_text()[-2000:])
    return wall, peak, proc.returncode


def exe(args, tool, name):
    base = {"deepToolsR": args.r_bin, "3.5.6": args.v356_bin, "4.0.0": args.v400_bin}[tool]
    if not base:
        raise SystemExit(f"--tools includes {tool}: pass its --*-bin directory")
    name = EXE[tool].get(name, name)
    found = shutil.which(name, path=str(Path(base).expanduser()))
    if not found:
        raise SystemExit(f"{name} not found in {base}")
    return found


def cases(args, w):
    """Yield (group, case, threads, tool, [cmds], note). Several cmds = sequential steps."""
    chip, rna = args.chip_bam, args.rna_bam
    for bs in (50, 1):
        for p in (1, 4, 8):
            for tool in TOOLS:
                out = w / f"chip_bs{bs}" / f"{tool}_p{p}.bw"
                cmd = [exe(args, tool, "bamCoverage"), "-b", chip, "-o", str(out),
                       "--binSize", str(bs), "--normalizeUsing", "CPM", "-p", str(p)]
                yield "bamcov_chip", f"bamCoverage ChIP bs{bs}", p, tool, [cmd], ""
    for bs, p in ((bs, p) for bs in (10, 1) for p in (1, 4, 8)):
        for tool in TOOLS:
            if tool == "3.5.6" and bs == 1 and p != 8:
                continue  # owner decision: 3.5.6 bs1 RNA (two strand runs) at -p 8 only
            d = w / f"rna_bs{bs}"
            common = ["-b", rna, "--binSize", str(bs), "-p", str(p)]
            if tool == "deepToolsR":
                cmds = [[exe(args, tool, "bamCoverage"), *common, "-o", str(d / f"{tool}.bw"),
                         "--normalizeUsing", "read-count", "--strandedness", "reverse",
                         "--filterRNAstrand", "split"]]
                note = ("one run, --normalizeUsing read-count --strandedness reverse "
                        "--filterRNAstrand split")
            else:
                cmds = [[exe(args, tool, "bamCoverage"), *common, "-o", str(d / f"{tool}.{s}.bw"),
                         "--filterRNAstrand", s] for s in ("forward", "reverse")]
                note = "two runs (--filterRNAstrand forward, reverse)"
            yield "rna", f"RNA stranded bs{bs}", p, tool, cmds, note
    genes = gene_sets(args)
    for size, (_n, samples, mode, opts) in MATRIX_SIZES.items():
        bws = args.matrix_bigwigs[:samples]
        labels = [f"S{i + 1}" for i in range(samples)]
        for p in (1, 4, 8):
            for tool in TOOLS:
                out = w / f"matrix_{size}" / f"{tool}_p{p}.gz"
                cmd = [exe(args, tool, "computeMatrix"), mode, "-S", *bws, "-R", genes[size],
                       *opts, "--binSize", "10", "--samplesLabel", *labels,
                       "-o", str(out), "-p", str(p)]
                yield "matrix", f"computeMatrix {size}", p, tool, [cmd], ""
    for size in STRANDED_SIZES:
        _n, samples, mode, opts = MATRIX_SIZES[size]
        plus, minus = args.matrix_bigwigs[:samples], args.matrix_bigwigs_minus[:samples]
        common = [*opts, "--binSize", "10", "--samplesLabel", *[f"S{i + 1}" for i in range(samples)]]
        for p in (1, 4, 8) if size == "medium" else (8,):  # owner: stranded big at -p 8 only
            for tool in TOOLS:
                out = w / f"matrix_stranded_{size}" / f"{tool}_p{p}.gz"
                cm = exe(args, tool, "computeMatrix")
                if tool == "deepToolsR":
                    cmds = [[cm, mode, "-S", *plus, "--scoreFileNameMinus", *minus,
                             "-R", genes[size], *common, "-o", str(out), "-p", str(p)]]
                    note = "one run, --scoreFileNameMinus"
                else:
                    parts = [out.with_suffix(f".{s}.gz") for s in ("plus", "minus")]
                    cmds = [[cm, mode, "-S", *bws, "-R", genes[f"{size}.{s}"], *common,
                             "-o", str(part), "-p", str(p)]
                            for s, bws, part in (("plus", plus, parts[0]), ("minus", minus, parts[1]))]
                    cmds.append([exe(args, tool, "computeMatrixOperations"), "rbind", "-m",
                                 *map(str, parts), "-o", str(out)])
                    note = "computeMatrix on + genes and on - genes, then computeMatrixOperations rbind"
                yield "matrix_stranded", f"computeMatrix stranded {size}", p, tool, cmds, note
    for size in MATRIX_SIZES:
        mat = str(w / f"input_matrix_{size}.gz")
        for name in ("plotHeatmap", "plotProfile"):
            for tool in TOOLS:
                # deepTools plotting is single-threaded; deepToolsR defaults to -p auto
                # (half the cores), so it always gets an explicit -p.
                for p in (1, 8) if tool == "deepToolsR" else (1,):
                    out = w / f"plot_{name}_{size}" / f"{tool}_p{p}.png"
                    cmd = [exe(args, tool, name), "-m", mat, "-o", str(out),
                           *PLOT_OPTS[name](tool, size)]
                    if tool == "deepToolsR":
                        cmd += ["-p", str(p)]
                    yield "plot", f"{name} {size}", p, tool, [cmd], ""


# Figure geometry for the plot benchmarks. deepTools 3.5.6 is the reference: plotHeatmap
# --heatmapHeight 28 --heatmapWidth 4, plotProfile --plotHeight 7 --plotWidth 11, all at
# --dpi 200. The tools map these centimetres to pixels differently (3.5.6 and 4.0.0 size
# the whole figure grid, deepToolsR sizes the data panel), so deepToolsR's and 4.0.0's
# values were calibrated so that each heatmap/profile panel has the same pixel size as in
# 3.5.6 (measured from the frame lines of the PNGs; see RESULTS.md).
HEATMAP_CM = {  # tool -> (height, {size: per-sample width})
    "3.5.6": (28, {"small": 4, "medium": 4, "big": 4}),
    "deepToolsR": (21.34, {"small": 2.69, "medium": 2.69, "big": 3.18}),
    "4.0.0": (28.96, {"small": 4.64, "medium": 4.40, "big": 4.54}),
}
PROFILE_CM = {  # tool -> (height, {size: per-sample width})
    "3.5.6": (7, {"small": 11, "medium": 11, "big": 11}),
    "deepToolsR": (4.71, {"small": 9.54, "medium": 9.54, "big": 10.26}),
    "4.0.0": (7.60, {"small": 10.73, "medium": 10.73, "big": 10.88}),
}
PLOT_OPTS = {
    "plotHeatmap": lambda tool, size: [
        "--heatmapHeight", str(HEATMAP_CM[tool][0]), "--heatmapWidth", str(HEATMAP_CM[tool][1][size]),
        "--dpi", "200", "--sortRegions", "descend"],
    "plotProfile": lambda tool, size: [
        "--plotHeight", str(PROFILE_CM[tool][0]), "--plotWidth", str(PROFILE_CM[tool][1][size]),
        "--dpi", "200"],
}

# size -> (genes, samples, computeMatrix mode, mode options); gene sets are nested.
MATRIX_SIZES = {
    "small": (3000, 1, "reference-point", ["--referencePoint", "TSS", "-b", "3000", "-a", "3000"]),
    "medium": (10000, 1, "reference-point", ["--referencePoint", "TSS", "-b", "3000", "-a", "3000"]),
    "big": (20000, 4, "scale-regions", ["-m", "4000", "-b", "3000", "-a", "3000"]),
}
STRANDED_SIZES = ("medium", "big")


def gene_sets(args):
    """Nested seeded random gene draws (small in medium in big), written beside --regions."""
    import numpy as np
    src = Path(args.regions)
    lines = [x for x in src.read_text().splitlines() if x.strip()]
    order = np.random.default_rng(args.seed).permutation(len(lines))
    out = {}
    for size, (n, *_rest) in MATRIX_SIZES.items():
        path = src.with_name(f"{src.stem}.{size}_{n}.seed{args.seed}.bed")
        if not path.exists():
            pick = sorted(order[:n])
            path.write_text("\n".join(lines[i] for i in pick) + "\n")
        out[size] = str(path)
        for strand, sign in (("plus", "+"), ("minus", "-")):
            part = path.with_suffix(f".{strand}.bed")
            if not part.exists():
                part.write_text("".join(x + "\n" for x in path.read_text().splitlines()
                                        if x.split("\t")[5] == sign))
            out[f"{size}.{strand}"] = str(part)
    return out


def prepare_inputs(w, tools):
    """Stage each size's -p 8 matrix from the first selected tool (deepToolsR by
    default) as the plots' shared input, so every tool plots the same file."""
    for size in MATRIX_SIZES:
        src, dst = w / f"matrix_{size}" / f"{tools[0]}_p8.gz", f"input_matrix_{size}.gz"
        if src.exists() and not (w / dst).exists():
            shutil.copy(src, w / dst)


def machine_info(args):
    """Machine and tool versions, recorded once per workdir."""
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
    elif platform.system() == "Linux":
        m = re.search(r"model name\s*:\s*(.*)", Path("/proc/cpuinfo").read_text())
        cpu = m.group(1) if m else cpu or platform.machine()
    info = {"os": platform.platform(), "mac_ver": platform.mac_ver()[0], "machine": platform.machine(),
            "cpu": cpu, "cores_logical": psutil.cpu_count(),
            "cores_physical": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "python": platform.python_version(), "tools": {}}
    for tool in args.tools:
        info["tools"][tool] = {}
        for name in ("bamCoverage", "computeMatrix", "plotHeatmap", "plotProfile"):
            r = subprocess.run([exe(args, tool, name), "--version"], capture_output=True, text=True)
            info["tools"][tool][name] = (r.stdout + r.stderr).strip().splitlines()[-1]
    return info


def cmd_run(args):
    w = Path(args.workdir).expanduser()
    w.mkdir(parents=True, exist_ok=True)
    log = w / "runs.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            done[(r["case"], r["threads"], r["tool"], r["rep"])] = r
    global IDLE_CPU, NO_WAIT
    IDLE_CPU, NO_WAIT = args.idle_cpu, args.no_wait
    (w / "machine.json").write_text(json.dumps(machine_info(args), indent=1))
    for group, case, p, tool, cmds, note in cases(args, w):
        if (args.only and group not in args.only) or tool not in args.tools:
            continue
        prepare_inputs(w, args.tools)
        for rep in range(1, args.reps + 1):
            key = (case, p, tool, rep)
            if key in done:
                continue
            if rep > 1 and done.get((case, p, tool, 1), {}).get("status") == "timeout":
                break
            for c in cmds:
                Path(c[c.index("-o") + 1]).parent.mkdir(parents=True, exist_ok=True)
            wall, rss, status = 0.0, 0, 0
            for c in cmds:
                t, r, status = measure(c, args.timeout - wall)
                wall, rss = wall + t, max(rss, r)
                if status:
                    break
            rec = {"group": group, "case": case, "threads": p, "tool": tool, "rep": rep,
                   "wall_s": round(wall, 3), "peak_rss_mb": round(rss / 2**20, 1),
                   "status": status, "steps": len(cmds), "note": note,
                   "cmd": [" ".join(c) for c in cmds]}
            done[key] = rec
            with log.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{case:32s} p{p} {tool:10s} rep{rep} {wall:8.1f}s {rss / 2**20:8.0f} MB {status}",
                  flush=True)
            if status == "timeout":
                break
    prepare_inputs(w, args.tools)


# ---------------------------------------------------------------- parity
PARITY_CHROMS = ("chr19", "chr21", "chr22")


def bw_diff(a, b):
    import numpy as np
    import pyBigWig
    fa, fb = pyBigWig.open(str(a)), pyBigWig.open(str(b))
    worst, nbad, n = 0.0, 0, 0
    for c in PARITY_CHROMS:
        va = np.nan_to_num(fa.values(c, 0, fa.chroms(c), numpy=True))
        vb = np.nan_to_num(fb.values(c, 0, fb.chroms(c), numpy=True))
        d = np.abs(va - vb)
        worst = max(worst, float(d.max()))
        nbad += int((d > 1e-3 * np.maximum(1, np.abs(vb))).sum())
        n += d.size
    return {"max_abs_diff": worst, "bases_over_tol": nbad, "bases": n}


def read_matrix(path):
    import numpy as np
    rows = {}
    with gzip.open(path, "rt") as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            rows[tuple(f[:3] + f[5:6])] = np.array([float(x) for x in f[6:]])
    return rows


def matrix_diff(a, b):
    import numpy as np
    ra, rb = read_matrix(a), read_matrix(b)
    common = ra.keys() & rb.keys()
    worst = 0.0
    for k in common:
        va, vb = ra[k], rb[k]
        if va.shape != vb.shape:
            worst = float("inf")
            continue
        both = ~(np.isnan(va) & np.isnan(vb))
        d = np.abs(np.nan_to_num(va[both], nan=np.inf) - np.nan_to_num(vb[both], nan=np.inf))
        if d.size:
            worst = max(worst, float(np.nan_to_num(d.max(), posinf=np.inf)))
    return {"rows_a": len(ra), "rows_b": len(rb), "rows_common": len(common),
            "max_abs_diff": worst}


def cmd_parity(args):
    w = Path(args.workdir).expanduser()
    res = []
    for bs in (50, 1):
        for other in ("3.5.6", "4.0.0"):
            b = w / f"chip_bs{bs}" / f"{other}_p8.bw"
            if b.exists():
                res.append({"check": f"bamCoverage ChIP bs{bs} p8", "vs": other,
                            **bw_diff(w / f"chip_bs{bs}" / "deepToolsR_p8.bw", b)})
    for bs in (10, 1):
        d = w / f"rna_bs{bs}"
        for other in ("3.5.6", "4.0.0"):
            for mine, s in (("plus", "forward"), ("minus", "reverse")):
                a, b = d / f"deepToolsR.{mine}.bw", d / f"{other}.{s}.bw"
                if a.exists() and b.exists():
                    res.append({"check": f"RNA bs{bs} {s} (R split .{mine})", "vs": other,
                                **bw_diff(a, b)})
        a, b = d / "R_tworun.forward.bw", d / "deepToolsR.plus.bw"
        if a.exists():
            res.append({"check": f"RNA bs{bs} split vs deepToolsR two-run forward", "vs": "deepToolsR",
                        **bw_diff(b, a)})
    for size in STRANDED_SIZES:
        for other in ("3.5.6", "4.0.0"):
            b = w / f"matrix_stranded_{size}" / f"{other}_p8.gz"
            if b.exists():
                res.append({"check": f"computeMatrix stranded {size} p8 (rows matched by region)",
                            "vs": f"{other} rbind", **matrix_diff(
                                w / f"matrix_stranded_{size}" / "deepToolsR_p8.gz", b)})
    for size in MATRIX_SIZES:
        for other in ("3.5.6", "4.0.0"):
            b = w / f"matrix_{size}" / f"{other}_p8.gz"
            if b.exists():
                res.append({"check": f"computeMatrix {size} p8", "vs": other,
                            **matrix_diff(w / f"matrix_{size}" / "deepToolsR_p8.gz", b)})
    (w / "parity.json").write_text(json.dumps(res, indent=1))
    for r in res:
        print(json.dumps(r))


# ---------------------------------------------------------------- report
def cmd_report(args):
    import statistics
    from plot_results import mean_ci, plot_all
    w, out = Path(args.workdir).expanduser(), Path(args.out)
    recs = [json.loads(x) for x in (w / "runs.jsonl").read_text().splitlines()]
    # Published copies carry no local paths (the RNA data set is unpublished).
    (out / "runs.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "cmd"} for r in recs], indent=1))
    info = json.loads((w / "machine.json").read_text())
    for t in info["tools"].values():
        t.pop("path", None)
    (out / "machine.json").write_text(json.dumps(info, indent=1))
    if (w / "parity.json").exists():
        shutil.copy(w / "parity.json", out / "parity.json")
    rows, keys = [], []
    for r in recs:
        k = (r["group"], r["case"], r["threads"], r["tool"])
        if k not in keys:
            keys.append(k)
    for k in keys:
        rs = [r for r in recs if (r["group"], r["case"], r["threads"], r["tool"]) == k]
        ok = [r for r in rs if r["status"] == 0]
        if not ok:
            rows.append({"group": k[0], "case": k[1], "threads": k[2], "tool": k[3], "n": 0,
                         "status": str(rs[0]["status"])})
            continue
        t = [r["wall_s"] for r in ok]
        m = [r["peak_rss_mb"] for r in ok]
        tm, tlo, thi = mean_ci(t)
        mm, mlo, mhi = mean_ci(m)
        rows.append({"group": k[0], "case": k[1], "threads": k[2], "tool": k[3], "n": len(ok),
                     "status": "ok", "wall_mean_s": round(tm, 3), "wall_ci95_lo_s": round(tlo, 3),
                     "wall_ci95_hi_s": round(thi, 3), "rss_mean_mb": round(mm, 1),
                     "rss_ci95_lo_mb": round(mlo, 1), "rss_ci95_hi_mb": round(mhi, 1),
                     "wall_median_s": statistics.median(t),
                     "wall_min_s": min(t), "wall_max_s": max(t),
                     "rss_median_mb": statistics.median(m), "rss_min_mb": min(m),
                     "rss_max_mb": max(m)})
    cols = ["group", "case", "threads", "tool", "n", "status", "wall_mean_s", "wall_ci95_lo_s",
            "wall_ci95_hi_s", "wall_median_s", "wall_min_s", "wall_max_s", "rss_mean_mb",
            "rss_ci95_lo_mb", "rss_ci95_hi_mb", "rss_median_mb", "rss_min_mb", "rss_max_mb"]
    with (out / "summary.csv").open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    with (out / "runs.csv").open("w") as fh:
        rc = ["group", "case", "threads", "tool", "rep", "wall_s", "peak_rss_mb", "status"]
        fh.write(",".join(rc) + "\n")
        for r in recs:
            fh.write(",".join(str(r[c]) for c in rc) + "\n")
    plot_all(recs, out / "figures")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--chip-bam", required=True)
    r.add_argument("--rna-bam", required=True)
    r.add_argument("--regions", required=True,
                   help="gene BED; nested random subsets are written beside it")
    r.add_argument("--matrix-bigwigs", nargs=4, required=True, metavar="BW",
                   help="4 plus-strand bigWigs for computeMatrix (the first is the 1-sample input)")
    r.add_argument("--matrix-bigwigs-minus", nargs=4, required=True, metavar="BW",
                   help="the 4 matching minus-strand bigWigs (stranded computeMatrix)")
    r.add_argument("--seed", type=int, default=20260930, help="gene-subset seed")
    r.add_argument("--workdir", required=True)
    r.add_argument("--r-bin", help="directory holding bamCoverageR etc.")
    r.add_argument("--v356-bin", help="directory holding deepTools 3.5.6 executables")
    r.add_argument("--v400-bin", help="directory holding deepTools 4.0.0 executables")
    r.add_argument("--tools", nargs="+", choices=TOOLS, default=list(TOOLS))
    r.add_argument("--idle-cpu", type=float, default=15.0,
                   help="system CPU %% below which a run may start (default 15)")
    r.add_argument("--no-wait", action="store_true",
                   help="skip the idle gate (smoke tests only; timings are not valid)")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--timeout", type=float, default=1800, help="seconds per run (all steps)")
    r.add_argument("--only", nargs="*", choices=["bamcov_chip", "rna", "matrix", "matrix_stranded", "plot"])
    p = sub.add_parser("parity")
    p.add_argument("--workdir", required=True)
    q = sub.add_parser("report")
    q.add_argument("--workdir", required=True)
    q.add_argument("--out", default=str(Path(__file__).parent))
    a = ap.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))
    {"run": cmd_run, "parity": cmd_parity, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
