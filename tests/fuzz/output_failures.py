"""Output-failure harness: every tool and output type, with writes that fail.

Each command first runs unlimited to measure its outputs. It then runs again
with every output destination pre-filled, under write limits from tiny to
just below the full size of each output:

- ``rlimit`` (default): RLIMIT_FSIZE in the child, with SIGXFSZ ignored so a
  write past the limit fails with EFBIG instead of killing the process;
- ``enospc``: outputs and TMPDIR on a small filesystem (``--mount DIR``, for
  example a tmpfs) that is filled until only the limit is free. Linux only;
  runs serially.

A run is classified as ``ok-error`` (non-zero exit, one clean stderr line),
``noisy-error`` (several stderr lines), ``traceback``, ``crash``, ``hang``,
``silent-success`` (exit 0 with outputs different from the unlimited run) or
``success-same``. ``clobbered`` (a pre-existing output changed) and
``leftover`` (temporary or partial files in the output directory or TMPDIR)
are reported in addition.

Run it directly (``python -m tests.fuzz.output_failures OUTDIR [--mount DIR]``)
or through ``pytest tests/fuzz --fuzz``.
"""

import collections
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tests.fuzz import synthetic
from tests.fuzz.corrupt_inputs import LEFTOVER

TIMEOUT = 60
PREEXISTING = b"pre-existing output\n"
BC = ("bamCoverageR", "-b", "@pe.bam", "-bs", "1")
CMO = ("computeMatrixOperationsR",)
CM = ("-S", "@mb.bw", "@mb2.bw", "-R", "@mr1.bed", "@mr2.bed", "-bs", "50")
COMMANDS = {
    "bc-bigwig": (*BC, "-o", "out.bw"),
    "bc-bedgraph": (*BC, "-o", "out.bg", "-of", "bedgraph"),
    "bc-bedgraph-gz": (*BC, "-o", "out.bg.gz", "-of", "bedgraph"),
    "bc-split-bigwig": (*BC, "-o", "out.bw", "--filterRNAstrand", "split",
                        "--strandedness", "reverse"),
    "bc-split-bedgraph": (*BC, "-o", "out.bg", "-of", "bedgraph", "--filterRNAstrand",
                          "split", "--strandedness", "reverse"),
    "bwo-scale": ("bigWigOperationsR", "scale", "-b", "@mb.bw", "-o", "out.bw", "-s", "2"),
    "bwo-merge": ("bigWigOperationsR", "merge", "-b", "@mb.bw", "@mb2.bw", "-o", "out.bw"),
    "bwo-chromsizes": ("bigWigOperationsR", "info", "--chromSizes", "out.sizes", "@mb.bw"),
    "cm-scale-all": ("computeMatrixR", "scale-regions", *CM, "-m", "1000", "-o", "out.mat.gz",
                     "--outFileNameMatrix", "out.tab", "--outFileSortedRegions", "out.bed"),
    "cm-reference": ("computeMatrixR", "reference-point", *CM, "-a", "500", "-b", "500",
                     "-o", "out.mat.gz"),
    "cmo-relabel": (*CMO, "relabel", "-m", "@mA.mat.gz", "-o", "out.mat.gz",
                    "--sampleLabels", "a", "b"),
    "cmo-subset": (*CMO, "subset", "-m", "@mA.mat.gz", "-o", "out.mat.gz", "--samples", "mb2"),
    "cmo-filterStrand": (*CMO, "filterStrand", "-m", "@mA.mat.gz", "-o", "out.mat.gz",
                         "--strand", "+"),
    "cmo-filterValues": (*CMO, "filterValues", "-m", "@mA.mat.gz", "-o", "out.mat.gz",
                         "--min", "1"),
    "cmo-rbind": (*CMO, "rbind", "-m", "@mA.mat.gz", "@mB.mat.gz", "-o", "out.mat.gz"),
    "cmo-cbind": (*CMO, "cbind", "-m", "@mA.mat.gz", "@mC.mat.gz", "-o", "out.mat.gz"),
    "cmo-sort": (*CMO, "sort", "-m", "@mA.mat.gz", "-R", "@mr2.bed", "@mr1.bed",
                 "-o", "out.mat.gz"),
    "deeptoolsr-options": ("deeptoolsr", "options", "--config", "out.cfg",
                           "--fontMultiplier", "2"),
}
for _tool in ("plotHeatmapR", "plotProfileR"):
    for _ext in ("png", "pdf", "svg"):
        COMMANDS[f"{_tool}-{_ext}"] = (_tool, "-m", "@mA.mat.gz", "-o", f"out.{_ext}")
COMMANDS["plotHeatmapR-sidecars"] = ("plotHeatmapR", "-m", "@mA.mat.gz", "-o", "out.png",
                                     "--outFileSortedRegions", "out.bed",
                                     "--outFileNameMatrix", "out.tab")
COMMANDS["plotProfileR-sidecars"] = ("plotProfileR", "-m", "@mA.mat.gz", "-o", "out.png",
                                     "--outFileNameData", "out.tab",
                                     "--outFileSortedRegions", "out.bed")
LIMITER = ("import os, resource, signal, sys\n"
           "signal.signal(signal.SIGXFSZ, signal.SIG_IGN)\n"
           "n = int(sys.argv[1])\n"
           "resource.setrlimit(resource.RLIMIT_FSIZE, (n, n))\n"
           "os.execvp(sys.argv[2], sys.argv[2:])\n")


def _argv(cmd, inputs):
    return [str(inputs / a[1:]) if a.startswith("@") else a for a in cmd]


def _snapshot(path):
    return {p.name: p.read_bytes() for p in sorted(path.iterdir()) if p.is_file()}


def _fill(mount, free):
    """Leave about ``free`` bytes available on ``mount``."""
    filler = mount / "filler"
    stat = os.statvfs(mount)
    size = stat.f_bavail * stat.f_frsize - free
    if size > 0:
        with open(filler, "wb") as handle:
            os.posix_fallocate(handle.fileno(), 0, size)
    return filler


def run_one(rundir, inputs, cmd, limit=None, outputs=(), seed=b"", mount=None):
    """Run ``cmd`` in ``rundir/out`` with every output pre-filled; classify the run."""
    out, tmp = rundir / "out", rundir / "tmp"
    out.mkdir(parents=True)
    tmp.mkdir()
    for name in outputs:
        (out / name).write_bytes(seed)
    before = _snapshot(out)
    env = dict(os.environ, TMPDIR=str(tmp), PYTHONFAULTHANDLER="1")
    argv = _argv(cmd, inputs)
    if limit is not None and mount is None:
        argv = [sys.executable, "-c", LIMITER, str(limit), *argv]
    filler = _fill(mount, limit) if mount is not None and limit is not None else None
    try:
        proc = subprocess.run(argv, cwd=out, capture_output=True, timeout=TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"class": "hang", "rc": None, "stderr": "", "flags": []}
    finally:
        if filler is not None:
            filler.unlink(missing_ok=True)
    stderr = proc.stderr.decode("utf-8", "replace")
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    rc = proc.returncode
    after = _snapshot(out)
    if rc < 0 or rc >= 128:
        cls = "crash"
    elif "Traceback (most recent call last)" in stderr:
        cls = "traceback"
    elif rc:
        cls = "ok-error" if len(lines) == 1 else "noisy-error"
    else:
        cls = "success"
    flags = []
    if rc and any(after.get(n) != data for n, data in before.items()):
        flags.append("clobbered:" + ",".join(
            n for n, data in before.items() if after.get(n) != data))
    extra = sorted(set(after) - set(before))
    if rc == 0:
        extra = [n for n in extra if not n.startswith("out.") or LEFTOVER.search(n)]
    if extra:
        flags.append("leftover:" + ",".join(extra))
    tmp_left = sorted(os.listdir(tmp))
    if tmp_left:
        flags.append("tmp-leftover:" + ",".join(tmp_left))
    return {"class": cls, "rc": rc, "stderr": stderr[-2000:], "flags": flags,
            "outputs": {n: d for n, d in after.items() if n.startswith("out.")}}


def _limits(sizes):
    top = max(sizes)
    limits = {1, min(4096, top // 4), top // 2} | {s - 1 for s in sizes if s > 1}
    return sorted(x for x in limits if 0 < x < top)


def signature(stderr):
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    last = lines[-1] if lines else "<empty stderr>"
    return re.sub(r"\d+", "N", re.sub(r"/\S+", "<path>", last))[:160]


def run_all(workdir, jobs=6, mount=None, only=None):
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    inputs = workdir / "inputs"
    synthetic.write_all(inputs)
    runroot = Path(mount) if mount else workdir / "runs"
    names = [n for n in COMMANDS if not only or n in only]
    full = {}
    for name in names:
        res = run_one(workdir / "full" / name, inputs, COMMANDS[name])
        assert res["class"] == "success", (name, res["stderr"])
        full[name] = res["outputs"]
    todo = [(name, limit) for name in names
            for limit in _limits([len(d) for d in full[name].values()])]
    config = (inputs / "cfg.txt").read_bytes()

    def one(item):
        name, limit = item
        rundir = runroot / f"{name}@{limit}"
        seed = config if name == "deeptoolsr-options" else PREEXISTING
        res = run_one(rundir, inputs, COMMANDS[name], limit, tuple(full[name]), seed,
                      Path(mount) if mount else None)
        if res["class"] == "success":
            res["class"] = ("success-same" if res["outputs"] == full[name]
                            else "silent-success")
        res.pop("outputs", None)
        if mount or (res["class"] == "ok-error" and not res["flags"]):
            shutil.rmtree(rundir)
        res.update(name=name, limit=limit, cmd=" ".join(COMMANDS[name]),
                   sizes={n: len(d) for n, d in full[name].items()},
                   rundir=str(rundir), sig=signature(res["stderr"]))
        return res

    with concurrent.futures.ThreadPoolExecutor(1 if mount else jobs) as pool:
        results = list(pool.map(one, todo))
    (workdir / "results.json").write_text(json.dumps(results, indent=1))
    return results


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    mount = argv[argv.index("--mount") + 1] if "--mount" in argv else None
    workdir = Path(argv[0] if argv else "output-failures-work")
    results = run_all(workdir, jobs=int(os.environ.get("FUZZ_JOBS", "6")), mount=mount)
    counts = collections.Counter((r["name"], r["class"]) for r in results)
    for key in sorted(counts):
        print(*key, counts[key], sep="\t")
    print()
    groups = collections.defaultdict(list)
    for r in results:
        if r["class"] != "ok-error" or r["flags"]:
            groups[(r["class"], r["name"], r["sig"], tuple(r["flags"]))].append(r)
    for (cls, name, sig, flags), rs in sorted(groups.items()):
        print(f"[{cls}] {name} :: {sig} :: {list(flags)} :: limits="
              f"{[r['limit'] for r in rs]} sizes={rs[0]['sizes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
