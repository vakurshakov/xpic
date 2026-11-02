#!/usr/bin/env python3
"""Runs the same config through two xpic versions and writes
before/after tables per simulation to output/results.md.

Usage: python3 -I tests/benchmark/compare_versions/run.py <before> [<after>] [--repeats N]
       [--sims basic,ecsim,ecsimcorr,eccapfim] [--keep] [--workdir DIR]

<before>, <after>: an existing checkout directory (used as is) or a git revision (checked out into
a detached worktree under the workdir and removed at the end unless --keep). <after> defaults to
the current tree. Each checkout is built with `cmake --build build/Release --target xpic.out`.
Repeats are the outer loop and before/after are interleaved, so slow drifts of the machine
are spread evenly. Total threads are kept at 4: 1 rank x 4, 2 ranks x 2.
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile

FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[3]
OUT = FILE.parent / "output"
TEMPLATE = (FILE.parent / "config.template.json").read_text()
WORK = pathlib.Path()  # scratch directory, set in main()

SNES = (
    "-snes_type anderson "
    "-snes_anderson_m 40 "
    "-snes_anderson_restart 40 "
    "-snes_anderson_beta 0.65 "
    "-snes_atol 1e-7 "
    "-snes_rtol 1e-7 "
    "-snes_stol 1e-6 "
    "-snes_max_it 100"
).split()

# n: cells per axis, ppc: particles per cell
CASES = {
    "basic": dict(n=16, ppc=64, steps=20, dx=0.05, dt=0.025, extra=[]),
    "ecsim": dict(n=16, ppc=64, steps=20, dx=0.5, dt=1.5, extra=[]),
    "ecsimcorr": dict(n=16, ppc=64, steps=20, dx=0.5, dt=1.5, extra=[]),
    "eccapfim": dict(n=12, ppc=32, steps=3, dx=0.5, dt=1.5, extra=SNES),
}

RANKS = [1, 2]

PERF_EVENTS = ["cycles", "instructions", "L1-dcache-load-misses"]

# events (summed over all stages) shown when present in a run
EVENTS = [
    "first_push",
    "second_push",
    "fill_ecsim_current",
    "final_update"
    "prepare_storage",
    "form_iteration",
    "update_cells",
    "update_cells_with_assembly",
]


def config(sim, tag):
    c = CASES[sim]
    out = f"{WORK}/runs/{sim}_{tag}"
    text = TEMPLATE
    d = dict(
        SIMULATION=sim,
        OUT=out,
        N=c["n"],
        STEPS=c["steps"],
        DX=c["dx"],
        DT=c["dt"],
        PPC=c["ppc"]
    )
    for key, val in d.items():
        text = text.replace(f"@{key}@", str(val))
    path = WORK / "runs" / f"{sim}_{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path, out


def command(exe, cfg, sim, ranks, wrapper):
    return [*wrapper, "mpiexec", "-np", str(ranks), "--oversubscribe", "--bind-to", "none",
            str(exe), str(cfg), "-log_view", *CASES[sim]["extra"],
            *(["-da_processors_x", str(ranks)] if ranks > 1 else [])]


def env(ranks):
    return dict(os.environ, OMP_NUM_THREADS=str(4 // ranks), OMP_PROC_BIND="spread", OMP_PLACES="threads")


def parse_log_view(text):
    """Returns {metric: seconds} from the last PETSc summary: wall time, stages, selected events."""
    res = {}
    res["wall"] = float(re.search(r"^Time \(sec\):\s+(\S+)", text, re.M).group(1))
    summary = text[text.rindex("Summary of Stages:"):]
    for m in re.finditer(r"^\s*\d+:\s+(.+?):\s+(\S+)\s+[\d.]+%\s", summary.split("Phase summary")[0], re.M):
        res["stage: " + m.group(1)] = float(m.group(2))
    for m in re.finditer(r"^(\w+)\s+(\d+) 1\.0 (\S+) ", text, re.M):
        if m.group(1) in EVENTS:
            res["event: " + m.group(1)] = res.get("event: " + m.group(1), 0) + float(m.group(3))
    return res


def parse_perf_stat(path):
    res = {}
    for line in pathlib.Path(path).read_text().splitlines():
        f = line.split(",")
        if len(f) > 2 and f[0].isdigit():
            res[f[2].split(":")[0]] = float(f[0])
    return res


def last_row(path, columns):
    lines = pathlib.Path(path).read_text().splitlines()
    head = lines[0].split()
    row = lines[-1].split()
    return {c: float(row[head.index(c)]) for c in columns}


def run_once(exe, sim, ranks, tag):
    cfg, out = config(sim, tag)
    stat = f"{WORK}/runs/{sim}_{tag}.stat"
    wrapper = ["perf", "stat", "-x,", "-e", ",".join(PERF_EVENTS), "-o", stat]
    cmd = command(exe, cfg, sim, ranks, wrapper)
    done = subprocess.run(cmd, env=env(ranks), cwd=exe.parents[2], capture_output=True, text=True, check=True)
    res = parse_log_view(done.stdout + done.stderr)
    res.update({"perf: " + k: v for k, v in parse_perf_stat(stat).items()})
    res["IPC"] = res["perf: instructions"] / res["perf: cycles"]
    energy = last_row(f"{out}/temporal/energy.txt", ["wE", "wK_electrons"])
    res["final wK"] = energy["wK_electrons"]
    res["final wE"] = energy["wE"]
    return res


def perf_profile(exe, sim, tag, top=12):
    cfg, _ = config(sim, tag)
    data = f"{WORK}/runs/{sim}_{tag}.perf"
    cmd = command(exe, cfg, sim, 1, ["perf", "record", "-F", "499", "-o", data, "--"])
    subprocess.run(cmd, env=env(1), cwd=exe.parents[2], capture_output=True, text=True, check=True)
    rep = subprocess.run(["perf", "report", "-i", data, "--no-children", "--stdio", "--sort", "symbol",
                          "--percent-limit", "1"], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in rep.splitlines():
        m = re.match(r"\s*([\d.]+)%\s+\[\.\]\s+(.*)", line)
        if m:
            rows.append((float(m.group(1)), re.sub(r"\(.*", "", m.group(2)).strip()[:70]))
    return rows[:top]


def fmt(vals, scale=1.0, digits=1):
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals) * scale:.{digits}f} ± {sd * scale:.{digits}f}"


def table(sim, ranks, before, after):
    steps = CASES[sim]["steps"]
    keys = sorted({k for r in before + after for k in r},
                  key=lambda k: (not k.startswith("stage"), not k.startswith("event"), k.startswith("perf"), k))
    lines = ["| metric | before | after | after/before |", "| ------ | ------ | ----- | ------------ |"]
    for k in keys:
        b = [r[k] for r in before if k in r]
        a = [r[k] for r in after if k in r]
        if k.startswith(("stage", "event", "wall")):
            scale, digits, unit = 1e3 / steps, 2, " [ms/step]"
        elif k.startswith("perf"):
            scale, digits, unit = 1e-9 / steps, 3, " [G/step]"
        else:
            scale, digits, unit = 1.0, 4, ""
        cb = fmt(b, scale, digits) if b else "-"
        ca = fmt(a, scale, digits) if a else "-"
        ratio = f"{statistics.mean(a) / statistics.mean(b):.3f}" if a and b and statistics.mean(b) else "-"
        lines.append(f"| {k}{unit} | {cb} | {ca} | {ratio} |")
    return "\n".join(lines)


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def resolve(arg):
    """Returns (checkout dir, is_worktree); the revision is only checked here, not checked out."""
    if pathlib.Path(arg).is_dir():
        return pathlib.Path(arg).resolve(), False
    rev = git("rev-parse", "--verify", f"{arg}^{{commit}}")
    if rev.returncode:
        sys.exit(f"'{arg}' is neither a directory nor a git revision")
    return WORK / rev.stdout.strip()[:8], True


def checkout(arg, path):
    git("worktree", "prune")
    if not path.exists():
        done = git("worktree", "add", "--detach", str(path), arg)
        if done.returncode:
            sys.exit(done.stderr)
    if not (path / "external").exists():
        (path / "external").symlink_to(ROOT / "external")


def build(path):
    cmd = ("source ./header.sh && cmake -S . -B build/Release -D CMAKE_BUILD_TYPE=Release && "
           "cmake --build build/Release --parallel --target xpic.out")
    print(f"building {path}", flush=True)
    done = subprocess.run(["bash", "-c", cmd], cwd=path, capture_output=True, text=True)
    if done.returncode:
        sys.exit(f"build of {path} failed:\n" + "\n".join((done.stdout + done.stderr).splitlines()[-30:]))


def describe(path):
    done = git("describe", "--always", "--dirty", cwd=path)
    return f"{path} ({done.stdout.strip()})" if done.returncode == 0 else str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("before", help=f"directory with the source code or git revision to compare with")
    ap.add_argument("after", nargs="?", default=str(ROOT), help=f"directory with the source code or git revision, default is {ROOT}")
    ap.add_argument("--repeats", type=int, default=3, help="number of repetitions, to check mean and stdev")
    ap.add_argument("--sims", default=",".join(CASES), help="list of simulation names to compare")
    ap.add_argument("--keep", action="store_true", help="keep the worktrees created for revisions")
    ap.add_argument("--workdir", default=str(pathlib.Path(tempfile.gettempdir()) / "xpic_compare"), help="directory, where simulation will be ran")
    args = ap.parse_args()

    global WORK
    WORK = pathlib.Path(args.workdir).resolve()
    WORK.mkdir(parents=True, exist_ok=True)

    sims = args.sims.split(",")
    repeats = args.repeats
    sides = {"before": args.before, "after": args.after}
    resolved = {ver: resolve(arg) for ver, arg in sides.items()}
    created = []
    try:
        for ver, (path, is_worktree) in resolved.items():
            if is_worktree:
                checkout(sides[ver], path)
                created.append(path)
            build(path)
        exes = {ver: path / "build/Release/xpic.out" for ver, (path, _) in resolved.items()}
        run(exes, sims, repeats, {ver: describe(path) for ver, (path, _) in resolved.items()})
    finally:
        if not args.keep:
            for path in set(created):
                git("worktree", "remove", "--force", str(path))
            git("worktree", "prune")


def run(exes, sims, repeats, revisions):
    results = {(s, r, v): [] for s in sims for r in RANKS for v in exes}
    for rep in range(repeats):
        for sim in sims:
            for ranks in RANKS:
                for ver, exe in exes.items():
                    print(f"repeat {rep} {sim} ranks={ranks} {ver}", flush=True)
                    results[(sim, ranks, ver)].append(run_once(exe, sim, ranks, f"{ver}{ranks}"))

    profiles = {}
    for sim in sims:
        for ver, exe in exes.items():
            print(f"perf record {sim} {ver}", flush=True)
            profiles[(sim, ver)] = perf_profile(exe, sim, f"{ver}1")

    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {f"{s}/{r}/{v}": x for (s, r, v), x in results.items()}, indent=1))
    md = ["# xpic version comparison", "",
          *(f"- {ver}: {rev}" for ver, rev in revisions.items()),
          f"- date: {datetime.date.today()}, threads: 4 in total (1 rank x 4, 2 ranks x 2)",
          f"- command: `{' '.join(sys.argv)}`", ""]
    for sim in sims:
        c = CASES[sim]
        md += [f"### {sim}", "",
               f"`-n {c['n']} -ppc {c['ppc']}` ({c['n'] ** 3 * c['ppc']} particles), {c['steps']} steps, "
               f"dx={c['dx']}, dt={c['dt']}, {repeats} repeats, `mean ± sd`.", ""]
        for ranks in RANKS:
            md += [f"**{ranks} rank(s) x {4 // ranks} threads**", "",
                   table(sim, ranks, results[(sim, ranks, "before")], results[(sim, ranks, "after")]), ""]
        md += ["**perf record, 1 rank (self time, user space)**", "",
               "| before % | before symbol | after % | after symbol |", "| --- | --- | --- | --- |"]
        pb, pa = profiles[(sim, "before")], profiles[(sim, "after")]
        for i in range(max(len(pb), len(pa))):
            b = pb[i] if i < len(pb) else ("", "")
            a = pa[i] if i < len(pa) else ("", "")
            md.append(f"| {b[0]} | `{b[1]}` | {a[0]} | `{a[1]}` |")
        md.append("")
    (OUT / "results.md").write_text("\n".join(md))
    print(f"written {OUT / 'results.md'}")


if __name__ == "__main__":
    main()
