#!/usr/bin/env python3
"""Runs storage_spike.out repeatedly and writes averaged markdown tables to
output/storage_spike.md next to this script.

Usage: run_storage_spike.py <build/Release/tests/benchmark/storage_spike.out> [repeats] [steps] [collide]

With `collide` only the collision variants are run (collisions every step and
every 10th step) and the tables go to output/storage_spike_collide.md.

Repeats are the outer loop, so slow drifts of the machine load are spread
evenly over all variants. Each value is `mean ± standard deviation`, ms/step.
"""

import os
import pathlib
import re
import statistics
import subprocess
import sys

PHASES = ["prepare", "push", "deposit", "migrate", "sort", "collide", "total"]

TABLE_PHASES = ["prepare", "push", "deposit", "migrate", "sort", "update_cells", "total"]
COLLISION_PHASES = ["prepare", "push", "deposit", "update_cells", "collide", "total"]

RANKS = [1, 2, 4, 8]

# (label, extra args, ranks it can run on)
VARIANTS = [
    ("l", ["-backends", "l"], [1, 2, 4, 8]),
    ("v", ["-backends", "v"], [1, 2, 4, 8]),
    ("a", ["-backends", "a"], [1]),
    ("b", ["-backends", "b"], [1]),
    ("c", ["-backends", "c"], [1, 2, 4, 8]),
    ("d", ["-backends", "d"], [1, 2, 4, 8]),
    ("e", ["-backends", "e", "-resort", "0"], [1, 2, 4, 8]),
    ("e, resort 10", ["-backends", "e", "-resort", "10"], [1, 2, 4, 8]),
]

# (label, extra args), run with `-collide <period>` for each of PERIODS
COLLISION_VARIANTS = [
    ("v", ["-backends", "v"]),
    ("c", ["-backends", "c"]),
    ("e + sort", ["-backends", "e", "-collide_index", "0"]),
    ("e + index", ["-backends", "e", "-collide_index", "1"]),
]

PERIODS = [1, 10]

LINE = re.compile(r"^(list|vector|swarm_\w)\s+(prepare.*)$")

OUTPUT = pathlib.Path(__file__).resolve().parent / "output" / "storage_spike.md"
OUTPUT_COLLIDE = OUTPUT.with_name("storage_spike_collide.md")


def run(exe, ranks, args, steps):
    cmd = [
        "mpiexec", "-n", str(ranks), "--oversubscribe", "--bind-to", "none", exe,
            "-n", "32", "-ppc", "16", "-steps", str(steps), "-vth", "0.05", *args
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    for line in out.stdout.splitlines():
        if LINE.match(line):
            values = re.findall(r"(\w+)\s+([\d.]+)", line)
            return {k: float(v) for k, v in values if k in PHASES}
    raise RuntimeError(f"no result for {cmd}:\n{out.stdout}\n{out.stderr}")


def row(label, runs, phases):
    cells = []
    for phase in phases:
        if phase == "update_cells":
            xs = [r["migrate"] + r["sort"] for r in runs]
        else:
            xs = [r[phase] for r in runs]
        sd = statistics.stdev(xs) if len(xs) > 1 else 0
        cells.append(f"{statistics.mean(xs):.1f} ± {sd:.1f}")
    return f"| {label} | " + " | ".join(cells) + " |"


def table(samples, key, label_names, phases):
    header = ["id", *phases]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("-" * len(h) for h in header) + " |",
    ]
    for label in label_names:
        runs = samples.get(key(label))
        if runs:
            lines.append(row(label, runs, phases))
    return lines


def main_collide(exe, repeats, steps):
    samples = {}
    for _ in range(repeats):
        for ranks in RANKS:
            for period in PERIODS:
                for label, args in COLLISION_VARIANTS:
                    samples.setdefault((ranks, period, label), []).append(
                        run(exe, ranks, [*args, "-collide", str(period)], steps))

    lines = [
        "# Storage spike results: collisions",
        "",
        f"Setup: `-n 32 -ppc 16 -vth 0.05`, 2 OpenMP threads per rank, {repeats} runs of {steps} steps each. "
        "Values are ms/step, `mean ± standard deviation`; `update_cells` = `migrate` + `sort`. "
        "`collide` includes the grouping by cell for `e` (the cell order of `e` is stale), "
        "and is amortised over the steps (collisions every `period` steps).",
    ]
    for period in PERIODS:
        for ranks in RANKS:
            lines += ["", f"## collisions every {period} step(s), {ranks} rank{'s' if ranks > 1 else ''}", ""]
            lines += table(samples, lambda label: (ranks, period, label),
                           [v[0] for v in COLLISION_VARIANTS], COLLISION_PHASES)

    OUTPUT_COLLIDE.parent.mkdir(exist_ok=True)
    OUTPUT_COLLIDE.write_text("\n".join(lines) + "\n")
    print(f"written {OUTPUT_COLLIDE}")


def main():
    exe = sys.argv[1]
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    if len(sys.argv) > 4 and sys.argv[4] == "collide":
        return main_collide(exe, repeats, steps)

    samples = {}
    for _ in range(repeats):
        for ranks in RANKS:
            for label, args, allowed in VARIANTS:
                if ranks in allowed:
                    samples.setdefault((ranks, label), []).append(run(exe, ranks, args, steps))

    lines = [
        "# Storage spike results",
        "",
        f"Setup: `-n 32 -ppc 16 -vth 0.05`, 2 OpenMP threads per rank, {repeats} runs of {steps} steps each. "
        "Values are ms/step, `mean ± standard deviation`; `update_cells` = `migrate` + `sort`.",
    ]
    for ranks in RANKS:
        lines += ["", f"## {ranks} rank{'s' if ranks > 1 else ''}", ""]
        lines += table(samples, lambda label: (ranks, label), [v[0] for v in VARIANTS], TABLE_PHASES)

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"written {OUTPUT}")


if __name__ == "__main__":
    main()
