#!/usr/bin/env python3
"""Builds and runs eccapfim_warm_plasma_suite.out repeatedly and writes a markdown table
row per rank count to output/eccapfim_warm_plasma_suite.md next to this script:
`mean ± standard deviation` of ms/step over the repeats.

Usage: python3 -I tests/benchmark/eccapfim_warm_plasma/run.py <label> [repeats] [steps]

Repeats are the outer loop, so slow drifts of the machine load are spread
evenly over all rank counts. The total number of threads is kept at 4: 1 rank x 4, 2 ranks x 2, 4 ranks x 1.
"""

import os
import pathlib
import re
import statistics
import subprocess
import sys

FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[3]

TARGET = "eccapfim_warm_plasma_suite.out"
EXE = ROOT / "build" / "Release" / "tests" / "benchmark" / TARGET

PHASES = [
    "prepare",
    "interpolate",
    "cn_push",
    "decompose",
    "other",
    "form_iteration",
    "field solve",
    "update_cells",
    "total",
]

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

RANKS = [1, 2, 4]

OUTPUT = FILE.parent / "output" / "eccapfim_warm_plasma_suite.md"


def build():
    cmd = ("source ./header.sh && cmake -S . -B build/Release -D CMAKE_BUILD_TYPE=Release &&"
           f"cmake --build build/Release --parallel --target {TARGET}")
    subprocess.run(["bash", "-c", cmd], cwd=ROOT, check=True)


def run(exe, ranks, steps):
    env = dict(os.environ, OMP_NUM_THREADS=str(4 // ranks))
    cmd = ["mpiexec", "-n", str(ranks), "--oversubscribe", "--bind-to", "none",
           exe, "-n", "12", "-ppc", "32", "-steps", str(steps), *SNES]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("prepare "))
    return [float(x) for x in re.findall(r"(-?\d+\.\d+)", line)]


def main():
    build()
    exe = str(EXE)
    label = sys.argv[1]
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    results = {ranks: [] for ranks in RANKS}
    for _ in range(repeats):
        for ranks in RANKS:
            results[ranks].append(run(exe, ranks, steps))

    lines = [
        "# eccapfim warm plasma suite results",
        "",
        f"Setup: `-n 12 -ppc 32`, 4 threads in total, {repeats} runs of {steps} steps each. "
        "Values are ms/step, `mean ± standard deviation`.",
        "",
        "| ranks | " + " | ".join(PHASES) + " |",
        "| ----- | " + " | ".join("-" * len(p) for p in PHASES) + " |",
    ]
    for ranks in RANKS:
        cells = []
        for k in range(len(PHASES)):
            vals = [x[k] for x in results[ranks]]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0
            cells.append(f"{statistics.mean(vals):.1f} ± {sd:.1f}")
        lines.append(f"| {label} {ranks} | " + " | ".join(cells) + " |")

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"written {OUTPUT}")


if __name__ == "__main__":
    main()
