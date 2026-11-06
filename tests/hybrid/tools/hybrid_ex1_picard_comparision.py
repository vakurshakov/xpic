#!/usr/bin/env python3
"""Run and plot classic/Aitken Picard at dt=100, t=100000, mi/me=1836.

Build the Release hybrid target first. --run compiles two pusher variants,
links them against the same remaining objects, and runs them sequentially.
Without --run only analyse the saved runs and regenerate the figures.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

import matplotlib

matplotlib.use("Agg")

from hybrid_ex1_energy import load_energies, load_timing
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "build/Release"
RESULTS = ROOT / "tests/hybrid/output/picard_mi1836_dt100_t100000"
FIGURES = ROOT / ".claude/figures"
CASES = (("classic", 0, "tab:red"), ("aitken", 1, "tab:green"))
RUN_PREFIX = "picard_long"
FIGURE_PREFIX = "preconditioner_picard_long"
PUSH = re.compile(r'Push "electrons": avg it=([0-9.]+),.* retries=(\d+)')


def output_directory(name):
    return ROOT / ("tests/hybrid/output/"
                   f"hybrid_ex1_hybrid_kinetic_{RUN_PREFIX}_{name}_mi1836_dt100_t100000")


def run_cases():
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, _, _ in CASES:
        if output_directory(name).exists():
            raise RuntimeError(f"Preserving existing output: {output_directory(name)}")
    database = json.loads((BUILD / "compile_commands.json").read_text())
    entry = next(e for e in database if e["file"].endswith("/drift_kinetic_push.cpp"))
    link = shlex.split((BUILD / "CMakeFiles/xpic.dir/link.txt").read_text())
    cpus = sorted(os.sched_getaffinity(0))[:4]
    manifest = dict(dt=100, time=100000, steps=1000, mi=1836, npc=50, dx=10,
                    field_atol=1e-13, field_rtol=1e-13, field_stol=0,
                    particle_atol=1e-16, particle_rtol=1e-13,
                    mpi_ranks=1, omp_threads=4, cpu_affinity=cpus,
                    preconditioner="mass_matrix", builds=[], runs=[],
                    source_sha256={})
    for path in ("src/algorithms/drift_kinetic_push.cpp",
                 "src/algorithms/drift_kinetic_push.h",
                 "src/impls/drift_kinetic/particles.cpp",
                 "src/impls/hybrid_kinetic/simulation.cpp",
                 "tests/hybrid/hybrid_ex1_comparison.h"):
        manifest["source_sha256"][path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    manifest_path = RESULTS / "manifest.json"
    for name, mode, _ in CASES:
        directory = BUILD / RUN_PREFIX / name
        directory.mkdir(parents=True, exist_ok=True)
        obj = directory / "drift_kinetic_push.cpp.o"
        command = shlex.split(entry["command"])
        command.insert(1, f"-DDRIFT_KINETIC_AITKEN={mode}")
        command[command.index("-o") + 1] = str(obj)
        manifest["builds"].append(command)
        subprocess.run(command, cwd=entry["directory"], check=True)
        command = [str(obj) if arg == entry["output"] else arg for arg in link]
        command = [arg for arg in command if not arg.startswith("-Wl,--dependency-file=")]
        library = directory / "libxpic.so"
        command[command.index("-o") + 1] = str(library)
        manifest["builds"].append(command)
        subprocess.run(command, cwd=BUILD, check=True)
        print(f"Built {name}, mode={mode}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for name, mode, _ in CASES:
        tag = f"{RUN_PREFIX}_{name}_mi1836_dt100_t100000"
        command = ["taskset", "-c", ",".join(map(str, cpus)),
                   str(BUILD / "tests/hybrid/hybrid_ex1.out"),
                   "-mi", "1836", "-npc", "50", "-dx", "10",
                   "-tau", "100", "-time", "100000", "-precond", "mass_matrix",
                   "-snes_atol", "1e-13", "-snes_rtol", "1e-13", "-snes_stol", "0",
                   "-tag", tag]
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="4", OMP_PROC_BIND="spread", OMP_PLACES="threads",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   LD_LIBRARY_PATH=str(BUILD / RUN_PREFIX / name) + ":" +
                                   env.get("LD_LIBRARY_PATH", ""))
        logfile = RESULTS / f"{name}.log"
        run = dict(name=name, mode=mode, command=command, status="running",
                   log=str(logfile.relative_to(ROOT)),
                   output=str(output_directory(name).relative_to(ROOT)))
        manifest["runs"].append(run)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Starting {name}: dt=100, t=100000, 1000 steps", flush=True)
        start = time.monotonic()
        with logfile.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT)
        run.update(status="finished", exit_code=result.returncode,
                   elapsed_seconds=time.monotonic() - start)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Finished {name}: exit={result.returncode}, "
              f"elapsed={run['elapsed_seconds']:.3f} s", flush=True)
        if result.returncode:
            raise RuntimeError(f"Failed {name}; inspect {logfile}")


def analyse():
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = []
    curves = []
    reference = None
    for name, _, _ in CASES:
        directory = output_directory(name)
        config = json.loads((directory / "config.json").read_text())
        if reference is not None:
            for key in ("Geometry", "Particles", "Presets", "FieldPreconditioner"):
                assert config[key] == reference[key]
        reference = config
        steps, _, WK0, _, _, _, dW, charge, iterations, _ = load_energies(str(directory), config)
        times = steps * config["Geometry"]["dt"]
        assert times[-1] == 100000 and len(times) == 1001
        timing = load_timing(str(directory))
        increments = np.cumsum(dW)
        error = (increments - increments[0]) / WK0
        assert np.all(np.isfinite(error))
        log = (RESULTS / f"{name}.log").read_text()
        pushes = [(float(m[1]), int(m[2])) for m in PUSH.finditer(log)]
        failures = [int(value) for value in re.findall(r"leaf_failures=(\d+)", log)]
        final_push_failures = []
        for block in re.split(r"Timestep = [^\n]*\[dt\]", log)[1:]:
            final_pushes = re.findall(r'Push "electrons": [^\n]*leaf failures=(\d+)', block)
            assert final_pushes
            final_push_failures.append(int(final_pushes[-1]))
        assert len(final_push_failures) == 1000
        row = dict(name=name, wall=timing["Calculate"], steps=int(timing["Steps"]),
                   snes=float(np.mean(iterations[1:])),
                   picard=float(np.mean([p[0] for p in pushes])) if pushes else None,
                   retries=sum(p[1] for p in pushes),
                   terminal_failures=sum(failures),
                   final_push_failure_steps=sum(value > 0 for value in final_push_failures),
                   final_push_failure_events=sum(final_push_failures),
                   max_energy_error=float(np.max(np.abs(error))),
                   max_charge_residual=float(np.max(np.abs(charge))))
        metrics.append(row)
        curves.append((times, error))
        print(json.dumps(row), flush=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(9, 5.8))
    for x, (row, (_, _, color)) in enumerate(zip(metrics, CASES)):
        ax.plot(x, row["wall"], "o", ms=10, color=color)
        ax.annotate(f"{row['wall']:.1f} s", (x, row["wall"]), xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom", color=color, fontsize=17)
        ax.annotate(f"{row['snes']:.1f} SNES it", (x, row["wall"]), xytext=(0, -12),
                    textcoords="offset points", ha="center", va="top", color=color, fontsize=17)
    wall_max = max(row["wall"] for row in metrics)
    ax.set(xlim=(-0.65, 1.65), ylim=(0, wall_max * 1.25))
    ax.set_xticks([0, 1], [plot_label(row["name"]) for row in metrics])
    ax.set_ylabel("wall time, s", fontsize=22)
    ax.tick_params(labelsize=16)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, FIGURE_PREFIX + "_speed")

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    for (times, error), (name, _, color) in zip(curves, CASES):
        label = plot_label(name)
        ax.plot(times, error, color=color, lw=1.2, label=label)
    ax.axhline(0, color="0.4", lw=0.7, zorder=0)
    ax.set_xlim(0, 100000)
    ax.margins(y=0.15)
    ax.set_xlabel(r"$\omega_{pe}t$", fontsize=22)
    ax.set_ylabel(r"$(W(t)-W(0))/W_K(0)$", fontsize=22)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(18)
    ax.yaxis.get_offset_text().set_fontsize(18)
    ax.tick_params(labelsize=18)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=15, loc="best", framealpha=1)
    fig.tight_layout()
    save_figure(fig, FIGURE_PREFIX + "_energy")
    np.savetxt(FIGURES / (FIGURE_PREFIX + "_energy.csv"),
               np.column_stack((curves[0][0], curves[0][1], curves[1][1])),
               delimiter=",", header="omega_pe_t," + ",".join(name for name, _, _ in CASES), comments="")


def plot_label(name):
    suffix = {"classic": "", "aitken": " + Aitken"}[name]
    return "hybrid + mm, Picard" + suffix


def save_figure(fig, name):
    for extension in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=180,
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run_cases()
    analyse()
