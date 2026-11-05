#!/usr/bin/env python3
"""Cost of the ex1 runs in a (tau, dx, m_i/m_e) scan.

For every output directory of hybrid_ex1_{hybrid_kinetic,eccapfim}
(see the `-tau`, `-dx`, `-mi`, `-tag` options of the tests) reads the field
iterations per step N_it and the wall time, and plots (a) N_it against
omega_pe * tau, colored by the Courant number c * tau / dx, (b) the wall time
per unit of physical time against tau, (c) the speedup of the hybrid model
S = C_kinetic / min C_hybrid over tau <= 0.1 / Omega_i, with the cost
C = wall time / physical time.

Usage: hybrid_ex1_speedup.py output_directory ...
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_ex1_energy import (load_energies, load_timing, preset_B,
    style_axes, LABELSIZE, LEGENDSIZE, TITLESIZE)
from lib.plot_utils import figure, subplot
import matplotlib.pyplot as plt

MARKERS = {"hybrid_kinetic": "o", "eccapfim": "s", "drift_kinetic": "^"}
COLORS = {25: "tab:purple", 100: "tab:blue", 400: "tab:green",
          1836: "tab:red"}


def read_run(out):
    with open(os.path.join(out, "config.json")) as f:
        config = json.load(f)

    geometry = config["Geometry"]
    sorts = {s["sort_name"]: s for s in config["Particles"]}
    electrons, ions = sorts["electrons"], sorts["ions"]
    omega_e = abs(electrons["q"]) * np.linalg.norm(preset_B(config)) \
        / electrons["m"]
    omega_pe = np.sqrt(electrons["n"] * electrons["q"]**2 / electrons["m"])

    iterations = load_energies(out, config)[8]
    timing = load_timing(out)
    tau = float(geometry["dt"])
    return {
        "dir": os.path.basename(os.path.normpath(out)),
        "simulation": config["Simulation"],
        "mi": round(ions["m"] / electrons["m"]),
        "tau": tau * omega_e,
        "wpe_tau": tau * omega_pe,
        "dx": float(geometry["dx"]),
        "courant": tau / float(geometry["dx"]),
        "N_it": np.mean(iterations[1:]),
        "per_step": timing["PerStep"],
        "per_iteration": timing["PerStep"] / np.mean(iterations[1:]),
        "cost": timing["PerStep"] / tau,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", help="output directories")
    parser.add_argument("-o", "--output", default="hybrid_ex1_speedup",
                        help="figure path without extension")
    args = parser.parse_args()
    runs = []
    for d in args.dirs:
        if os.path.exists(os.path.join(d, "temporal", "timing.txt")):
            runs.append(read_run(d))
        else:
            print(f"skipped {d}: no timing.txt, the run failed or is running")
    runs.sort(key=lambda r: (r["simulation"], r["mi"], r["tau"], r["dx"]))

    print(f"{'run':<44} {'mi':>5} {'tau':>6} {'dx':>6} {'C':>5}"
          f" {'N_it':>7} {'s/iter':>9} {'s/step':>9} {'s/t_pe':>9}")
    for r in runs:
        print(f"{r['dir']:<44} {r['mi']:>5} {r['tau']:>6g} {r['dx']:>6g}"
              f" {r['courant']:>5.2g} {r['N_it']:>7.1f}"
              f" {r['per_iteration']:>9.3e} {r['per_step']:>9.3e}"
              f" {r['cost']:>9.3e}")

    fig, gs = figure(ncols=3, nrows=1, figsize=(24, 7.5))

    # (a) Field iterations per step.
    ax = subplot(fig, gs, 0, 0)
    for simulation, marker in MARKERS.items():
        group = [r for r in runs if r["simulation"] == simulation]
        if not group:
            continue
        points = ax.scatter([r["wpe_tau"] for r in group],
            [r["N_it"] for r in group], c=[r["courant"] for r in group],
            cmap="viridis", norm=plt.matplotlib.colors.LogNorm(0.1, 10),
            marker=marker, s=90, edgecolors="black", label=simulation)
    fig.colorbar(points, ax=ax, label=r"$c\tau/\Delta x$")

    unit = [r for r in runs
            if r["simulation"] == "hybrid_kinetic" and r["courant"] == 1]
    if len(unit) > 1:
        x = np.log([r["wpe_tau"] for r in unit])
        y = np.log([r["N_it"] for r in unit])
        slope, shift = np.polyfit(x, y, 1)
        grid = np.exp(np.linspace(x.min(), x.max(), 50))
        ax.plot(grid, np.exp(shift) * grid**slope, "k--", lw=1.2,
                label=rf"$N_{{it}} \propto (\omega_{{pe}}\tau)^{{{slope:.2f}}}$")
        print(f"hybrid, C = 1: N_it ~ (omega_pe tau)^{slope:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\omega_{pe}\tau$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$N_{it}$ per step", fontsize=LABELSIZE)
    ax.legend(loc="best", fontsize=LEGENDSIZE)
    style_axes(ax, "(a)")

    # (b) Wall time per unit of physical time.
    ax = subplot(fig, gs, 1, 0)
    for r in runs:
        ax.scatter(r["tau"], r["cost"], marker=MARKERS[r["simulation"]],
            color=COLORS.get(r["mi"], "black"), s=90, edgecolors="black")
    for mi in sorted({r["mi"] for r in runs}):
        ax.axvline(0.1 * mi, color=COLORS.get(mi, "black"), ls="--", lw=1.2,
                   label=rf"$m_i/m_e = {mi}$, $\tau = 0.1/\Omega_i$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau,\ \Omega_e^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"wall time / $t$, s$\cdot\omega_{pe}$",
                  fontsize=LABELSIZE)
    ax.legend(loc="best", fontsize=LEGENDSIZE)
    style_axes(ax, "(b)")

    # (c) Speedup of the hybrid model at its best admissible step.
    ax = subplot(fig, gs, 2, 0)
    speedup = []
    for mi in sorted({r["mi"] for r in runs}):
        kinetic = [r["cost"] for r in runs
                   if r["simulation"] == "eccapfim" and r["mi"] == mi]
        hybrid = [r for r in runs if r["simulation"] == "hybrid_kinetic"
                  and r["mi"] == mi and r["tau"] <= 0.1 * mi * (1 + 1e-9)]
        if kinetic and hybrid:
            best = min(hybrid, key=lambda r: r["cost"])
            speedup.append((mi, min(kinetic) / best["cost"]))
            print(f"m_i/m_e = {mi}: S = {speedup[-1][1]:.2f}"
                  f" (hybrid tau = {best['tau']:g}, dx = {best['dx']:g})")
    if speedup:
        ax.plot(*zip(*speedup), "ko-", lw=1.5, ms=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$m_i/m_e$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$S = C_{kinetic} / C_{hybrid}$", fontsize=LABELSIZE)
    style_axes(ax, "(c)")

    fig.suptitle("Cost of the kinetic and hybrid models in a uniform plasma",
                 fontsize=TITLESIZE)
    fig.tight_layout(w_pad=2.0)
    for extension in ("png", "pdf"):
        fig.savefig(f"{args.output}.{extension}", dpi=150,
                    bbox_inches="tight", pad_inches=0.12)
    print(f"saved figure to {args.output}.png")


if __name__ == "__main__":
    main()
