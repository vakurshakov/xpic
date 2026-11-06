#!/usr/bin/env python3
"""SNES tolerance scan of hybrid_ex1_{hybrid_kinetic,eccapfim}.

Reads the runs tagged `s3_tol<tolerance>` and `s3mm_tol<tolerance>` (with
`"FieldPreconditioner": "mass_matrix"`), `-snes_atol` = `-snes_rtol`, and
plots (a) the field iterations per step and (b) the wall time against the
tolerance for all models, (c) the energy error |W - W0| / W_K0 of the hybrid
model for all tolerances, dotted with the preconditioner.

Usage: hybrid_ex1_tolerance_comparision.py [output_directory ...]
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_ex1_energy import (TEST_DIR, TITLES, load_energies,
    load_timing, style_axes, LABELSIZE, LEGENDSIZE, TITLESIZE)
from lib.plot_utils import figure, subplot
import matplotlib.pyplot as plt

STYLES = {
    "hybrid_kinetic": ("o-", "tab:red"),
    "hybrid_kinetic+mass_matrix": ("o:", "tab:green"),
    "eccapfim": ("s-", "tab:blue"),
    "ecsimcorr": ("D-", "tab:orange"),
}
LINESTYLES = {"none": "-", "mass_matrix": ":"}


def read_run(out):
    with open(os.path.join(out, "config.json")) as f:
        config = json.load(f)
    steps, W0, WK0, dE, dB, dK, dW, R_rho, iterations, sorts = \
        load_energies(out, config)
    preconditioner = config.get("FieldPreconditioner", "none")
    return {
        "simulation": config["Simulation"]
            + ("" if preconditioner == "none" else "+" + preconditioner),
        "preconditioner": preconditioner,
        "tolerance": float(re.search(r"tol([0-9.e+-]+)$",
                                     os.path.normpath(out)).group(1)),
        "t": steps * float(config["Geometry"]["dt"]),
        "delta_W": np.abs(np.cumsum(dW)) / WK0,
        "N_it": np.mean(iterations[1:]),
        "wall": load_timing(out)["Calculate"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="*", default=sorted(glob.glob(
        os.path.join(TEST_DIR, "output", "hybrid_ex1_*_s3*_tol*"))))
    parser.add_argument("-o", "--output", default=os.path.join(TEST_DIR,
        "output", "hybrid_ex1_tolerance"))
    args = parser.parse_args()

    runs = []
    for d in args.dirs:
        if os.path.exists(os.path.join(d, "temporal", "timing.txt")):
            runs.append(read_run(d))
        else:
            print(f"skipped {d}: no timing.txt, the run failed or is running")
    runs.sort(key=lambda r: (r["simulation"], -r["tolerance"]))

    print(f"{'simulation':<16} {'tol':>7} {'N_it':>7} {'wall, s':>9}"
          f" {'max dW/WK0':>11}")
    for r in runs:
        print(f"{r['simulation']:<16} {r['tolerance']:>7.0e} {r['N_it']:>7.1f}"
              f" {r['wall']:>9.1f} {np.max(r['delta_W']):>11.3e}")

    fig, gs = figure(ncols=3, nrows=1, figsize=(24, 7.5))
    simulations = [s for s in STYLES if any(r["simulation"] == s for r in runs)]

    for panel, key, ylabel in ((0, "N_it", r"$N_{it}$ per step"),
                              (1, "wall", "wall time, s")):
        ax = subplot(fig, gs, panel, 0)
        for simulation in simulations:
            group = [r for r in runs if r["simulation"] == simulation]
            style, color = STYLES[simulation]
            name, _, preconditioner = simulation.partition("+")
            label = TITLES[name]
            if preconditioner:
                label += f", {preconditioner.replace('_', ' ')}"
            ax.plot([r["tolerance"] for r in group], [r[key] for r in group],
                    style, color=color, lw=2, ms=8, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"SNES tolerance $\varepsilon_a = \varepsilon_r$",
                      fontsize=LABELSIZE)
        ax.set_ylabel(ylabel, fontsize=LABELSIZE)
        ax.legend(loc="best", fontsize=LEGENDSIZE)
        style_axes(ax, f"({'ab'[panel]})")

    ax = subplot(fig, gs, 2, 0)
    hybrid = [r for r in runs if r["simulation"].startswith("hybrid_kinetic")]
    tolerances = sorted({r["tolerance"] for r in hybrid}, reverse=True)
    colors = dict(zip(tolerances,
        plt.cm.viridis(np.linspace(0, 0.9, max(len(tolerances), 1)))))
    for r in hybrid:
        label = rf"$\varepsilon = 10^{{{np.log10(r['tolerance']):.0f}}}$"
        preconditioner = r["preconditioner"]
        ax.semilogy(r["t"][1:], r["delta_W"][1:], color=colors[r["tolerance"]],
                    lw=1.5, ls=LINESTYLES[preconditioner],
                    label=label + ("" if preconditioner == "none"
                                   else f", {preconditioner.replace('_', ' ')}"))
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$|W - W_0| / W_{K0}$", fontsize=LABELSIZE)
    ax.legend(loc="best", fontsize=LEGENDSIZE, ncol=2)
    style_axes(ax, "(c)")

    fig.suptitle("SNES tolerance scan: kinetic vs hybrid", fontsize=TITLESIZE)
    fig.tight_layout(w_pad=2.0)
    for extension in ("png", "pdf"):
        fig.savefig(f"{args.output}.{extension}", dpi=150,
                    bbox_inches="tight", pad_inches=0.12)
    print(f"saved figure to {args.output}.png")


if __name__ == "__main__":
    main()
