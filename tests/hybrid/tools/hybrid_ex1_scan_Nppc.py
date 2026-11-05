#!/usr/bin/env python3
"""Particles-per-cell scan of hybrid_ex1_{hybrid_kinetic,eccapfim,ecsimcorr}.

Reads the runs tagged `<prefix>*` (see the `-npc`, `-tau`, `-mi`, `-precond`,
`-tag` options of the tests) at the step Omega_e * tau = --tau-hybrid for the
hybrid models and --tau-kinetic for the fully kinetic ones, and plots the wall
time against the number of particles per cell of a sort in log-log scale,
labeled with the wall time above and the field iterations per step (and, with
--logs, the mean Picard iterations of the electron push) below each point.
A failed run is marked by a cross at the level of the last converged run of
the model, a run still in progress is skipped (see hybrid_ex1_scan_dt.py).

Usage: hybrid_ex1_scan_Nppc.py [-p prefix] [--logs dir] [-o output] [output_directory ...]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_ex1_energy import TEST_DIR, LABELSIZE, LEGENDSIZE, TICKSIZE
from hybrid_ex1_scan_dt import STYLES, LABEL, read_run
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator


def read_npc(out):
    with open(os.path.join(out, "config.json")) as f:
        config = json.load(f)
    return next(s for s in config["Particles"] if s["sort_name"] == "electrons")["Np"]


def plot_runs(ax, runs):
    for model, (style, color, label) in STYLES.items():
        group = [r for r in runs if r["model"] == model]
        done = [r for r in group if r["converged"]]
        if not done:
            continue

        # The kinetic models are dashed lines without markers in STYLES.
        marker = "" if style[0] == "o" else "s"
        ax.plot([r["npc"] for r in done], [r["wall"] for r in done], marker + style,
                color=color, lw=2, ms=8,
                label=f"{label}, $\\Omega_e \\tau = {done[0]['tau']:g}$")
        # The wall times of the hybrid and kinetic models are close: the labels
        # of the hybrid ones go to the left of the points, the kinetic ones to the right.
        dx, ha = (-12, "right") if marker == "" else (12, "left")
        for r in done:
            labels = [f"{r['wall']:.0f} s"]
            labels += [f"{r['N_it']:.1f} SNES it"] if not np.isnan(r["N_it"]) else []
            labels += [f"{r['picard']:.1f} Picard it"] if not np.isnan(r["picard"]) else []
            for i, text in enumerate(labels):
                ax.annotate(text, (r["npc"], r["wall"]), textcoords="offset points",
                            xytext=(dx, 8 - 17 * i), ha=ha, va="center", color=color, **LABEL)

        failed = [r for r in group if not r["converged"]]
        if failed:
            last = done[-1]
            ax.plot([last["npc"], failed[0]["npc"]], [last["wall"]] * 2, ":", color=color, lw=1.5)
            ax.plot(failed[0]["npc"], last["wall"], "x", color=color, ms=14, mew=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="*")
    parser.add_argument("-p", "--prefix", default="mi1836",
                        help="tag prefix of the runs, used if no directories are given")
    parser.add_argument("--logs", help="directory with the run logs <output_directory>.log "
                        "to label the mean electron Picard iterations")
    parser.add_argument("--tau-hybrid", type=float, default=10.0)
    parser.add_argument("--tau-kinetic", type=float, default=0.2)
    parser.add_argument("-o", "--output", default=os.path.join(TEST_DIR, "output",
        "hybrid_ex1_scan_Nppc"))
    args = parser.parse_args()

    dirs = args.dirs or sorted(glob.glob(os.path.join(TEST_DIR, "output",
        f"hybrid_ex1_*_{args.prefix}*")))
    runs = []
    for d in dirs:
        run = read_run(d, args.logs)
        tau = args.tau_hybrid if run["model"].startswith("hybrid_kinetic") else args.tau_kinetic
        if np.isclose(run["tau"], tau) and not run["running"]:
            run["npc"] = read_npc(d)
            runs.append(run)
    runs.sort(key=lambda r: r["npc"])

    print(f"{'model':<28} {'Nppc':>5} {'N_it':>7} {'wall, s':>9} {'Picard':>7}")
    for r in sorted(runs, key=lambda r: list(STYLES).index(r["model"])):
        status = f"{r['N_it']:>7.1f} {r['wall']:>9.1f} {r['picard']:>7.1f}" if r["converged"] else "failed"
        print(f"{r['model']:<28} {r['npc']:>5} {status}")

    fig, ax = plt.subplots(figsize=(12, 9))
    plot_runs(ax, runs)
    npcs = sorted({r["npc"] for r in runs})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(npcs, [str(n) for n in npcs])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(npcs[0] / 1.3, npcs[-1] * 1.3)
    ax.margins(y=0.15)
    ax.tick_params(labelsize=TICKSIZE)
    ax.grid(alpha=0.3, which="both")
    ax.set_xlabel("particles per cell", fontsize=LABELSIZE)
    ax.set_ylabel("wall time, s", fontsize=LABELSIZE)
    ax.legend(loc="best", fontsize=LEGENDSIZE - 6, handlelength=2.5, borderpad=0.3,
              labelspacing=0.2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(f"{args.output}.{extension}", dpi=150, bbox_inches="tight", pad_inches=0.12)
    print(f"saved figure to {args.output}.png")


if __name__ == "__main__":
    main()
