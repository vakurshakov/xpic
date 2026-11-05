#!/usr/bin/env python3
"""Time step scan of hybrid_ex1_{hybrid_kinetic,eccapfim,ecsimcorr}.

Reads the runs tagged `<prefix>_dt<tau>` (see the `-tau`, `-mi`, `-precond`,
`-tag` options of the tests) and plots the wall time against the step
Omega_e * tau: the hybrid models as lines, labeled with the wall time above
and the field iterations per step (and, with --logs, the mean Picard iterations
of the electron push) below each point, the fully kinetic models
as dashed horizontal lines. A run without temporal/timing.txt has failed
(the field solver did not converge), it is marked by a cross at the level of
the last converged step of the model. With --logs, the first step at which the
electron push halves its step (Picard has not converged) is marked by a
vertical dashed line. A run without timing.txt whose output
was updated within the last RUNNING_SECONDS is still in progress, it is skipped.
The electron push with Aitken relaxation (DRIFT_KINETIC_AITKEN=1) is a compile
time switch, not in config.json: its runs are marked by `_aa_` in the tag,
e.g. `-tag aa_dt<tau>`, and plotted as a separate model.

Usage: hybrid_ex1_scan_dt.py [-p prefix ...] [--logs dir] [-o output] [output_directory ...]
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_ex1_energy import (TEST_DIR, load_energies, load_timing,
    preset_B, LABELSIZE, LEGENDSIZE, TICKSIZE)
import matplotlib.pyplot as plt

STYLES = {
    "hybrid_kinetic": ("o-", "tab:red", "hybrid"),
    "hybrid_kinetic+mass_matrix": ("o-", "tab:green", "hybrid + mass matrix preconditioner"),
    "hybrid_kinetic+anderson": ("s-", "tab:brown", "hybrid + Anderson push"),
    "hybrid_kinetic+mass_matrix+anderson": ("s-", "tab:purple",
                                            "hybrid + mass matrix preconditioner + Anderson push"),
    "eccapfim": ("--", "tab:blue", "kinetic (eccapfim)"),
    "ecsimcorr": ("--", "tab:orange", "kinetic (ecsimcorr)"),
}
TEXT = LEGENDSIZE - 4
LABEL = dict(fontsize=TEXT, bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
RUNNING_SECONDS = 300
BREAK_RATIO = 3.0
PUSH = re.compile(r'Push "electrons": avg it=([0-9.]+),.* retries=(\d+)')


def push_statistics(log):
    """Mean Picard iterations of the electron push over all field iterations and
    the total number of step halvings (retries) of the adaptive substepping."""
    with open(log) as f:
        values = np.array([(float(m.group(1)), int(m.group(2)))
                           for line in f if (m := PUSH.search(line))]).reshape(-1, 2)
    return (values[:, 0].mean(), values[:, 1].sum()) if len(values) else (np.nan, 0)


def read_run(out, logs):
    with open(os.path.join(out, "config.json")) as f:
        config = json.load(f)
    electrons = next(s for s in config["Particles"] if s["sort_name"] == "electrons")
    omega_e = abs(electrons["q"]) * np.linalg.norm(preset_B(config)) / electrons["m"]
    preconditioner = config.get("FieldPreconditioner", "none")
    anderson = "_aa_" in os.path.basename(os.path.normpath(out))
    run = {
        "model": config["Simulation"] + ("" if preconditioner == "none" else "+" + preconditioner)
                 + ("+anderson" if anderson else ""),
        "tau": float(config["Geometry"]["dt"]) * omega_e,
        "converged": os.path.exists(os.path.join(out, "temporal", "timing.txt")),
    }
    temporal = os.path.join(out, "temporal")
    run["running"] = not run["converged"] and \
        time.time() - max(os.path.getmtime(os.path.join(temporal, f))
                          for f in os.listdir(temporal)) < RUNNING_SECONDS
    if run["converged"]:
        iterations = load_energies(out, config)[8]
        run["wall"] = load_timing(out)["Calculate"]
        run["N_it"] = np.mean(iterations[1:])
        log = os.path.join(logs, os.path.basename(os.path.normpath(out)) + ".log") if logs else ""
        run["picard"], run["retries"] = push_statistics(log) if os.path.exists(log) \
            else (np.nan, 0)
    return run


def plot_runs(ax, runs):
    taus = [r["tau"] for r in runs]
    for model, (style, color, label) in STYLES.items():
        group = [r for r in runs if r["model"] == model]
        done = [r for r in group if r["converged"]]
        if not done:
            continue

        if model in ("eccapfim", "ecsimcorr"):
            r = done[0]
            ax.axhline(r["wall"], color=color, ls=style, lw=2, label=f"{label}, $\\tau = {r['tau']:g}$")
            ax.annotate(f"{r['wall']:.0f} s", (max(taus), r["wall"]), textcoords="offset points",
                        xytext=(0, 6), ha="right", color=color, **LABEL)
            continue

        ax.plot([r["tau"] for r in done], [r["wall"] for r in done], style,
                color=color, lw=2, ms=8, label=label)
        for r in done:
            ax.annotate(f"{r['wall']:.0f} s", (r["tau"], r["wall"]), textcoords="offset points",
                        xytext=(0, 10), ha="center", color=color, **LABEL)
            ax.annotate(f"{r['N_it']:.1f} SNES it", (r["tau"], r["wall"]), textcoords="offset points",
                        xytext=(0, -12), ha="center", va="top", color=color, **LABEL)
            if not np.isnan(r["picard"]):
                ax.annotate(f"{r['picard']:.1f} Picard it", (r["tau"], r["wall"]),
                            textcoords="offset points", xytext=(0, -30), ha="center", va="top",
                            color=color, **LABEL)

        substeps = [r for r in done if r["retries"] > 0]
        if substeps:
            ax.axvline(substeps[0]["tau"], color=color, ls="--", lw=1.5)

        failed = [r for r in group if not r["converged"]]
        if failed:
            last = done[-1]
            ax.plot([last["tau"], failed[0]["tau"]], [last["wall"]] * 2, ":", color=color, lw=1.5)
            ax.plot(failed[0]["tau"], last["wall"], "x", color=color, ms=14, mew=3)


def label_substepping(ax, runs):
    labels = 0
    for model, (_, color, _) in STYLES.items():
        substeps = [r for r in runs if r["model"] == model and r["converged"] and r["retries"] > 0]
        if substeps:
            tau = substeps[0]["tau"]
            ax.annotate(f"substepping from $\\Omega_e \\tau = {tau:g}$", (tau, 0),
                        xycoords=("data", "axes fraction"), textcoords="offset points",
                        xytext=(5, 8 + 22 * labels), ha="left", va="bottom", color=color, **LABEL)
            labels += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="*")
    parser.add_argument("-p", "--prefix", nargs="+", default=["mi1836"],
                        help="tag prefixes of the runs, used if no directories are given")
    parser.add_argument("--logs", help="directory with the run logs <output_directory>.log "
                        "to label the mean electron Picard iterations")
    parser.add_argument("--tau-max", type=float, default=300.0,
                        help="larger steps are not plotted: at Omega_e tau = 400 the fastest "
                             "electrons cross ~2 cells per step and the 25 steps of the run are "
                             "too few for the field noise to saturate")
    parser.add_argument("-o", "--output", default=os.path.join(TEST_DIR, "output",
        "hybrid_ex1_scan_dt"))
    args = parser.parse_args()

    dirs = args.dirs or sorted({d for prefix in args.prefix for d in glob.glob(
        os.path.join(TEST_DIR, "output", f"hybrid_ex1_*_{prefix}*_dt*"))})
    runs = sorted((r for r in (read_run(d, args.logs) for d in dirs) if r["tau"] <= args.tau_max and not r["running"]),
                  key=lambda r: r["tau"])

    print(f"{'model':<28} {'tau':>6} {'N_it':>7} {'wall, s':>9} {'Picard':>7}")
    for r in runs:
        status = f"{r['N_it']:>7.1f} {r['wall']:>9.1f} {r['picard']:>7.1f}" if r["converged"] else "failed"
        print(f"{r['model']:<28} {r['tau']:>6g} {status}")

    # The wall times form clusters (hybrid + preconditioner vs the rest) far
    # apart: the y axis is broken at the largest gap if it exceeds BREAK_RATIO.
    walls = np.sort([r["wall"] for r in runs if r["converged"]])
    ranges = [walls]
    if len(walls) > 1:
        gaps = walls[1:] / walls[:-1]
        k = np.argmax(gaps)
        if gaps[k] > BREAK_RATIO:
            ranges = [walls[k + 1:], walls[:k + 1]]

    fig, axes = plt.subplots(len(ranges), 1, sharex=True, figsize=(18, 4.5 * len(ranges) + 1.5),
                             squeeze=False)
    axes = axes[:, 0]
    for ax, values in zip(axes, ranges):
        plot_runs(ax, runs)
        span = max(values[-1] - values[0], 0.3 * values[-1])
        ax.set_ylim(values[0] - 0.45 * span, values[-1] + 0.2 * span)
        ax.tick_params(labelsize=TICKSIZE)
        ax.grid(alpha=0.3)

    # The substepping labels go to the bottom of the lowest panel only.
    label_substepping(axes[-1], runs)
    for upper, lower in zip(axes[:-1], axes[1:]):
        upper.spines.bottom.set_visible(False)
        lower.spines.top.set_visible(False)
        upper.tick_params(bottom=False)
        marks = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=14, linestyle="none",
                     color="k", mec="k", mew=1, clip_on=False)
        upper.plot([0, 1], [0, 0], transform=upper.transAxes, **marks)
        lower.plot([0, 1], [1, 1], transform=lower.transAxes, **marks)

    axes[-1].set_xlabel(r"$\Omega_e \tau$", fontsize=LABELSIZE)
    fig.supylabel("wall time, s", fontsize=LABELSIZE)
    axes[0].legend(loc="best", fontsize=LEGENDSIZE - 6, handlelength=1.5, borderpad=0.3,
                   labelspacing=0.2)
    fig.tight_layout(h_pad=0.3)
    for extension in ("png", "pdf"):
        fig.savefig(f"{args.output}.{extension}", dpi=150, bbox_inches="tight", pad_inches=0.12)
    print(f"saved figure to {args.output}.png")


if __name__ == "__main__":
    main()
