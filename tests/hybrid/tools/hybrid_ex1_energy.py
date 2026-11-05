#!/usr/bin/env python3
"""Energy and charge conservation in a uniform Maxwellian plasma.

Plots and checks the output of hybrid_ex1_{drift_kinetic,hybrid_kinetic,eccapfim,ecsimcorr}:
(a) change of the total energy relative to the initial particle energy W_K0
(the field energy is dominated by the uniform B0, so W0 would hide the error),
(b) residual of the discrete continuity equation, (c) changes of the kinetic,
electric and magnetic energies, (d) for the hybrid model, the residuals of the
per-sort energy balance dK_s - tau * E * J_s. The wall time of the run is read
from temporal/timing.txt, the number of field iterations per step from the
diagnostics.

Usage: hybrid_ex1_energy.py [--no-check] [test or output directory ...]
tests: hybrid_ex1_drift_kinetic, hybrid_ex1_hybrid_kinetic, hybrid_ex1_eccapfim,
hybrid_ex1_ecsimcorr
"""

import argparse
import json
import os
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.append(os.path.join(TEST_DIR, "../../tools"))

from lib.plot_utils import figure, subplot
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True

CASES = ("hybrid_ex1_drift_kinetic", "hybrid_ex1_hybrid_kinetic", "hybrid_ex1_eccapfim",
         "hybrid_ex1_ecsimcorr")

TITLES = {
    "hybrid_kinetic": "kinetic ions, drift-kinetic electrons",
    "drift_kinetic": "drift-kinetic ions and electrons",
    "eccapfim": "kinetic ions and electrons",
    "ecsimcorr": "kinetic ions and electrons, semi-implicit",
}

# Limits for max |delta_W| (relative to W_K0) and max R_rho, set by the
# iterative solvers.
LIMITS = {
    "hybrid_kinetic": (1e-10, 1e-12),
    "drift_kinetic": (1e-10, 1e-12),
    "eccapfim": (1e-10, 1e-12),
    "ecsimcorr": (1e-10, 1e-12),
}

COLOR_K = "tab:red"
COLOR_E = "tab:blue"
COLOR_B = "tab:green"

LABELSIZE = 22
TICKSIZE = 20
LEGENDSIZE = 18
PANELSIZE = 22
TITLESIZE = 22


def read_table(path):
    # Titles may contain single spaces ("wEB + wK"), columns are split by two.
    with open(path) as f:
        header = re.split(r"\s{2,}", f.readline().strip())
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    if data.shape[1] != len(header):
        raise SystemExit(f"unexpected table layout: {path}")
    return {name: data[:, i] for i, name in enumerate(header)}


def preset_B(config):
    for preset in config["Presets"]:
        if preset.get("command") == "SetField" and preset.get("field") == "B":
            return np.array(preset["setter"]["value"], dtype=float)
    raise KeyError("SetField preset for \"B\" is not found in the config")


def load_energies(out, config):
    """Returns time steps, the total and particle energies W0, W_K0 at t = 0,
    the per-step changes of the electric, magnetic, kinetic and total
    energies, R_rho, the field iterations per step and the per-sort residuals
    of the energy balance (hybrid model only, None otherwise).

    The absolute energies are printed with 7 digits only, so the balance is
    accumulated from the per-step changes, which keep the full precision.
    """
    temporal = os.path.join(out, "temporal")
    simulation = config["Simulation"]

    if simulation in ("hybrid_kinetic", "drift_kinetic"):
        name = {"hybrid_kinetic": "hybrid_diagnostic.txt",
                "drift_kinetic": "dk_diagnostic.txt"}[simulation]
        table = read_table(os.path.join(temporal, name))
        W0 = table["wEB + wK"][0]
        sorts = None
        if simulation == "hybrid_kinetic":
            sorts = {s: table[f"dK-tEJ_{s}"] for s in ("ions", "electrons")}
        return (table["Time"], W0, table["wK"][0], table["dE"], table["dB"],
                table["dK"], table["dE+dB+dK"], table["N2dQ_tot"],
                table["AvgFieldIt"], sorts)

    # `Energy` of eccapfim and ecsimcorr stores |B - B0|^2 / 2, the uniform
    # background is restored here to normalize delta_W by the same total energy.
    energy = read_table(os.path.join(temporal, "energy.txt"))
    balance = read_table(os.path.join(temporal, "energy_conservation.txt"))
    charge = read_table(os.path.join(temporal, "charge_conservation.txt"))

    geometry = config["Geometry"]
    cells = np.prod([round(geometry[a] / geometry["d" + a]) for a in "xyz"])
    wB0 = 0.5 * np.dot(preset_B(config), preset_B(config)) * cells

    wK0 = sum(column[0] for name, column in energy.items()
              if name.startswith("wK_"))
    dK = sum(column for name, column in balance.items()
             if name.startswith("dK_"))
    W0 = energy["wE"][0] + energy["wB"][0] + wB0 + wK0
    # The table ends with a variable-length convergence history after "|".
    # ecsimcorr solves linear field equations and has no such table.
    history = os.path.join(temporal, "convergence_history.txt")
    iterations = np.loadtxt(history, skiprows=1, usecols=1, ndmin=1) \
        if os.path.exists(history) else np.full(len(balance["Time"]), np.nan)
    return (balance["Time"], W0, wK0, balance["dE"], balance["dB"], dK,
            balance["dE+dB+dK"], charge["N2dQ_tot"], iterations, None)


def load_timing(out):
    with open(os.path.join(out, "temporal", "timing.txt")) as f:
        header = f.readline().split()
        values = f.readline().split()
    return dict(zip(header, map(float, values)))


def running_average(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def check_error(name, error, tolerance):
    print(f"{name:<28} = {error:.3e} (limit {tolerance:.1e})")
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(
            f"{name} is {error:.3e}, expected at most {tolerance:.1e}")


def style_axes(ax, panel):
    ax.minorticks_on()
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(axis="both", which="both", direction="in",
        top=True, bottom=True, left=True, right=True, labelsize=TICKSIZE)
    ax.yaxis.get_offset_text().set_fontsize(TICKSIZE)
    ax.text(0.02, 0.95, panel, transform=ax.transAxes,
        ha="left", va="top", fontsize=PANELSIZE, zorder=20,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6,
            boxstyle="round,pad=0.2"))


def process(case, check):
    out = case if os.path.isdir(case) else \
        os.path.join(TEST_DIR, "output", case)
    test = os.path.basename(os.path.normpath(out))
    with open(os.path.join(out, "config.json")) as f:
        config = json.load(f)

    simulation = config["Simulation"]
    dt = float(config["Geometry"]["dt"])
    electrons = next(s for s in config["Particles"]
                     if s["sort_name"] == "electrons")
    omega_e = abs(electrons["q"]) * np.linalg.norm(preset_B(config)) \
        / electrons["m"]

    steps, W0, WK0, dE, dB, dK, dW, R_rho, iterations, sorts = \
        load_energies(out, config)
    timing = load_timing(out)

    t = steps * dt
    delta_W = np.cumsum(dW) / WK0
    N_it = np.mean(iterations[1:])

    print(f"{test}: {TITLES[simulation]}")
    print(f"  tau = {dt * omega_e:g} / Omega_e, dx = {config['Geometry']['dx']:g},"
          f" steps = {int(timing['Steps'])}")
    print(f"  wall time = {timing['Total']:.3f} s"
          f" (calculate {timing['Calculate']:.3f} s,"
          f" {timing['PerStep']:.3e} s per step)")
    print(f"  field iterations per step = {N_it:.1f},"
          f" {timing['PerStep'] / N_it:.3e} s per iteration,"
          f" {timing['Calculate'] / t[-1]:.3e} s per 1 / omega_pe")
    print(f"  max |W - W0| / W0 = {np.max(np.abs(delta_W)) * WK0 / W0:.3e}")

    ncols = 3 if sorts is None else 4
    fig, gs = figure(ncols=ncols, nrows=1, figsize=(8 * ncols, 7.5))
    window = max(1, len(t) // 20)

    # (a) Relative change of the total energy.
    ax = subplot(fig, gs, 0, 0)
    mean_W = np.mean(delta_W[1:]) if len(delta_W) > 1 else 0.0
    ax.plot(t, delta_W, color="black", lw=1.5)
    ax.axhline(0, color="0.4", ls="--", lw=1.2)
    ax.axhline(mean_W, color=COLOR_K, ls="--", lw=1.2,
               label=f"mean = {mean_W:.1e}")
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$\delta_W = (W - W_0) / W_{K0}$", fontsize=LABELSIZE)
    ax.legend(loc="best", fontsize=LEGENDSIZE)
    style_axes(ax, "(a)")

    # (b) Residual of the discrete continuity equation.
    ax = subplot(fig, gs, 1, 0)
    mean_R = np.mean(R_rho[1:]) if len(R_rho) > 1 else 0.0
    ax.semilogy(t[1:], R_rho[1:], color="black", lw=1.5)
    ax.axhline(mean_R, color=COLOR_K, ls="--", lw=1.2,
               label=f"mean = {mean_R:.1e}")
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$\mathcal{R}_\rho$", fontsize=LABELSIZE)
    ax.legend(loc="best", fontsize=LEGENDSIZE)
    style_axes(ax, "(b)")

    # (c) Energy exchange between particles and fields.
    ax = subplot(fig, gs, 2, 0)
    for values, color, label in (
            (np.cumsum(dK), COLOR_K, r"$\Delta W_K$"),
            (np.cumsum(dE), COLOR_E, r"$\Delta W_E$"),
            (np.cumsum(dB), COLOR_B, r"$\Delta W_B$")):
        ax.plot(t, values, color=color, lw=0.8, alpha=0.3)
        ax.plot(t, running_average(values, window), color=color, lw=2.5,
                label=label)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$\Delta W$", fontsize=LABELSIZE)
    ax.legend(loc="best", fontsize=LEGENDSIZE)
    style_axes(ax, "(c)")

    # (d) Energy balance of each sort: dK_s = tau * E * J_s.
    if sorts is not None:
        ax = subplot(fig, gs, 3, 0)
        for (name, values), color in zip(sorts.items(), (COLOR_K, COLOR_E)):
            ax.semilogy(t[1:], np.abs(values[1:]) / WK0, color=color, lw=1.0,
                        label=name)
        ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
        ax.set_ylabel(r"$|\Delta K_s - \tau\,\mathbf{E}\cdot\mathbf{J}_s|"
                      r" / W_{K0}$", fontsize=LABELSIZE)
        ax.legend(loc="best", fontsize=LEGENDSIZE)
        style_axes(ax, "(d)")

    fig.suptitle(
        f"{TITLES[simulation]}: $\\tau = {dt * omega_e:g}/\\Omega_e$, "
        f"{int(timing['Steps'])} steps, wall time {timing['Total']:.1f} s "
        f"({timing['PerStep']:.2e} s per step)", fontsize=TITLESIZE)
    fig.tight_layout(w_pad=2.0)

    for extension in ("png", "pdf"):
        image = os.path.join(out, f"{test}.{extension}")
        fig.savefig(image, dpi=150, bbox_inches="tight", pad_inches=0.12)
        print(f"  saved figure to {image}")
    plt.close(fig)

    if check:
        W_limit, R_limit = LIMITS[simulation]
        check_error("max |delta_W|", np.max(np.abs(delta_W)), W_limit)
        check_error("max R_rho", np.max(R_rho), R_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="*", default=CASES,
                        help="test name or output directory")
    parser.add_argument("--no-check", action="store_true",
                        help="skip the limits, e.g. for a tolerance scan")
    args = parser.parse_args()
    for case in args.cases:
        if case not in CASES and not os.path.isdir(case):
            parser.error(f"unknown case or directory: {case}")
        process(case, not args.no_check)


if __name__ == "__main__":
    main()
