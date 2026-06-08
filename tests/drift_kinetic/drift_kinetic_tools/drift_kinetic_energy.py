#!/usr/bin/env python3
"""Plot energy time series from a drift_kinetic run.

Reads ``temporal/dk_diagnostic.txt`` and produces a 3x2 figure:
- top-left:     delta wE and delta wB (electromagnetic field energies);
- top-right:    delta wK for every particle sort (kinetic energies).
- middle-left:  total-energy conservation residual dE + dB + dK = Delta W;
- middle-right: continuity-equation residual N1dQ_tot.
- bottom-left:  average field iteration count.
- bottom-right: cumulative sum of ``dK-dMB+dt*dEJ``.

Each curve on the left column has its initial value subtracted, so the
y-axis shows the *change* in energy from t=0. That makes the conservation
balance directly readable: sum of top-left + bottom-left curves should stay
near zero in a closed system, which is exactly what the top-right panel
plots from the diagnostic's own column ``dE+dB+dK``. The bottom-right
panel shows ``N1dQ_tot`` (continuity-equation residual summed over sorts)
as-is, without any baseline subtraction.

The Time column is in iteration units; we convert to ``omega_pe * t`` using
``dt`` from the config so the x-axis matches the field/density frames.
"""

from __future__ import annotations

import os
import re
import sys
import argparse

import numpy as np

TOOLS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS_DIR)

from lib.plot_utils import figure, subplot
from lib.plot import bbox, labelsize, ticksize
from lib.constants import const, init_constants

import shutil
import matplotlib.pyplot as plt

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out-subdir", default="energy",
                        help="Subdirectory under <out_dir>/processed for the PNG")
    parser.add_argument("--filename", default="energy.png",
                        help="Output PNG name inside the subdirectory")
    return parser.parse_args()


def load_energy_table(path: str):
    """Return (header_list, data_2d_array).

    dk_diagnostic.txt has columns separated by 2+ spaces; some column names
    contain single spaces (e.g. ``dt * a_EJ``, ``wEB + wK``), so we cannot use
    naive whitespace splitting on the header row.
    """
    with open(path, "r", encoding="utf-8") as fh:
        header_line = fh.readline().rstrip("\n")
    header = re.split(r" {2,}", header_line.strip())
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(header):
        raise RuntimeError(
            f"Header has {len(header)} columns but data has {data.shape[1]} — "
            f"file format may have changed: {path}")
    return header, data


def column(header, data, name):
    if name not in header:
        return None
    return data[:, header.index(name)]


def main():
    args = parse_args()
    init_constants(args.config)

    energy_path = os.path.join(const.in_dir, "temporal", "dk_diagnostic.txt")
    if not os.path.isfile(energy_path):
        print(f"[error] {energy_path} not found")
        return

    header, data = load_energy_table(energy_path)
    if data.shape[0] < 1:
        print(f"[error] {energy_path} has no data rows")
        return

    time_iter = column(header, data, "Time")
    if time_iter is None:
        print(f"[error] no 'Time' column in {energy_path}")
        return
    t_wpe = time_iter * const.dt

    wE = column(header, data, "wE")
    wB = column(header, data, "wB")
    sort_columns = [
        (name[len("wK_"):], data[:, i])
        for i, name in enumerate(header)
        if name.startswith("wK_")
    ]
    if not sort_columns:
        print("[warn] no wK_<sort> columns found; bottom panel will be empty")

    fig, gs = figure(3, 2, figsize=(24, 9))
    ax_top = subplot(fig, gs, 0, 0)
    ax_bot = subplot(fig, gs, 0, 1)
    ax_dW = subplot(fig, gs, 1, 0)
    ax_dQ = subplot(fig, gs, 1, 1)
    ax_avg = subplot(fig, gs, 2, 0)
    ax_cum = subplot(fig, gs, 2, 1)

    def delta(arr):
        return arr - arr[0]

    if wE is not None:
        ax_top.plot(t_wpe, delta(wE), label=r"$\Delta w_E$",
                    color="tab:blue", linewidth=2)
    if wB is not None:
        ax_top.plot(t_wpe, delta(wB), label=r"$\Delta w_B$",
                    color="tab:red", linewidth=2)
    ax_top.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax_top.set_ylabel(r"$\Delta w_{E,B}$", fontsize=labelsize)
    ax_top.tick_params(labelsize=ticksize)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=labelsize, loc="best")
    ax_top.set_title(r"EM field energy",
                     fontsize=labelsize, bbox=bbox)

    cmap = plt.get_cmap("tab10")
    for k, (sort_name, wK) in enumerate(sort_columns):
        ax_bot.plot(t_wpe, delta(wK),
                    label=rf"$\Delta w_K^{{\mathrm{{{sort_name}}}}}$",
                    color=cmap(2 + k), linewidth=2)
    ax_bot.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax_bot.set_xlabel(r"$\omega_{pe}\,t$", fontsize=labelsize)
    ax_bot.set_ylabel(r"$\Delta w_K$", fontsize=labelsize)
    ax_bot.tick_params(labelsize=ticksize)
    ax_bot.grid(True, alpha=0.3)
    if sort_columns:
        ax_bot.legend(fontsize=labelsize, loc="best")
    ax_bot.set_title(r"Kinetic energy",
                     fontsize=labelsize, bbox=bbox)

    # Right column — diagnostic-native conservation residuals. Total energy
    # W(t) = wE + wB + sum_s wK_s is recomputed from the per-component
    # columns and plotted as the relative drift (W - W0) / W0, where
    # W0 = W(t=0). N1dQ_tot is the L1 norm of the continuity residual
    # summed over sorts — plotted as-is.
    W = np.zeros_like(t_wpe)
    have_W = False
    for nm in ("wE", "wB"):
        c = column(header, data, nm)
        if c is not None:
            W = W + c
            have_W = True
    for _, wK in sort_columns:
        W = W + wK
        have_W = True
    if have_W:
        W0 = float(W[0])
        if W0 == 0.0:
            print("[warn] W0 = 0; plotting W - W0 unnormalised")
            ax_dW.plot(t_wpe, W - W0, color="tab:purple", linewidth=2,
                       label=r"$W - W_0$")
            ax_dW.set_ylabel(r"$W - W_0$", fontsize=labelsize)
        else:
            ax_dW.plot(t_wpe, (W - W0) / W0, color="tab:purple",
                       linewidth=2, label=r"$(W - W_0) / W_0$")
            ax_dW.set_ylabel(r"$(W - W_0) / W_0$", fontsize=labelsize)
        ax_dW.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_dW.legend(fontsize=labelsize, loc="best")
    else:
        print("[warn] no wE/wB/wK_* columns found; total-energy panel empty")
        ax_dW.set_ylabel(r"$(W - W_0) / W_0$", fontsize=labelsize)
    ax_dW.tick_params(labelsize=ticksize)
    ax_dW.grid(True, alpha=0.3)
    ax_dW.set_title(r"Total energy conservation",
                    fontsize=labelsize, bbox=bbox)

    dQ_label = r"$\|\partial_t \rho + \nabla\cdot J\|_1$"
    dQ = column(header, data, "N1dQ_tot")
    if dQ is not None:
        ax_dQ.plot(t_wpe, dQ, color="tab:olive", linewidth=2, label=dQ_label)
        ax_dQ.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_dQ.legend(fontsize=labelsize, loc="best")
    else:
        print("[warn] no 'N1dQ_tot' column found; continuity panel empty")
    ax_dQ.set_xlabel(r"$\omega_{pe}\,t$", fontsize=labelsize)
    ax_dQ.set_ylabel(dQ_label, fontsize=labelsize)
    ax_dQ.tick_params(labelsize=ticksize)
    ax_dQ.grid(True, alpha=0.3)
    ax_dQ.set_title(r"Continuity residual",
                    fontsize=labelsize, bbox=bbox)

    avg_field_it = column(header, data, "AvgFieldIt")
    if avg_field_it is not None:
        ax_avg.plot(t_wpe, avg_field_it, color="tab:cyan", linewidth=2,
                    label="AvgFieldIt")
        ax_avg.legend(fontsize=labelsize, loc="best")
    else:
        print("[warn] no 'AvgFieldIt' column found; panel empty")
    ax_avg.set_xlabel(r"$\omega_{pe}\,t$", fontsize=labelsize)
    ax_avg.set_ylabel("AvgFieldIt", fontsize=labelsize)
    ax_avg.tick_params(labelsize=ticksize)
    ax_avg.grid(True, alpha=0.3)
    ax_avg.set_title("Average field iterations",
                     fontsize=labelsize, bbox=bbox)

    balance = column(header, data, "dK-dMB+dt*dEJ")
    balance_label = "cumsum(dK-dMB+dt*dEJ)"
    if balance is not None:
        ax_cum.plot(t_wpe, np.cumsum(balance), color="tab:pink", linewidth=2,
                    label=balance_label)
        ax_cum.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_cum.legend(fontsize=labelsize, loc="best")
    else:
        print("[warn] no dK-dMB+dt*dEJ column found; cumulative panel empty")
    ax_cum.set_xlabel(ax_avg.get_xlabel(), fontsize=labelsize)
    ax_cum.set_ylabel(balance_label, fontsize=labelsize)
    ax_cum.tick_params(labelsize=ticksize)
    ax_cum.grid(True, alpha=0.3)
    ax_cum.set_title("Cumulative particle balance",
                     fontsize=labelsize, bbox=bbox)

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.filename)

    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Energy figure written to {out_path}")


if __name__ == "__main__":
    main()
