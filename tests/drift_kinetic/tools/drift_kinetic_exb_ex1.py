#!/usr/bin/env python3

import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.append(os.path.join(TEST_DIR, "../../tools"))

from lib.plot_utils import figure, subplot

OUT = os.path.join(TEST_DIR, "output", "drift_kinetic_exb_ex1")
DK_DIR = os.path.join(OUT, "drift_kinetic")
KIN_DIR = os.path.join(OUT, "eccapfim")

COLOR_KIN = "black"
COLOR_DK = "red"
COLOR_TH = "black"

LABELSIZE = 17
TICKSIZE = 13
LEGENDSIZE = 12
PANELSIZE = 17
TITLESIZE = 16
OFFSETSIZE = 15


def load_config(path):
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


def preset_value(config, command):
    for preset in config["Presets"]:
        if preset.get("command") == command:
            return np.array(preset["setter"]["value"], dtype=float)
    raise KeyError(f"Preset \"{command}\" is not found in the config")


def load_trace(path):
    return np.loadtxt(os.path.join(path, "temporal", "particle_trace.txt"), skiprows=1)


def mark_ends(ax, xs, ys, color):
    # Hollow dot at the start of the trajectory, solid dot at the end.
    ax.plot(xs[0], ys[0], "o", mfc="white", mec=color, mew=1.5, ms=7, zorder=5)
    ax.plot(xs[-1], ys[-1], "o", color=color, ms=7, zorder=5)


def style_axes(ax, panel):
    ax.minorticks_on()
    # Ticks on every side, pointing inward; x labels only on the bottom.
    ax.tick_params(axis="both", which="both", direction="in",
        top=True, bottom=True, left=True, right=True, labelsize=TICKSIZE)
    ax.tick_params(axis="x", which="both", labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", which="both", labelleft=True, labelright=False)
    ax.text(0.88, 0.98, panel, transform=ax.transAxes,
        ha="right", va="top", fontsize=PANELSIZE,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6,
            boxstyle="round,pad=0.2"))


def main():
    cfg_dk = load_config(DK_DIR)
    cfg_kin = load_config(KIN_DIR)

    E = preset_value(cfg_dk, "SetElectricField")
    B = preset_value(cfg_dk, "SetMagneticField")
    q = float(cfg_dk["Particles"][0]["q"])
    m = float(cfg_dk["Particles"][0]["m"])

    # Steps are stored in 1/w_pe; express them in units of 1/Omega_e using the
    # gyrofrequency at the particles' initial point (uniform field => |B|).
    omega_e = abs(q) * np.linalg.norm(B) / m
    tau_kin = float(cfg_kin["Geometry"]["dt"]) * omega_e
    tau_dk = float(cfg_dk["Geometry"]["dt"]) * omega_e
    label_kin = rf"kinetic: $\tau = {tau_kin:g}/\Omega_e$"
    label_dk = rf"drift-kinetic: $\tau = {tau_dk:g}/\Omega_e$"

    v_drift = np.cross(E, B) / np.dot(B, B)
    v_drift_x = v_drift[0]

    dk = load_trace(DK_DIR)
    kin = load_trace(KIN_DIR)

    t_dk, x_dk, y_dk, p_perp_dk = dk[:, 0], dk[:, 1], dk[:, 2], dk[:, 5]
    t_kin, x_kin, y_kin = kin[:, 0], kin[:, 1], kin[:, 2]

    # Guiding-center x and the Larmor radius give the gyration envelope used in
    # the deviation panel (b).
    x0 = x_dk[0]
    rho = p_perp_dk[0] / (abs(q) * np.linalg.norm(B))

    fig, gs = figure(ncols=3, nrows=1, figsize=(14, 4.5))

    # (a) Trajectory in the (x, y) plane.
    ax = subplot(fig, gs, 0, 0)
    ax.plot(x_kin, y_kin, color=COLOR_KIN, lw=1.0, label=label_kin)
    ax.plot(x_dk, y_dk, color=COLOR_DK, lw=2.0, label=label_dk)
    mark_ends(ax, x_kin, y_kin, COLOR_KIN)
    mark_ends(ax, x_dk, y_dk, COLOR_DK)
    ax.set_xlabel(r"$x,\ c/\omega_{pe}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$y,\ c/\omega_{pe}$", fontsize=LABELSIZE)
    ax.set_ylim(0.5, 1.2)
    ax.text(0.97, 0.86, r"$\mathbf{E}\times\mathbf{B}$ drift",
        transform=ax.transAxes, ha="right", va="top", fontsize=TITLESIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(a)")

    # (b) Drift-subtracted x(t) - V_E t; the gyration stays within +-rho_e.
    ax = subplot(fig, gs, 1, 0)
    x_kin_dev = x_kin - v_drift_x * t_kin
    x_dk_dev = x_dk - v_drift_x * t_dk
    ax.plot(t_kin, x_kin_dev, color=COLOR_KIN, lw=1.0, label=label_kin)
    ax.plot(t_dk, x_dk_dev, color=COLOR_DK, lw=2.0, label=label_dk)
    ax.axhline(x0 + rho, color="gray", ls=":", lw=1.0, label=r"theory $\pm\ \rho_e$")
    ax.axhline(x0 - rho, color="gray", ls=":", lw=1.0)
    mark_ends(ax, t_kin, x_kin_dev, COLOR_KIN)
    mark_ends(ax, t_dk, x_dk_dev, COLOR_DK)
    ax.set_ylim(x0 - 1.5 * rho, x0 + 2.5 * rho)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$x(t) - V_E t,\ c/\omega_{pe}$", fontsize=LABELSIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(b)")

    # (c) Relative drift-velocity error of the drift-kinetic run.
    ax = subplot(fig, gs, 2, 0)
    v_dk = np.gradient(x_dk, t_dk)
    rel_err = (v_dk - v_drift_x) / v_drift_x
    ax.plot(t_dk, rel_err, color=COLOR_DK, lw=1.5, label=label_dk)
    ax.axhline(0.0, color=COLOR_TH, ls="--", lw=1.0)
    ymax = np.max(np.abs(rel_err))
    if ymax > 0:
        ax.set_ylim(-1.1 * ymax, 1.1 * ymax)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$(V(t) - V_E) / V_E$", fontsize=LABELSIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(OFFSETSIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(c)")

    print(f"v_drift (theory)        = {v_drift_x:.6e}")
    print(f"v_drift (drift-kinetic) = {np.mean(v_dk):.6e}")
    print(f"max relative error      = {np.max(np.abs(rel_err)):.3e}")

    fig.tight_layout(pad=1.2, w_pad=2.5)
    image = os.path.join(OUT, "drift_kinetic_exb_ex1.png")
    fig.savefig(image, dpi=150)
    print(f"Saved figure to {image}")


if __name__ == "__main__":
    main()
