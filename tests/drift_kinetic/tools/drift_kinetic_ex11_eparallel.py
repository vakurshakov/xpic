#!/usr/bin/env python3

import json
import os
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

OUT = os.path.join(TEST_DIR, "output", "drift_kinetic_ex11_eparallel")
DK_DIR = os.path.join(OUT, "drift_kinetic")
KIN_DIR = os.path.join(OUT, "eccapfim")

COLOR_KIN = "black"
COLOR_DK = "red"
COLOR_TH = "black"

LABELSIZE = 28
TICKSIZE = 27
LEGENDSIZE = 26
PANELSIZE = 26
TITLESIZE = 26
OFFSETSIZE = 27

MOTION_REL_TOL = 1e-8
ENERGY_REL_TOL = 1e-8


def load_config(path):
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


def preset_value(config, field):
    for preset in config["Presets"]:
        if preset.get("command") == "SetField" and preset.get("field") == field:
            return np.array(preset["setter"]["value"], dtype=float)
    raise KeyError(f"SetField preset for \"{field}\" is not found in the config")


def load_trace(path):
    return np.loadtxt(os.path.join(path, "temporal", "particle_trace.txt"), skiprows=1)


def check_error(name, error, tolerance):
    print(f"{name:<28} = {error:.3e} (limit {tolerance:.1e})")
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(
            f"{name} is {error:.3e}, expected at most {tolerance:.1e}")


def mark_ends(ax, xs, ys, color):
    # Hollow dot at the start of the trajectory, solid dot at the end.
    ax.plot(xs[0], ys[0], "o", mfc="white", mec=color, mew=1.5, ms=7, zorder=5)
    ax.plot(xs[-1], ys[-1], "o", color=color, ms=7, zorder=5)


def style_axes(ax, panel):
    ax.minorticks_on()
    ax.set_box_aspect(1)
    ax.grid(True, which="both", alpha=0.25)
    # Ticks on every side, pointing inward; x labels only on the bottom.
    ax.tick_params(axis="both", which="both", direction="in",
        top=True, bottom=True, left=True, right=True, labelsize=TICKSIZE)
    ax.tick_params(axis="x", which="both", labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", which="both", labelleft=True, labelright=False)
    ax.xaxis.get_offset_text().set_fontsize(OFFSETSIZE)
    ax.yaxis.get_offset_text().set_fontsize(OFFSETSIZE)
    ax.text(0.97, 0.97, panel, transform=ax.transAxes,
        ha="right", va="top", fontsize=PANELSIZE, zorder=20,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6,
            boxstyle="round,pad=0.2"))


def main():
    cfg_dk = load_config(DK_DIR)
    cfg_kin = load_config(KIN_DIR)

    E = preset_value(cfg_dk, "E")
    B = preset_value(cfg_dk, "B")
    q = float(cfg_dk["Particles"][0]["q"])
    m = float(cfg_dk["Particles"][0]["m"])
    a = q * E[2] / m

    dk = load_trace(DK_DIR)
    kin = load_trace(KIN_DIR)

    t_dk, z_dk, v_dk = dk[:, 0], dk[:, 3], dk[:, 4] / m
    t_kin, z_kin = kin[:, 0], kin[:, 3]

    # Uniform acceleration along the field line.
    z_th = z_dk[0] + v_dk[0] * t_dk + 0.5 * a * t_dk ** 2
    v_th = v_dk[0] + a * t_dk

    # Energy m v_par^2 / 2 + mu |B| - q E z, the last term is the potential.
    energy = 0.5 * m * v_dk ** 2 + dk[:, 6] * np.linalg.norm(B) - q * E[2] * z_dk
    energy_dk = energy / energy[0] - 1.0

    omega_e = abs(q) * np.linalg.norm(B) / m
    tau_kin = float(cfg_kin["Geometry"]["dt"]) * omega_e
    tau_dk = float(cfg_dk["Geometry"]["dt"]) * omega_e
    label_kin = rf"kinetic: $\tau = {tau_kin:g}/\Omega_e$"
    label_dk = rf"drift-kinetic: $\tau = {tau_dk:g}/\Omega_e$"

    fig, gs = figure(ncols=3, nrows=1, figsize=(25.0, 8.1))

    # (a) Motion along the field line, reversed by the parallel electric field.
    ax = subplot(fig, gs, 0, 0)
    ax.plot(t_kin, z_kin, color=COLOR_KIN, lw=1.0, label=label_kin)
    ax.plot(t_dk, z_dk, "o", color=COLOR_DK, ms=7, label=label_dk)
    mark_ends(ax, t_kin, z_kin, COLOR_KIN)
    z_span = np.ptp(z_dk)
    ax.set_ylim(z_dk.min() - 0.1 * z_span, z_dk.max() + 0.8 * z_span)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$z,\ c/\omega_{pe}$", fontsize=LABELSIZE)
    ax.text(0.97, 0.66, r"$E_\parallel$ acceleration",
        transform=ax.transAxes, ha="right", va="top", fontsize=TITLESIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(a)")

    # (b) Deviation of the kinetic and drift-kinetic runs from the theory.
    ax = subplot(fig, gs, 1, 0)
    z_kin_th = z_kin[0] + kin[0, 6] * t_kin + 0.5 * a * t_kin ** 2
    ax.plot(t_kin, z_kin - z_kin_th, color=COLOR_KIN, lw=1.0, label=label_kin)
    ax.plot(t_dk, z_dk - z_th, "o-", color=COLOR_DK, lw=2.0, ms=7,
        label=label_dk)
    ax.axhline(0.0, color=COLOR_TH, ls="--", lw=1.5)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$z - z_{\mathrm{th}},\ c/\omega_{pe}$", fontsize=LABELSIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(b)")

    # (c) Energy of the drift-kinetic particle, including the work of E.
    ax = subplot(fig, gs, 2, 0)
    ax.axhline(0.0, color="0.25", ls="--", lw=1.5, zorder=1)
    ax.plot(t_dk, energy_dk, "o-", color=COLOR_DK, lw=1.5, ms=5,
        label=label_dk)
    energy_max = max(np.max(np.abs(energy_dk)), np.finfo(float).eps)
    ax.set_ylim(-1.3 * energy_max, 2.0 * energy_max)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$W / W_0 - 1$", fontsize=LABELSIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(c)")

    # Errors are scaled by the excursion along z and the velocity change.
    z_error = np.max(np.abs(z_dk - z_th)) / np.ptp(z_th)
    v_error = np.max(np.abs(v_dk - v_th)) / np.ptp(v_th)
    work = q * E[2] * (z_dk - z_dk[0])
    kinetic_change = 0.5 * m * (v_dk ** 2 - v_dk[0] ** 2)
    energy_error = np.max(np.abs(kinetic_change - work)) / np.max(np.abs(work))

    check_error("z relative error", z_error, MOTION_REL_TOL)
    check_error("v_par relative error", v_error, MOTION_REL_TOL)
    check_error("work relative error", energy_error, ENERGY_REL_TOL)

    fig.tight_layout(w_pad=1.8)
    for extension in ("png", "pdf"):
        image = os.path.join(OUT, f"drift_kinetic_ex11_eparallel.{extension}")
        fig.savefig(image, dpi=150, bbox_inches="tight", pad_inches=0.12)
        print(f"Saved figure to {image}")


if __name__ == "__main__":
    main()
