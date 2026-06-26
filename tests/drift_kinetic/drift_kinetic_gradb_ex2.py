#!/usr/bin/env python3

import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../tools"))

from lib.plot_utils import figure, subplot

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "drift_kinetic_gradb_ex2")
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


def gradient_field(config):
    # Returns (value, length, axis) of the SetGradientField magnetic preset.
    for preset in config["Presets"]:
        if preset.get("command") != "SetMagneticField":
            continue
        setter = preset["setter"]
        if setter.get("name") == "SetGradientField":
            value = np.array(setter["value"], dtype=float)
            length = float(setter["length"])
            axis = {"X": 0, "x": 0, "Y": 1, "y": 1, "Z": 2, "z": 2}[
                setter.get("axis", "X")]
            return value, length, axis
    raise KeyError("SetGradientField preset is not found in the config")


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

    q = float(cfg_dk["Particles"][0]["q"])
    m = float(cfg_dk["Particles"][0]["m"])

    # B = value * (1 + s/length) along `axis`; here value || z, axis = x, so the
    # magnitude gradient is |value|/length along x and the drift is along y.
    value, length, _ = gradient_field(cfg_dk)
    B0 = np.linalg.norm(value)
    grad_b = B0 / length

    dk = load_trace(DK_DIR)
    kin = load_trace(KIN_DIR)

    t_dk, x_dk, y_dk = dk[:, 0], dk[:, 1], dk[:, 2]
    p_perp_dk, mu_dk = dk[:, 5], dk[:, 6]
    t_kin, x_kin, y_kin = kin[:, 0], kin[:, 1], kin[:, 2]

    # Field magnitude at the guiding center, taken from the conserved invariants
    # (B = m p_perp^2 / 2mu) so it matches exactly what the pusher interpolated
    # off the grid, independent of any Yee half-cell offset.
    x_gc, y_gc = x_dk[0], y_dk[0]
    b_gc = m * p_perp_dk[0] ** 2 / (2.0 * mu_dk[0])
    omega_e = abs(q) * b_gc / m
    rho = p_perp_dk[0] / (abs(q) * b_gc)

    # Grad-B drift (along y): V = mu (b x grad|B|) / (q B) = mu (B0/L) / (q B) y.
    v_drift = mu_dk[0] * grad_b / (q * b_gc)

    # Steps are stored in 1/w_pe; express them in units of 1/Omega_e using the
    # gyrofrequency at the particles' initial point.
    tau_kin = float(cfg_kin["Geometry"]["dt"]) * omega_e
    tau_dk = float(cfg_dk["Geometry"]["dt"]) * omega_e
    label_kin = rf"kinetic: $\tau = {tau_kin:g}/\Omega_e$"
    label_dk = rf"drift-kinetic: $\tau = {tau_dk:g}/\Omega_e$"

    fig, gs = figure(ncols=3, nrows=1, figsize=(14, 4.5))

    # (a) Trajectory in the (y, x) plane: the drift (y) is horizontal, the
    # gradient/gyration direction (x) is vertical.
    ax = subplot(fig, gs, 0, 0)
    ax.plot(y_kin, x_kin, color=COLOR_KIN, lw=1.0, label=label_kin)
    ax.plot(y_dk, x_dk, color=COLOR_DK, lw=2.0, label=label_dk)
    mark_ends(ax, y_kin, x_kin, COLOR_KIN)
    mark_ends(ax, y_dk, x_dk, COLOR_DK)
    ax.set_xlabel(r"$y,\ c/\omega_{pe}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$x,\ c/\omega_{pe}$", fontsize=LABELSIZE)
    ax.set_ylim(x_gc - 1.5 * rho, x_gc + 3.0 * rho)
    ax.text(0.97, 0.86, r"$\nabla B$ drift",
        transform=ax.transAxes, ha="right", va="top", fontsize=TITLESIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(a)")

    # (b) Drift-subtracted y(t) - V_gradB t; the gyration stays within +-rho_e.
    ax = subplot(fig, gs, 1, 0)
    y_kin_dev = y_kin - v_drift * t_kin
    y_dk_dev = y_dk - v_drift * t_dk
    ax.plot(t_kin, y_kin_dev, color=COLOR_KIN, lw=1.0, label=label_kin)
    ax.plot(t_dk, y_dk_dev, color=COLOR_DK, lw=2.0, label=label_dk)
    ax.axhline(y_gc + rho, color="gray", ls=":", lw=1.0, label=r"theory $\pm\ \rho_e$")
    ax.axhline(y_gc - rho, color="gray", ls=":", lw=1.0)
    mark_ends(ax, t_kin, y_kin_dev, COLOR_KIN)
    mark_ends(ax, t_dk, y_dk_dev, COLOR_DK)
    ax.set_ylim(y_gc - 1.5 * rho, y_gc + 2.5 * rho)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$y(t) - V_{\nabla B} t,\ c/\omega_{pe}$", fontsize=LABELSIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(b)")

    # (c) Relative drift-velocity error of the drift-kinetic run.
    ax = subplot(fig, gs, 2, 0)
    v_dk = np.gradient(y_dk, t_dk)
    rel_err = (v_dk - v_drift) / v_drift
    ax.plot(t_dk, rel_err, color=COLOR_DK, lw=1.5, label=label_dk)
    ax.axhline(0.0, color=COLOR_TH, ls="--", lw=1.0)
    ymax = np.max(np.abs(rel_err))
    if ymax > 0:
        ax.set_ylim(-1.1 * ymax, 1.1 * ymax)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$(V(t) - V_{\nabla B}) / V_{\nabla B}$", fontsize=LABELSIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(OFFSETSIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(c)")

    print(f"v_drift (theory)        = {v_drift:.6e}")
    print(f"v_drift (drift-kinetic) = {np.mean(v_dk):.6e}")
    print(f"max relative error      = {np.max(np.abs(rel_err)):.3e}")

    fig.tight_layout(pad=1.2, w_pad=2.5)
    image = os.path.join(OUT, "drift_kinetic_gradb_ex2.png")
    fig.savefig(image, dpi=150)
    print(f"Saved figure to {image}")


if __name__ == "__main__":
    main()
