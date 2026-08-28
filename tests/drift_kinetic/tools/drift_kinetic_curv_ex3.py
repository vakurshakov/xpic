#!/usr/bin/env python3

import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.append(os.path.join(TEST_DIR, "../../tools"))

from lib.plot_utils import figure, subplot

OUT = os.path.join(TEST_DIR, "output", "drift_kinetic_curv_ex3")
DK_DIR = os.path.join(OUT, "drift_kinetic")
KIN_DIR = os.path.join(OUT, "eccapfim")

COLOR_KIN = "red"
COLOR_DK = "blue"
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


def azimuthal_center(config):
    # Returns the axis location of the SetAzimuthalField magnetic preset.
    for preset in config["Presets"]:
        if preset.get("command") != "SetMagneticField":
            continue
        setter = preset["setter"]
        if setter.get("name") == "SetAzimuthalField":
            return np.array(setter["center"], dtype=float)
    raise KeyError("SetAzimuthalField preset is not found in the config")


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

    center = azimuthal_center(cfg_dk)
    cx, cy = center[0], center[1]

    dk = load_trace(DK_DIR)
    kin = load_trace(KIN_DIR)

    t_dk, x_dk, y_dk, z_dk = dk[:, 0], dk[:, 1], dk[:, 2], dk[:, 3]
    p_par_dk, p_perp_dk, mu_dk = dk[:, 4], dk[:, 5], dk[:, 6]
    t_kin, x_kin, y_kin, z_kin = kin[:, 0], kin[:, 1], kin[:, 2], kin[:, 3]

    # Field magnitude at the guiding center, from the conserved invariants
    # (B = m p_perp^2 / 2mu) so it matches what the pusher interpolated.
    b_gc = m * p_perp_dk[0] ** 2 / (2.0 * mu_dk[0])
    omega_e = abs(q) * b_gc / m
    rho = p_perp_dk[0] / (abs(q) * b_gc)

    # Curvature radius = guiding-center distance from the axis; parallel
    # velocity from p_par. Curvature drift (along z): V = m v_par^2 / (q B R).
    R = np.hypot(x_dk[0] - cx, y_dk[0] - cy)
    v_par = p_par_dk[0] / m
    v_drift = m * v_par ** 2 / (q * b_gc * R)
    z_gc = z_dk[0]

    # Steps are stored in 1/w_pe; Omega_e = |q| B0 / m = 1 by design, so the
    # step in units of 1/Omega_e is just the configured dt (use it directly to
    # keep exact 0.1 / 10 instead of interpolation-noise digits from b_gc).
    tau_kin = float(cfg_kin["Geometry"]["dt"])
    tau_dk = float(cfg_dk["Geometry"]["dt"])
    label_kin = rf"kinetic: $\tau = {tau_kin:g}/\Omega_e$"
    label_dk = rf"drift-kinetic: $\tau = {tau_dk:g}/\Omega_e$"

    fig, gs = figure(ncols=3, nrows=1, figsize=(14, 4.5))

    # (a) Trajectory projected on the (x, y) plane: the guiding center follows
    # the circular field line of radius R; the drift is out of plane (along z).
    ax = subplot(fig, gs, 0, 0)
    ax.plot(x_kin, y_kin, color=COLOR_KIN, lw=1.0, label=label_kin)
    ax.plot(x_dk, y_dk, color=COLOR_DK, lw=2.0, label=label_dk)
    mark_ends(ax, x_kin, y_kin, COLOR_KIN)
    mark_ends(ax, x_dk, y_dk, COLOR_DK)
    ax.set_xlabel(r"$x,\ c/\omega_{pe}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$y,\ c/\omega_{pe}$", fontsize=LABELSIZE)
    # Frame on the actual particle trajectory, not the whole model domain;
    # use an equal span on both axes so the panel stays square.
    xs = np.concatenate([x_kin, x_dk])
    ys = np.concatenate([y_kin, y_dk])
    pad = 0.3
    xc = 0.5 * (xs.min() + xs.max())
    yc = 0.5 * (ys.min() + ys.max())
    half = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min()) + pad
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(yc - half, yc + half)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.03, 0.22, r"curvature drift",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=TITLESIZE)
    ax.legend(loc="lower left", fontsize=LEGENDSIZE)
    style_axes(ax, "(a)")

    # (b) Drift-subtracted z(t) - V_kappa t; the gyration stays within +-rho_e.
    ax = subplot(fig, gs, 1, 0)
    z_kin_dev = z_kin - v_drift * t_kin
    z_dk_dev = z_dk - v_drift * t_dk
    ax.plot(t_kin, z_kin_dev, color=COLOR_KIN, lw=1.0, label=label_kin)
    ax.plot(t_dk, z_dk_dev, color=COLOR_DK, lw=2.0, label=label_dk)
    ax.axhline(z_gc + rho, color="gray", ls=":", lw=1.0, label=r"theory $\pm\ \rho_e$")
    ax.axhline(z_gc - rho, color="gray", ls=":", lw=1.0)
    mark_ends(ax, t_kin, z_kin_dev, COLOR_KIN)
    mark_ends(ax, t_dk, z_dk_dev, COLOR_DK)
    ax.set_ylim(z_gc - 1.5 * rho, z_gc + 2.5 * rho)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$z(t) - V_\kappa t,\ c/\omega_{pe}$", fontsize=LABELSIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(b)")

    # (c) Relative drift-velocity error of the drift-kinetic run.
    ax = subplot(fig, gs, 2, 0)
    v_dk = np.gradient(z_dk, t_dk)
    rel_err = (v_dk - v_drift) / v_drift
    ax.plot(t_dk, rel_err, color=COLOR_DK, lw=1.5, label=label_dk)
    ax.axhline(0.0, color=COLOR_TH, ls="--", lw=1.0)
    ymax = np.max(np.abs(rel_err))
    if ymax > 0:
        ax.set_ylim(-1.1 * ymax, 1.1 * ymax)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$(V(t) - V_\kappa) / V_\kappa$", fontsize=LABELSIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(OFFSETSIZE)
    ax.legend(fontsize=LEGENDSIZE)
    style_axes(ax, "(c)")

    print(f"|B| at guiding center  = {b_gc:.6e}")
    print(f"v_drift (theory)        = {v_drift:.6e}")
    print(f"v_drift (drift-kinetic) = {np.mean(v_dk):.6e}")
    print(f"max relative error      = {np.max(np.abs(rel_err)):.3e}")

    fig.tight_layout(pad=1.2, w_pad=2.5)
    image = os.path.join(OUT, "drift_kinetic_curv_ex3.png")
    fig.savefig(image, dpi=150)
    print(f"Saved figure to {image}")


if __name__ == "__main__":
    main()
