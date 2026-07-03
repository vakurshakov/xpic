#!/usr/bin/env python3

import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../tools"))

from lib.plot_utils import figure, subplot

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "drift_kinetic_curv_ex3")
DK_DIR = os.path.join(OUT, "drift_kinetic")
KIN_DIR = os.path.join(OUT, "eccapfim")

# Convergence runs (drift-kinetic), refining the time step and dx together.
# Each lives in DK_DIR/<name>/particle_trace.txt; tuple is (dir, tau, dx, color).
CONV_RUNS = [
    ("temporal_10_0.5",   10.0, 0.5,   "#9467bd"),
    ("temporal_5_0.25",    5.0, 0.25,  "#2ca02c"),
    ("temporal_2.5_0.125", 2.5, 0.125, "#ff7f0e"),
]
# Panels (a), (b) show the coarsest drift-kinetic run (tau = 10) vs the kinetic
# reference, which uses tau = 0.1 and lives in the same case subdir.
AB_RUN = CONV_RUNS[0]
TAU_KIN = 0.1

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


def azimuthal_center(config):
    # Returns the axis location of the SetAzimuthalField magnetic preset.
    for preset in config["Presets"]:
        if preset.get("command") != "SetMagneticField":
            continue
        setter = preset["setter"]
        if setter.get("name") == "SetAzimuthalField":
            return np.array(setter["center"], dtype=float)
    raise KeyError("SetAzimuthalField preset is not found in the config")


def load_trace(path, subdir="temporal"):
    return np.loadtxt(os.path.join(path, subdir, "particle_trace.txt"), skiprows=1)


def curvature_drift(dk, cx, cy, q, m):
    # Theoretical curvature drift V = m v_par^2 / (q B R) from the run's initial
    # invariants (|B| = m p_perp^2 / 2mu so it matches the interpolated field).
    p_par, p_perp, mu = dk[:, 4], dk[:, 5], dk[:, 6]
    b_gc = m * p_perp[0] ** 2 / (2.0 * mu[0])
    R = np.hypot(dk[0, 1] - cx, dk[0, 2] - cy)
    v_par = p_par[0] / m
    return m * v_par ** 2 / (q * b_gc * R), b_gc


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

    q = float(cfg_dk["Particles"][0]["q"])
    m = float(cfg_dk["Particles"][0]["m"])

    center = azimuthal_center(cfg_dk)
    cx, cy = center[0], center[1]

    # Panels (a), (b): coarse drift-kinetic run (tau = 10) vs the kinetic
    # reference (tau = 0.1), stored in the same case subdir as the drift run.
    # The paired run shares the geometry (so the z-baseline matches); a fallback
    # run with a different dx sits on a different z and drops out of panel (b).
    if os.path.isdir(os.path.join(KIN_DIR, AB_RUN[0])):
        kin_subdir = AB_RUN[0]
    else:
        kin_subdir = "temporal"
        print(f"WARNING: {os.path.join(KIN_DIR, AB_RUN[0])} not found; "
            f"falling back to '{kin_subdir}' (mismatched z-baseline in panel b)")
    dk = load_trace(DK_DIR, AB_RUN[0])
    kin = load_trace(KIN_DIR, kin_subdir)

    t_dk, x_dk, y_dk, z_dk = dk[:, 0], dk[:, 1], dk[:, 2], dk[:, 3]
    p_perp_dk, mu_dk = dk[:, 5], dk[:, 6]
    t_kin, x_kin, y_kin, z_kin = kin[:, 0], kin[:, 1], kin[:, 2], kin[:, 3]

    v_drift, b_gc = curvature_drift(dk, cx, cy, q, m)
    rho = p_perp_dk[0] / (abs(q) * b_gc)
    z_gc = z_dk[0]

    # Steps are stored in 1/w_pe; Omega_e = |q| B0 / m = 1 by design, so the
    # step in units of 1/Omega_e is just the dt of each run.
    label_kin = rf"kinetic: $\tau = {TAU_KIN:g}/\Omega_e$"
    label_dk = rf"drift-kinetic: $\tau = {AB_RUN[1]:g}/\Omega_e$"

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

    # (c) Magnitude of the relative drift-velocity error for the three runs,
    # refining dt and dx together (convergence of the drift-kinetic pusher).
    ax = subplot(fig, gs, 2, 0)
    err_max = 0.0
    for run_dir, tau, run_dx, color in CONV_RUNS:
        run = load_trace(DK_DIR, run_dir)
        t, z = run[:, 0], run[:, 3]
        v_run, _ = curvature_drift(run, cx, cy, q, m)
        v_num = np.gradient(z, t)
        rel_err = np.abs((v_num - v_run) / v_run)
        err_max = max(err_max, float(np.max(rel_err)))
        label = rf"$\tau = {tau:g}/\Omega_e,\ \Delta x = {run_dx:g}\,c/\omega_{{pe}}$"
        ax.plot(t, rel_err, color=color, lw=2.2, label=label)
        print(f"{run_dir:>20}: max |rel. error| = {np.max(rel_err):.3e}")
    ax.axhline(0.0, color=COLOR_TH, ls="--", lw=1.0)
    ax.set_xlim(0.0, 300.0)
    ax.set_ylim(0.0, 2e-3)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$\delta_V(t)$", fontsize=LABELSIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(OFFSETSIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(c)")

    fig.tight_layout(pad=1.2, w_pad=2.5)
    image = os.path.join(OUT, "drift_kinetic_curv_ex3_convergence.png")
    fig.savefig(image, dpi=150)
    print(f"Saved figure to {image}")


if __name__ == "__main__":
    main()
