#!/usr/bin/env python3

import glob
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

OUT = os.path.join(TEST_DIR, "output", "drift_kinetic_ex10_bounce")
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

FREQUENCY_REL_TOL = 1e-6
ENERGY_REL_TOL = 1e-8
ORDER_MIN = 1.8


def load_config(path):
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


def mirror_field(config):
    # Returns (value, curvature, center) of the SetMirrorField magnetic preset.
    for preset in config["Presets"]:
        if preset.get("command") != "SetField" or preset.get("field") != "B":
            continue
        setter = preset["setter"]
        if setter.get("name") == "SetMirrorField":
            return (float(setter["value"]), float(setter["curvature"]),
                np.array(setter["center"], dtype=float))
    raise KeyError("SetMirrorField preset is not found in the config")


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


def drift_runs():
    # Drift-kinetic runs sorted by the time step, largest first.
    paths = glob.glob(os.path.join(OUT, "drift_kinetic_dt*"))
    if not paths:
        raise RuntimeError("drift-kinetic run directories are missing")
    return sorted(paths, key=lambda path: -float(load_config(path)["Geometry"]["dt"]))


def bounce_frequency(t, s, v, omega_b):
    # Mean phase velocity on the (s, v / omega_b) plane.
    phase = np.unwrap(np.arctan2(-v / omega_b, s))
    return (phase[-1] - phase[0]) / (t[-1] - t[0])


def analyze(path, B0, kappa, center, m):
    # Returns (dt, trace, omega_b, omega_h, omega_dk, amplitude, W / W_0 - 1).
    dt = float(load_config(path)["Geometry"]["dt"])
    dk = load_trace(path)
    s, v, mu = dk[:, 3] - center[2], dk[:, 4] / m, dk[:, 6]

    # Bounce frequency on the axis and its implicit-midpoint counterpart.
    omega_b = np.sqrt(2.0 * mu[0] * B0 * kappa / m)
    omega_h = 2.0 / dt * np.arctan(0.5 * omega_b * dt)

    # The midpoint rule conserves s^2 + (v/omega_b)^2 exactly; every step
    # advances the phase on this ellipse by omega_h dt.
    amplitude = np.hypot(s[0], v[0] / omega_b)
    omega_dk = bounce_frequency(dk[:, 0], s, v, omega_b)

    # |B| of the paraxial mirror, including the small off-axis radial part.
    r2 = (dk[:, 1] - center[0]) ** 2 + (dk[:, 2] - center[1]) ** 2
    b = B0 * np.sqrt((1.0 + kappa * s ** 2) ** 2 + (kappa * s) ** 2 * r2)
    energy = 0.5 * m * v ** 2 + mu * b
    return dt, dk, omega_b, omega_h, omega_dk, amplitude, \
        energy / energy[0] - 1.0


def main():
    cfg_kin = load_config(KIN_DIR)

    q = float(cfg_kin["Particles"][0]["q"])
    m = float(cfg_kin["Particles"][0]["m"])
    B0, kappa, center = mirror_field(cfg_kin)
    omega_e = abs(q) * B0 / m

    runs = [analyze(path, B0, kappa, center, m) for path in drift_runs()]
    dt = np.array([run[0] for run in runs])
    omega_b = runs[0][2]
    omega_dk = np.array([run[4] for run in runs])
    omega_h = np.array([run[3] for run in runs])

    # Panels (a), (b) show the largest drift-kinetic step.
    dt_dk, dk, _, _, _, amplitude, energy_dk = runs[0]
    kin = load_trace(KIN_DIR)

    t_dk, s_dk = dk[:, 0], dk[:, 3] - center[2]
    t_kin, s_kin, v_kin = kin[:, 0], kin[:, 3] - center[2], kin[:, 6]
    omega_kin = bounce_frequency(t_kin, s_kin, v_kin, omega_b)

    dt_kin = float(cfg_kin["Geometry"]["dt"])
    tau_kin = dt_kin * omega_e
    label_kin = rf"kinetic: $\tau = {tau_kin:g}/\Omega_e$"
    label_dk = rf"drift-kinetic: $\tau = {dt_dk * omega_e:g}/\Omega_e$"

    fig, gs = figure(ncols=3, nrows=1, figsize=(25.0, 8.1))

    # (a) Bounce motion s(t) = z(t) - z_c between the turning points.
    ax = subplot(fig, gs, 0, 0)
    ax.plot(t_kin, s_kin, color=COLOR_KIN, lw=2.0, label=label_kin)
    ax.plot(t_dk, s_dk, "o", color=COLOR_DK, ms=5, label=label_dk)
    ax.axhline(+amplitude, color="0.25", ls=":", lw=2.0, zorder=3,
               label="theory")
    ax.axhline(-amplitude, color="0.25", ls=":", lw=2.0, zorder=3)
    mark_ends(ax, t_kin, s_kin, COLOR_KIN)
    mark_ends(ax, t_dk, s_dk, COLOR_DK)
    ax.set_ylim(-1.3 * amplitude, 2.0 * amplitude)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$z - z_c,\ c/\omega_{pe}$", fontsize=LABELSIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(a)")

    # (b) Bounce frequency error against the time step.
    ax = subplot(fig, gs, 1, 0)
    rel_err = np.abs(omega_dk / omega_b - 1.0)
    rel_err_kin = abs(omega_kin / omega_b - 1.0)
    ax.loglog(dt * omega_e, rel_err[-1] * (dt / dt[-1]) ** 2, color=COLOR_TH,
        ls="--", lw=1.5, label=r"$O(\Delta t^2)$")
    ax.loglog(dt * omega_e, rel_err, "^", color=COLOR_DK, ms=10,
        label="drift-kinetic")
    ax.loglog(tau_kin, rel_err_kin, "s", color=COLOR_KIN, ms=9,
        label="kinetic")
    ax.set_xlim(min(dt.min(), dt_kin) / 1.25 * omega_e,
        dt.max() * 1.25 * omega_e)
    ax.set_ylim(rel_err.min() / 3.0, rel_err.max() * 30.0)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"$\tau,\ \Omega_e^{-1}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$|\omega/\omega_b - 1|$",
        fontsize=LABELSIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(b)")

    # (c) Energy m v_par^2 / 2 + mu |B| of the drift-kinetic particle.
    ax = subplot(fig, gs, 2, 0)
    ax.axhline(0.0, color="0.25", ls="--", lw=1.5, zorder=1)
    ax.plot(t_dk, energy_dk, "o-", color=COLOR_DK, lw=1.5, ms=5,
        label=label_dk)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=LABELSIZE)
    energy_max = np.max(np.abs(energy_dk))
    ax.set_ylim(-1.3 * energy_max, 2.0 * energy_max)
    ax.set_ylabel(r"$W / W_0 - 1$", fontsize=LABELSIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(c)")

    order = np.polyfit(np.log(dt), np.log(rel_err), 1)[0]
    print(f"omega_b (theory)        = {omega_b:.6e}")
    print(f"omega (drift-kinetic)   = {np.array2string(omega_dk, formatter={'float': '{:.6e}'.format})}")
    print(f"fitted order            = {order:.3f} (limit {ORDER_MIN})")
    if order < ORDER_MIN:
        raise AssertionError(
            f"fitted order is {order:.3f}, expected at least {ORDER_MIN}")
    check_error("frequency relative error", np.max(np.abs(omega_dk / omega_h - 1.0)), FREQUENCY_REL_TOL)
    check_error("DK energy relative change",
        max(np.max(np.abs(run[6])) for run in runs), ENERGY_REL_TOL)

    fig.tight_layout(w_pad=1.8)
    for extension in ("png", "pdf"):
        image = os.path.join(OUT, f"drift_kinetic_ex10_bounce.{extension}")
        fig.savefig(image, dpi=150, bbox_inches="tight", pad_inches=0.12)
        print(f"Saved figure to {image}")


if __name__ == "__main__":
    main()
