#!/usr/bin/env python3

import argparse
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

OUT = os.path.join(TEST_DIR, "output", "drift_kinetic_ex9_nonlin_gradb")

COLOR_KIN = "black"
COLOR_DK = "red"
COLOR_TH = "black"

LABELSIZE = 28
TICKSIZE = 27
LEGENDSIZE = 26
PANELSIZE = 26
TITLESIZE = 26
OFFSETSIZE = 27

ORDER_MIN = 1.8
TOLERANCES = {"aligned": (5e-4, 1e-8), "oblique": (1e-3, 1e-9)}


def load_config(path):
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


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


# Yee offsets of the B components, in cells.
YEE_B = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])


def periodic_field(config):
    # Returns (B0, b, modulation, k) of the SetUniformField and
    # SetGeneralCosineField magnetic presets, B = B0 (1 + eps cos(k r)) b.
    value = amplitude = n = None
    for preset in config["Presets"]:
        if preset.get("command") != "SetField" or preset.get("field") != "B":
            continue
        setter = preset["setter"]
        if setter.get("name") == "SetUniformField":
            value = np.array(setter["value"], dtype=float)
        elif setter.get("name") == "SetGeneralCosineField":
            amplitude = np.array(setter["amplitude"], dtype=float)
            n = np.array(setter["wave_number"], dtype=float)
    if value is None or amplitude is None:
        raise KeyError("periodic field presets are not found in the config")
    B0 = np.linalg.norm(value)
    L = np.array([float(config["Geometry"][name]) for name in "xyz"])
    return B0, value / B0, np.linalg.norm(amplitude) / B0, 2.0 * np.pi * n / L


def spline2(s):
    s = np.abs(s)
    return np.where(s <= 0.5, 0.75 - s * s,
        np.where(s <= 1.5, 0.5 * (1.5 - s) ** 2, 0.0))


def grid_field(r, config, B0, b, eps, k):
    # |B| interpolated by the S2 kernel from the Yee grid, as the pusher does.
    # The field does not depend on z, so only the x and y stencils are summed.
    h = np.array([float(config["Geometry"][name]) for name in ("dx", "dy")])
    B = np.zeros((len(r), 3))
    for c in range(3):
        u = r[:, :2] / h - YEE_B[c, :2]
        i0 = np.round(u)
        for a in (-1, 0, 1):
            for e in (-1, 0, 1):
                node = i0 + np.array([a, e])
                w = spline2(node[:, 0] - u[:, 0]) * spline2(node[:, 1] - u[:, 1])
                phase = (node + YEE_B[c, :2]) * h @ k[:2]
                B[:, c] += w * b[c] * B0 * (1.0 + eps * np.cos(phase))
    return np.linalg.norm(B, axis=1)


def unwrapped_coordinates(trace, config):
    # Coordinates continued across the periodic boundaries.
    return np.stack([np.unwrap(trace[:, 1 + i],
        period=float(config["Geometry"][name])) for i, name in enumerate("xyz")],
        axis=1)


def drift_runs(out):
    # Drift-kinetic runs sorted by the grid spacing, coarsest first.
    paths = glob.glob(os.path.join(out, "drift_kinetic_n*"))
    if not paths:
        raise RuntimeError("drift-kinetic run directories are missing")
    return sorted(paths, key=lambda path: -float(load_config(path)["Geometry"]["dx"]))


def main(case):
    out = os.path.join(OUT, case)
    kin_dir = os.path.join(out, "eccapfim")
    finest_rel_tol, energy_rel_tol = TOLERANCES[case]
    cfg_kin = load_config(kin_dir)
    q = float(cfg_kin["Particles"][0]["q"])
    m = float(cfg_kin["Particles"][0]["m"])
    B0, b, eps, k = periodic_field(cfg_kin)
    omega_e = abs(q) * B0 / m

    # Gradient direction g, drift direction d = b x g; both are oblique to the
    # grid as b is.
    g = k / np.linalg.norm(k)
    d = np.cross(b, g)

    def field(r):
        return B0 * (1.0 + eps * np.cos(r @ k))

    # Grad-B drift along d at the guiding center r_gc, with the exact profile:
    # V = mu (b x grad B) / (q B), grad B = -B0 eps sin(k r) k.
    dx, rel_err, v_par, traces = [], [], [], {}
    for path in drift_runs(out):
        cfg = load_config(path)
        dk = load_trace(path)
        r = unwrapped_coordinates(dk, cfg)
        t, v, mu = dk[:, 0], dk[:, 4] / m, dk[:, 6]
        grad_b = -B0 * eps * np.sin(r[0] @ k) * np.linalg.norm(k)
        v_drift = mu[0] * grad_b / (q * field(r[0]))
        v_dk = (r[-1] - r[0]) @ d / (t[-1] - t[0])

        # The discrete energy uses |B| interpolated from the grid.
        magnetic_field = (field(r) if case == "aligned" else
            grid_field(r, cfg, B0, b, eps, k))
        energy = 0.5 * m * v ** 2 + mu * magnetic_field
        dx.append(float(cfg["Geometry"]["dx"]))
        rel_err.append(abs(v_dk / v_drift - 1.0))
        v_par.append(np.max(np.abs(v)))
        traces[cfg["Geometry"]["dx"]] = (r, dk, cfg, v_drift,
            energy / energy[0] - 1.0)
    dx, rel_err, v_par = np.array(dx), np.array(rel_err), np.array(v_par)

    # Panel (a) shows the drift-kinetic run on the grid of the kinetic one.
    dx_kin = float(cfg_kin["Geometry"]["dx"])
    r_dk, dk, cfg_dk, v_drift, energy_dk = traces[cfg_kin["Geometry"]["dx"]]
    kin = load_trace(kin_dir)
    r_kin = unwrapped_coordinates(kin, cfg_kin)
    t_dk, t_kin = dk[:, 0], kin[:, 0]

    # Kinetic guiding center r_gc = r + (p x b) / (q B) filters out the
    # gyration from the drift velocity.
    r_gc = r_kin + m * np.cross(kin[:, 4:7], b) / (q * field(r_kin))[:, None]
    rel_err_kin = abs(np.polyfit(t_kin, r_gc @ d, 1)[0] / v_drift - 1.0)

    rho = dk[0, 5] / (abs(q) * field(r_dk[0]))

    tau_kin = float(cfg_kin["Geometry"]["dt"]) * omega_e
    tau_dk = float(cfg_dk["Geometry"]["dt"]) * omega_e
    label_kin = rf"kinetic: $\tau = {tau_kin:g}/\Omega_e$"
    label_dk = rf"drift-kinetic: $\tau = {tau_dk:g}/\Omega_e$"

    fig, gs = figure(ncols=3, nrows=1, figsize=(25.0, 8.1))

    # (a) Trajectory in the (d, g) plane: the drift (d) is horizontal, the
    # gradient direction (g) is vertical.
    ax = subplot(fig, gs, 0, 0)
    ax.plot(r_kin @ d, r_kin @ g, color=COLOR_KIN, lw=1.0, label=label_kin)
    ax.plot(r_dk @ d, r_dk @ g, color=COLOR_DK, lw=2.0, label=label_dk)
    mark_ends(ax, r_kin @ d, r_kin @ g, COLOR_KIN)
    mark_ends(ax, r_dk @ d, r_dk @ g, COLOR_DK)
    ax.set_xlabel(r"$\mathbf{r}\cdot[\mathbf{b}\times\hat{\mathbf{k}}],\ c/\omega_{pe}$",
        fontsize=LABELSIZE)
    ax.set_ylabel(r"$\mathbf{r}\cdot\hat{\mathbf{k}},\ c/\omega_{pe}$",
        fontsize=LABELSIZE)
    ax.set_ylim(r_dk[0] @ g - 1.5 * rho, r_dk[0] @ g + 3.0 * rho)
    ax.text(0.97, 0.70, rf"{case} $\nabla B$ drift",
        transform=ax.transAxes, ha="right", va="top", fontsize=TITLESIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(a)")

    # (b) Spatial convergence of the drift velocity at a fixed dt.
    ax = subplot(fig, gs, 1, 0)
    ax.loglog(dx, rel_err[-1] * (dx / dx[-1]) ** 2, color=COLOR_TH, ls="--",
        lw=1.5, label=r"$O(\Delta x^2)$")
    ax.loglog(dx, rel_err, "^", color=COLOR_DK, ms=10, label="drift-kinetic")
    ax.loglog(dx_kin, rel_err_kin, "s", color=COLOR_KIN, ms=9,
        label="kinetic")
    ax.set_xlim(dx.min() / 1.5, dx.max() * 1.5)
    ax.set_ylim(rel_err.min() / 3.0, rel_err.max() * 10.0)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel((r"$\Delta x,\ c/\omega_{pe}$" if case == "aligned" else
        r"$\Delta x = \Delta y,\ c/\omega_{pe}$"), fontsize=LABELSIZE)
    ax.set_ylabel(r"$|V_{\nabla B}/V^{\mathrm{th}}_{\nabla B} - 1|$",
        fontsize=LABELSIZE)
    ax.legend(loc="upper left", fontsize=LEGENDSIZE)
    style_axes(ax, "(b)")

    # (c) Energy m v_par^2 / 2 + mu |B_p| of the drift-kinetic particle.
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

    order = np.polyfit(np.log(dx), np.log(rel_err), 1)[0]
    print(f"relative errors         = {np.array2string(rel_err, formatter={'float': '{:.3e}'.format})}")
    print(f"parasitic max |v_par|   = {np.array2string(v_par, formatter={'float': '{:.3e}'.format})}")
    print(f"kinetic relative error  = {rel_err_kin:.3e}")
    print(f"fitted order            = {order:.3f} (limit {ORDER_MIN})")
    if order < ORDER_MIN:
        raise AssertionError(
            f"fitted order is {order:.3f}, expected at least {ORDER_MIN}")
    check_error("finest drift error", rel_err[-1], finest_rel_tol)
    check_error("DK energy relative change",
        max(np.max(np.abs(run[4])) for run in traces.values()), energy_rel_tol)

    fig.tight_layout(w_pad=1.8)
    for extension in ("png", "pdf"):
        image = os.path.join(out, f"drift_kinetic_ex9_nonlin_gradb.{extension}")
        fig.savefig(image, dpi=150, bbox_inches="tight", pad_inches=0.12)
        print(f"Saved figure to {image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Non-linear grad-B convergence")
    parser.add_argument("--case", required=True, choices=TOLERANCES)
    main(parser.parse_args().case)
