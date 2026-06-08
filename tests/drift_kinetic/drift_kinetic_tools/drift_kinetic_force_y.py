#!/usr/bin/env python3
"""1D-along-x radial-force-balance diagnostic for drift_kinetic runs.

Uses **plane-Y** FieldView slices (one fixed y, with x and z varying) and
collapses each (Nz, Nx) slice to a 1D profile vs x by either averaging
along z (default) or picking the central z plane.

Quantities (computed per cell on the (Nz, Nx) slice, then reduced to 1D):

    p_perp = -M . B                       (full dot product)
    B^2    = Bx^2 + By^2 + Bz^2
    J_phi  = (J + curl M)_y               (azimuthal at y=cy is +/- y-comp)

On a plane-Y slice we lack y-derivatives, so for an axisymmetric,
axially-aligned configuration we use
    (curl M)_y ~ dM_x/dz - dM_z/dx.

The four panels of the figure are:
    (1) R(x)        = -dp_perp/dx + J_phi * B_z
    (2) -dp_perp/dx and -J_phi * B_z (current sign-flipped so the two
                                       lines visually cancel)
    (3) (p_perp + B^2/2) - (p_perp + B^2/2)|_{t = t_0}
    (4) p_perp  and  B_0^2/2 - B^2/2,  where B_0^2/2 is the magnetic
        pressure of the very first available snapshot (t = 0)

J and M are stored half a step ahead of B/E (snap k holds them at step
k - 1/2), so we backward-average two consecutive snapshots,
X_eff(step k) = (X[k-1] + X[k]) / 2. The first frame uses the raw value
(zero anyway).

`--start-idx` skips snapshots before the given index from drawing,
time-averaging and from the t_0 baseline of panel 3 (the first frame
with idx >= start_idx is the baseline). Snapshots before start_idx are
still iterated through to keep the half-step backward averages of J/M
consistent.

`--time-average` overlays a running mean of R(x) on panel 1 (black,
dotted) on top of the instantaneous curve; the other three panels are
always instantaneous. The mean uses snapshots with
idx >= --tavg-start-idx (default: same as --start-idx). Frames between
start_idx and tavg_start_idx are still drawn but do not contribute to
the average and the dotted line only appears once tavg_start_idx is
reached.

The t = 0 baseline of panel 3 is **always** the very first available
snapshot (idx = common[0]), regardless of --start-idx.

One PNG per snapshot is written to <out_dir>/processed/<out-subdir>/.
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np

TOOLS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from lib.plot_utils import figure, subplot
from lib.plot import bbox, labelsize, ticksize
from lib.constants import const, init_constants

import shutil
import matplotlib.pyplot as plt

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)

from drift_kinetic_equilibrium import auto_center, list_steps


def find_plane_y_dir(field: str):
    """Path to the plane-Y FieldView directory for `field`."""
    for diag in const.config.get("Diagnostics", []):
        if diag.get("diagnostic") != "FieldView":
            continue
        if diag.get("field") != field:
            continue
        region = diag.get("region", {})
        if region.get("type") != "2D" or region.get("plane") != "Y":
            continue
        sub = diag.get("out_dir", field)
        path = os.path.join(const.in_dir, sub)
        if os.path.isdir(path):
            return path
    for sub in (f"{field}_planeY", f"{field}_Y"):
        path = os.path.join(const.in_dir, sub)
        if os.path.isdir(path):
            return path
    # Some configs append a position suffix to the directory name; pick
    # the last lexicographic match.
    for base in (f"{field}_planeY", field):
        parent = const.in_dir
        if not os.path.isdir(parent):
            continue
        prefix_matches = sorted(
            os.path.join(parent, name)
            for name in os.listdir(parent)
            if name.startswith(base + "_") and
               os.path.isdir(os.path.join(parent, name))
        )
        if prefix_matches:
            return prefix_matches[-1]
    return None


def load_plane_y_vec(path: str, name: str):
    """Plane-Y FieldView: shape (Nz, Nx, 3), float32."""
    h, w = const.Nz, const.Nx
    raw = np.fromfile(os.path.join(path, name),
                      dtype=np.float32, count=h * w * 3)
    return raw.reshape(h, w, 3)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--center-x", type=float, default=None,
                        help="Cylinder axis x-coordinate (default: auto)")
    parser.add_argument("--z-mode", choices=("avg", "center"), default="avg",
                        help="How to reduce the (Nz, Nx) slice to a 1D "
                             "profile vs x. 'avg' averages over z; "
                             "'center' uses Nz//2 (use this when z-averaging "
                             "is not appropriate).")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Skip snapshots with idx < START_IDX from "
                             "drawing, from time-averaging and from the "
                             "t_0 baseline of panel 3 (default: 0). "
                             "Snapshots before that are still iterated "
                             "through to keep the half-step backward "
                             "averages of J/M consistent.")
    parser.add_argument("--time-average", action="store_true",
                        help="Overlay a running time-mean of R(x) on the "
                             "first panel as a black dotted line (other "
                             "panels stay instantaneous). The mean uses "
                             "snapshots with idx >= --tavg-start-idx.")
    parser.add_argument("--tavg-start-idx", type=int, default=None,
                        help="First idx that contributes to the running "
                             "mean of R(x) (default: same as --start-idx). "
                             "Useful to draw a long range of frames but "
                             "average only the late, quasi-steady tail.")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out-subdir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    init_constants(args.config)

    b_dir = find_plane_y_dir("B")
    j_dir = find_plane_y_dir("J")
    m_dir = find_plane_y_dir("M")
    if any(d is None for d in (b_dir, j_dir, m_dir)):
        print(f"[error] need plane-Y FieldView for B, J and M "
              f"(got B={b_dir}, J={j_dir}, M={m_dir})")
        return

    plane_size = const.Nx * const.Nz * 3 * 4

    b_steps = dict(list_steps(b_dir, plane_size))
    j_steps = dict(list_steps(j_dir, plane_size))
    m_steps = dict(list_steps(m_dir, plane_size))

    common = sorted(set(b_steps) & set(j_steps) & set(m_steps))
    if not common:
        print("[error] no timesteps common to B / J / M plane-Y")
        return

    if args.center_x is not None:
        cx = args.center_x
    else:
        cx, _ = auto_center()

    tavg_start_idx = (args.tavg_start_idx if args.tavg_start_idx is not None
                      else args.start_idx)
    if tavg_start_idx < args.start_idx:
        tavg_start_idx = args.start_idx

    print(f"Cylinder axis x: {cx:.3f}   (z-mode: {args.z_mode}, "
          f"time-avg: {args.time_average}, "
          f"start_idx: {args.start_idx}, "
          f"tavg_start_idx: {tavg_start_idx})")

    xs = (np.arange(const.Nx) + 0.5) * const.dx

    out_subdir = args.out_subdir or (
        "force_y_timeavg" if args.time_average else "force_y")
    out_dir = os.path.join(const.out_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    def reduce_z(field2d):
        if args.z_mode == "avg":
            return field2d.mean(axis=0)
        return field2d[const.Nz // 2]

    def step_quantities(B, J_eff, M_eff):
        Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
        Mx, My, Mz = M_eff[..., 0], M_eff[..., 1], M_eff[..., 2]

        p_perp_2d = -(Mx * Bx + My * By + Mz * Bz)
        B2_half_2d = 0.5 * (Bx * Bx + By * By + Bz * Bz)

        # curl M on plane-Y (no y-derivatives available).
        dMx_dz = np.gradient(Mx, const.dz, axis=0)
        dMz_dx = np.gradient(Mz, const.dx, axis=1)
        curlM_y_2d = dMx_dz - dMz_dx

        Jy_tot_2d = J_eff[..., 1] + curlM_y_2d
        dp_dx_2d = np.gradient(p_perp_2d, const.dx, axis=1)

        return {
            "p_perp": reduce_z(p_perp_2d),
            "B2_half": reduce_z(B2_half_2d),
            "neg_dp_dx": reduce_z(-dp_dx_2d),
            "JyBz": reduce_z(Jy_tot_2d * Bz),
        }

    def half_step_average(loader, steps_dict, k_pos, prev):
        idx = common[k_pos]
        cur = loader(idx, steps_dict)
        if k_pos == 0:
            return cur, cur
        return 0.5 * (prev + cur), cur

    def load_M(idx, steps): return load_plane_y_vec(m_dir, steps[idx])
    def load_J(idx, steps): return load_plane_y_vec(j_dir, steps[idx])

    fig, gs = figure(2, 2, figsize=(16, 12))
    ax_R = subplot(fig, gs, 0, 0)
    ax_T = subplot(fig, gs, 1, 0)
    ax_dP = subplot(fig, gs, 0, 1)
    ax_PB = subplot(fig, gs, 1, 1)

    # Baseline (t = 0) profiles taken from the very first available
    # snapshot, regardless of --start-idx. J and M at the first frame use
    # raw values (no backward averaging — same convention as the loop's
    # k_pos = 0 case).
    idx0 = common[0]
    B_0 = load_plane_y_vec(b_dir, b_steps[idx0])
    M_0 = load_plane_y_vec(m_dir, m_steps[idx0])
    J_0 = load_plane_y_vec(j_dir, j_steps[idx0])
    q0 = step_quantities(B_0, J_0, M_0)
    P0 = q0["p_perp"] + q0["B2_half"]
    B0_2_half = q0["B2_half"].copy()

    n_drawn = 0
    n_avg = 0
    R_running_sum = None

    prev_M = None
    prev_J = None
    for k_pos, idx in enumerate(common):
        M_eff, prev_M = half_step_average(load_M, m_steps, k_pos, prev_M)
        J_eff, prev_J = half_step_average(load_J, j_steps, k_pos, prev_J)

        if idx < args.start_idx:
            continue

        B = load_plane_y_vec(b_dir, b_steps[idx])
        q = step_quantities(B, J_eff, M_eff)
        R_now = q["neg_dp_dx"] + q["JyBz"]
        P_total = q["p_perp"] + q["B2_half"]

        n_drawn += 1
        if args.time_average and idx >= tavg_start_idx:
            if R_running_sum is None:
                R_running_sum = R_now.copy()
            else:
                R_running_sum += R_now
            n_avg += 1
            R_avg = R_running_sum / n_avg

        deltaP = P_total - P0
        deltaB2 = B0_2_half - q["B2_half"]

        r_panel_lines = [(R_now, "tab:purple", 2.0, "-", r"$R(x)$")]
        if args.time_average and n_avg > 0:
            r_panel_lines.append(
                (R_avg, "k", 1.8, ":",
                 rf"$\langle R\rangle_t$  ($N={n_avg}$)"))

        YLIM_R = (-1.3e-4, 1.3e-4)
        YLIM_T = (-1.3e-4, 1.3e-4)
        YLIM_dP = (-1e-3, 10e-3)
        YLIM_PB = (-1e-3, 10e-3)

        panels = (
            (ax_R, r_panel_lines,
             r"$R(x) = -dp_{\perp}/dx + J_{\varphi} B_z$",
             YLIM_R),
            (ax_T,
             [(q["neg_dp_dx"], "tab:blue", 2.0, "-", r"$-dp_{\perp}/dx$"),
              (-q["JyBz"], "tab:red", 2.0, "-", r"$-J_{\varphi} B_z$")],
             r"force-balance",
             YLIM_T),
            (ax_dP,
             [(deltaP, "tab:green", 2.0, "-",
               r"$P(x)$")],
             r"$P(x)=(p_\perp + B^2/2) - (\cdot)|_{t=0}$",
             YLIM_dP),
            (ax_PB,
             [(q["p_perp"], "tab:blue", 2.0, "-", r"$p_{\perp}$"),
              (deltaB2, "tab:red", 2.0, "-",
               r"$B_0^2/2 - B^2/2$")],
             r"$p_{\perp}$ and  $B_0^2/2 - B^2/2$",
             YLIM_PB),
        )

        for ax, plots, title, ylim in panels:
            ax.clear()
            ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
            ax.axvline(cx, color="k", linewidth=0.6, alpha=0.3,
                       linestyle=":")
            for y, color, lw, ls, label in plots:
                ax.plot(xs, y, color=color, linewidth=lw, linestyle=ls,
                        label=label)
            ax.set_xlabel(r"$x,~c/\omega_{pe}$", fontsize=labelsize)
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(*ylim)
            ax.tick_params(labelsize=ticksize)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=labelsize, loc="best")
            ax.set_title(title, fontsize=labelsize, bbox=bbox)

        fig.suptitle(
            rf"$\omega_{{pe}}\,t = {idx * const.dts:.2f}$",
            x=0.5, y=0.995, bbox=bbox, fontsize=labelsize,
        )

        finite = np.isfinite(R_now)
        if finite.any():
            r_in = R_now[finite]
            print(f"  idx={idx}  R(x): "
                  f"min={r_in.min():+.3e}  max={r_in.max():+.3e}")

        figname = os.path.join(out_dir, f"{idx:04d}.png")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(figname, dpi=args.dpi)

    if n_drawn == 0:
        print(f"[warn] no snapshots with idx >= {args.start_idx}; "
              f"nothing was drawn.")

    print(f"Frames written to {out_dir}")


if __name__ == "__main__":
    main()
