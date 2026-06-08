#!/usr/bin/env python3
"""Radial force-balance diagnostic for drift_kinetic runs.

For an axisymmetric, static equilibrium the radial component of the
momentum equation reduces (with the rho*E_r term dropped — quasi-neutral
plasma, drift-kinetic gyrocenters carry the polarization separately) to

    R(r) = - d p_perp / d r  +  J_phi^{total} * B_z   ==  0,

where p_perp = -M_z * B_z (z-component closure on the plane-Z slice; same
as in drift_kinetic_equilibrium.py) and the azimuthal current is the
*total* current J_total = J + curl M. On a plane-Z slice we lack
z-derivatives, so for an axisymmetric, axially-aligned configuration we
use (curl M)_x = dM_z/dy, (curl M)_y = -dM_z/dx.

Per cell on the plane-Z slice we form B_z, M_z, p_perp, the local radial
derivative dp_perp/dr (= dp/dx cos phi + dp/dy sin phi) and finally
R(x, y, t). The per-cell quantities are then azimuthally averaged as
single scalar profiles.

Each frame plots a 2x2 grid:

    top-left     : R(r) = -dp_perp/dr + J_phi^{total} * B_z (instantaneous)
                   and R_M(r) = -dp_perp/dr + (curl M)_phi * B_z;
    top-right    : the equilibrium force-balance terms separately,
                   -dp_perp/dr,   -J_phi^{total} * B_z   and
                   -(curl M)_phi * B_z (currents drawn with a flipped
                   sign so they sit on the same side as the pressure
                   gradient and visually balance it);
    bottom-left  : time-averaged R(r) and R_M(r), running mean from
                   --start-idx up to the current frame;
    bottom-right : cumulative radial integral of R(r) and R_M(r) at the
                   current step,  I(r) = int_0^r R(r') dr'.

J and M are stored half a step ahead of B/E (snap k holds them at step
k - 1/2), so we backward-average two consecutive snapshots,
X_eff(step k) = (X[k-1] + X[k]) / 2. The first frame uses the raw value
(zero anyway).

One PNG per snapshot is written to <out_dir>/processed/force_<method>/.
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

from drift_kinetic_equilibrium import (
    auto_center, find_plane_z_dir, list_steps, load_plane_z_vec,
    radial_average, polar_average,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--center", nargs=2, type=float, default=None,
                        metavar=("CX", "CY"),
                        help="Cylinder axis (default: auto-detect)")
    parser.add_argument("--n-bins", type=int, default=None,
                        help="Number of radial bins (default: max(Nx,Ny)//2)")
    parser.add_argument("--r-max", type=float, default=None,
                        help="Maximum radius for the profile [c/w_pe] "
                             "(default: half of min(Lx, Ly))")
    parser.add_argument("--method", choices=("dec", "pol"), default="dec",
                        help="dec: bin Cartesian cells; pol: polar bilinear "
                             "resampling with --n-phi samples per ring.")
    parser.add_argument("--n-phi", type=int, default=64,
                        help="Azimuthal samples per ring for --method pol")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out-subdir", default=None)
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Skip snapshots with idx < START_IDX before "
                             "drawing (default: 0). Snapshots before that "
                             "are still walked through to keep the half-step "
                             "backward averages of J/M consistent.")
    return parser.parse_args()


def main():
    args = parse_args()
    init_constants(args.config)

    b_dir = find_plane_z_dir("B")
    j_dir = find_plane_z_dir("J")
    m_dir = find_plane_z_dir("M")
    if any(d is None for d in (b_dir, j_dir, m_dir)):
        print(f"[error] need plane-Z FieldView for B, J, M "
              f"(got B={b_dir}, J={j_dir}, M={m_dir})")
        return

    plane_size = const.Nx * const.Ny * 3 * 4

    b_steps = dict(list_steps(b_dir, plane_size))
    j_steps = dict(list_steps(j_dir, plane_size))
    m_steps = dict(list_steps(m_dir, plane_size))

    common = sorted(set(b_steps) & set(j_steps) & set(m_steps))
    if not common:
        print("[error] no timesteps common to B/J/M planeZ")
        return

    cx, cy = args.center if args.center else auto_center()
    print(f"Cylinder axis: ({cx:.2f}, {cy:.2f})")

    xs = (np.arange(const.Nx) + 0.5) * const.dx
    ys = (np.arange(const.Ny) + 0.5) * const.dy
    X, Y = np.meshgrid(xs, ys)
    R = np.hypot(X - cx, Y - cy)
    # azimuth at every cell, used to project Cartesian E and J into radial
    # and azimuthal components.
    Phi = np.arctan2(Y - cy, X - cx)
    cos_phi = np.cos(Phi)
    sin_phi = np.sin(Phi)

    r_max = args.r_max if args.r_max is not None \
        else 0.5 * min(const.Lx, const.Ly)
    n_bins = args.n_bins if args.n_bins is not None \
        else max(const.Nx, const.Ny) // 2
    edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    out_subdir = args.out_subdir or f"force_{args.method}"
    out_dir = os.path.join(const.out_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Method: {args.method}"
          + (f"  (n_phi={args.n_phi})" if args.method == "pol" else ""))

    def average(scalar2d):
        if args.method == "dec":
            _, prof = radial_average(scalar2d, R, edges)
        else:
            prof = polar_average(scalar2d, cx, cy, r_centers, args.n_phi)
        return prof

    def step_R_profile(B, J_eff, M_eff):
        # 2D scalar fields at the z-center plane
        B_z = B[..., 2]
        M_z = M_eff[..., 2]

        # p_perp from MHD-style closure: -M_z * B_z (matches equilibrium.py)
        p_perp = -M_z * B_z

        # Magnetization (bound) current: J_M = curl M. On a plane-Z slice
        # we lack z-derivatives; for an axisymmetric, axially-aligned
        # configuration the in-plane components of M are negligible and
        # M_z dominates, so the in-plane components of curl M reduce to
        # (curl M)_x = dM_z/dy, (curl M)_y = -dM_z/dx.
        dMz_dy = np.gradient(M_z, const.dy, axis=0)
        dMz_dx = np.gradient(M_z, const.dx, axis=1)
        curlMx = dMz_dy
        curlMy = -dMz_dx
        Jx_tot = J_eff[..., 0] + curlMx
        Jy_tot = J_eff[..., 1] + curlMy
        J_phi_tot = -Jx_tot * sin_phi + Jy_tot * cos_phi
        curlM_phi = -curlMx * sin_phi + curlMy * cos_phi

        # Per-cell radial derivative of p_perp:
        #   dp/dr = dp/dx cos phi + dp/dy sin phi
        dp_dy = np.gradient(p_perp, const.dy, axis=0)
        dp_dx = np.gradient(p_perp, const.dx, axis=1)
        dp_dr = dp_dx * cos_phi + dp_dy * sin_phi

        # Equilibrium force-balance terms and their (signed) sums.
        term_grad_2d = -dp_dr
        term_jxb_2d = J_phi_tot * B_z
        term_mxb_2d = curlM_phi * B_z
        R_2d = term_grad_2d + term_jxb_2d
        R_M_2d = term_grad_2d + term_mxb_2d
        return (average(R_2d), average(R_M_2d),
                average(term_grad_2d),
                average(term_jxb_2d), average(term_mxb_2d))

    # M and J are stored at half-steps; backward-average two consecutive
    # snapshots to align with B. First frame: use raw values (zero anyway).
    def half_step_average(loader, steps_dict, k_pos, prev):
        idx = common[k_pos]
        cur = loader(idx, steps_dict)
        if k_pos == 0:
            return cur, cur
        return 0.5 * (prev + cur), cur

    def load_M(idx, steps): return load_plane_z_vec(m_dir, steps[idx])
    def load_J(idx, steps): return load_plane_z_vec(j_dir, steps[idx])

    fig, gs = figure(2, 2, figsize=(16, 12))
    ax_R = subplot(fig, gs, 0, 0)
    ax_T = subplot(fig, gs, 1, 0)
    ax_R_avg = subplot(fig, gs, 0, 1)
    ax_int = subplot(fig, gs, 1, 1)

    def cum_trapz_from_zero(y, x):
        # Cumulative trapezoid from r=0 to each r_centers[k]; treat
        # NaN bins as zero contribution so the integral stays defined.
        y = np.where(np.isfinite(y), y, 0.0)
        out = np.empty_like(y)
        out[0] = 0.5 * y[0] * x[0]
        increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
        out[1:] = out[0] + np.cumsum(increments)
        return out

    n_drawn = 0
    R_running_sum = None
    R_M_running_sum = None

    prev_M = None
    prev_J = None
    for k_pos, idx in enumerate(common):
        M_eff, prev_M = half_step_average(load_M, m_steps, k_pos, prev_M)
        J_eff, prev_J = half_step_average(load_J, j_steps, k_pos, prev_J)

        if idx < args.start_idx:
            continue

        B = load_plane_z_vec(b_dir, b_steps[idx])
        (R_prof, R_M_prof, grad_prof,
         jxb_prof, mxb_prof) = step_R_profile(B, J_eff, M_eff)
        n_drawn += 1

        if R_running_sum is None:
            R_running_sum = np.array(R_prof, copy=True)
            R_M_running_sum = np.array(R_M_prof, copy=True)
        else:
            R_running_sum += R_prof
            R_M_running_sum += R_M_prof
        R_avg = R_running_sum / n_drawn
        R_M_avg = R_M_running_sum / n_drawn

        R_int = cum_trapz_from_zero(R_prof, r_centers)
        R_M_int = cum_trapz_from_zero(R_M_prof, r_centers)

        ax_R.clear()
        ax_R.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_R.plot(r_centers, R_prof, color="tab:purple", linewidth=2,
                  label=r"$R(r)$ (full $J^{tot}$)")
        ax_R.plot(r_centers, R_M_prof, color="tab:green", linewidth=2,
                  linestyle="--",
                  label=r"$R_M(r)$ ($\nabla\!\times\!M$ only)")
        ax_R.set_xlabel(r"$r,~c/\omega_{pe}$", fontsize=labelsize)
        ax_R.set_xlim(0.0, r_max)
        ax_R.tick_params(labelsize=ticksize)
        ax_R.grid(True, alpha=0.3)
        ax_R.legend(fontsize=labelsize, loc="best")
        ax_R.set_title(
            r"$R(r) = -dp_{\perp}/dr + J_{\varphi}^{\bullet} B_z$",
            fontsize=labelsize, bbox=bbox)

        ax_T.clear()
        ax_T.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_T.plot(r_centers, grad_prof, color="tab:blue", linewidth=2,
                  label=r"$-dp_{\perp}/dr$")
        ax_T.plot(r_centers, -jxb_prof, color="tab:red", linewidth=2,
                  label=r"$-J_{\varphi}^{tot} B_z$")
        ax_T.plot(r_centers, -mxb_prof, color="tab:orange", linewidth=2,
                  linestyle="--",
                  label=r"$-(\nabla\!\times\!M)_{\varphi} B_z$")
        ax_T.set_xlabel(r"$r,~c/\omega_{pe}$", fontsize=labelsize)
        ax_T.set_xlim(0.0, r_max)
        ax_T.tick_params(labelsize=ticksize)
        ax_T.grid(True, alpha=0.3)
        ax_T.legend(fontsize=labelsize, loc="best")
        ax_T.set_title(
            r"force-balance terms (currents flipped)",
            fontsize=labelsize, bbox=bbox)

        ax_R_avg.clear()
        ax_R_avg.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_R_avg.plot(r_centers, R_avg, color="tab:purple", linewidth=2,
                      label=r"$\langle R(r)\rangle_t$ (full $J^{tot}$)")
        ax_R_avg.plot(r_centers, R_M_avg, color="tab:green", linewidth=2,
                      linestyle="--",
                      label=r"$\langle R_M(r)\rangle_t$ ($\nabla\!\times\!M$ only)")
        ax_R_avg.set_xlabel(r"$r,~c/\omega_{pe}$", fontsize=labelsize)
        ax_R_avg.set_xlim(0.0, r_max)
        ax_R_avg.tick_params(labelsize=ticksize)
        ax_R_avg.grid(True, alpha=0.3)
        ax_R_avg.legend(fontsize=labelsize, loc="best")
        ax_R_avg.set_title(
            rf"time-averaged $R(r)$,  $N={n_drawn}$,  "
            rf"start_idx$={args.start_idx}$",
            fontsize=labelsize, bbox=bbox)

        ax_int.clear()
        ax_int.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_int.plot(r_centers, R_int, color="tab:purple", linewidth=2,
                    label=r"$\int_0^r R(r')\,dr'$")
        ax_int.plot(r_centers, R_M_int, color="tab:green", linewidth=2,
                    linestyle="--",
                    label=r"$\int_0^r R_M(r')\,dr'$")
        ax_int.set_xlabel(r"$r,~c/\omega_{pe}$", fontsize=labelsize)
        ax_int.set_xlim(0.0, r_max)
        ax_int.tick_params(labelsize=ticksize)
        ax_int.grid(True, alpha=0.3)
        ax_int.legend(fontsize=labelsize, loc="best")
        ax_int.set_title(
            r"cumulative radial integral of $R$",
            fontsize=labelsize, bbox=bbox)

        fig.suptitle(
            rf"$\omega_{{pe}}\,t = {idx * const.dts:.2f}$",
            x=0.5, y=0.995, bbox=bbox, fontsize=labelsize,
        )

        finite = np.isfinite(R_prof)
        if finite.any():
            r_in = R_prof[finite]
            m_in = R_M_prof[np.isfinite(R_M_prof)]
            print(f"  idx={idx}  R(r): "
                  f"min={r_in.min():+.3e}  max={r_in.max():+.3e}  "
                  f"|  R_M(r): "
                  f"min={m_in.min():+.3e}  max={m_in.max():+.3e}")

        figname = os.path.join(out_dir, f"{idx:04d}.png")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(figname, dpi=args.dpi)

    if n_drawn == 0:
        print(f"[warn] no snapshots with idx >= {args.start_idx}; "
              f"nothing was drawn.")

    print(f"Frames written to {out_dir}")


if __name__ == "__main__":
    main()
