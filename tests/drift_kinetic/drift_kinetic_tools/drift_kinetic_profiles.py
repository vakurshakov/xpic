#!/usr/bin/env python3
"""Mean radial profiles of E_r, J_phi and B_z for drift_kinetic runs.

Reads plane-Z FieldView snapshots of E, J, B, projects E and J onto the
cylindrical basis around the cylinder axis (auto-detected the same way as
drift_kinetic_equilibrium.py), then azimuthally averages each quantity into
a 1D radial profile. All three profiles are drawn on a single figure with
three y-axes (E_r on the left, B_z on the right, J_phi on a second right
axis) sharing the same r axis.

J is stored half a step ahead of E/B (snap k holds J at step k - 1/2), so we
backward-average two consecutive snapshots, J_eff(step k) = (J[k-1] + J[k])/2;
the first frame uses the raw J (zero anyway).

One PNG per snapshot is written to <out_dir>/processed/profiles_<method>/.
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
    return parser.parse_args()


def main():
    args = parse_args()
    init_constants(args.config)

    e_dir = find_plane_z_dir("E")
    j_dir = find_plane_z_dir("J")
    b_dir = find_plane_z_dir("B")
    if any(d is None for d in (e_dir, j_dir, b_dir)):
        print(f"[error] need plane-Z FieldView for E, J, B "
              f"(got E={e_dir}, J={j_dir}, B={b_dir})")
        return

    plane_size = const.Nx * const.Ny * 3 * 4
    e_steps = dict(list_steps(e_dir, plane_size))
    j_steps = dict(list_steps(j_dir, plane_size))
    b_steps = dict(list_steps(b_dir, plane_size))

    common = sorted(set(e_steps) & set(j_steps) & set(b_steps))
    if not common:
        print("[error] no timesteps common to E/J/B planeZ")
        return

    cx, cy = args.center if args.center else auto_center()
    print(f"Cylinder axis: ({cx:.2f}, {cy:.2f})")

    xs = (np.arange(const.Nx) + 0.5) * const.dx
    ys = (np.arange(const.Ny) + 0.5) * const.dy
    X, Y = np.meshgrid(xs, ys)
    R = np.hypot(X - cx, Y - cy)
    Phi = np.arctan2(Y - cy, X - cx)
    cos_phi = np.cos(Phi)
    sin_phi = np.sin(Phi)

    r_max = args.r_max if args.r_max is not None \
        else 0.5 * min(const.Lx, const.Ly)
    n_bins = args.n_bins if args.n_bins is not None \
        else max(const.Nx, const.Ny) // 2
    edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    out_subdir = args.out_subdir or f"profiles_{args.method}"
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

    # J is at half-steps; backward-average to align with E and B.
    def half_step_average(loader, k_pos, prev):
        idx = common[k_pos]
        cur = loader(idx)
        if k_pos == 0:
            return cur, cur
        return 0.5 * (prev + cur), cur

    def load_J(idx):
        return load_plane_z_vec(j_dir, j_steps[idx])

    fig, gs = figure(1, 1)
    ax_E = subplot(fig, gs, 0, 0)
    ax_B = ax_E.twinx()
    ax_J = ax_E.twinx()
    # Offset the J axis so it doesn't overlap with the B axis.
    ax_J.spines["right"].set_position(("axes", 1.18))

    color_E = "tab:blue"
    color_B = "tab:red"
    color_J = "tab:green"

    prev_J = None
    for k_pos, idx in enumerate(common):
        E = load_plane_z_vec(e_dir, e_steps[idx])
        B = load_plane_z_vec(b_dir, b_steps[idx])
        J_eff, prev_J = half_step_average(load_J, k_pos, prev_J)

        E_r = E[..., 0] * cos_phi + E[..., 1] * sin_phi
        J_phi = -J_eff[..., 0] * sin_phi + J_eff[..., 1] * cos_phi
        B_z = B[..., 2]

        Er_r = average(E_r)
        Jphi_r = average(J_phi)
        Bz_r = average(B_z)

        ax_E.clear()
        ax_B.clear()
        ax_J.clear()

        ax_E.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        l_E, = ax_E.plot(r_centers, Er_r, color=color_E, linewidth=2,
                         label=r"$\langle E_r\rangle_{\varphi}$")
        l_B, = ax_B.plot(r_centers, Bz_r, color=color_B, linewidth=2,
                         label=r"$\langle B_z\rangle_{\varphi}$")
        l_J, = ax_J.plot(r_centers, Jphi_r, color=color_J, linewidth=2,
                         label=r"$\langle J_{\varphi}\rangle_{\varphi}$")

        ax_E.set_xlabel(r"$r,~c/\omega_{pe}$", fontsize=labelsize)
        ax_E.set_ylabel(r"$E_r$", color=color_E, fontsize=labelsize)
        ax_B.set_ylabel(r"$B_z$", color=color_B, fontsize=labelsize)
        ax_J.set_ylabel(r"$J_{\varphi}$", color=color_J, fontsize=labelsize)

        ax_E.tick_params(axis="y", labelcolor=color_E, labelsize=ticksize)
        ax_B.tick_params(axis="y", labelcolor=color_B, labelsize=ticksize)
        ax_J.tick_params(axis="y", labelcolor=color_J, labelsize=ticksize)
        ax_E.tick_params(axis="x", labelsize=ticksize)
        ax_E.set_xlim(0.0, r_max)
        ax_E.grid(True, alpha=0.3)

        lines = [l_E, l_B, l_J]
        ax_E.legend(lines, [ln.get_label() for ln in lines],
                    fontsize=labelsize, loc="best")

        fig.suptitle(
            rf"$\omega_{{pe}}\,t = {idx * const.dts:.2f}$",
            x=0.5, y=0.995, bbox=bbox, fontsize=labelsize,
        )

        for name, prof in (("E_r", Er_r), ("J_phi", Jphi_r), ("B_z", Bz_r)):
            finite = np.isfinite(prof)
            if finite.any():
                p = prof[finite]
                print(f"  idx={idx}  {name}: min={p.min():+.3e}  "
                      f"max={p.max():+.3e}")

        figname = os.path.join(out_dir, f"{idx:04d}.png")
        fig.tight_layout(rect=(0, 0, 1, 1))
        fig.savefig(figname, dpi=args.dpi)

    print(f"Frames written to {out_dir}")


if __name__ == "__main__":
    main()
