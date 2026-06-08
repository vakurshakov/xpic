#!/usr/bin/env python3
"""Cylindrical pressure-tensor diagonals (Pi_rr, Pi_phi phi, Pi_zz) for
every species, plotted vs r.

For each species the script overlays two estimates per component:
  - "full"  — directly from the ``momentum_flux_diag_cyl`` diagnostic
              (component [0] = m * v_r^2, [1] = m * v_phi^2, [2] = m * v_z^2,
              accumulated with the same n/Np factor as other moments, so
              the stored field is already n * m * <v_i^2> = Pi_{ii}).
  - "flow"  — the cold-fluid / bulk-flow part inferred from J and n,
              Pi_{ii}^flow = m * <v_i>^2 * n
                           = m * (J_i / (q * n))^2 * n
                           = m * J_i^2 / (q^2 * n).
              Thermal part = full - flow.

Loading mirrors drift_kinetic_force_pressure_3D_phi.py: 3D fields are
z-reduced (avg or center) into 2D (Ny, Nx, ...), vectors projected to
cylindrical (r, phi, z) before binning, then averaged in phi over radial
bins. One figure per species, three panels (rr, phi phi, zz).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

TOOLS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from lib.constants import const, init_constants
from lib.plot import bbox, labelsize, ticksize

import matplotlib.pyplot as plt

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)

from drift_kinetic_equilibrium import auto_center, list_steps
from drift_kinetic_force_pressure_3D_y import find_field_dir, find_moment_dir
from drift_kinetic_force_pressure_3D_phi import (
    load_vec_zred,
    load_scalar_zred,
    project_phi,
    phi_average,
)


SPECIES = ("electrons", "ions")
COMPONENTS = ("rr", "phiphi", "zz")
COMPONENT_LABELS = {
    "rr":     r"$\Pi_{rr}$",
    "phiphi": r"$\Pi_{\varphi\varphi}$",
    "zz":     r"$\Pi_{zz}$",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--center", type=float, nargs=2, default=None,
                        metavar=("CX", "CY"),
                        help="Cylinder axis (cx, cy). Default: auto.")
    parser.add_argument("--z-mode", choices=("avg", "center"), default="avg")
    parser.add_argument("--mode", choices=("full", "J"), default="full",
                        help="full: overlay momentum_flux_diag_cyl ('full') "
                             "with the bulk-flow Pi_{ii}=m J_i^2/(q^2 n) and "
                             "their difference. J: skip the diagnostic and "
                             "plot only the J-based estimate (single curve "
                             "per component, all three components).")
    parser.add_argument("--r-max", type=float, default=None,
                        help="Max radius. Default: min(Lx, Ly) / 2.")
    parser.add_argument("--n-r", type=int, default=None,
                        help="Number of radial bins. Default: r_max / dx.")
    parser.add_argument("--idx", type=int, default=None,
                        help="Snapshot index to draw. Default: last common.")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out-subdir", default="pressure_cyl",
                        help="Subdir under <out_dir>/processed.")
    return parser.parse_args()


def load_pi_vec(path, name, z_mode):
    """3-component cell-centered DistributionMoment (Pi_rr, Pi_phi phi,
    Pi_zz), z-reduced to (Ny, Nx, 3)."""
    return load_vec_zred(path, name, z_mode)


def main():
    args = parse_args()
    init_constants(args.config)

    j_dirs = [find_field_dir(f"{s}/J") for s in SPECIES]
    n_dirs = [find_moment_dir(s, "density") for s in SPECIES]
    if args.mode == "full":
        pi_dirs = [find_moment_dir(s, "momentum_flux_diag_cyl")
                   for s in SPECIES]
    else:
        pi_dirs = [None] * len(SPECIES)

    required = []
    required += [(f"{s}/J", d) for s, d in zip(SPECIES, j_dirs)]
    required += [(f"{s}/density", d) for s, d in zip(SPECIES, n_dirs)]
    if args.mode == "full":
        required += [(f"{s}/momentum_flux_diag_cyl", d)
                     for s, d in zip(SPECIES, pi_dirs)]
    missing = [name for name, p in required if p is None]
    if missing:
        print(f"[error] missing diagnostics: {missing}")
        return

    vec_size = const.Nx * const.Ny * const.Nz * 3 * 4
    sca_size = const.Nx * const.Ny * const.Nz * 4
    j_steps = [dict(list_steps(d, vec_size)) for d in j_dirs]
    n_steps = [dict(list_steps(d, sca_size)) for d in n_dirs]
    if args.mode == "full":
        pi_steps = [dict(list_steps(d, vec_size)) for d in pi_dirs]
    else:
        pi_steps = [{} for _ in pi_dirs]

    common = set(j_steps[0])
    iter_sets = (*j_steps, *n_steps)
    if args.mode == "full":
        iter_sets = iter_sets + tuple(pi_steps)
    for s in iter_sets:
        common &= set(s)
    common = sorted(common)
    if not common:
        print("[error] no timesteps common to J / Pi / density of all species")
        return

    idx = args.idx if args.idx is not None else common[-1]
    if idx not in common:
        print(f"[error] idx {idx} not in common timesteps; "
              f"have {common[0]}..{common[-1]}")
        return
    print(f"Snapshot idx = {idx}  (t = {idx * const.dts:.4g})")

    if args.center is not None:
        cx, cy = float(args.center[0]), float(args.center[1])
    else:
        cx, cy = auto_center()
    print(f"Cylinder axis: ({cx:.3f}, {cy:.3f})  (z-mode: {args.z_mode})")

    cfg_particles = const.config.get("Particles", [])

    def species_param(name, key):
        for s in cfg_particles:
            if s.get("sort_name") == name:
                return float(s[key])
        raise KeyError(f"sort '{name}' / key '{key}' not in config")

    sp_q = [species_param(s, "q") for s in SPECIES]
    sp_m = [species_param(s, "m") for s in SPECIES]

    # Per-cell (x, y) → (r, theta).
    xs_2d = (np.arange(const.Nx)[None, :] + 0.5) * const.dx
    ys_2d = (np.arange(const.Ny)[:, None] + 0.5) * const.dy
    dx_2d = np.broadcast_to(xs_2d - cx, (const.Ny, const.Nx))
    dy_2d = np.broadcast_to(ys_2d - cy, (const.Ny, const.Nx))
    r2d = np.sqrt(dx_2d * dx_2d + dy_2d * dy_2d)
    theta_2d = np.arctan2(dy_2d, dx_2d)
    cos_th = np.cos(theta_2d)
    sin_th = np.sin(theta_2d)

    r_max = args.r_max if args.r_max is not None \
        else 0.5 * min(const.Lx, const.Ly)
    n_r = args.n_r if args.n_r is not None \
        else max(2, int(round(r_max / const.dx)))
    r_edges = np.linspace(0.0, r_max, n_r + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    species_results = {}
    for s, j_dir, pi_dir, n_dir, jst, pist, nst, q, m in zip(
            SPECIES, j_dirs, pi_dirs, n_dirs,
            j_steps, pi_steps, n_steps, sp_q, sp_m):
        J = load_vec_zred(j_dir, jst[idx], args.z_mode)        # (Ny, Nx, 3)
        n_2d = load_scalar_zred(n_dir, nst[idx], args.z_mode)  # (Ny, Nx)

        # Cartesian -> cylindrical projection for J.
        Jx, Jy, Jz = J[..., 0], J[..., 1], J[..., 2]
        J_r_2d = Jx * cos_th + Jy * sin_th
        J_phi_2d = project_phi(Jx, Jy, cos_th, sin_th)
        J_z_2d = Jz

        # Flow part Pi_{ii}^flow = m * J_i^2 / (q^2 * n_local).
        # n_local is the species density diagnostic; small values near the
        # boundary or in nearly-empty cells are masked to avoid blowups.
        n_safe = np.where(n_2d > 1e-12, n_2d, np.nan)
        pi_rr_flow_2d   = m * J_r_2d   * J_r_2d   / (q * q * n_safe)
        pi_phph_flow_2d = m * J_phi_2d * J_phi_2d / (q * q * n_safe)
        pi_zz_flow_2d   = m * J_z_2d   * J_z_2d   / (q * q * n_safe)

        def bin_r(field2d):
            return phi_average(field2d, r2d, r_edges)

        entry = {
            "rr_flow":     bin_r(pi_rr_flow_2d),
            "phiphi_flow": bin_r(pi_phph_flow_2d),
            "zz_flow":     bin_r(pi_zz_flow_2d),
        }
        if args.mode == "full":
            Pi = load_pi_vec(pi_dir, pist[idx], args.z_mode)   # (Ny, Nx, 3)
            entry["rr_full"]     = bin_r(Pi[..., 0])
            entry["phiphi_full"] = bin_r(Pi[..., 1])
            entry["zz_full"]     = bin_r(Pi[..., 2])
        species_results[s] = entry

    # One PNG per species, three panels per figure.
    for s in SPECIES:
        res = species_results[s]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
        for ax, comp in zip(axes, COMPONENTS):
            flow = res[f"{comp}_flow"]
            if args.mode == "full":
                full = res[f"{comp}_full"]
                therm = full - flow
                ax.plot(r_centers, full, color="black", linewidth=2.0,
                        label=COMPONENT_LABELS[comp] + r"$\ (\mathrm{full})$")
                ax.plot(r_centers, flow, color="tab:red", linewidth=2.0,
                        linestyle="--",
                        label=r"flow $= m J_i^2 / (q^2 n)$")
                ax.plot(r_centers, therm, color="tab:blue", linewidth=1.4,
                        linestyle=":",
                        label=r"thermal $=$ full $-$ flow")
            else:
                ax.plot(r_centers, flow, color="tab:red", linewidth=2.0,
                        label=COMPONENT_LABELS[comp]
                              + r"$ = m J_i^2 / (q^2 n)$")
            ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
            ax.set_xlim(0.0, r_max)
            ax.set_xlabel(r"$r\ (c/\omega_{pe})$", fontsize=labelsize)
            ax.set_ylabel(COMPONENT_LABELS[comp], fontsize=labelsize)
            ax.tick_params(labelsize=ticksize)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=ticksize)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                                useMathText=True)
        fig.suptitle(
            rf"{s}:  $\omega_{{pe}}\,t = {idx * const.dts:.2f}$",
            bbox=bbox, fontsize=labelsize)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        suffix = "_J" if args.mode == "J" else ""
        out_path = os.path.join(
            out_dir, f"pressure_cyl_{s}{suffix}_{idx:04d}.png")
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"  {s}: {out_path}")


if __name__ == "__main__":
    main()
