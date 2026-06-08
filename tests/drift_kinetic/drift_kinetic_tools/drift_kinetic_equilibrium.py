#!/usr/bin/env python3
"""Radial pressure-balance diagnostic for drift_kinetic runs.

For an axisymmetric, axially-aligned configuration the static MHD equilibrium
across magnetic surfaces is

    p_perp(r) + B_z^2(r) / 2 = const

(in xpic-normalized units, where mu_0 absorbs into the normalization).

This script reads the Z-plane cuts of B and M, takes their z-center slice and
forms the per-cell scalars

    p_perp = - M_z * B_z
    w_B    =   B_z^2 / 2

then averages each over the azimuthal angle around the cylinder center to get
1D radial profiles. The sum p_perp + w_B should be flat in r at equilibrium;
the script overlays the curves so any deviation is visible.

M is stored half a step ahead of B (snap k holds M at step k - 1/2), so we
backward-average two consecutive snapshots to align with B at integer steps.

One PNG per snapshot is written to <out_dir>/processed/equilibrium_<method>/.
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np

TOOLS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS_DIR)

from lib.plot_utils import figure, subplot
from lib.plot import bbox, labelsize, ticksize
from lib.constants import const, init_constants

import shutil
import matplotlib.pyplot as plt

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--center", nargs=2, type=float, default=None,
                        metavar=("CX", "CY"),
                        help="Cylinder axis in (x, y) [c/w_pe]. Default: "
                             "auto-detect from FieldsDamping/SetParticles "
                             "CylinderGeometry, else domain center")
    parser.add_argument("--n-bins", type=int, default=None,
                        help="Number of radial bins (default: max(Nx, Ny)//2)")
    parser.add_argument("--r-max", type=float, default=None,
                        help="Maximum radius for the profile [c/w_pe] "
                             "(default: half of min(Lx, Ly))")
    parser.add_argument("--method", choices=("dec", "pol"), default="dec",
                        help="dec: bin Cartesian cells by radius (fast, "
                             "noisy near r=0). pol: bilinear-interpolate "
                             "field on a polar grid with fixed n_phi samples "
                             "per ring (uniform precision in r, including "
                             "near r=0).")
    parser.add_argument("--n-phi", type=int, default=64,
                        help="Number of azimuthal samples per ring for "
                             "--method pol (default: 64)")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out-subdir", default=None,
                        help="Subdir under processed/ (default depends on "
                             "--method: equilibrium_dec or equilibrium_pol)")
    return parser.parse_args()


def auto_center():
    """Find cylinder axis from config: FieldsDamping → SetParticles → domain."""
    for preset in const.config.get("StepPresets", []) + const.config.get("Presets", []):
        cmd = preset.get("command")
        geom = None
        if cmd == "FieldsDamping":
            geom = preset.get("geometry", {})
        elif cmd == "SetParticles":
            geom = preset.get("coordinate", {})
        if geom and geom.get("name") == "CylinderGeometry" and "center" in geom:
            c = geom["center"]
            return float(c[0]), float(c[1])
    return 0.5 * const.Lx, 0.5 * const.Ly


def find_plane_z_dir(field: str):
    """Return path to the directory holding plane-Z FieldView frames for
    `field`, by scanning Diagnostics in config."""
    for diag in const.config.get("Diagnostics", []):
        if diag.get("diagnostic") != "FieldView":
            continue
        if diag.get("field") != field:
            continue
        region = diag.get("region", {})
        if region.get("type") != "2D" or region.get("plane") != "Z":
            continue
        sub = diag.get("out_dir", field)
        path = os.path.join(const.in_dir, sub)
        if os.path.isdir(path):
            return path
    # fallback to a few common naming patterns
    for sub in (f"{field}_planeZ", field, f"{field}_Z"):
        path = os.path.join(const.in_dir, sub)
        if os.path.isdir(path):
            return path
    return None


def find_density_dir(species: str):
    for diag in const.config.get("Diagnostics", []):
        if diag.get("diagnostic") != "DistributionMoment":
            continue
        if diag.get("particles") != species or diag.get("moment") != "density":
            continue
        sub = diag.get("out_dir", f"{species}/density")
        path = os.path.join(const.in_dir, sub)
        if os.path.isdir(path):
            return path
    path = os.path.join(const.in_dir, species, "density")
    return path if os.path.isdir(path) else None


def list_steps(dir_path: str, expected_size: int):
    out = []
    for name in os.listdir(dir_path):
        full = os.path.join(dir_path, name)
        if os.path.isfile(full) and name.isdigit() \
                and os.path.getsize(full) == expected_size:
            out.append((int(name), name))
    out.sort()
    return out


def load_plane_z_vec(path: str, name: str):
    """Plane-Z FieldView: shape (Ny, Nx, 3), float32."""
    h, w = const.Ny, const.Nx
    raw = np.fromfile(os.path.join(path, name), dtype=np.float32, count=h * w * 3)
    return raw.reshape(h, w, 3)


def load_density_3d_zcenter(path: str, name: str):
    """3D density: shape (Nz, Ny, Nx). Return z-center slice (Ny, Nx)."""
    nz, ny, nx = const.Nz, const.Ny, const.Nx
    raw = np.fromfile(os.path.join(path, name), dtype=np.float32, count=nz * ny * nx)
    return raw.reshape(nz, ny, nx)[nz // 2]


def radial_average(field2d: np.ndarray, r2d: np.ndarray, edges: np.ndarray):
    """Mean of field2d over cells whose radius falls in each bin.
    Returns (centers, mean_per_bin); empty bins are NaN."""
    flat_f = field2d.ravel()
    flat_r = r2d.ravel()
    sums = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1)
    idx = np.digitize(flat_r, edges) - 1
    mask = (idx >= 0) & (idx < len(sums))
    np.add.at(sums, idx[mask], flat_f[mask])
    np.add.at(counts, idx[mask], 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, means


def bilinear_periodic(field2d: np.ndarray, xq: np.ndarray, yq: np.ndarray):
    """Bilinear sample of `field2d` (shape Ny, Nx) at world points (xq, yq)
    [c/w_pe], assuming periodic boundaries in x and y. Returns array shaped
    like xq/yq with the interpolated values."""
    ny, nx = field2d.shape
    # cell-centered coordinates → fractional grid index
    u = xq / const.dx - 0.5
    v = yq / const.dy - 0.5
    i0 = np.floor(u).astype(np.int64)
    j0 = np.floor(v).astype(np.int64)
    wu = u - i0
    wv = v - j0
    i0m = i0 % nx
    j0m = j0 % ny
    i1m = (i0 + 1) % nx
    j1m = (j0 + 1) % ny
    f00 = field2d[j0m, i0m]
    f01 = field2d[j0m, i1m]
    f10 = field2d[j1m, i0m]
    f11 = field2d[j1m, i1m]
    return ((1 - wu) * (1 - wv) * f00 + wu * (1 - wv) * f01 +
            (1 - wu) * wv * f10 + wu * wv * f11)


def polar_average(field2d: np.ndarray, cx: float, cy: float,
                  r_arr: np.ndarray, n_phi: int):
    """Bilinear-interpolate field2d on a (n_r, n_phi) polar grid centered at
    (cx, cy), then average over phi. Returns array of length len(r_arr)."""
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    Rg, Phig = np.meshgrid(r_arr, phi, indexing="ij")  # (n_r, n_phi)
    xq = cx + Rg * np.cos(Phig)
    yq = cy + Rg * np.sin(Phig)
    F = bilinear_periodic(field2d, xq, yq)
    return F.mean(axis=1)


def main():
    args = parse_args()
    init_constants(args.config)

    b_dir = find_plane_z_dir("B")
    m_dir = find_plane_z_dir("M")
    if b_dir is None or m_dir is None:
        print(f"[error] need plane-Z FieldView for both B and M "
              f"(got B={b_dir}, M={m_dir})")
        return

    plane_size = const.Nx * const.Ny * 3 * 4
    b_steps = dict(list_steps(b_dir, plane_size))
    m_steps = dict(list_steps(m_dir, plane_size))

    common = sorted(set(b_steps) & set(m_steps))
    if not common:
        print("[error] no timesteps common to B planeZ and M planeZ")
        return

    cx, cy = args.center if args.center else auto_center()
    print(f"Cylinder axis: ({cx:.2f}, {cy:.2f})")

    # cell-centered (x, y) coordinates in physical units
    xs = (np.arange(const.Nx) + 0.5) * const.dx
    ys = (np.arange(const.Ny) + 0.5) * const.dy
    X, Y = np.meshgrid(xs, ys)
    R = np.hypot(X - cx, Y - cy)

    r_max = args.r_max if args.r_max is not None \
        else 0.5 * min(const.Lx, const.Ly)
    n_bins = args.n_bins if args.n_bins is not None \
        else max(const.Nx, const.Ny) // 2
    edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    out_subdir = args.out_subdir or f"equilibrium_{args.method}"
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

    def compute_profiles(B, M_eff):
        # p_perp from MHD-style closure: -M_z * B_z (z-components only)
        p_perp = -M_eff[..., 2] * B[..., 2]
        bz2 = B[..., 2] ** 2
        # For the equilibrium check we average the per-cell SUM
        # (p_perp + B_z^2/2) so any local cancellation between p_perp and
        # B_z^2/2 is preserved before the radial average. Components are
        # still averaged separately for the right-hand panel.
        p_r = average(p_perp)
        bz2_r = average(bz2)
        total_r = average(p_perp + 0.5 * bz2)
        return p_r, bz2_r, total_r

    # M is stored half a step ahead of B: snap k holds M at step k - 1/2. To
    # align with B at full step k we use M_eff(step k) = (M[k-1] + M[k]) / 2.
    # The first snapshot has no predecessor; M is exactly zero there at t=0.
    def m_eff_for(k_pos: int, prev_M):
        idx = common[k_pos]
        M_now = load_plane_z_vec(m_dir, m_steps[idx])
        if k_pos == 0:
            return M_now, M_now
        return 0.5 * (prev_M + M_now), M_now

    # Reference (t=0) profiles
    B0 = load_plane_z_vec(b_dir, b_steps[common[0]])
    M_eff_0, prev_M = m_eff_for(0, None)
    p0_r, bz2_0_r, total0 = compute_profiles(B0, M_eff_0)

    fig, gs = figure(2, 1, figsize=(20, 8))
    ax_total = subplot(fig, gs, 0, 0)
    ax_components = subplot(fig, gs, 1, 0)

    for k_pos, idx in enumerate(common):
        if k_pos == 0:
            B = B0
            M_eff = M_eff_0
        else:
            B = load_plane_z_vec(b_dir, b_steps[idx])
            M_eff, prev_M = m_eff_for(k_pos, prev_M)
        p_r, bz2_r, t_r = compute_profiles(B, M_eff)

        # left panel: total p_perp + B_z^2/2 (with t=0 reference)
        ax_total.clear()
        ax_total.plot(r_centers, t_r, color="tab:green", linewidth=2,
                      label=r"$p_{\perp} + B_z^2/2$")
        ax_total.plot(r_centers, total0, color="tab:gray", linewidth=1.5,
                      linestyle="--",
                      label=r"$(p_{\perp} + B_z^2/2)|_{t=0}$")
        ax_total.set_xlabel(r"$r,~c/\omega_{pe}$", fontsize=labelsize)
        ax_total.set_xlim(0.0, r_max)
        ax_total.tick_params(labelsize=ticksize)
        ax_total.grid(True, alpha=0.3)
        ax_total.legend(fontsize=labelsize, loc="best")
        ax_total.set_title(r"$p_{\perp} + B_z^2/2$",
                           fontsize=labelsize, bbox=bbox)

        # right panel: each component separately
        ax_components.clear()
        ax_components.plot(r_centers, p_r, color="tab:blue", linewidth=2,
                           label=r"$p_{\perp}$")
        ax_components.plot(r_centers, 0.5 * bz2_r, color="tab:red", linewidth=2,
                           label=r"$B_z^2/2$")
        ax_components.set_xlabel(r"$r,~c/\omega_{pe}$", fontsize=labelsize)
        ax_components.set_xlim(0.0, r_max)
        ax_components.tick_params(labelsize=ticksize)
        ax_components.grid(True, alpha=0.3)
        ax_components.legend(fontsize=labelsize, loc="best")
        ax_components.set_title(r"components: $p_{\perp}$ and $B_z^2/2$",
                                fontsize=labelsize, bbox=bbox)

        fig.suptitle(
            rf"$\omega_{{pe}}\,t = {idx * const.dts:.2f}$",
            x=0.5, y=0.995, bbox=bbox, fontsize=labelsize,
        )

        t_delta = t_r - total0
        finite = np.isfinite(t_delta)
        if finite.any():
            d_in = t_delta[finite]
            print(f"  idx={idx}  delta(p+B_z^2/2): "
                  f"min={d_in.min():+.3e}  max={d_in.max():+.3e}")

        figname = os.path.join(out_dir, f"{idx:04d}.png")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(figname, dpi=args.dpi)

    print(f"Frames written to {out_dir}")


if __name__ == "__main__":
    main()
