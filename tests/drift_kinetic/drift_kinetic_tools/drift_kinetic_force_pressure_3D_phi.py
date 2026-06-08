#!/usr/bin/env python3
"""Two-panel (force-balance, magnetic-field profile) diagnostic computed
from 3D FieldView and DistributionMoment outputs and *azimuthally*
averaged around the cylinder axis.

Mirrors drift_kinetic_force_pressure_3D_y.py panel-for-panel, but instead
of taking a y-slice at y = Ny/2 (so profiles vs x), the 3D fields are
reduced over z (avg or center) and then averaged in phi about the
cylinder axis (cx, cy). All profiles are plotted vs r.

Vector quantities are projected to cylindrical (r, phi, z) before
binning, so the φ-component of J / rotM is the true azimuthal
component, not just J_y. p_phiphi is the per-cell flow stress
sum_s m_s * J_phi_s^2 / (q_s^2 * n_s) (J_phi_s computed by the same
projection), then averaged in phi.
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
from lib.plot import bbox, labelsize

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)

from drift_kinetic_equilibrium import auto_center, list_steps
from drift_kinetic_force_pressure_y import draw_frame
from drift_kinetic_force_pressure_3D_y import (
    SPECIES,
    find_field_dir,
    find_moment_dir,
)


def load_vec_zred(path: str, name: str, z_mode: str) -> np.ndarray:
    """Read a 3D vector FieldView (Nz, Ny, Nx, 3) and reduce along z.
    ``z_mode == 'avg'`` → mean over z, ``'center'`` → z = Nz // 2. Returns
    array of shape (Ny, Nx, 3)."""
    Nx, Ny, Nz = const.Nx, const.Ny, const.Nz
    raw = np.fromfile(os.path.join(path, name), dtype=np.float32,
                      count=Nx * Ny * Nz * 3)
    arr = raw.reshape(Nz, Ny, Nx, 3)
    if z_mode == "avg":
        return arr.mean(axis=0)
    return arr[Nz // 2]


def load_scalar_zred(path: str, name: str, z_mode: str) -> np.ndarray:
    """Same as ``load_vec_zred`` but for a 3D scalar DistributionMoment."""
    Nx, Ny, Nz = const.Nx, const.Ny, const.Nz
    raw = np.fromfile(os.path.join(path, name), dtype=np.float32,
                      count=Nx * Ny * Nz)
    arr = raw.reshape(Nz, Ny, Nx)
    if z_mode == "avg":
        return arr.mean(axis=0)
    return arr[Nz // 2]


def sum_species_zred(dirs, steps, idx, z_mode, loader):
    out = None
    for d, s in zip(dirs, steps):
        cur = loader(d, s[idx], z_mode)
        out = cur if out is None else out + cur
    return out


def project_phi(Vx_2d, Vy_2d, cos_th, sin_th):
    """Azimuthal projection of an (x, y) vector field at every cell:
    V_phi = -Vx sin(theta) + Vy cos(theta)."""
    return -Vx_2d * sin_th + Vy_2d * cos_th


def phi_average(field2d, r2d, r_edges):
    """Bin a 2D (Ny, Nx) field over radial bins and return the per-bin
    mean. Empty bins return NaN."""
    flat_f = field2d.ravel()
    flat_r = r2d.ravel()
    nb = len(r_edges) - 1
    sums = np.zeros(nb)
    counts = np.zeros(nb)
    idx = np.digitize(flat_r, r_edges) - 1
    mask = (idx >= 0) & (idx < nb) & np.isfinite(flat_f)
    np.add.at(sums, idx[mask], flat_f[mask])
    np.add.at(counts, idx[mask], 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def step_quantities(B, J_eff, J_eff_sp, rotM_eff, p_perp_2d,
                    cos_th, sin_th, r2d, r_edges, r_centers, dr,
                    sp_q, sp_m, sp_n):
    """Compute the same dict drift_kinetic_force_pressure_y.draw_frame
    expects, but with every entry already a radial profile vs r."""
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
    B2_half_2d = 0.5 * (Bx * Bx + By * By + Bz * Bz)

    curlM_phi_2d = project_phi(rotM_eff[..., 0], rotM_eff[..., 1],
                               cos_th, sin_th)
    J_phi_tot_2d = project_phi(J_eff[..., 0], J_eff[..., 1],
                               cos_th, sin_th) + curlM_phi_2d

    p_phiphi_2d = np.zeros_like(p_perp_2d)
    for J_s, q_s, m_s, n_s in zip(J_eff_sp, sp_q, sp_m, sp_n):
        J_phi_s = project_phi(J_s[..., 0], J_s[..., 1], cos_th, sin_th)
        p_phiphi_2d += m_s * J_phi_s * J_phi_s / (q_s * q_s * n_s)

    # Phi-averaged 1D profiles.
    p_perp_r = phi_average(p_perp_2d, r2d, r_edges)
    p_phiphi_r = phi_average(p_phiphi_2d, r2d, r_edges)
    B2_half_r = phi_average(B2_half_2d, r2d, r_edges)
    Bz_r = phi_average(Bz, r2d, r_edges)
    J_phi_tot_r = phi_average(J_phi_tot_2d, r2d, r_edges)
    curlM_phi_r = phi_average(curlM_phi_2d, r2d, r_edges)

    dp_dr = np.gradient(p_perp_r, dr)
    return {
        "p_perp": p_perp_r,
        "p_phiphi": p_phiphi_r,
        "B2_half": B2_half_r,
        "neg_dp_dx": -dp_dr,
        "JyBz": J_phi_tot_r * Bz_r,
        "curlM_Bz": curlM_phi_r * Bz_r,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--center", type=float, nargs=2, default=None,
                        metavar=("CX", "CY"),
                        help="Cylinder axis (cx, cy). Default: auto from "
                             "config.")
    parser.add_argument("--z-mode", choices=("avg", "center"), default="avg",
                        help="Reduction along z before phi-averaging. "
                             "'avg' uses the z-mean, 'center' the z = Nz//2 "
                             "plane.")
    parser.add_argument("--r-max", type=float, default=None,
                        help="Maximum radius (c/w_pe) included in profiles. "
                             "Default: min(Lx, Ly) / 2.")
    parser.add_argument("--n-r", type=int, default=None,
                        help="Number of radial bins. Default: "
                             "round(r_max / dx).")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="First diagnostic-frame index to draw "
                             "(inclusive). Earlier frames are still "
                             "iterated to keep the half-step backward "
                             "averages of J / rotM consistent.")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Last diagnostic-frame index to draw "
                             "(inclusive). Default: last available frame.")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--anim-fps", type=int, default=10)
    parser.add_argument("--anim-name", default="force_pressure_3D_phi.mp4",
                        help="MP4 filename inside <out-subdir>")
    parser.add_argument("--mean-name", default="time_mean.png",
                        help="Time-averaged figure filename inside "
                             "<out-subdir>")
    parser.add_argument("--out-subdir", default="force_pressure_3D_phi",
                        help="Subdirectory under <out_dir>/processed for "
                             "PNGs / MP4")
    return parser.parse_args()


def main():
    args = parse_args()
    init_constants(args.config)

    b_dir = find_field_dir("B")
    j_dirs = [find_field_dir(f"{s}/J") for s in SPECIES]
    r_dirs = [find_field_dir(f"{s}/rotM") for s in SPECIES]
    t_dirs = [find_moment_dir(s, "temperature_perp") for s in SPECIES]

    required = [("B", b_dir)]
    required += [(f"{s}/J", d) for s, d in zip(SPECIES, j_dirs)]
    required += [(f"{s}/rotM", d) for s, d in zip(SPECIES, r_dirs)]
    required += [(f"{s}/temperature_perp", d)
                 for s, d in zip(SPECIES, t_dirs)]
    missing = [name for name, p in required if p is None]
    if missing:
        print(f"[error] missing diagnostics: {missing}")
        return

    vec_size = const.Nx * const.Ny * const.Nz * 3 * 4
    sca_size = const.Nx * const.Ny * const.Nz * 4
    b_steps = dict(list_steps(b_dir, vec_size))
    j_steps = [dict(list_steps(d, vec_size)) for d in j_dirs]
    r_steps = [dict(list_steps(d, vec_size)) for d in r_dirs]
    t_steps = [dict(list_steps(d, sca_size)) for d in t_dirs]

    common = set(b_steps)
    for s in (*j_steps, *r_steps, *t_steps):
        common &= set(s)
    common = sorted(common)
    if not common:
        print("[error] no timesteps common to all required diagnostics")
        return

    end_idx = args.end_idx if args.end_idx is not None else common[-1]
    if args.start_idx > end_idx:
        print(f"[error] start_idx {args.start_idx} > end_idx {end_idx}")
        return

    if args.center is not None:
        cx, cy = float(args.center[0]), float(args.center[1])
    else:
        cx, cy = auto_center()
    print(f"Cylinder axis: ({cx:.3f}, {cy:.3f})   "
          f"(z-mode: {args.z_mode}, start: {args.start_idx}, end: {end_idx})")

    # Per-species (q, m, n) from config, same convention as the y-slice
    # variant: J = q * n * <v>, so V_phi_s = J_phi_s / (q_s * n_s).
    cfg_particles = const.config.get("Particles", [])

    def species_param(name, key):
        for s in cfg_particles:
            if s.get("sort_name") == name:
                return float(s[key])
        raise KeyError(f"sort '{name}' / key '{key}' not in config")

    sp_q = [species_param(s, "q") for s in SPECIES]
    sp_m = [species_param(s, "m") for s in SPECIES]
    sp_n = [species_param(s, "n") for s in SPECIES]
    sp_Np = [species_param(s, "Np") for s in SPECIES]
    for s, q, m, n, Np in zip(SPECIES, sp_q, sp_m, sp_n, sp_Np):
        print(f"  {s}: q={q:+g} m={m:g} n={n:g} Np={Np:g}")

    # Per-cell (x, y) → (r, theta).
    xs_2d = (np.arange(const.Nx)[None, :] + 0.5) * const.dx
    ys_2d = (np.arange(const.Ny)[:, None] + 0.5) * const.dy
    dx_2d = xs_2d - cx
    dy_2d = ys_2d - cy
    r2d = np.sqrt(dx_2d * dx_2d + dy_2d * dy_2d)
    theta_2d = np.arctan2(dy_2d, dx_2d)
    cos_th = np.cos(theta_2d)
    sin_th = np.sin(theta_2d)
    # Broadcast (Ny, Nx) → (Ny, Nx) for stacking with components.
    cos_th = np.broadcast_to(cos_th, (const.Ny, const.Nx))
    sin_th = np.broadcast_to(sin_th, (const.Ny, const.Nx))
    r2d = np.broadcast_to(r2d, (const.Ny, const.Nx))

    r_max = args.r_max if args.r_max is not None \
        else 0.5 * min(const.Lx, const.Ly)
    n_r = args.n_r if args.n_r is not None \
        else max(2, int(round(r_max / const.dx)))
    r_edges = np.linspace(0.0, r_max, n_r + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = float(r_edges[1] - r_edges[0])
    print(f"Radial bins: n_r = {n_r}, r in [0, {r_max:.3f}], dr = {dr:.4g}")

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    def load_J_per_species(idx):
        return [load_vec_zred(d, s[idx], args.z_mode)
                for d, s in zip(j_dirs, j_steps)]

    def load_J(idx):
        return sum_species_zred(j_dirs, j_steps, idx, args.z_mode,
                                load_vec_zred)

    def load_rotM(idx):
        return sum_species_zred(r_dirs, r_steps, idx, args.z_mode,
                                load_vec_zred)

    def load_p_perp(idx):
        # temperature_perp moment stores n*m*<v_perp^2> = 2*p_perp, so the
        # per-species pressure is 0.5 * temperature_perp; summed over sorts.
        p = None
        for t_dir, t_st in zip(t_dirs, t_steps):
            T = load_scalar_zred(t_dir, t_st[idx], args.z_mode)
            term = 0.5 * T
            p = term if p is None else p + term
        return p

    idx0 = common[0]
    B_0 = load_vec_zred(b_dir, b_steps[idx0], args.z_mode)
    Bx0, By0, Bz0 = B_0[..., 0], B_0[..., 1], B_0[..., 2]
    B0_2_half_2d = 0.5 * (Bx0 * Bx0 + By0 * By0 + Bz0 * Bz0)
    B0_2_half = phi_average(B0_2_half_2d, r2d, r_edges)

    drawn = []
    prev_J = None
    prev_J_sp = None
    prev_rotM = None
    for k, idx in enumerate(common):
        if idx > end_idx:
            break
        cur_J = load_J(idx)
        cur_J_sp = load_J_per_species(idx)
        cur_rotM = load_rotM(idx)
        if k == 0:
            J_eff = cur_J
            J_eff_sp = cur_J_sp
            rotM_eff = cur_rotM
        else:
            J_eff = 0.5 * (prev_J + cur_J)
            J_eff_sp = [0.5 * (pj + cj)
                        for pj, cj in zip(prev_J_sp, cur_J_sp)]
            rotM_eff = 0.5 * (prev_rotM + cur_rotM)
        prev_J, prev_J_sp, prev_rotM = cur_J, cur_J_sp, cur_rotM

        if idx < args.start_idx:
            continue

        B = load_vec_zred(b_dir, b_steps[idx], args.z_mode)
        p_perp = load_p_perp(idx)

        q = step_quantities(B, J_eff, J_eff_sp, rotM_eff, p_perp,
                            cos_th, sin_th, r2d, r_edges, r_centers, dr,
                            sp_q, sp_m, sp_n)
        drawn.append((idx, q))

    if not drawn:
        print(f"[warn] no snapshots in [{args.start_idx}, {end_idx}].")
        return

    fig, (ax_F, ax_P) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Hoop-stress fold (same as the y-slice variant), but in r.
    r_mask = r_centers > 0.5 * dr

    def fold_hoop_into_grad(q):
        delta = np.zeros_like(q["p_phiphi"])
        delta[r_mask] = q["p_phiphi"][r_mask] / r_centers[r_mask]
        out = dict(q)
        out["neg_dp_perp_dx"] = q["neg_dp_dx"]
        out["neg_dp_dx"] = q["neg_dp_dx"] + delta
        return out

    def relabel_axes():
        # draw_frame hardcodes "x" on the bottom panel and uses cx to draw
        # a vertical "axis" marker. Both panels share x, so updating ax_P
        # is sufficient. The vertical line at cx ends up at r = 0 — still
        # the axis, so semantically correct.
        ax_P.set_xlabel(r"$r~(c/\omega_{pe})$",
                        fontsize=labelsize + 4)

    def render(k):
        idx, q = drawn[k]
        draw_frame(ax_F, ax_P, r_centers, 0.0, fold_hoop_into_grad(q),
                   B0_2_half, time_avg=True,
                   grad_label=r"$-dp_{\perp}/dr + \Pi_{\varphi\varphi}/r$")
        relabel_axes()
        fig.suptitle(rf"$\omega_{{pe}}\,t = {idx * const.dts:.2f}$",
                     bbox=bbox, fontsize=labelsize)
        figname = os.path.join(out_dir, f"{idx:04d}.png")
        print(f"Processing {figname} (idx={idx})")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(figname, dpi=args.dpi)
        return [ax_F, ax_P]

    anim = FuncAnimation(fig, render, frames=len(drawn), blit=False)
    anim_path = os.path.join(out_dir, args.anim_name)
    anim.save(anim_path, writer=FFMpegWriter(fps=args.anim_fps),
              dpi=args.dpi)

    keys = ("p_perp", "p_phiphi", "B2_half", "neg_dp_dx", "JyBz", "curlM_Bz")
    accum = {key: np.zeros_like(drawn[0][1][key]) for key in keys}
    for _, q in drawn:
        for key in keys:
            accum[key] += q[key]
    mean_q = {key: v / len(drawn) for key, v in accum.items()}

    draw_frame(ax_F, ax_P, r_centers, 0.0, fold_hoop_into_grad(mean_q),
               B0_2_half, time_avg=True,
               grad_label=r"$-dp_{\perp}/dr + \Pi_{\varphi\varphi}/r$")
    relabel_axes()
    fig.suptitle("")
    fig.tight_layout()
    mean_path = os.path.join(out_dir, args.mean_name)
    fig.savefig(mean_path, dpi=args.dpi)
    plt.close(fig)

    last_q = drawn[-1][1]
    grad = fold_hoop_into_grad(last_q)["neg_dp_dx"]
    grad_scale = float(np.linalg.norm(grad))
    res_J = grad + last_q["JyBz"]
    res_M = grad + last_q["curlM_Bz"]
    err_J = float(np.linalg.norm(res_J) / grad_scale * 100.0) \
        if grad_scale > 0.0 else float("nan")
    err_M = float(np.linalg.norm(res_M) / grad_scale * 100.0) \
        if grad_scale > 0.0 else float("nan")

    B0_sq = 2.0 * B0_2_half
    B_theory = np.sqrt(np.maximum(B0_sq - 2.0 * last_q["p_perp"], 0.0))
    B_model = np.sqrt(np.maximum(2.0 * last_q["B2_half"], 0.0))
    diff_B = np.abs(B_model - B_theory)
    i_max = int(np.argmax(diff_B))

    i_center = 0
    B0_center = float(np.sqrt(2.0 * B0_2_half[i_center]))
    B_theory_center = float(B_theory[i_center])
    theory_drop = B0_center - B_theory_center
    err_B = float(diff_B[i_max] / theory_drop * 100.0) \
        if theory_drop > 0.0 else float("nan")

    print(f"Frames written to {out_dir}")
    print(f"Animation: {anim_path}")
    print(f"Time-mean: {mean_path}")
    print(f"Errors evaluated on last frame (idx = {drawn[-1][0]})")
    print(f"Force-balance error (||res||_2 / ||grad p||_2):")
    print(f"  J_phi B_z         : {err_J:+.3f}%")
    print(f"  (rot M)_phi B_z   : {err_M:+.3f}%")
    print(f"Equilibrium error max|B_model - B_theory| / (B0 - B_theory) at "
          f"r=0: {err_B:+.3f}%  (drop = {theory_drop:.4g}, "
          f"max diff = {float(diff_B[i_max]):.4g} "
          f"at r={float(r_centers[i_max]):.3f})")


if __name__ == "__main__":
    main()
