#!/usr/bin/env python3
"""Two-panel (force-balance, magnetic-field profile) diagnostic computed
from 3D FieldView and DistributionMoment outputs, sliced to plane Y.

Mirrors drift_kinetic_force_pressure_y.py panel-for-panel, keeping the
animation/time-mean PNG outputs and the same styling. The only changes are
the data sources:
    - 3D B           (FieldView "B")
    - 3D J_total     = sum_species (FieldView "<sort>/J")
    - 3D rotM_total  = sum_species (FieldView "<sort>/rotM")
    - 3D temperature_perp per species (DistributionMoment)
all sliced at ``y = --y-idx`` (default Ny // 2). The pressure used in the
force balance is the particle-moment estimate
    p_perp(x, z) = 0.5 * sum_species temperature_perp_s(x, z)
rather than the MHD closure -M·B used in the 2D version. The factor 0.5 is
because the ``temperature_perp`` moment is *not* a per-particle temperature:
distribution_moment.cpp accumulates ``m * v_perp^2`` weighted by the
particle's grid contribution without dividing by density, so the stored
field is already n * m * <v_perp^2> = 2 * p_perp. Multiplying it by the
density diagnostic would double-count n and make p_perp exceed B0^2 / 2.

J and rotM live half a step ahead of B (they are evaluated at n + 1/2),
so consecutive snapshots are backward-averaged
    X_eff(k) = (X[k-1] + X[k]) / 2
to put them at integer step n. The very first frame uses the raw values
(both are zero at t = 0 in equilibrium anyway).
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
from matplotlib.animation import FFMpegWriter, FuncAnimation

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)

from drift_kinetic_equilibrium import auto_center, list_steps
from drift_kinetic_force_pressure_y import draw_frame


SPECIES = ("electrons", "ions")


def find_field_dir(field_name: str):
    """Return the directory of the 3D FieldView with ``field == field_name``,
    or ``None`` if no matching diagnostic is configured. 2D entries are
    skipped — only full-domain (or explicitly 3D) regions qualify."""
    for diag in const.config.get("Diagnostics", []):
        if diag.get("diagnostic") != "FieldView":
            continue
        if diag.get("field") != field_name:
            continue
        region = diag.get("region", {})
        if region.get("type") == "2D":
            continue
        sub = diag.get("out_dir", field_name)
        path = os.path.join(const.in_dir, sub)
        if os.path.isdir(path):
            return path
    return None


def find_moment_dir(species: str, moment: str):
    for diag in const.config.get("Diagnostics", []):
        if diag.get("diagnostic") != "DistributionMoment":
            continue
        if diag.get("particles") != species:
            continue
        if diag.get("moment") != moment:
            continue
        sub = diag.get("out_dir", f"{species}/{moment}")
        path = os.path.join(const.in_dir, sub)
        if os.path.isdir(path):
            return path
    return None


def load_vec_slice_y(path: str, name: str, y_idx: int,
                     avg_zy: bool = False) -> np.ndarray:
    """Read a 3D vector FieldView and reduce it to (Nz, Nx, 3). By default
    the y = ``y_idx`` slice is returned; with ``avg_zy`` the field is
    averaged over y instead (``y_idx`` is ignored)."""
    Nx, Ny, Nz = const.Nx, const.Ny, const.Nz
    raw = np.fromfile(os.path.join(path, name), dtype=np.float32,
                      count=Nx * Ny * Nz * 3)
    arr = raw.reshape(Nz, Ny, Nx, 3)
    return arr.mean(axis=1) if avg_zy else arr[:, y_idx, :, :]


def load_scalar_slice_y(path: str, name: str, y_idx: int,
                        avg_zy: bool = False) -> np.ndarray:
    """Read a 3D scalar DistributionMoment file and reduce it to (Nz, Nx).
    By default the y = ``y_idx`` slice is returned; with ``avg_zy`` the
    field is averaged over y instead (``y_idx`` is ignored)."""
    Nx, Ny, Nz = const.Nx, const.Ny, const.Nz
    raw = np.fromfile(os.path.join(path, name), dtype=np.float32,
                      count=Nx * Ny * Nz)
    arr = raw.reshape(Nz, Ny, Nx)
    return arr.mean(axis=1) if avg_zy else arr[:, y_idx, :]


def sum_species(dirs, steps, idx, y_idx, loader, avg_zy=False):
    out = None
    for d, s in zip(dirs, steps):
        cur = loader(d, s[idx], y_idx, avg_zy)
        out = cur if out is None else out + cur
    return out


def central_diff_x(field2d, dx):
    """First derivative along x (axis = 1) using the centered stencil
    (f[i+1] - f[i-1]) / (2 dx), with one-sided differences at the two
    x-boundaries."""
    d = np.empty_like(field2d)
    d[..., 1:-1] = (field2d[..., 2:] - field2d[..., :-2]) / (2.0 * dx)
    d[..., 0] = (field2d[..., 1] - field2d[..., 0]) / dx
    d[..., -1] = (field2d[..., -1] - field2d[..., -2]) / dx
    return d


def shift_right_x(profile):
    """Shift a 1D x-profile one cell to the right (profile[i] <- profile[i-1]).
    The first cell keeps its original value instead of wrapping around.

    Used to realign the pressure-gradient curve with the J*B curve on the
    force-balance panel: the moment pressure sits on the integer x-node while
    the Yee-staggered B/J (FieldView) effectively land half a cell to its
    right, so the pressure side reads one cell left of the current side."""
    s = np.roll(profile, 1)
    s[0] = profile[0]
    return s


def step_quantities(B, J_eff, rotM_eff, p_perp_2d, p_phiphi_2d, z_mode,
                    grad_pressure_2d=None):
    """Per-frame reductions matching drift_kinetic_force_pressure_y's dict
    layout so ``draw_frame`` can render the result unchanged. ``rotM_eff``
    is the species-summed simulation diagnostic (∇×M), so its y-component
    replaces the finite-difference curl from the 2D variant.

    ``p_perp_2d`` is the perpendicular-pressure scalar from
    temperature_perp; it always feeds the bottom panel's B_theory.
    ``p_phiphi_2d`` is the φφ-component computed from the bulk azimuthal
    flow J_y/(q*n) summed over species; the +p_φφ/r centrifugal/hoop-
    stress term lives separately in the returned dict so the top panel
    can overlay it. ``grad_pressure_2d`` (default = ``p_perp_2d``) is the
    field whose radial gradient enters the force-balance curve on the top
    panel; passing Π_{rr} here lets the caller swap the pressure used for
    the gradient without changing the B_theory baseline."""
    if grad_pressure_2d is None:
        grad_pressure_2d = p_perp_2d
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
    B2_half_2d = 0.5 * (Bx * Bx + By * By + Bz * Bz)
    curlM_y_2d = rotM_eff[..., 1]

    Jy_tot_2d = J_eff[..., 1] + curlM_y_2d
    dp_dx_2d = central_diff_x(grad_pressure_2d, const.dx)

    def red(field2d):
        if z_mode == "avg":
            return field2d.mean(axis=0)
        return field2d[const.Nz // 2]

    return {
        "p_perp": red(p_perp_2d),
        "p_phiphi": red(p_phiphi_2d),
        "B2_half": red(B2_half_2d),
        "neg_dp_dx": red(-dp_dx_2d),
        "JyBz": red(Jy_tot_2d * Bz),
        "curlM_Bz": red(curlM_y_2d * Bz),
        # Bare gyrocenter J_phi (no curl-M) and bare (rot M)_phi, both summed
        # over species in the upstream `sum_species` calls. Reduced to 1D.
        "J_phi": red(J_eff[..., 1]),
        "curlM_phi": red(curlM_y_2d),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--center-x", type=float, default=None,
                        help="Cylinder axis x-coordinate (default: auto)")
    parser.add_argument("--y-idx", type=int, default=None,
                        help="Cell index along y where the 3D fields are "
                             "sliced. Default: Ny // 2.")
    parser.add_argument("--z-mode", choices=("avg", "center"), default="avg",
                        help="Reduction of the (Nz, Nx) slice to a 1D "
                             "profile vs x. 'avg' averages along z; "
                             "'center' uses Nz // 2.")
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
    parser.add_argument("--anim-name", default="force_pressure_3D_y.mp4",
                        help="MP4 filename inside <out-subdir>")
    parser.add_argument("--mean-name", default="time_mean.png",
                        help="Time-averaged figure filename inside "
                             "<out-subdir>")
    parser.add_argument("--out-subdir", default="force_pressure_3D_y",
                        help="Subdirectory under <out_dir>/processed for "
                             "PNGs / MP4")
    parser.add_argument("--pi-mode", choices=("perp", "Pi"), default="perp",
                        help="Pressure used on the top (force-balance) "
                             "panel. 'perp' (default): p_perp from "
                             "temperature_perp; hoop fold = "
                             "-dp_perp/dr + Pi_{phi phi}/r. 'Pi': replace "
                             "p_perp with the flow Pi_{rr} = "
                             "sum_s m_s J_{x,s}^2/(q_s^2 n_s); hoop fold = "
                             "-dPi_{rr}/dr + Pi_{phi phi}/r - Pi_{rr}/r.")
    parser.add_argument("--avg-zy", action="store_true",
                        help="Average the 3D fields over y as well as z "
                             "instead of slicing at --y-idx. The z reduction "
                             "still follows --z-mode; --y-idx is ignored.")
    parser.add_argument("--pi", action="store_true",
                        help="Take Pi_{phi phi} from the second component of "
                             "the per-species momentum_flux_diag_cyl moment "
                             "(already n*m*<v_phi^2>) instead of the bulk-flow "
                             "estimate sum_s m_s J_{y,s}^2/(q_s^2 n_s).")
    parser.add_argument("--j", action="store_true",
                        help="Do not backward-average J / rotM in time: use "
                             "the raw n+1/2 snapshot at each frame instead of "
                             "X_eff = (X[k-1] + X[k]) / 2.")
    return parser.parse_args()


def main():
    args = parse_args()
    init_constants(args.config)
    y_idx = args.y_idx if args.y_idx is not None else const.Ny // 2

    b_dir = find_field_dir("B")
    j_dirs = [find_field_dir(f"{s}/J") for s in SPECIES]
    r_dirs = [find_field_dir(f"{s}/rotM") for s in SPECIES]
    t_dirs = [find_moment_dir(s, "temperature_perp") for s in SPECIES]
    tpar_dirs = [find_moment_dir(s, "temperature_parallel") for s in SPECIES]
    n_dirs = [find_moment_dir(s, "density") for s in SPECIES]
    pi_dirs = ([find_moment_dir(s, "momentum_flux_diag_cyl") for s in SPECIES]
               if args.pi else [None] * len(SPECIES))

    required = [("B", b_dir)]
    required += [(f"{s}/J", d) for s, d in zip(SPECIES, j_dirs)]
    required += [(f"{s}/rotM", d) for s, d in zip(SPECIES, r_dirs)]
    required += [(f"{s}/temperature_perp", d)
                 for s, d in zip(SPECIES, t_dirs)]
    if args.pi:
        required += [(f"{s}/momentum_flux_diag_cyl", d)
                     for s, d in zip(SPECIES, pi_dirs)]
    missing = [name for name, p in required if p is None]
    if missing:
        print(f"[error] missing diagnostics: {missing}")
        return

    # Per-species profile figure is optional: only drawn if both
    # temperature_parallel and density moments are configured.
    have_profiles = (all(d is not None for d in tpar_dirs)
                     and all(d is not None for d in n_dirs))
    if not have_profiles:
        miss = []
        miss += [f"{s}/temperature_parallel" for s, d in
                 zip(SPECIES, tpar_dirs) if d is None]
        miss += [f"{s}/density" for s, d in zip(SPECIES, n_dirs)
                 if d is None]
        print(f"[warn] species-profile figure skipped, missing: {miss}")

    vec_size = const.Nx * const.Ny * const.Nz * 3 * 4
    sca_size = const.Nx * const.Ny * const.Nz * 4
    b_steps = dict(list_steps(b_dir, vec_size))
    j_steps = [dict(list_steps(d, vec_size)) for d in j_dirs]
    r_steps = [dict(list_steps(d, vec_size)) for d in r_dirs]
    t_steps = [dict(list_steps(d, sca_size)) for d in t_dirs]
    pi_steps = ([dict(list_steps(d, vec_size)) for d in pi_dirs]
                if args.pi else None)
    if have_profiles:
        tpar_steps = [dict(list_steps(d, sca_size)) for d in tpar_dirs]
        n_steps = [dict(list_steps(d, sca_size)) for d in n_dirs]
    else:
        tpar_steps = n_steps = None

    common = set(b_steps)
    for s in (*j_steps, *r_steps, *t_steps):
        common &= set(s)
    if args.pi:
        for s in pi_steps:
            common &= set(s)
    if have_profiles:
        for s in (*tpar_steps, *n_steps):
            common &= set(s)
    common = sorted(common)
    if not common:
        print("[error] no timesteps common to all required diagnostics")
        return

    end_idx = args.end_idx if args.end_idx is not None else common[-1]
    if args.start_idx > end_idx:
        print(f"[error] start_idx {args.start_idx} > end_idx {end_idx}")
        return

    cx = args.center_x if args.center_x is not None else auto_center()[0]
    y_red = "avg" if args.avg_zy else f"idx {y_idx}"
    print(f"Cylinder axis x: {cx:.3f}   (z-mode: {args.z_mode}, "
          f"y: {y_red}, start: {args.start_idx}, end: {end_idx}, "
          f"pi: {args.pi}, j-shift: {not args.j})")

    # Per-species (q, m, n) from config. Used to convert J_y_s -> V_phi_s and
    # form p_phiphi = sum_s m_s * n_s * V_phi_s^2 = sum_s m_s * J_y_s^2 /
    # (q_s^2 * n_s). The per-macro (n/Np) factor required by
    # `distribution_moment.cpp` is already baked into the J diagnostic
    # (see Shape::cache * particles.n_Np(point) on collect()), so it is not
    # re-applied here.
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

    xs = (np.arange(const.Nx) + 0.5) * const.dx

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    def load_J_per_species(idx):
        return [load_vec_slice_y(d, s[idx], y_idx, args.avg_zy)
                for d, s in zip(j_dirs, j_steps)]

    def load_J(idx):
        return sum_species(j_dirs, j_steps, idx, y_idx, load_vec_slice_y,
                           args.avg_zy)

    def load_rotM(idx):
        return sum_species(r_dirs, r_steps, idx, y_idx, load_vec_slice_y,
                           args.avg_zy)

    def load_rotM_per_species(idx):
        return [load_vec_slice_y(d, s[idx], y_idx, args.avg_zy)
                for d, s in zip(r_dirs, r_steps)]

    def load_p_perp_per_species(idx):
        # temperature_perp is n*m*<v_perp^2> = 2*p_perp; per-species pressure
        # is 0.5 * temperature_perp_s. The sum across species is taken by
        # the caller when the total is needed.
        return [0.5 * load_scalar_slice_y(t_dir, t_st[idx], y_idx, args.avg_zy)
                for t_dir, t_st in zip(t_dirs, t_steps)]

    def load_p_perp(idx):
        return sum(load_p_perp_per_species(idx))

    # Baseline magnetic pressure at t = 0 (always the first snapshot, even
    # if --start-idx skips it), used by draw_frame to derive B_theory.
    idx0 = common[0]
    B_0 = load_vec_slice_y(b_dir, b_steps[idx0], y_idx, args.avg_zy)
    Bx0, By0, Bz0 = B_0[..., 0], B_0[..., 1], B_0[..., 2]
    B0_2_half_2d = 0.5 * (Bx0 ** 2 + By0 ** 2 + Bz0 ** 2)
    B0_2_half = (B0_2_half_2d.mean(axis=0)
                 if args.z_mode == "avg" else
                 B0_2_half_2d[const.Nz // 2])

    def red(field2d):
        """1D reduction (Nz, Nx) -> (Nx,) matching ``step_quantities``."""
        if args.z_mode == "avg":
            return field2d.mean(axis=0)
        return field2d[const.Nz // 2]

    # Per-frame storage for the species-profile figure. Each entry is a
    # list (one array per species) of 1D radial profiles. p_perp /
    # p_phiphi / J_phi / curlM_phi already live in `drawn`'s `q` dict, so
    # we don't duplicate them here.
    species_T_perp_frames = []
    species_T_par_frames = []
    species_n_frames = []

    drawn = []
    prev_J = None
    prev_J_sp = None
    prev_rotM = None
    prev_rotM_sp = None
    for k, idx in enumerate(common):
        if idx > end_idx:
            break
        cur_J = load_J(idx)
        cur_J_sp = load_J_per_species(idx)
        cur_rotM = load_rotM(idx)
        cur_rotM_sp = load_rotM_per_species(idx)
        if k == 0 or args.j:
            J_eff = cur_J
            J_eff_sp = cur_J_sp
            rotM_eff = cur_rotM
            rotM_eff_sp = cur_rotM_sp
        else:
            J_eff = 0.5 * (prev_J + cur_J)
            J_eff_sp = [0.5 * (pj + cj)
                        for pj, cj in zip(prev_J_sp, cur_J_sp)]
            rotM_eff = 0.5 * (prev_rotM + cur_rotM)
            rotM_eff_sp = [0.5 * (pr + cr)
                           for pr, cr in zip(prev_rotM_sp, cur_rotM_sp)]
        prev_J, prev_J_sp = cur_J, cur_J_sp
        prev_rotM, prev_rotM_sp = cur_rotM, cur_rotM_sp

        if idx < args.start_idx:
            continue

        B = load_vec_slice_y(b_dir, b_steps[idx], y_idx, args.avg_zy)
        p_perp_sp = load_p_perp_per_species(idx)
        p_perp = sum(p_perp_sp)

        if args.pi:
            # Pi_{phi phi} straight from the momentum_flux_diag_cyl moment:
            # component [1] is already the azimuthal pressure n*m*<v_phi^2>
            # (full tensor diagonal, not just the bulk-flow part). It is an
            # integer-step moment, so no half-step time average is applied.
            p_phiphi_sp = [
                load_vec_slice_y(pi_dir, pi_st[idx], y_idx, args.avg_zy)[..., 1]
                for pi_dir, pi_st in zip(pi_dirs, pi_steps)]
        else:
            # p_phiphi(x, z) = sum_s m_s * n_s * V_phi_s^2, with
            # V_phi_s = J_y_s / (q_s * n_s) and n_s, q_s, m_s from config.
            # Equivalent to sum_s m_s * J_y_s^2 / (q_s^2 * n_s). The macro-
            # particle (n/Np) factor is implicit in J
            # (distribution_moment.cpp:146).
            p_phiphi_sp = []
            for J_s, q_s, m_s, n_s in zip(J_eff_sp, sp_q, sp_m, sp_n):
                Jy_s = J_s[..., 1]
                p_phiphi_sp.append(m_s * Jy_s * Jy_s / (q_s * q_s * n_s))
        p_phiphi = sum(p_phiphi_sp)

        if args.pi_mode == "Pi":
            # On the y = y_idx slice, e_r ≈ ±e_x, so J_r ≈ ±J_x; squared,
            # J_r^2 = J_x^2. Pi_{rr} = sum_s m_s J_{x,s}^2 / (q_s^2 n_s).
            p_rr = np.zeros_like(p_perp)
            for J_s, q_s, m_s, n_s in zip(J_eff_sp, sp_q, sp_m, sp_n):
                Jx_s = J_s[..., 0]
                p_rr += m_s * Jx_s * Jx_s / (q_s * q_s * n_s)
        else:
            p_rr = None

        if args.pi_mode == "Pi":
            # Top panel uses Pi_{rr} as the pressure whose radial gradient
            # enters the force balance; p_perp stays the temperature-perp
            # moment so the bottom panel (B_theory) is unchanged.
            grad_source = p_rr
        else:
            grad_source = p_perp

        q = step_quantities(B, J_eff, rotM_eff, p_perp, p_phiphi, args.z_mode,
                            grad_pressure_2d=grad_source)
        if p_rr is not None:
            # Reduce p_rr to 1D the same way step_quantities does so the
            # hoop-fold formula has access to it.
            q["p_rr"] = red(p_rr)
        # Per-species 1D profiles for the species figure: each is a list
        # (one array per species) of length-Nx radial profiles.
        q["p_perp_sp"] = [red(pp) for pp in p_perp_sp]
        q["p_phiphi_sp"] = [red(pp) for pp in p_phiphi_sp]
        q["J_phi_sp"] = [red(J_s[..., 1]) for J_s in J_eff_sp]
        q["curlM_phi_sp"] = [red(rM[..., 1]) for rM in rotM_eff_sp]
        drawn.append((idx, q))

        if have_profiles:
            frame_T_perp = []
            frame_T_par = []
            frame_n = []
            for t_dir, t_st, tpar_dir, tpar_st, nd, n_st in zip(
                    t_dirs, t_steps, tpar_dirs, tpar_steps,
                    n_dirs, n_steps):
                T_perp_diag = load_scalar_slice_y(
                    t_dir, t_st[idx], y_idx, args.avg_zy)
                T_par_diag = load_scalar_slice_y(
                    tpar_dir, tpar_st[idx], y_idx, args.avg_zy)
                ns_field = load_scalar_slice_y(
                    nd, n_st[idx], y_idx, args.avg_zy)
                # temperature_perp diagnostic = n*m*<v_perp^2> (2 DoF), so
                # per-particle T_perp = m<v_perp^2>/2 = T_diag / (2 n).
                # temperature_parallel diagnostic = n*m*<v_par^2> (1 DoF),
                # so T_par = m<v_par^2> = T_diag / n. ns_field is the
                # diagnostic density (n_Np-weighted unity moment).
                eps = 1e-30
                T_perp_per = T_perp_diag / np.maximum(2.0 * ns_field, eps)
                T_par_per = T_par_diag / np.maximum(ns_field, eps)
                frame_T_perp.append(red(T_perp_per))
                frame_T_par.append(red(T_par_per))
                frame_n.append(red(ns_field))
            species_T_perp_frames.append(frame_T_perp)
            species_T_par_frames.append(frame_T_par)
            species_n_frames.append(frame_n)

    if not drawn:
        print(f"[warn] no snapshots in [{args.start_idx}, {end_idx}].")
        return

    fig, (ax_F, ax_P) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Cylindrical radius for the +p_phiphi/r hoop-stress term. The user
    # asked to fold this into the pressure-gradient curve (full cylindrical
    # radial divergence  -1/r d(r p_rr)/dr  = -dp_rr/dr + (p_phiphi - p_rr)/r
    # but using the simpler -dp_rr/dr + p_phiphi/r the user requested).
    # Cells within half a cell of the axis are masked to dodge 1/r.
    r_xs = np.abs(xs - cx)
    r_mask = r_xs > 0.5 * const.dx

    if args.pi_mode == "Pi":
        grad_label = (r"$-d\Pi_{rr}/dr + \Pi_{\varphi\varphi}/r"
                      r" - \Pi_{rr}/r$")
    else:
        grad_label = r"$-dp_{\perp}/dr + \Pi_{\varphi\varphi}/r$"

    def fold_hoop_into_grad(q):
        """Return a shallow-copy dict with ``neg_dp_dx`` replaced by the
        chosen radial pressure divergence. Two variants:
          perp: neg_dp_dx + p_phiphi/r
          Pi  : neg_dp_dx + (p_phiphi - p_rr)/r   (full cylindrical div
                of Pi_{rr} including the centrifugal Pi_{φφ}/r and the
                cylindrical-correction -Pi_{rr}/r terms)
        Does not mutate the input so the same ``q`` can be reduced into
        the time-mean accumulator afterwards."""
        if args.pi_mode == "Pi" and "p_rr" in q:
            numer = q["p_phiphi"] - q["p_rr"]
        else:
            numer = q["p_phiphi"]
        delta = np.zeros_like(numer)
        delta[r_mask] = numer[r_mask] / r_xs[r_mask]
        out = dict(q)
        # Shift the pressure-gradient curves one cell to the right in x so they
        # line up with the Yee-staggered J*B / (rot M)*B curves (the moment
        # pressure sits on the integer x-node, the fields half a cell to its
        # right). Both the raw -dp/dr and the hoop-folded curve are shifted.
        out["neg_dp_perp_dx"] = shift_right_x(q["neg_dp_dx"])
        out["neg_dp_dx"] = shift_right_x(q["neg_dp_dx"] + delta)
        return out

    def render(k):
        idx, q = drawn[k]
        draw_frame(ax_F, ax_P, xs, cx, fold_hoop_into_grad(q),
                   B0_2_half, time_avg=True,
                   grad_label=grad_label,
                   grad_raw_label=(r"$-d\Pi_{rr}/dr$"
                                   if args.pi_mode == "Pi"
                                   else r"$-dp_{\perp}/dr$"))
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

    keys = ["p_perp", "p_phiphi", "B2_half", "neg_dp_dx", "JyBz", "curlM_Bz",
            "J_phi", "curlM_phi"]
    if args.pi_mode == "Pi":
        keys.append("p_rr")
    accum = {key: np.zeros_like(drawn[0][1][key]) for key in keys}
    for _, q in drawn:
        for key in keys:
            accum[key] += q[key]
    mean_q = {key: v / len(drawn) for key, v in accum.items()}

    draw_frame(ax_F, ax_P, xs, cx, fold_hoop_into_grad(mean_q),
               B0_2_half, time_avg=True,
               grad_label=grad_label,
               grad_raw_label=(r"$-d\Pi_{rr}/dr$"
                               if args.pi_mode == "Pi"
                               else r"$-dp_{\perp}/dr$"))
    fig.suptitle("")
    fig.tight_layout()
    mean_path = os.path.join(out_dir, args.mean_name)
    fig.savefig(mean_path, dpi=args.dpi)
    plt.close(fig)

    # Species-profile figure: 3 rows x 2 columns, one quantity per cell.
    # All cells share the same x-axis; only the bottom row keeps tick
    # labels. Legends live inside each cell.
    #
    #   Row 1: T_perp/T_par per species   |  n per species
    #   Row 2: p_perp                     |  Pi_phiphi
    #   Row 3: J_phi (gyrocenter sum)     |  (rot M)_phi sum
    species_out_dir = os.path.join(out_dir, "species")
    os.makedirs(species_out_dir, exist_ok=True)

    fig_sp, axs_sp = plt.subplots(3, 2, figsize=(18, 15), sharex=True)
    (ax_T, ax_n_panel), (ax_pp, ax_Pi), (ax_J, ax_M) = axs_sp
    # The total gyrocenter current is plotted on its own right-side axis so
    # its scale doesn't squash the per-species curves on the left. Right
    # axis keeps its tick numbers but has no y-label (per request).
    ax_J_R = ax_J.twinx()

    sp_colors = ("tab:blue", "tab:red")

    def _short(s):
        return s[:-1] if s.endswith("s") else s

    def draw_species_panels(idx_T_perp, idx_T_par, idx_n,
                            p_perp_1d, p_phiphi_1d,
                            J_phi_1d, curlM_phi_1d,
                            p_perp_sp=None, p_phiphi_sp=None,
                            J_phi_sp=None, curlM_phi_sp=None):
        for ax in (ax_T, ax_n_panel, ax_pp, ax_Pi, ax_J, ax_J_R, ax_M):
            ax.clear()

        if have_profiles:
            for s, color, T_perp, T_par, ns in zip(
                    SPECIES, sp_colors, idx_T_perp, idx_T_par, idx_n):
                short = _short(s)
                ax_T.plot(xs, T_perp, color=color, linestyle="-",
                          label=rf"$T_{{\perp,{short}}}$")
                ax_T.plot(xs, T_par, color=color, linestyle="--",
                          label=rf"$T_{{\parallel,{short}}}$")
                ax_n_panel.plot(xs, ns, color=color, linestyle="-",
                                label=rf"$n_{{{short}}}$")
        else:
            for ax in (ax_T, ax_n_panel):
                ax.text(0.5, 0.5,
                        "temperature_parallel / density not configured",
                        transform=ax.transAxes,
                        ha="center", va="center")

        # Totals drawn thick + dark; per-species contributions overlaid with
        # the same blue/red colour code as the T/n panels but dashed.
        ax_pp.plot(xs, p_perp_1d, color="darkgreen", linewidth=2.0,
                   label=r"$p_{\perp}$")
        if p_perp_sp is not None:
            for s, color, pp_s in zip(SPECIES, sp_colors, p_perp_sp):
                ax_pp.plot(xs, pp_s, color=color, linestyle="--",
                           linewidth=1.0,
                           label=rf"$p_{{\perp,{_short(s)}}}$")

        ax_Pi.plot(xs, p_phiphi_1d, color="purple", linewidth=2.0,
                   label=r"$\Pi_{\varphi\varphi}$")
        if p_phiphi_sp is not None:
            for s, color, Pi_s in zip(SPECIES, sp_colors, p_phiphi_sp):
                ax_Pi.plot(xs, Pi_s, color=color, linestyle="--",
                           linewidth=1.0,
                           label=rf"$\Pi_{{\varphi\varphi,{_short(s)}}}$")

        # Total on the right axis (own scale); per-species contributions on
        # the left axis so they aren't dwarfed when the total is much larger
        # (e.g. when species partially cancel each other in the sum).
        ax_J_R.plot(xs, J_phi_1d, color="saddlebrown", linewidth=2.0,
                    label=r"$J_{\varphi, g}$")
        if J_phi_sp is not None:
            for s, color, J_s in zip(SPECIES, sp_colors, J_phi_sp):
                ax_J.plot(xs, J_s, color=color, linestyle="--",
                          linewidth=1.0,
                          label=rf"$J_{{\varphi, g, {_short(s)}}}$")

        ax_M.plot(xs, curlM_phi_1d, color="teal", linewidth=2.0,
                  label=r"$(\nabla\times M)_{\varphi}$")
        if curlM_phi_sp is not None:
            for s, color, cM_s in zip(SPECIES, sp_colors, curlM_phi_sp):
                ax_M.plot(xs, cM_s, color=color, linestyle="--",
                          linewidth=1.0,
                          label=rf"$(\nabla\times M)_{{\varphi,{_short(s)}}}$")

        ax_T.set_ylabel(r"$T$", fontsize=labelsize)
        ax_n_panel.set_ylabel(r"$n$", fontsize=labelsize)
        ax_pp.set_ylabel(r"$p_{\perp}$", fontsize=labelsize)
        ax_Pi.set_ylabel(r"$\Pi_{\varphi\varphi}$", fontsize=labelsize)
        ax_J.set_ylabel(r"$J_{\varphi}$", fontsize=labelsize)
        ax_M.set_ylabel(r"$(\nabla\times M)_{\varphi}$",
                       fontsize=labelsize)
        ax_J.set_xlabel(r"$x$", fontsize=labelsize)
        ax_M.set_xlabel(r"$x$", fontsize=labelsize)

        legend_kwargs = dict(
            fontsize=ticksize * 1.1,
            loc="center",
            bbox_to_anchor=(1.0, 1.0),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=1.0,
            handlelength=1.5,
            handletextpad=0.4,
            borderaxespad=0.0,
        )

        for ax in (ax_T, ax_n_panel, ax_pp, ax_Pi, ax_J, ax_M):
            ax.axvline(cx, color="k", linewidth=0.6, alpha=0.4)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=ticksize)
            if ax is ax_J:
                # Merge handles from left (per-species) and right (total)
                # axes so they share one legend.
                hL, lL = ax_J.get_legend_handles_labels()
                hR, lR = ax_J_R.get_legend_handles_labels()
                ax_J.legend(hL + hR, lL + lR, **legend_kwargs)
            else:
                ax.legend(**legend_kwargs)

        # Right-side axis on the J panel: keep tick numbers, drop the label.
        ax_J_R.tick_params(labelsize=ticksize)
        ax_J_R.set_ylabel("")

    def render_species(k):
        idx, q = drawn[k]
        idx_T_perp = species_T_perp_frames[k] if have_profiles else None
        idx_T_par = species_T_par_frames[k] if have_profiles else None
        idx_n = species_n_frames[k] if have_profiles else None
        draw_species_panels(idx_T_perp, idx_T_par, idx_n,
                            q["p_perp"], q["p_phiphi"],
                            q["J_phi"], q["curlM_phi"],
                            p_perp_sp=q["p_perp_sp"],
                            p_phiphi_sp=q["p_phiphi_sp"],
                            J_phi_sp=q["J_phi_sp"],
                            curlM_phi_sp=q["curlM_phi_sp"])
        fig_sp.suptitle(rf"$\omega_{{pe}}\,t = {idx * const.dts:.2f}$",
                        bbox=bbox, fontsize=labelsize)
        figname = os.path.join(species_out_dir, f"{idx:04d}.png")
        print(f"Processing {figname} (idx={idx})")
        fig_sp.tight_layout(rect=(0, 0, 1, 0.97))
        fig_sp.savefig(figname, dpi=args.dpi)
        return [ax_T, ax_n_panel, ax_pp, ax_Pi, ax_J, ax_J_R, ax_M]

    anim_sp = FuncAnimation(fig_sp, render_species, frames=len(drawn),
                            blit=False)
    sp_anim_path = os.path.join(out_dir, "species_profiles.mp4")
    anim_sp.save(sp_anim_path, writer=FFMpegWriter(fps=args.anim_fps),
                 dpi=args.dpi)

    # Time-mean profiles.
    if have_profiles:
        mean_T_perp = [np.mean([species_T_perp_frames[k][si]
                                for k in range(len(drawn))], axis=0)
                       for si in range(len(SPECIES))]
        mean_T_par = [np.mean([species_T_par_frames[k][si]
                               for k in range(len(drawn))], axis=0)
                      for si in range(len(SPECIES))]
        mean_n = [np.mean([species_n_frames[k][si]
                           for k in range(len(drawn))], axis=0)
                  for si in range(len(SPECIES))]
    else:
        mean_T_perp = mean_T_par = mean_n = None

    mean_J_phi = mean_q.get("J_phi") if "J_phi" in mean_q else \
        np.mean([q["J_phi"] for _, q in drawn], axis=0)
    mean_curlM_phi = np.mean([q["curlM_phi"] for _, q in drawn], axis=0)

    def _mean_sp(key):
        return [np.mean([q[key][si] for _, q in drawn], axis=0)
                for si in range(len(SPECIES))]

    mean_p_perp_sp = _mean_sp("p_perp_sp")
    mean_p_phiphi_sp = _mean_sp("p_phiphi_sp")
    mean_J_phi_sp = _mean_sp("J_phi_sp")
    mean_curlM_phi_sp = _mean_sp("curlM_phi_sp")

    draw_species_panels(mean_T_perp, mean_T_par, mean_n,
                        mean_q["p_perp"], mean_q["p_phiphi"],
                        mean_J_phi, mean_curlM_phi,
                        p_perp_sp=mean_p_perp_sp,
                        p_phiphi_sp=mean_p_phiphi_sp,
                        J_phi_sp=mean_J_phi_sp,
                        curlM_phi_sp=mean_curlM_phi_sp)
    fig_sp.suptitle("")
    fig_sp.tight_layout()
    species_path = os.path.join(out_dir, "species_profiles_mean.png")
    fig_sp.savefig(species_path, dpi=args.dpi)
    plt.close(fig_sp)

    # Errors are computed on the LAST snapshot rather than the time mean,
    # so the figure-level diagnostics reflect the final equilibrium state.
    last_q = drawn[-1][1]

    # Relative L2 force-balance error over the whole diagnostic, not the
    # pointwise max — dominated by the bulk of the profile, not by a single
    # noisy cell. The pressure side uses the full cylindrical pressure
    # divergence -dp_perp/dr + Pi_{phi phi}/r, i.e. the same curve drawn on
    # the figure (`fold_hoop_into_grad`).
    grad = fold_hoop_into_grad(last_q)["neg_dp_dx"]
    grad_scale = float(np.linalg.norm(grad))
    res_J = grad + last_q["JyBz"]         # -dp/dr - (-J*B) = neg_dp_dx + JyBz
    res_M = grad + last_q["curlM_Bz"]     # same against -(rot M)*B_z
    err_J = float(np.linalg.norm(res_J) / grad_scale * 100.0) \
        if grad_scale > 0.0 else float("nan")
    err_M = float(np.linalg.norm(res_M) / grad_scale * 100.0) \
        if grad_scale > 0.0 else float("nan")

    # Equilibrium error: max |B_model - B_theory| normalised by the
    # theoretical on-axis field drop B0_center - B_theory_center. The drop
    # is what the theory itself predicts the deconfinement to do, so the
    # ratio answers "how much of the predicted dent does the model miss?".
    B0_sq = 2.0 * B0_2_half
    B_theory = np.sqrt(np.maximum(B0_sq - 2.0 * last_q["p_perp"], 0.0))
    B_model = np.sqrt(np.maximum(2.0 * last_q["B2_half"], 0.0))
    diff_B = np.abs(B_model - B_theory)
    i_max = int(np.argmax(diff_B))

    i_center = int(np.argmin(np.abs(xs - cx)))
    B0_center = float(np.sqrt(2.0 * B0_2_half[i_center]))
    B_theory_center = float(B_theory[i_center])
    theory_drop = B0_center - B_theory_center
    err_B = float(diff_B[i_max] / theory_drop * 100.0) \
        if theory_drop > 0.0 else float("nan")

    print(f"Frames written to {out_dir}")
    print(f"Animation: {anim_path}")
    print(f"Time-mean: {mean_path}")
    print(f"Species frames: {species_out_dir}")
    print(f"Species animation: {sp_anim_path}")
    print(f"Species time-mean: {species_path}")
    print(f"Errors evaluated on last frame (idx = {drawn[-1][0]})")
    print(f"Force-balance error (||res||_2 / ||grad p_perp||_2):")
    print(f"  J_phi B_z         : {err_J:+.3f}%")
    print(f"  (rot M)_phi B_z   : {err_M:+.3f}%")
    print(f"Equilibrium error max|B_model - B_theory| / (B0 - B_theory) at "
          f"center x={float(xs[i_center]):.3f}: {err_B:+.3f}%  "
          f"(drop = {theory_drop:.4g}, max diff = {float(diff_B[i_max]):.4g} "
          f"at x={float(xs[i_max]):.3f})")


if __name__ == "__main__":
    main()
