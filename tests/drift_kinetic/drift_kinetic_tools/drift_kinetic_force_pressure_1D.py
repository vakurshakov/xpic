#!/usr/bin/env python3
"""Two-panel (force-balance, magnetic-field profile) diagnostic reduced to a
single 1D profile vs x by averaging the 3D fields over **both** z and y.

A simplified sibling of drift_kinetic_force_pressure_3D_y.py: instead of
slicing at one y and folding a cylindrical hoop term, fp1d treats the
column as a planar (Cartesian) force balance in x and averages everything
over the y-z cross-section. The figure layout, styling and time-mean /
animation outputs are taken verbatim from the 3D_y variant (via the shared
drift_kinetic_force_pressure_y.draw_frame).

Quantities (per frame, all reduced to a 1D profile vs x):
    p_perp(x) = sum_s 0.5 * temperature_perp_s   (= sum_s n_s * T_perp,s,
                since the temperature_perp moment stores n*m*<v_perp^2> =
                2 p_perp), averaged over z and y.

  Top panel (force balance):
    -dp_perp/dx                       central finite difference in x.
    -J_phi^tot B_z = -(J_y + (rot M)_y) B_z  and  -(rot M)_y B_z, where the
        Yee staggering is removed before the product so every factor lives
        on the integer (i, j) cell node:
            B_z  is Sh-in-x and Sh-in-y  -> averaged to the node in x and y
            J_y, (rot M)_y are Sh-in-y   -> averaged to the node in y
            (both are already No-in-x, co-located with p_perp / -dp/dx)
        The products are formed per cell, then averaged over z and y.

  Bottom panel (unchanged from 3D_y): B_theory = sqrt(B_0^2 - 2 p_perp)
    against |B| = sqrt(2 * B^2/2), with B^2/2 from the raw (un-staggered) B
    and B_0^2/2 the z-y-averaged magnetic pressure of the first snapshot.

J and rot M live half a step ahead of B (evaluated at n + 1/2), so they are
backward-averaged between consecutive snapshots, X_eff(k) = (X[k-1] +
X[k]) / 2; the first frame uses the raw value.
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
from drift_kinetic_force_pressure_y import draw_frame, style_axes
from drift_kinetic_force_pressure_3D_y import SPECIES, find_field_dir, find_moment_dir


def load_vec_3d(path: str, name: str) -> np.ndarray:
    """Read a 3D vector FieldView as (Nz, Ny, Nx, 3), float32."""
    Nx, Ny, Nz = const.Nx, const.Ny, const.Nz
    raw = np.fromfile(os.path.join(path, name), dtype=np.float32,
                      count=Nx * Ny * Nz * 3)
    return raw.reshape(Nz, Ny, Nx, 3)


def load_scalar_3d(path: str, name: str) -> np.ndarray:
    """Read a 3D scalar DistributionMoment as (Nz, Ny, Nx), float32."""
    Nx, Ny, Nz = const.Nx, const.Ny, const.Nz
    raw = np.fromfile(os.path.join(path, name), dtype=np.float32,
                      count=Nx * Ny * Nz)
    return raw.reshape(Nz, Ny, Nx)


def sum_species_3d(dirs, steps, idx, loader):
    out = None
    for d, s in zip(dirs, steps):
        cur = loader(d, s[idx])
        out = cur if out is None else out + cur
    return out


def destagger_node(field, axis):
    """Move a +1/2-cell Yee-shifted component onto the integer grid node
    along ``axis``: 1/2 (F[i] + F[i-1]). The first index along ``axis``
    keeps its raw value (one-sided, no wrap-around)."""
    out = 0.5 * (field + np.roll(field, 1, axis=axis))
    sl = [slice(None)] * field.ndim
    sl[axis] = 0
    out[tuple(sl)] = np.take(field, 0, axis=axis)
    return out


def backward_diff_1d(profile, dx):
    """Left (backward) first derivative of a 1D profile: (f[i]-f[i-1])/dx,
    with a one-sided forward difference at the first cell."""
    d = np.empty_like(profile)
    d[1:] = (profile[1:] - profile[:-1]) / dx
    d[0] = (profile[1] - profile[0]) / dx
    return d


def central_diff_1d(profile, dx):
    """Central first derivative of a 1D profile: (f[i+1]-f[i-1])/(2 dx),
    one-sided at the two ends."""
    d = np.empty_like(profile)
    d[1:-1] = (profile[2:] - profile[:-2]) / (2.0 * dx)
    d[0] = (profile[1] - profile[0]) / dx
    d[-1] = (profile[-1] - profile[-2]) / dx
    return d


def reduce_zy(field3d):
    """Average a (Nz, Ny, Nx) field over z (axis 0) and y (axis 1) -> (Nx,)."""
    return field3d.mean(axis=(0, 1))


# Axis indices for the (Nz, Ny, Nx[, 3]) layout.
Z_AX, Y_AX, X_AX = 0, 1, 2


def step_quantities(B, J_eff, rotM_eff, p_perp_3d):
    """1D (vs x) reductions matching drift_kinetic_force_pressure_y.draw_frame's
    dict layout. Field products are de-staggered onto the cell node before the
    z-y average so they co-locate with the node-centred moment pressure."""
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]

    # Bottom panel: raw |B|^2/2 (no de-staggering), z-y averaged.
    B2_half_1d = reduce_zy(0.5 * (Bx * Bx + By * By + Bz * Bz))

    # B_z is Sh in x and y -> bring it onto the integer (i, j) node.
    Bz_node = destagger_node(destagger_node(Bz, axis=X_AX), axis=Y_AX)
    # J_y and (rot M)_y are Sh in y only -> de-stagger in y.
    Jy_node = destagger_node(J_eff[..., 1], axis=Y_AX)
    curlMy_node = destagger_node(rotM_eff[..., 1], axis=Y_AX)
    Jy_tot_node = Jy_node + curlMy_node

    # p_perp lives on the node already; average over z and y, then -d/dx
    # (left / backward difference in x).
    p_perp_1d = reduce_zy(p_perp_3d)
    neg_dp_dx = -backward_diff_1d(p_perp_1d, const.dx)

    # Ampere comparison: J_y^tot vs -dB_z/dx. B_z is node-de-staggered and
    # z-y averaged like p_perp, but its x-derivative uses a central difference.
    Bz_node_1d = reduce_zy(Bz_node)

    return {
        "p_perp": p_perp_1d,
        "B2_half": B2_half_1d,
        "neg_dp_dx": neg_dp_dx,
        "JyBz": reduce_zy(Jy_tot_node * Bz_node),
        "curlM_Bz": reduce_zy(curlMy_node * Bz_node),
        "Jy_tot": reduce_zy(Jy_tot_node),
        "neg_dBz_dx": -central_diff_1d(Bz_node_1d, const.dx),
    }


def _short(s):
    return s[:-1] if s.endswith("s") else s


def draw_ampere_figure(drawn, xs, cx, out_dir, args):
    """Separate single-panel comparison (in the spirit of the displacement
    panel) of the total azimuthal current J_y^tot against -dB_z/dx, the
    dominant (curl B)_y term on the z-y-averaged column. Both curves are
    z-y averaged; -dB_z/dx is built exactly like -dp_perp/dx (node-de-
    staggered B_z, then a left/backward x-difference). Writes a per-frame
    animation and a time-mean PNG."""
    fig_a, ax_a = plt.subplots(figsize=(10, 6))

    def draw(q, avg_sub):
        ax_a.cla()
        ax_a.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_a.axvline(cx, color="k", linewidth=0.6, alpha=0.3, linestyle=":")
        ax_a.plot(xs, q["Jy_tot"], color="saddlebrown", linewidth=2.0,
                  label=r"$J_y^{\mathrm{tot}}$")
        ax_a.plot(xs, q["neg_dBz_dx"], color="darkgreen", linewidth=2.0,
                  label=r"$-dB_z/dx$")
        ax_a.set_xlim(xs[0], xs[-1])
        ax_a.set_xlabel(r"$x~(c/\omega_{pe})$", fontsize=labelsize)
        ax_a.set_ylabel(rf"$\langle\,\cdot\,\rangle_{{{avg_sub}}}$"
                        r" ($e n_0 c$)", fontsize=labelsize)
        ax_a.tick_params(labelsize=ticksize)
        ax_a.grid(True, alpha=0.3)
        ax_a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                              useMathText=True)
        ax_a.legend(fontsize=ticksize + 2, loc="best")

    ampere_dir = os.path.join(out_dir, "ampere")
    os.makedirs(ampere_dir, exist_ok=True)

    def render(k):
        idx, q = drawn[k]
        draw(q, avg_sub=r"z,y")
        fig_a.suptitle(rf"$\omega_{{pe}}\,t = {idx * const.dts:.2f}$",
                       bbox=bbox, fontsize=labelsize)
        figname = os.path.join(ampere_dir, f"{idx:04d}.png")
        print(f"Processing {figname} (idx={idx})")
        fig_a.tight_layout(rect=(0, 0, 1, 0.97))
        fig_a.savefig(figname, dpi=args.dpi)
        return [ax_a]

    anim = FuncAnimation(fig_a, render, frames=len(drawn), blit=False)
    anim_path = os.path.join(out_dir, "ampere.mp4")
    anim.save(anim_path, writer=FFMpegWriter(fps=args.anim_fps), dpi=args.dpi)

    mean_q = {
        "Jy_tot": np.mean([q["Jy_tot"] for _, q in drawn], axis=0),
        "neg_dBz_dx": np.mean([q["neg_dBz_dx"] for _, q in drawn], axis=0),
    }
    draw(mean_q, avg_sub=r"z,y,t")
    fig_a.suptitle("")
    fig_a.tight_layout()
    mean_path = os.path.join(out_dir, "ampere_mean.png")
    fig_a.savefig(mean_path, dpi=args.dpi)
    plt.close(fig_a)
    print(f"Ampere frames: {ampere_dir}")
    print(f"Ampere animation: {anim_path}")
    print(f"Ampere time-mean: {mean_path}")


def draw_parallel_correction_figure(drawn, xs, cx, out_dir, args, B0_2_half):
    """Two-panel diagnostic isolating the spurious curvature-drift current.

    The push uses the proxy grad|B|/B in place of the true line curvature
    kappa = (b.grad)b. For the straight field of this test kappa = 0 exactly,
    so the curvature drift should vanish; the proxy instead injects a parasitic
    current
        J_y^par = p_par / B_z^2 * dB_z/dx ,   p_par = sum_s n m <v_par^2> ,
    which (via Ampere) is exactly the residual the bare force balance leaves.

    The picture intentionally keeps the same visual contract as the main
    force/B-balance figure. The top-panel brown current curve is corrected
    by subtracting J_y^par, but remains labelled as -J_y^tot B_z so the
    resulting frame can be compared directly to the uncorrected balance.
    """
    def parasitic_force(q):
        # f_par = (p_par/B_z) dB_z/dx. The corrected force is
        # -J_y^tot B_z + f_par.
        return q["p_par"] / q["Bz_zy"] * central_diff_1d(q["Bz_zy"], const.dx)

    def supplemented_B(p_perp, p_par):
        # RK4 integration of dB/dx = -B/(B^2 + p_par) dp_perp/dx from the edge
        # field B_0; independent of the measured B (uses only p_perp, p_par).
        dpdx = central_diff_1d(p_perp, const.dx)
        B0 = float(np.sqrt(2.0 * B0_2_half[0]))
        h = xs[1] - xs[0]

        def f(x, B):
            pp = np.interp(x, xs, p_par)
            g = np.interp(x, xs, dpdx)
            return -B / (B * B + pp) * g

        B = np.empty_like(xs)
        B[0] = B0
        for i in range(len(xs) - 1):
            x0, b = xs[i], B[i]
            k1 = f(x0, b)
            k2 = f(x0 + 0.5 * h, b + 0.5 * h * k1)
            k3 = f(x0 + 0.5 * h, b + 0.5 * h * k2)
            k4 = f(x0 + h, b + h * k3)
            B[i + 1] = b + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return B

    fig, (ax_F, ax_P) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    def draw(q, avg_sub):
        ax_F.cla()
        ax_P.cla()
        style_axes(ax_F, ax_P, xs, cx, time_avg=True, force_sym=r"F_x",
                   avg_sub=avg_sub)
        ax_F.set_ylim(-1.0e-4, 1.0e-4)
        ax_P.set_ylim(0.15, 0.21)
        f_par = parasitic_force(q)
        leg_fs = ticksize + 1
        lbl = labelsize + 4
        red = "red"
        blue = "blue"
        red_width = 3.0
        blue_width = 1.9

        ax_F.plot(xs, q["neg_dp_dx"], color=red, linewidth=red_width,
                  label=r"$-dp_{\perp}/dx$")
        ax_F.plot(xs, -q["curlM_Bz"], color=blue, linewidth=blue_width,
                  label=r"$-(\mathrm{rot}\,M)_{y}\,B_z$")
        ax_F.plot(xs, -q["JyBz"] + f_par, color=blue, linewidth=blue_width,
                  linestyle=":", marker=".", markersize=10.0,
                  markevery=max(1, len(xs) // 50),
                  label=r"$-J_{y}^{\mathrm{tot}} B_z$")
        ax_F.legend(fontsize=leg_fs, loc="lower right").set_zorder(10)

        B_model = np.sqrt(np.maximum(2.0 * q["B2_half"], 0.0))
        B_corr = supplemented_B(q["p_perp"], q["p_par"])

        ax_P.plot(xs, B_model, color=red, linewidth=red_width,
                  label=r"$B_z~-$ модель")
        ax_P.plot(xs, B_corr, color=blue, linewidth=blue_width,
                  linestyle=":", marker=".", markersize=10.0,
                  markevery=max(1, len(xs) // 50),
                  label=r"$B_z = \sqrt{B_0^2 - 2 p_{\perp}}~-$ теория")
        ax_P.legend(fontsize=leg_fs, loc="lower right").set_zorder(10)

        panel_bbox = dict(facecolor="white", edgecolor="none",
                          alpha=0.75, boxstyle="round,pad=0.25")
        ax_F.text(0.03, 0.05, "(а)", transform=ax_F.transAxes,
                  ha="left", va="bottom", fontsize=lbl, bbox=panel_bbox)
        ax_P.text(0.03, 0.05, "(б)", transform=ax_P.transAxes,
                  ha="left", va="bottom", fontsize=lbl, bbox=panel_bbox)

    corr_dir = os.path.join(out_dir, "parallel_correction")
    os.makedirs(corr_dir, exist_ok=True)

    def render(k):
        idx, q = drawn[k]
        draw(q, avg_sub=r"z,y")
        fig.suptitle(rf"$\omega_{{pe}}\,t = {idx * const.dts:.2f}$",
                     bbox=bbox, fontsize=labelsize)
        figname = os.path.join(corr_dir, f"{idx:04d}.png")
        print(f"Processing {figname} (idx={idx})")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(figname, dpi=args.dpi)
        return [ax_F, ax_P]

    anim = FuncAnimation(fig, render, frames=len(drawn), blit=False)
    anim_path = os.path.join(out_dir, "parallel_correction.mp4")
    anim.save(anim_path, writer=FFMpegWriter(fps=args.anim_fps), dpi=args.dpi)

    # Time-mean figure.
    keys = ("p_perp", "B2_half", "neg_dp_dx", "JyBz", "curlM_Bz",
            "Bz_zy", "p_par")
    mean_q = {key: np.mean([q[key] for _, q in drawn], axis=0) for key in keys}
    draw(mean_q, avg_sub=r"z,y,t")
    fig.suptitle("")
    fig.tight_layout()
    mean_path = os.path.join(out_dir, "parallel_correction_mean.png")
    fig.savefig(mean_path, dpi=args.dpi)
    plt.close(fig)

    # Force-balance residual on the time mean, raw vs parasitic-corrected
    # (rel. L2 vs ||dp_perp/dx||). err_corr << err_raw confirms the diagnosis.
    grad = mean_q["neg_dp_dx"]
    scale = float(np.linalg.norm(grad))
    f_par = (mean_q["p_par"] / mean_q["Bz_zy"]
             * central_diff_1d(mean_q["Bz_zy"], const.dx))
    err_raw = (float(np.linalg.norm(grad + mean_q["JyBz"]) / scale * 100.0)
               if scale > 0.0 else float("nan"))
    err_corr = (float(np.linalg.norm(grad + mean_q["JyBz"] - f_par)
                      / scale * 100.0) if scale > 0.0 else float("nan"))
    print(f"Force-balance residual raw       : {err_raw:.3f}%")
    print(f"Force-balance residual corrected : {err_corr:.3f}%")
    print(f"Parallel-correction frames: {corr_dir}")
    print(f"Parallel-correction animation: {anim_path}")
    print(f"Parallel-correction time-mean: {mean_path}")


def draw_pressure_figure(drawn, xs, cx, out_dir, args):
    """Single-panel comparison (same styling as draw_ampere_figure) of the two
    independent z-y-averaged perpendicular-pressure profiles:
        p_perp^T    from the kinetic temperature_perp moment (= 0.5 * sum_s
                    temperature_perp_s), and
        p_perp^{-M.B}  from the magnetization, p_perp = -sum_s M_s . B.
    They are equal in the continuum; their difference exposes the deposition-
    kernel mismatch (M uses a 2nd-order spline; the moment a 1st-order one),
    which is largest where the profile curvature is largest. Writes a per-frame
    animation plus a time-mean PNG."""
    fig_p, ax_p = plt.subplots(figsize=(10, 6))

    def draw(q, avg_sub):
        ax_p.cla()
        ax_p.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_p.axvline(cx, color="k", linewidth=0.6, alpha=0.3, linestyle=":")
        ax_p.plot(xs, q["p_perp_T"], color="darkgreen", linewidth=2.0,
                  label=r"$p_{\perp}$ (temperature$_{\perp}$)")
        ax_p.plot(xs, q["p_perp_MB"], color="saddlebrown", linewidth=2.0,
                  linestyle="--", label=r"$p_{\perp} = -\mathbf{M}\cdot\mathbf{B}$")
        ax_p.set_xlim(xs[0], xs[-1])
        ax_p.set_xlabel(r"$x~(c/\omega_{pe})$", fontsize=labelsize)
        ax_p.set_ylabel(rf"$\langle p_{{\perp}}\rangle_{{{avg_sub}}}$"
                        r" ($n_0 m_e c^2$)", fontsize=labelsize)
        ax_p.tick_params(labelsize=ticksize)
        ax_p.grid(True, alpha=0.3)
        ax_p.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                              useMathText=True)
        ax_p.legend(fontsize=ticksize + 2, loc="best")

    pressure_dir = os.path.join(out_dir, "pressure")
    os.makedirs(pressure_dir, exist_ok=True)

    def render(k):
        idx, q = drawn[k]
        draw(q, avg_sub=r"z,y")
        fig_p.suptitle(rf"$\omega_{{pe}}\,t = {idx * const.dts:.2f}$",
                       bbox=bbox, fontsize=labelsize)
        figname = os.path.join(pressure_dir, f"{idx:04d}.png")
        print(f"Processing {figname} (idx={idx})")
        fig_p.tight_layout(rect=(0, 0, 1, 0.97))
        fig_p.savefig(figname, dpi=args.dpi)
        return [ax_p]

    anim = FuncAnimation(fig_p, render, frames=len(drawn), blit=False)
    anim_path = os.path.join(out_dir, "pressure.mp4")
    anim.save(anim_path, writer=FFMpegWriter(fps=args.anim_fps), dpi=args.dpi)

    mean_q = {
        "p_perp_T": np.mean([q["p_perp_T"] for _, q in drawn], axis=0),
        "p_perp_MB": np.mean([q["p_perp_MB"] for _, q in drawn], axis=0),
    }
    draw(mean_q, avg_sub=r"z,y,t")
    fig_p.suptitle("")
    fig_p.tight_layout()
    mean_path = os.path.join(out_dir, "pressure_mean.png")
    fig_p.savefig(mean_path, dpi=args.dpi)
    plt.close(fig_p)

    # Relative L2 mismatch between the two definitions on the time mean.
    diff = mean_q["p_perp_T"] - mean_q["p_perp_MB"]
    scale = float(np.linalg.norm(mean_q["p_perp_T"]))
    err = float(np.linalg.norm(diff) / scale * 100.0) if scale > 0.0 \
        else float("nan")

    print(f"Pressure frames: {pressure_dir}")
    print(f"Pressure animation: {anim_path}")
    print(f"Pressure time-mean: {mean_path}")
    print(f"p_perp mismatch ||T - (-M.B)||_2 / ||T||_2 : {err:.3f}%")


def draw_species_figure(drawn, xs, cx, out_dir, args, have_profiles,
                        species_T_perp_frames, species_T_par_frames,
                        species_n_frames):
    """Per-species 3x2 figure mirroring drift_kinetic_force_pressure_3D_y.py,
    but every quantity is the z-y average and the Pi panel shows Pi_yy
    (rebuilt from Pi_rr / Pi_phiphi) instead of Pi_phiphi.

      Row 1: T_perp / T_par per species   |  n per species
      Row 2: p_perp                       |  Pi_yy
      Row 3: J_y (gyrocenter sum)         |  (rot M)_y sum
    """
    species_out_dir = os.path.join(out_dir, "species")
    os.makedirs(species_out_dir, exist_ok=True)

    fig_sp, axs_sp = plt.subplots(3, 2, figsize=(18, 15), sharex=True)
    (ax_T, ax_n_panel), (ax_pp, ax_Pi), (ax_J, ax_M) = axs_sp
    ax_J_R = ax_J.twinx()
    sp_colors = ("tab:blue", "tab:red")

    def draw_species_panels(idx_T_perp, idx_T_par, idx_n,
                            p_perp_1d, pi_yy_1d, J_y_1d, curlM_y_1d,
                            p_perp_sp=None, pi_yy_sp=None,
                            J_y_sp=None, curlM_y_sp=None):
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
                        transform=ax.transAxes, ha="center", va="center")

        ax_pp.plot(xs, p_perp_1d, color="darkgreen", linewidth=2.0,
                   label=r"$p_{\perp}$")
        if p_perp_sp is not None:
            for s, color, pp_s in zip(SPECIES, sp_colors, p_perp_sp):
                ax_pp.plot(xs, pp_s, color=color, linestyle="--",
                           linewidth=1.0, label=rf"$p_{{\perp,{_short(s)}}}$")

        ax_Pi.plot(xs, pi_yy_1d, color="purple", linewidth=2.0,
                   label=r"$\Pi_{yy}$")
        if pi_yy_sp is not None:
            for s, color, Pi_s in zip(SPECIES, sp_colors, pi_yy_sp):
                ax_Pi.plot(xs, Pi_s, color=color, linestyle="--",
                           linewidth=1.0, label=rf"$\Pi_{{yy,{_short(s)}}}$")

        ax_J_R.plot(xs, J_y_1d, color="saddlebrown", linewidth=2.0,
                    label=r"$J_{y, g}$")
        if J_y_sp is not None:
            for s, color, J_s in zip(SPECIES, sp_colors, J_y_sp):
                ax_J.plot(xs, J_s, color=color, linestyle="--",
                          linewidth=1.0, label=rf"$J_{{y, g, {_short(s)}}}$")

        ax_M.plot(xs, curlM_y_1d, color="teal", linewidth=2.0,
                  label=r"$(\nabla\times M)_{y}$")
        if curlM_y_sp is not None:
            for s, color, cM_s in zip(SPECIES, sp_colors, curlM_y_sp):
                ax_M.plot(xs, cM_s, color=color, linestyle="--",
                          linewidth=1.0,
                          label=rf"$(\nabla\times M)_{{y,{_short(s)}}}$")

        ax_T.set_ylabel(r"$T$", fontsize=labelsize)
        ax_n_panel.set_ylabel(r"$n$", fontsize=labelsize)
        ax_pp.set_ylabel(r"$p_{\perp}$", fontsize=labelsize)
        ax_Pi.set_ylabel(r"$\Pi_{yy}$", fontsize=labelsize)
        ax_J.set_ylabel(r"$J_{y}$", fontsize=labelsize)
        ax_M.set_ylabel(r"$(\nabla\times M)_{y}$", fontsize=labelsize)
        ax_J.set_xlabel(r"$x$", fontsize=labelsize)
        ax_M.set_xlabel(r"$x$", fontsize=labelsize)

        legend_kwargs = dict(
            fontsize=ticksize * 1.1, loc="center",
            bbox_to_anchor=(1.0, 1.0), frameon=True, facecolor="white",
            edgecolor="none", framealpha=1.0, handlelength=1.5,
            handletextpad=0.4, borderaxespad=0.0,
        )
        for ax in (ax_T, ax_n_panel, ax_pp, ax_Pi, ax_J, ax_M):
            ax.axvline(cx, color="k", linewidth=0.6, alpha=0.4)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=ticksize)
            if ax is ax_J:
                hL, lL = ax_J.get_legend_handles_labels()
                hR, lR = ax_J_R.get_legend_handles_labels()
                ax_J.legend(hL + hR, lL + lR, **legend_kwargs)
            else:
                ax.legend(**legend_kwargs)
        ax_J_R.tick_params(labelsize=ticksize)
        ax_J_R.set_ylabel("")

    def render_species(k):
        idx, q = drawn[k]
        draw_species_panels(
            species_T_perp_frames[k] if have_profiles else None,
            species_T_par_frames[k] if have_profiles else None,
            species_n_frames[k] if have_profiles else None,
            q["p_perp"], q["pi_yy"], q["J_y"], q["curlM_y"],
            p_perp_sp=q["p_perp_sp"], pi_yy_sp=q["pi_yy_sp"],
            J_y_sp=q["J_y_sp"], curlM_y_sp=q["curlM_y_sp"])
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

    n_drawn = len(drawn)

    def _mean(key):
        return np.mean([q[key] for _, q in drawn], axis=0)

    def _mean_sp(key):
        return [np.mean([q[key][si] for _, q in drawn], axis=0)
                for si in range(len(SPECIES))]

    if have_profiles:
        mean_T_perp = [np.mean([species_T_perp_frames[k][si]
                                for k in range(n_drawn)], axis=0)
                       for si in range(len(SPECIES))]
        mean_T_par = [np.mean([species_T_par_frames[k][si]
                               for k in range(n_drawn)], axis=0)
                      for si in range(len(SPECIES))]
        mean_n = [np.mean([species_n_frames[k][si]
                           for k in range(n_drawn)], axis=0)
                  for si in range(len(SPECIES))]
    else:
        mean_T_perp = mean_T_par = mean_n = None

    draw_species_panels(
        mean_T_perp, mean_T_par, mean_n,
        _mean("p_perp"), _mean("pi_yy"), _mean("J_y"), _mean("curlM_y"),
        p_perp_sp=_mean_sp("p_perp_sp"), pi_yy_sp=_mean_sp("pi_yy_sp"),
        J_y_sp=_mean_sp("J_y_sp"), curlM_y_sp=_mean_sp("curlM_y_sp"))
    fig_sp.suptitle("")
    fig_sp.tight_layout()
    species_path = os.path.join(out_dir, "species_profiles_mean.png")
    fig_sp.savefig(species_path, dpi=args.dpi)
    plt.close(fig_sp)
    print(f"Species frames: {species_out_dir}")
    print(f"Species animation: {sp_anim_path}")
    print(f"Species time-mean: {species_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--center-x", type=float, default=None,
                        help="Cylinder axis x-coordinate, only used to draw "
                             "the vertical marker (default: auto).")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="First diagnostic-frame index to draw "
                             "(inclusive). Earlier frames are still iterated "
                             "to keep the half-step backward averages of "
                             "J / rot M consistent.")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Last diagnostic-frame index to draw "
                             "(inclusive). Default: last available frame.")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--anim-fps", type=int, default=10)
    parser.add_argument("--anim-name", default="force_pressure_1D.mp4",
                        help="MP4 filename inside <out-subdir>")
    parser.add_argument("--mean-name", default="time_mean.png",
                        help="Time-averaged figure filename inside "
                             "<out-subdir>")
    parser.add_argument("--out-subdir", default="force_pressure_1D",
                        help="Subdirectory under <out_dir>/processed for "
                             "PNGs / MP4")
    return parser.parse_args()


def main():
    args = parse_args()
    init_constants(args.config)

    b_dir = find_field_dir("B")
    j_dirs = [find_field_dir(f"{s}/J") for s in SPECIES]
    r_dirs = [find_field_dir(f"{s}/rotM") for s in SPECIES]
    m_dirs = [find_field_dir(f"{s}/M") for s in SPECIES]
    t_dirs = [find_moment_dir(s, "temperature_perp") for s in SPECIES]
    tpar_dirs = [find_moment_dir(s, "temperature_parallel") for s in SPECIES]
    n_dirs = [find_moment_dir(s, "density") for s in SPECIES]
    pi_dirs = [find_moment_dir(s, "momentum_flux_diag_cyl") for s in SPECIES]

    required = [("B", b_dir)]
    required += [(f"{s}/J", d) for s, d in zip(SPECIES, j_dirs)]
    required += [(f"{s}/rotM", d) for s, d in zip(SPECIES, r_dirs)]
    required += [(f"{s}/temperature_perp", d)
                 for s, d in zip(SPECIES, t_dirs)]
    missing = [name for name, p in required if p is None]
    if missing:
        print(f"[error] missing diagnostics: {missing}")
        return

    # The per-species figure needs momentum_flux_diag_cyl (for Pi_yy via
    # Pi_rr / Pi_phiphi); its T/n row additionally needs temperature_parallel
    # and density. Either may be absent without blocking the main figure.
    have_pi = all(d is not None for d in pi_dirs)
    have_profiles = (all(d is not None for d in tpar_dirs)
                     and all(d is not None for d in n_dirs))
    # temperature_parallel alone (no density needed) gates the parallel-
    # curvature-correction figure, which needs p_par = sum_s n m <v_par^2>.
    have_tpar = all(d is not None for d in tpar_dirs)
    if not have_pi:
        print("[warn] species figure skipped: momentum_flux_diag_cyl not "
              "configured for all species")
    elif not have_profiles:
        miss = [f"{s}/temperature_parallel" for s, d in zip(SPECIES, tpar_dirs)
                if d is None]
        miss += [f"{s}/density" for s, d in zip(SPECIES, n_dirs) if d is None]
        print(f"[warn] species T/n row skipped, missing: {miss}")

    vec_size = const.Nx * const.Ny * const.Nz * 3 * 4
    sca_size = const.Nx * const.Ny * const.Nz * 4
    b_steps = dict(list_steps(b_dir, vec_size))
    j_steps = [dict(list_steps(d, vec_size)) for d in j_dirs]
    r_steps = [dict(list_steps(d, vec_size)) for d in r_dirs]
    # M is optional: only needed for the p_perp(temperature) vs p_perp(-M.B)
    # comparison figure. Absence simply skips that extra panel.
    have_m = all(d is not None for d in m_dirs)
    m_steps = ([dict(list_steps(d, vec_size)) for d in m_dirs]
               if have_m else None)
    t_steps = [dict(list_steps(d, sca_size)) for d in t_dirs]
    pi_steps = ([dict(list_steps(d, vec_size)) for d in pi_dirs]
                if have_pi else None)
    if have_tpar:
        tpar_steps = [dict(list_steps(d, sca_size)) for d in tpar_dirs]
    else:
        tpar_steps = None
    if have_profiles:
        n_steps = [dict(list_steps(d, sca_size)) for d in n_dirs]
    else:
        n_steps = None

    common = set(b_steps)
    for s in (*j_steps, *r_steps, *t_steps):
        common &= set(s)
    if have_m:
        for s in m_steps:
            common &= set(s)
    if have_pi:
        for s in pi_steps:
            common &= set(s)
    if have_tpar:
        for s in tpar_steps:
            common &= set(s)
    if have_profiles:
        for s in n_steps:
            common &= set(s)
    common = sorted(common)
    if not common:
        print("[error] no timesteps common to all required diagnostics")
        return

    end_idx = args.end_idx if args.end_idx is not None else common[-1]
    if args.start_idx > end_idx:
        print(f"[error] start_idx {args.start_idx} > end_idx {end_idx}")
        return

    cx_auto, cy = auto_center()
    cx = args.center_x if args.center_x is not None else cx_auto
    print(f"Cylinder axis: ({cx:.3f}, {cy:.3f})   (z-y averaged 1D, "
          f"start: {args.start_idx}, end: {end_idx})")

    xs = (np.arange(const.Nx) + 0.5) * const.dx

    # Per-cell cos^2 / sin^2 of the azimuthal angle theta = atan2(y-cy, x-cx),
    # used to rebuild Pi_yy = Pi_rr sin^2(theta) + Pi_phiphi cos^2(theta) from
    # the cylindrical pressure diagnostic (the off-diagonal Pi_{r phi} term is
    # dropped). On the axis (r = 0) the weights are set isotropic (1/2).
    xs_2d = (np.arange(const.Nx)[None, :] + 0.5) * const.dx
    ys_2d = (np.arange(const.Ny)[:, None] + 0.5) * const.dy
    ddx = xs_2d - cx
    ddy = ys_2d - cy
    r2 = ddx * ddx + ddy * ddy
    with np.errstate(invalid="ignore", divide="ignore"):
        sin2_th = np.where(r2 > 0.0, (ddy * ddy) / r2, 0.5)
        cos2_th = np.where(r2 > 0.0, (ddx * ddx) / r2, 0.5)
    sin2_th = np.broadcast_to(sin2_th, (const.Ny, const.Nx))
    cos2_th = np.broadcast_to(cos2_th, (const.Ny, const.Nx))

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    def load_J_sp(idx):
        return [load_vec_3d(d, s[idx]) for d, s in zip(j_dirs, j_steps)]

    def load_rotM_sp(idx):
        return [load_vec_3d(d, s[idx]) for d, s in zip(r_dirs, r_steps)]

    def load_p_perp_sp(idx):
        # temperature_perp = n*m*<v_perp^2> = 2 p_perp -> p_perp,s = 0.5*moment
        # (equivalently n_s * T_perp,s).
        return [0.5 * load_scalar_3d(t_dir, t_st[idx])
                for t_dir, t_st in zip(t_dirs, t_steps)]

    def load_p_perp_MB(idx, B):
        # Magnetization pressure: M = -(p_perp / B) b, so p_perp = -M.B.
        # Summed over species, de-staggering is unnecessary because M and B
        # share the same Yee magnetic staggering -> the dot product M.B is
        # already cell-consistent; we z-y average the scalar product.
        Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
        out = None
        for m_dir, m_st in zip(m_dirs, m_steps):
            M = load_vec_3d(m_dir, m_st[idx])
            mdotb = M[..., 0] * Bx + M[..., 1] * By + M[..., 2] * Bz
            out = mdotb if out is None else out + mdotb
        return reduce_zy(-out)

    def load_pi_yy_sp(idx):
        # Pi_yy,s(x, y, z) = Pi_rr,s sin^2(theta) + Pi_phiphi,s cos^2(theta),
        # with Pi_rr = component [0] and Pi_phiphi = component [1] of the
        # momentum_flux_diag_cyl moment. z-y averaged into a 1D profile.
        out = []
        for pi_dir, pi_st in zip(pi_dirs, pi_steps):
            Pi = load_vec_3d(pi_dir, pi_st[idx])
            pi_yy = Pi[..., 0] * sin2_th[None] + Pi[..., 1] * cos2_th[None]
            out.append(reduce_zy(pi_yy))
        return out

    # Baseline magnetic pressure (t = 0, first snapshot), z-y averaged.
    idx0 = common[0]
    B_0 = load_vec_3d(b_dir, b_steps[idx0])
    Bx0, By0, Bz0 = B_0[..., 0], B_0[..., 1], B_0[..., 2]
    B0_2_half = reduce_zy(0.5 * (Bx0 ** 2 + By0 ** 2 + Bz0 ** 2))

    # Per-species 1D (z-y-averaged) profiles for the species figure.
    species_T_perp_frames = []
    species_T_par_frames = []
    species_n_frames = []

    drawn = []
    prev_J_sp = None
    prev_rotM_sp = None
    for k, idx in enumerate(common):
        if idx > end_idx:
            break
        cur_J_sp = load_J_sp(idx)
        cur_rotM_sp = load_rotM_sp(idx)
        if k == 0:
            J_eff_sp = cur_J_sp
            rotM_eff_sp = cur_rotM_sp
        else:
            J_eff_sp = [0.5 * (p + c) for p, c in zip(prev_J_sp, cur_J_sp)]
            rotM_eff_sp = [0.5 * (p + c)
                           for p, c in zip(prev_rotM_sp, cur_rotM_sp)]
        prev_J_sp, prev_rotM_sp = cur_J_sp, cur_rotM_sp

        if idx < args.start_idx:
            continue

        J_eff = sum(J_eff_sp)
        rotM_eff = sum(rotM_eff_sp)
        B = load_vec_3d(b_dir, b_steps[idx])
        p_perp_sp = load_p_perp_sp(idx)
        p_perp = sum(p_perp_sp)
        q = step_quantities(B, J_eff, rotM_eff, p_perp)

        # Per-species 1D profiles (raw z-y average, mirroring fp3d_y's figure).
        q["p_perp_sp"] = [reduce_zy(pp) for pp in p_perp_sp]
        q["J_y_sp"] = [reduce_zy(J_s[..., 1]) for J_s in J_eff_sp]
        q["curlM_y_sp"] = [reduce_zy(rM[..., 1]) for rM in rotM_eff_sp]
        q["J_y"] = reduce_zy(J_eff[..., 1])
        q["curlM_y"] = reduce_zy(rotM_eff[..., 1])
        # Two independent z-y-averaged p_perp profiles for the comparison
        # panel: from the kinetic temperature_perp moment vs from the
        # magnetization -M.B (uses the instantaneous B of this frame).
        q["p_perp_T"] = reduce_zy(p_perp)
        if have_m:
            q["p_perp_MB"] = load_p_perp_MB(idx, B)
        # Bz_zy: node-de-staggered, z-y-averaged B_z (same reduction the
        # Ampere panel uses) -- needed by the parallel-correction figure.
        q["Bz_zy"] = reduce_zy(
            destagger_node(destagger_node(B[..., 2], axis=X_AX), axis=Y_AX))
        if have_tpar:
            # Parallel pressure p_par = sum_s n m <v_par^2> (1 DoF) =
            # sum_s temperature_parallel_s (no 1/2 factor, unlike p_perp).
            q["p_par"] = sum(
                reduce_zy(load_scalar_3d(td, ts[idx]))
                for td, ts in zip(tpar_dirs, tpar_steps))
        if have_pi:
            q["pi_yy_sp"] = load_pi_yy_sp(idx)
            q["pi_yy"] = sum(q["pi_yy_sp"])
        drawn.append((idx, q))

        if have_pi and have_profiles:
            frame_T_perp, frame_T_par, frame_n = [], [], []
            for t_dir, t_st, tpar_dir, tpar_st, nd, n_st in zip(
                    t_dirs, t_steps, tpar_dirs, tpar_steps, n_dirs, n_steps):
                T_perp_diag = load_scalar_3d(t_dir, t_st[idx])
                T_par_diag = load_scalar_3d(tpar_dir, tpar_st[idx])
                ns_field = load_scalar_3d(nd, n_st[idx])
                # temperature_perp = n*m*<v_perp^2> (2 DoF) -> T_perp =
                # diag/(2 n); temperature_parallel = n*m*<v_par^2> (1 DoF) ->
                # T_par = diag/n. ns_field is the diagnostic density.
                eps = 1e-30
                T_perp_per = T_perp_diag / np.maximum(2.0 * ns_field, eps)
                T_par_per = T_par_diag / np.maximum(ns_field, eps)
                frame_T_perp.append(reduce_zy(T_perp_per))
                frame_T_par.append(reduce_zy(T_par_per))
                frame_n.append(reduce_zy(ns_field))
            species_T_perp_frames.append(frame_T_perp)
            species_T_par_frames.append(frame_T_par)
            species_n_frames.append(frame_n)

    if not drawn:
        print(f"[warn] no snapshots in [{args.start_idx}, {end_idx}].")
        return

    fig, (ax_F, ax_P) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    grad_label = r"$-dp_{\perp}/dx$"
    jb_label = r"$-J_{y}^{\mathrm{tot}} B_z$"
    rot_label = r"$-(\mathrm{rot}\,M)_{y}\,B_z$"

    def render(k):
        idx, q = drawn[k]
        draw_frame(ax_F, ax_P, xs, cx, q, B0_2_half, time_avg=True,
                   grad_label=grad_label, jb_label=jb_label,
                   rot_label=rot_label, force_sym=r"F_x", avg_sub=r"z,y")
        fig.suptitle(rf"$\omega_{{pe}}\,t = {idx * const.dts:.2f}$",
                     bbox=bbox, fontsize=labelsize)
        figname = os.path.join(out_dir, f"{idx:04d}.png")
        print(f"Processing {figname} (idx={idx})")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(figname, dpi=args.dpi)
        return [ax_F, ax_P]

    anim = FuncAnimation(fig, render, frames=len(drawn), blit=False)
    anim_path = os.path.join(out_dir, args.anim_name)
    anim.save(anim_path, writer=FFMpegWriter(fps=args.anim_fps), dpi=args.dpi)

    # Time-averaged figure.
    keys = ("p_perp", "B2_half", "neg_dp_dx", "JyBz", "curlM_Bz")
    accum = {key: np.zeros_like(drawn[0][1][key]) for key in keys}
    for _, q in drawn:
        for key in keys:
            accum[key] += q[key]
    mean_q = {key: v / len(drawn) for key, v in accum.items()}

    draw_frame(ax_F, ax_P, xs, cx, mean_q, B0_2_half, time_avg=True,
               grad_label=grad_label, jb_label=jb_label, rot_label=rot_label,
               force_sym=r"F_x", avg_sub=r"z,y,t")
    fig.suptitle("")
    fig.tight_layout()
    mean_path = os.path.join(out_dir, args.mean_name)
    fig.savefig(mean_path, dpi=args.dpi)
    plt.close(fig)

    draw_ampere_figure(drawn, xs, cx, out_dir, args)

    if have_m:
        draw_pressure_figure(drawn, xs, cx, out_dir, args)
    else:
        print("[warn] pressure comparison skipped: per-species M FieldView "
              "not configured")

    if have_tpar:
        draw_parallel_correction_figure(drawn, xs, cx, out_dir, args,
                                        B0_2_half)
    else:
        print("[warn] parallel-curvature correction skipped: "
              "temperature_parallel not configured")

    if have_pi:
        draw_species_figure(
            drawn, xs, cx, out_dir, args, have_profiles,
            species_T_perp_frames, species_T_par_frames, species_n_frames)

    # Force-balance error on the time mean (rel. L2 vs ||grad p||).
    grad = mean_q["neg_dp_dx"]
    grad_scale = float(np.linalg.norm(grad))
    res_J = grad + mean_q["JyBz"]
    res_M = grad + mean_q["curlM_Bz"]
    err_J = float(np.linalg.norm(res_J) / grad_scale * 100.0) \
        if grad_scale > 0.0 else float("nan")
    err_M = float(np.linalg.norm(res_M) / grad_scale * 100.0) \
        if grad_scale > 0.0 else float("nan")

    # Magnetic-field profile error at the column center (x = cx), on the time
    # mean: |B_theory - B_model| / |B_theory - B_0| * 100%, with the same
    # definitions as the bottom panel (draw_frame):
    #   B_model  = sqrt(2 B2_half)            (raw |B|)
    #   B_theory = sqrt(B_0^2 - 2 p_perp)     (= sqrt(2 B0_2_half - 2 p_perp))
    #   B_0      = sqrt(2 B0_2_half)          (first-snapshot field)
    B_model_c = np.sqrt(2.0 * mean_q["B2_half"])
    B_theory_c = np.sqrt(np.maximum(2.0 * B0_2_half - 2.0 * mean_q["p_perp"],
                                    0.0))
    B_0_c = np.sqrt(2.0 * B0_2_half)
    ic = int(np.argmin(np.abs(xs - cx)))
    denom_B = abs(B_theory_c[ic] - B_0_c[ic])
    err_B = (abs(B_theory_c[ic] - B_model_c[ic]) / denom_B * 100.0
             if denom_B > 0.0 else float("nan"))

    print(f"Frames written to {out_dir}")
    print(f"Animation: {anim_path}")
    print(f"Time-mean: {mean_path}")
    print(f"Force-balance error (||res||_2 / ||grad p_perp||_2):")
    print(f"  J_phi B_z         : {err_J:+.3f}%")
    print(f"  (rot M)_phi B_z   : {err_M:+.3f}%")
    print(f"B-field error at center (x={xs[ic]:.2f}, i={ic}):")
    print(f"  |B_th - B_model| / |B_th - B_0| : {err_B:.3f}%")


if __name__ == "__main__":
    main()
