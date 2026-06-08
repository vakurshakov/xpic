#!/usr/bin/env python3
"""Two-panel (force-balance, perpendicular pressure) diagnostic for
drift_kinetic runs.

Splits the second column of drift_kinetic_force_y.py into a standalone
figure with two stacked panels (force-balance on top, pressure on
bottom) and draws PNG frames over a user-selected idx range
(``--start-idx``, ``--end-idx``). On top of the per-frame PNGs the script
writes:
    - an MP4 video out of those frames (excluding the time mean), and
    - a final extra PNG with the time-averaged values over the same
      idx range.

The plotted quantities and panel layout are taken verbatim from
drift_kinetic_force_y.py (panels 2 and 4); only the colors and
y-axis labels were changed:
    - p_perp-derived curves (-dp_perp/dx, p_perp) are drawn dark green
    - B-derived curves (-J_phi * B_z, B_0^2/2 - B^2/2) are brown
    - top panel y-label is "F_r", bottom panel y-label is "Давление"

J and M live half a step ahead of B/E, so they are backward-averaged
between consecutive snapshots, X_eff(k) = (X[k-1] + X[k]) / 2; the very
first frame uses the raw value (zero anyway).
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
from drift_kinetic_force_y import find_plane_y_dir, load_plane_y_vec


COLOR_P = "darkgreen"
COLOR_B = "saddlebrown"
COLOR_BL = "blue"

YLIM_F = (-1.3e-4, 1.3e-4)
# Bottom panel now shows B-fields (B_theory and |B|), set in main() from the
# observed B0 so the panel framing follows the actual field magnitude.
YLIM_P = (0.15, 0.25)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--center-x", type=float, default=None,
                        help="Cylinder axis x-coordinate (default: auto)")
    parser.add_argument("--z-mode", choices=("avg", "center"), default="avg",
                        help="Reduction of the (Nz, Nx) slice to a 1D "
                             "profile vs x. 'avg' averages along z; "
                             "'center' uses Nz//2.")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="First diagnostic-frame index to draw "
                             "(inclusive). Earlier frames are still "
                             "iterated to keep the half-step backward "
                             "averages of J/M consistent.")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Last diagnostic-frame index to draw "
                             "(inclusive). Default: last available frame.")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--anim-fps", type=int, default=10)
    parser.add_argument("--anim-name", default="force_pressure_y.mp4",
                        help="MP4 filename inside <out-subdir>")
    parser.add_argument("--mean-name", default="time_mean.png",
                        help="Time-averaged figure filename inside "
                             "<out-subdir>")
    parser.add_argument("--out-subdir", default="force_pressure_y",
                        help="Subdirectory under <out_dir>/processed for "
                             "PNGs / MP4")
    return parser.parse_args()


def step_quantities(B, J_eff, M_eff, z_mode):
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
    Mx, My, Mz = M_eff[..., 0], M_eff[..., 1], M_eff[..., 2]

    p_perp_2d = -(Mx * Bx + My * By + Mz * Bz)
    B2_half_2d = 0.5 * (Bx * Bx + By * By + Bz * Bz)

    dMx_dz = np.gradient(Mx, const.dz, axis=0)
    dMz_dx = np.gradient(Mz, const.dx, axis=1)
    curlM_y_2d = dMx_dz - dMz_dx

    Jy_tot_2d = J_eff[..., 1] + curlM_y_2d
    dp_dx_2d = np.gradient(p_perp_2d, const.dx, axis=1)

    def red(field2d):
        if z_mode == "avg":
            return field2d.mean(axis=0)
        return field2d[const.Nz // 2]

    return {
        "p_perp": red(p_perp_2d),
        "B2_half": red(B2_half_2d),
        "neg_dp_dx": red(-dp_dx_2d),
        "JyBz": red(Jy_tot_2d * Bz),
        "curlM_Bz": red(curlM_y_2d * Bz),
    }


def style_axes(ax_F, ax_P, xs, cx, time_avg=False, ylim_P=None,
               force_sym=r"F_r", avg_sub=None):
    sub = avg_sub if avg_sub is not None else ("z,t" if time_avg else "z")
    lbl = labelsize + 4 if time_avg else labelsize
    tks = ticksize + 4 if time_avg else ticksize
    for ax, ylim, ylabel in (
        (ax_F, YLIM_F,
         rf"$\langle {force_sym} \rangle_{{{sub}}}$"
         rf" ($n_0 m_e c \omega_{{pe}}$)"),
        (ax_P, ylim_P if ylim_P is not None else YLIM_P,
         rf"$\langle B \rangle_{{{sub}}}$ ($m_e c \omega_{{pe}}/e$)"),
    ):
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.axvline(cx, color="k", linewidth=0.6, alpha=0.3, linestyle=":")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=tks)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(ylabel, fontsize=lbl)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                            useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(tks)
    ax_P.set_xlabel(r"$x~(c/\omega_{pe})$", fontsize=lbl)


def draw_frame(ax_F, ax_P, xs, cx, q, B0_2_half, time_avg=False, ylim_P=None,
               grad_label=r"$-dp_{\perp}/dr$",
               grad_raw_label=r"$-dp_{\perp}/dr$",
               jb_label=r"$-J_{\varphi}^{\mathrm{tot}} B_z$",
               rot_label=None, force_sym=r"F_r", avg_sub=None):
    ax_F.cla()
    ax_P.cla()
    style_axes(ax_F, ax_P, xs, cx, time_avg=time_avg, ylim_P=ylim_P,
               force_sym=force_sym, avg_sub=avg_sub)

    # B_theory(x) = sqrt(B_0^2 - 2 p_perp(x)); |B|(x) = sqrt(2 * B^2/2).
    B0_sq = 2.0 * B0_2_half
    B_theory = np.sqrt(np.maximum(B0_sq - 2.0 * q["p_perp"], 0.0))
    B_mag = np.sqrt(np.maximum(2.0 * q["B2_half"], 0.0))

    lbl = labelsize + 4 if time_avg else labelsize
    leg_fs = ticksize + 1 if time_avg else ticksize
    if rot_label is None:
        rot_label = (r"$-(\mathrm{rot}\,M)_{\varphi}\,B_z$" if time_avg
                     else r"$-(\nabla\times M)_{\varphi}\,B_z$")
    B_theory_label = (r"$B_{\mathrm{theory}}$" if time_avg
                      else r"$B_{\mathrm{theory}} = "
                           r"\sqrt{B_0^2 - 2 p_{\perp}}$")
    B_mag_label = r"$B_{\mathrm{model}}$" if time_avg else r"$|B|$"

    ax_F.plot(xs, q["neg_dp_dx"], color=COLOR_P, linewidth=2.0,
              label=grad_label)
    if "neg_dp_perp_dx" in q:
        ax_F.plot(xs, q["neg_dp_perp_dx"], color="green", linewidth=2.0,
                  linestyle="--", label=grad_raw_label)
    ax_F.plot(xs, -q["JyBz"], color=COLOR_BL, linewidth=2.0,
              label=jb_label)
    ax_F.plot(xs, -q["curlM_Bz"], color=COLOR_B, linewidth=2.0,
              linestyle="--", label=rot_label)
    if not time_avg:
        ax_F.set_title(r"Баланс сил", fontsize=labelsize, bbox=bbox)
    ax_F.legend(fontsize=leg_fs, loc="center",
                bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0).set_zorder(10)

    ax_P.plot(xs, B_theory, color=COLOR_P, linewidth=2.0,
              label=B_theory_label)
    ax_P.plot(xs, B_mag, color=COLOR_B, linewidth=2.0, label=B_mag_label)
    if not time_avg:
        ax_P.set_title(r"Профиль магнитного поля",
                       fontsize=labelsize, bbox=bbox)
    ax_P.legend(fontsize=leg_fs + 4 if time_avg else leg_fs,
                loc="center", bbox_to_anchor=(1.0, 1.0),
                borderaxespad=0.0).set_zorder(10)

    if time_avg:
        panel_bbox = dict(facecolor="white", edgecolor="none",
                          alpha=0.75, boxstyle="round,pad=0.25")
        ax_F.text(0.03, 0.05, "(а)", transform=ax_F.transAxes,
                  ha="left", va="bottom", fontsize=lbl, bbox=panel_bbox)
        ax_P.text(0.03, 0.05, "(б)", transform=ax_P.transAxes,
                  ha="left", va="bottom", fontsize=lbl, bbox=panel_bbox)


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

    end_idx = args.end_idx if args.end_idx is not None else common[-1]
    if args.start_idx > end_idx:
        print(f"[error] start_idx {args.start_idx} > end_idx {end_idx}")
        return

    cx = args.center_x if args.center_x is not None else auto_center()[0]

    print(f"Cylinder axis x: {cx:.3f}   (z-mode: {args.z_mode}, "
          f"start_idx: {args.start_idx}, end_idx: {end_idx})")

    xs = (np.arange(const.Nx) + 0.5) * const.dx

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Baseline (t = 0) magnetic pressure for the deltaB^2 panel: always the
    # very first available snapshot, regardless of --start-idx. Same
    # convention as drift_kinetic_force_y.py.
    idx0 = common[0]
    B_0 = load_plane_y_vec(b_dir, b_steps[idx0])
    M_0 = load_plane_y_vec(m_dir, m_steps[idx0])
    J_0 = load_plane_y_vec(j_dir, j_steps[idx0])
    q0 = step_quantities(B_0, J_0, M_0, args.z_mode)
    B0_2_half = q0["B2_half"].copy()

    # Pre-compute reduced 1D quantities for every drawn frame so the
    # FuncAnimation render is cheap and the time mean is exact.
    drawn = []
    prev_M = None
    prev_J = None
    for k, idx in enumerate(common):
        if idx > end_idx:
            break
        cur_M = load_plane_y_vec(m_dir, m_steps[idx])
        cur_J = load_plane_y_vec(j_dir, j_steps[idx])
        if k == 0:
            M_eff = cur_M
            J_eff = cur_J
        else:
            M_eff = 0.5 * (prev_M + cur_M)
            J_eff = 0.5 * (prev_J + cur_J)
        prev_M = cur_M
        prev_J = cur_J

        if idx < args.start_idx:
            continue

        B = load_plane_y_vec(b_dir, b_steps[idx])
        q = step_quantities(B, J_eff, M_eff, args.z_mode)
        drawn.append((idx, q))

    if not drawn:
        print(f"[warn] no snapshots in [{args.start_idx}, {end_idx}].")
        return

    fig, (ax_F, ax_P) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    def render(k):
        idx, q = drawn[k]
        draw_frame(ax_F, ax_P, xs, cx, q, B0_2_half)
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

    # Time-averaged figure (separate from the animation).
    keys = ("p_perp", "B2_half", "neg_dp_dx", "JyBz", "curlM_Bz")
    accum = {key: np.zeros_like(drawn[0][1][key]) for key in keys}
    for _, q in drawn:
        for key in keys:
            accum[key] += q[key]
    mean_q = {key: v / len(drawn) for key, v in accum.items()}

    draw_frame(ax_F, ax_P, xs, cx, mean_q, B0_2_half, time_avg=True)
    fig.suptitle("")
    fig.tight_layout()
    mean_path = os.path.join(out_dir, args.mean_name)
    fig.savefig(mean_path, dpi=args.dpi)
    plt.close(fig)

    # Max relative errors on the time-averaged profiles.
    grad = mean_q["neg_dp_dx"]
    grad_scale = float(np.max(np.abs(grad)))
    res_J = grad + mean_q["JyBz"]         # -dp/dr - (-J*B) = neg_dp_dx + JyBz
    res_M = grad + mean_q["curlM_Bz"]     # same against -(rot M)*B_z
    err_J = float(np.max(np.abs(res_J)) / grad_scale * 100.0) \
        if grad_scale > 0.0 else float("nan")
    err_M = float(np.max(np.abs(res_M)) / grad_scale * 100.0) \
        if grad_scale > 0.0 else float("nan")

    B0_sq = 2.0 * B0_2_half
    B_theory_mean = np.sqrt(np.maximum(B0_sq - 2.0 * mean_q["p_perp"], 0.0))
    B_model_mean = np.sqrt(np.maximum(2.0 * mean_q["B2_half"], 0.0))
    diff_B = np.abs(B_model_mean - B_theory_mean)
    i_max = int(np.argmax(diff_B))
    B_at_max = float(B_theory_mean[i_max])
    err_B = float(diff_B[i_max] / B_at_max * 100.0) \
        if B_at_max > 0.0 else float("nan")

    print(f"Frames written to {out_dir}")
    print(f"Animation: {anim_path}")
    print(f"Time-mean: {mean_path}")
    print(f"Max force-balance error (rel. max|grad p_perp|):")
    print(f"  J_phi B_z         : {err_J:+.3f}%")
    print(f"  (rot M)_phi B_z   : {err_M:+.3f}%")
    print(f"Max equilibrium error |B_model - B_theory| / B_theory at "
          f"x={float(xs[i_max]):.3f}: {err_B:+.3f}%")


if __name__ == "__main__":
    main()
