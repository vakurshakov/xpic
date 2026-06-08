#!/usr/bin/env python3
"""Ballistic-spectrum diagnostic for ion-sound / Landau-damping runs.

For each species s with a 3D FieldView of `<s>/J`, this script computes
the z-Fourier mode-m time series of the (x, y)-averaged parallel
current J_z:

    J_c^(m)(t) = (2/Lz) * integral_0^Lz J_z_xy(z, t) * cos(m k z) dz
               ≈ 2 * mean_iz [ J_z_xy(z_iz, t) * cos(m k z_iz) ],
    J_s^(m)(t) = (2/Lz) * integral_0^Lz J_z_xy(z, t) * sin(m k z) dz
               ≈ 2 * mean_iz [ J_z_xy(z_iz, t) * sin(m k z_iz) ],

with k = 2 pi / Lz and J_z_xy = mean over (x, y) of J_z(x, y, z, t).

In the linearised-Vlasov ballistic limit, δf(v, z, t) ~ A(v) cos(k(z - vt))
yields

    J_c^(m)(t) = q_s integral v_‖ A_s(v_‖) cos(m k v_‖ t) dv_‖
              = q_s * F_c[ v · A_s(v) ](τ = m k t),

i.e. J_c^(m) (as a function of t) is the Fourier cosine transform of
v · A_s(v). Inverting back:

    |FFT_t[ J_c^(m)(t) ](ω)| ~ | v · A_s(v) |_{v = ω / (m k)}.

So the |FFT|-vs-(ω / m k) plot maps directly onto the velocity-space
amplitude function A(v_‖):
  - smooth Gaussian-like envelope  =>  smooth A(v), no filamentation
  - sharp narrow spikes at specific v_‖  =>  PIC recurrence / ballistic
    echo (sampling of f(v) coarse enough that filaments are visible)

Plotted (2x2):
  (a) ln|J^(m)(t)| time series for several m, both species
  (b) Re J^(1)(t) vs Im J^(1)(t) parametric (clean Landau = inward
      spiral; anomalies show as kinks/loops/outward jumps)
  (c) |FFT_t[ J_c^(spec_mode) ](ω)| vs v = ω/(spec_mode k), electrons
  (d) same for ions
Vertical guides on (c)/(d): ±c_s (ion-acoustic phase velocity), ±v_Te
(electron thermal velocity), ±v_Ti (ion thermal velocity).

Optional ``--subtract-wave`` fits an exp-damped cosine at the exact
ion-acoustic frequency to J_c^(spec_mode)(t) and subtracts it before
the FFT, exposing any sub-dominant ballistic peaks that would
otherwise be buried under the wave's dominant ω = ω_s line.
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

from lib.constants import const, init_constants, mec2
from lib.plot import bbox, labelsize, ticksize

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)

from drift_kinetic_equilibrium import list_steps
from drift_kinetic_density_z import (
    extract_plasma_params,
    find_sine_perturbation,
    solve_ion_sound_dispersion,
)


SPECIES = ("electrons", "ions")
SPECIES_COLOR = {"electrons": "red", "ions": "blue"}


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


def load_Jz_xy_avg(path: str, name: str) -> np.ndarray:
    """Read a 3D vector FieldView and return the z-profile J_z(z) after
    averaging over x and y. Returns shape (Nz,)."""
    Nx, Ny, Nz = const.Nx, const.Ny, const.Nz
    raw = np.fromfile(os.path.join(path, name), dtype=np.float32,
                      count=Nx * Ny * Nz * 3)
    return raw.reshape(Nz, Ny, Nx, 3)[..., 2].mean(axis=(1, 2))


def _ion_temperature():
    """Return the ion temperature T_i (keV, code units) from the Particles
    config; falls back to 0.0 if the ion sort cannot be identified."""
    for sp in const.config.get("Particles", []):
        if float(sp.get("q", 0.0)) > 0.0:
            if "T" in sp:
                return float(sp["T"])
            comps = [sp.get("Tx"), sp.get("Ty"), sp.get("Tz")]
            comps = [float(c) for c in comps if c is not None]
            return sum(comps) / len(comps) if comps else 0.0
    return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2, 3],
                        help="z-Fourier mode indices m for the time-series "
                             "panel (default: 1 2 3).")
    parser.add_argument("--spec-mode", type=int, default=1,
                        help="z-Fourier mode m used for the spectral panels "
                             "(c)/(d). Default: 1.")
    parser.add_argument("--subtract-wave", action="store_true",
                        help="Subtract a least-squares fit of "
                             "A cos(ω_s t) + B sin(ω_s t)) * exp(-Γ t) from "
                             "J_c^(spec_mode)(t) before the FFT, exposing "
                             "sub-dominant ballistic peaks.")
    parser.add_argument("--v-zoom", type=float, default=None,
                        help="Restrict the velocity axis of the spectral "
                             "panels to ±VZOOM (c). Default: full Nyquist.")
    parser.add_argument("--plot-skip", type=int, default=1,
                        help="Drop the first N frames on panels (a) and (b) "
                             "only (FFT is computed on the full time series). "
                             "Default 1 — kills the t=0 transient that "
                             "otherwise dominates both panels' scale.")
    parser.add_argument("--b-quantile", type=float, default=0.98,
                        help="Quantile of |J^(1)| used to set the (b)-panel "
                             "axis limits, after skipping --plot-skip "
                             "frames. Default 0.98 — clips the outermost "
                             "loops so the late-time spiral is readable.")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="First diagnostic-frame index to include "
                             "(default 0).")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Last diagnostic-frame index to include "
                             "(inclusive; default: last available).")
    parser.add_argument("--out-subdir", default="J_spectrum")
    parser.add_argument("--out-name", default="J_spectrum.png")
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    init_constants(args.config)

    # Same in_dir / out_dir fallback as the other drift_kinetic scripts: if
    # the run was produced on another machine, config.OutputDirectory points
    # at a non-existent path → fall back to the directory holding config.json.
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if not os.path.isdir(const.in_dir):
        const.in_dir = config_dir
        const.out_dir = os.path.join(config_dir, "processed")

    j_dirs = {s: find_field_dir(f"{s}/J") for s in SPECIES}
    missing = [s for s, d in j_dirs.items() if d is None]
    if missing:
        print(f"[error] no 3D FieldView of <species>/J for {missing}; "
              f"add one to the config to use this diagnostic.")
        return

    vec_size = const.Nx * const.Ny * const.Nz * 3 * 4
    j_steps = {s: dict(list_steps(j_dirs[s], vec_size)) for s in SPECIES}
    common = set(j_steps[SPECIES[0]])
    for s in SPECIES[1:]:
        common &= set(j_steps[s])
    common = sorted(common)
    if not common:
        print("[error] no timesteps common to all species J diagnostics.")
        return

    end_idx = args.end_idx if args.end_idx is not None else common[-1]
    common = [idx for idx in common
              if args.start_idx <= idx <= end_idx]
    if not common:
        print(f"[error] no frames in [{args.start_idx}, {end_idx}].")
        return

    z = (np.arange(const.Nz) + 0.5) * const.dz
    Lz = const.Lz
    k = 2.0 * np.pi / Lz

    modes = sorted(set(args.modes) | {args.spec_mode, 1})
    cos_zm = {m: np.cos(m * k * z) for m in modes}
    sin_zm = {m: np.sin(m * k * z) for m in modes}

    times = np.array([idx * const.dts for idx in common], dtype=float)
    Nt = len(common)

    # J_c[m][s][k_t] = z-cos mode-m of the xy-averaged J_z, at common[k_t].
    J_c = {m: {s: np.zeros(Nt) for s in SPECIES} for m in modes}
    J_s = {m: {s: np.zeros(Nt) for s in SPECIES} for m in modes}

    print(f"Loading {Nt} frames per species ({len(SPECIES)} species)…")
    for k_t, idx in enumerate(common):
        for s in SPECIES:
            prof = load_Jz_xy_avg(j_dirs[s], j_steps[s][idx])
            for m in modes:
                J_c[m][s][k_t] = 2.0 * float(np.mean(prof * cos_zm[m]))
                J_s[m][s][k_t] = 2.0 * float(np.mean(prof * sin_zm[m]))

    # --- Dispersion: omega_s, Gamma, v_te, c_s -----------------------------
    plasma = extract_plasma_params()
    _, wn_z = find_sine_perturbation()
    omega_s = Gamma = T_period = None
    k_wave = k
    if plasma is not None and wn_z:
        k_wave = 2.0 * np.pi * wn_z / Lz
        omega_s, Gamma = solve_ion_sound_dispersion(k_wave, plasma)
        T_period = 2.0 * np.pi / omega_s
    else:
        omega_s = float("nan")
        Gamma = 0.0
        T_period = float("nan")

    c_s = plasma["c_s"] if plasma is not None else float("nan")
    v_Te = plasma["v_Te"] if plasma is not None else float("nan")
    T_i = _ion_temperature()
    v_Ti = (np.sqrt(T_i / (plasma["m_i"] * mec2))
            if plasma is not None and T_i > 0.0 else float("nan"))

    print(f"Plasma / dispersion:")
    print(f"  k  (loaded)  = {k_wave:.6g}  rad/(c/wpe)")
    print(f"  c_s          = {c_s:.6g} c")
    print(f"  v_Te         = {v_Te:.6g} c")
    print(f"  v_Ti         = {v_Ti:.6g} c")
    print(f"  omega_s      = {omega_s:.6g} wpe")
    print(f"  Gamma        = {Gamma:.6g} wpe")
    print(f"  T_period     = {T_period:.6g} 1/wpe")
    print(f"  spec_mode    = {args.spec_mode}")
    print(f"  frames       = {Nt}  ({times[0]:.3g} .. {times[-1]:.3g} 1/wpe)")

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # --- Optional: subtract fitted ion-sound wave from J_c^(spec_mode) ----
    spec_m = args.spec_mode
    Jc_for_fft = {s: J_c[spec_m][s].copy() for s in SPECIES}
    if args.subtract_wave and np.isfinite(omega_s) and omega_s > 0.0:
        decay = np.exp(-Gamma * (times - times[0]))
        basis = np.column_stack([
            np.cos(omega_s * times) * decay,
            np.sin(omega_s * times) * decay,
        ])
        for s in SPECIES:
            y = Jc_for_fft[s]
            coeff, *_ = np.linalg.lstsq(basis, y, rcond=None)
            fit = basis @ coeff
            Jc_for_fft[s] = y - fit
            print(f"  wave fit ({s}): A_cos={coeff[0]:+.3e}, "
                  f"A_sin={coeff[1]:+.3e}; "
                  f"|residual|/|signal| = "
                  f"{np.linalg.norm(Jc_for_fft[s]) / max(np.linalg.norm(y), 1e-30):.3f}")

    # --- Time FFT: |FFT[J_c^(spec_mode)(t)](ω)| ----------------------------
    # Uniform sampling is required; we rely on the diagnostic period being
    # constant (drift_kinetic Simulation enforces this when diagnose_period
    # is set in Geometry).
    window = np.hanning(Nt)
    fft_spec = {}
    for s in SPECIES:
        # Subtract per-species mean to suppress the DC bin from any baseline
        # drift (numerical heating, secular trend) — it would otherwise
        # dominate the spectrum near v_‖ = 0.
        sig = Jc_for_fft[s] - np.mean(Jc_for_fft[s])
        fft_spec[s] = np.abs(np.fft.fftshift(np.fft.fft(sig * window)))
    omega_axis = np.fft.fftshift(np.fft.fftfreq(Nt, d=const.dts)) * 2.0 * np.pi
    v_axis = omega_axis / (spec_m * k)

    # --- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 11.0))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]
    panel_bbox = dict(facecolor="white", edgecolor="none",
                      alpha=0.75, boxstyle="round,pad=0.25")

    t_norm = T_period if np.isfinite(T_period) and T_period > 0.0 else 1.0
    t_label = r"$t/T$" if t_norm != 1.0 else r"$t\,\omega_{pe}$"

    # (a) ln|J^(m)(t)| for several m, both species ------------------------
    # Drop the first --plot-skip frames so the t≈0 transient (where J has
    # not yet relaxed and amplitudes are O(loading)) doesn't compress the
    # rest of the curve's dynamic range.
    skip = max(0, min(int(args.plot_skip), max(Nt - 2, 0)))
    t_plot = times[skip:]
    line_styles = ("-", "--", ":", "-.")
    for s in SPECIES:
        color = SPECIES_COLOR[s]
        for j, m in enumerate(sorted(args.modes)):
            ampl = np.hypot(J_c[m][s][skip:], J_s[m][s][skip:])
            ax_a.plot(t_plot / t_norm, np.log(np.maximum(ampl, 1e-30)),
                      color=color, linestyle=line_styles[j % len(line_styles)],
                      linewidth=1.6,
                      label=rf"{s[:1]}, $m={m}$")
    ax_a.set_xlabel(t_label, fontsize=labelsize)
    ax_a.set_ylabel(r"$\ln|J_z^{(m)}(t)|$", fontsize=labelsize)
    ax_a.grid(True, alpha=0.3)
    ax_a.tick_params(labelsize=ticksize)
    ax_a.legend(loc="upper right", fontsize=ticksize, ncol=2)
    ax_a.text(0.03, 0.97, "(а)", transform=ax_a.transAxes,
              ha="left", va="top", fontsize=labelsize, bbox=panel_bbox)

    # (b) Parametric Re J^(1)(t) vs Im J^(1)(t), per species --------------
    # Use a LineCollection coloured by t to highlight the time direction:
    # a clean Landau decay traces an inward spiral; anomalies show as
    # outward kinks/loops at a specific t.
    # Skip --plot-skip leading frames here too: the t=0 transient otherwise
    # sets the axis scale and crushes the late-time spiral into a single
    # pixel near the origin.
    lc = None
    for s in SPECIES:
        x = J_c[1][s][skip:]
        y = -J_s[1][s][skip:]              # Im J^(1) = -J_s
        if x.size < 2:
            continue
        points = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", linewidth=1.2, alpha=0.85)
        lc.set_array(t_plot[:-1] / t_norm)
        ax_b.add_collection(lc)
        ax_b.scatter(x[0], y[0], color=SPECIES_COLOR[s], marker="o", s=50,
                     zorder=5, edgecolor="black", linewidth=0.6,
                     label=rf"{s} start")
        ax_b.scatter(x[-1], y[-1], color=SPECIES_COLOR[s], marker="X", s=60,
                     zorder=5, edgecolor="black", linewidth=0.6,
                     label=rf"{s} end")
    ax_b.axhline(0, color="k", linewidth=0.6, alpha=0.5)
    ax_b.axvline(0, color="k", linewidth=0.6, alpha=0.5)
    # Axis limits from the post-skip data, clipped at the requested
    # quantile so a single outlier loop can't blow up the scale.
    all_xy = np.concatenate([J_c[1][s][skip:] for s in SPECIES]
                            + [-J_s[1][s][skip:] for s in SPECIES])
    q = float(np.clip(args.b_quantile, 0.0, 1.0))
    if all_xy.size and q > 0.0:
        lim = float(np.quantile(np.abs(all_xy), q)) * 1.1
    else:
        lim = float(np.max(np.abs(all_xy))) * 1.1 if all_xy.size else 1.0
    if not np.isfinite(lim) or lim <= 0.0:
        lim = 1.0
    ax_b.set_xlim(-lim, lim)
    ax_b.set_ylim(-lim, lim)
    ax_b.set_xlabel(r"$\mathrm{Re}\,J^{(1)}(t)$", fontsize=labelsize)
    ax_b.set_ylabel(r"$\mathrm{Im}\,J^{(1)}(t)$", fontsize=labelsize)
    ax_b.grid(True, alpha=0.3)
    ax_b.tick_params(labelsize=ticksize)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.legend(loc="upper right", fontsize=ticksize, framealpha=0.9)
    ax_b.text(0.03, 0.97, "(б)", transform=ax_b.transAxes,
              ha="left", va="top", fontsize=labelsize, bbox=panel_bbox)
    if lc is not None:
        cbar = fig.colorbar(lc, ax=ax_b, fraction=0.046, pad=0.04)
        cbar.set_label(t_label, fontsize=ticksize)
        cbar.ax.tick_params(labelsize=ticksize)

    # (c)/(d) FFT spectra vs v = ω / (spec_mode k) ------------------------
    v_lim = (args.v_zoom if args.v_zoom is not None
             else float(np.max(np.abs(v_axis))))
    for ax, s, letter in ((ax_c, "electrons", "(в)"),
                          (ax_d, "ions", "(г)")):
        color = SPECIES_COLOR[s]
        eps = max(np.max(fft_spec[s]) * 1e-12, 1e-30)
        ax.semilogy(v_axis, np.maximum(fft_spec[s], eps),
                    color=color, linewidth=1.4)
        ymin, ymax = ax.get_ylim()
        # Reference velocities: ion-sound phase, electron and ion thermal.
        refs = [
            (c_s,  r"$+c_s$",    "black", "--"),
            (-c_s, r"$-c_s$",    "black", "--"),
            (v_Te, r"$+v_{Te}$", "gray",  ":"),
            (-v_Te,r"$-v_{Te}$", "gray",  ":"),
            (v_Ti, r"$+v_{Ti}$", "gold",  ":"),
            (-v_Ti,r"$-v_{Ti}$", "gold",  ":"),
        ]
        for v_ref, label, lc_color, ls in refs:
            if not np.isfinite(v_ref) or abs(v_ref) > v_lim:
                continue
            ax.axvline(v_ref, color=lc_color, linestyle=ls, linewidth=1.0,
                       alpha=0.7)
            ax.text(v_ref, 0.95, label, color=lc_color, fontsize=ticksize,
                    transform=ax.get_xaxis_transform(),
                    ha="left", va="top")
        ax.set_xlim(-v_lim, v_lim)
        ax.set_xlabel(r"$v_\parallel = \omega\,/\,(m\,k)$  $(c)$",
                      fontsize=labelsize)
        ax.set_ylabel(rf"$|\widehat{{J_z^{{({spec_m})}}}}(\omega)|$",
                      fontsize=labelsize)
        ax.grid(True, alpha=0.3, which="both")
        ax.tick_params(labelsize=ticksize)
        ax.set_title(s, fontsize=labelsize, bbox=bbox)
        ax.text(0.03, 0.97, letter, transform=ax.transAxes,
                ha="left", va="top", fontsize=labelsize, bbox=panel_bbox)

    title = (rf"$k = {k_wave:.4g}\,\omega_{{pe}}/c,\ m_{{spec}} = {spec_m}"
             rf",\ T = {T_period:.4g}\,/\omega_{{pe}}"
             rf",\ \Gamma = {Gamma:.3e}\,\omega_{{pe}}$")
    if args.subtract_wave:
        title += r" — wave subtracted"
    fig.suptitle(title, fontsize=labelsize)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(out_dir, args.out_name)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Figure written to {out_path}")

    # --- CSV dump of the mode time series (downstream comparisons) -------
    csv_path = os.path.join(out_dir, "J_modes.csv")
    cols = [times]
    header = ["t"]
    for s in SPECIES:
        for m in sorted(args.modes):
            cols.append(J_c[m][s])
            cols.append(J_s[m][s])
            header.append(f"{s}_Jc_m{m}")
            header.append(f"{s}_Js_m{m}")
    np.savetxt(csv_path, np.column_stack(cols), delimiter=",",
               header=",".join(header), comments="", fmt="%.6e")
    print(f"CSV written to {csv_path}")

    # --- Quick numerical report -------------------------------------------
    # Cumulative phase angle of J^(1)(t): for a clean exp-damped cos wave
    # the unwrapped phase grows linearly at ω_s. A late-time jump (or a
    # discontinuity) is what a recurrence/echo looks like.
    if np.isfinite(omega_s) and omega_s > 0.0:
        for s in SPECIES:
            z_t = J_c[1][s] + 1j * J_s[1][s]
            mag = np.abs(z_t)
            phi = np.unwrap(np.angle(z_t))
            # Slope omega_fit ≈ (phi[-1] - phi[0]) / (t[-1] - t[0]).
            if len(times) > 2 and times[-1] > times[0]:
                omega_fit = (phi[-1] - phi[0]) / (times[-1] - times[0])
                rel = (omega_fit - omega_s) / omega_s * 100.0
                print(f"  {s}: <dphi/dt> = {omega_fit:+.4e}  "
                      f"(theory ω_s = {omega_s:+.4e}, "
                      f"rel diff {rel:+.3f}%)")
            # Late-time vs early-time amplitude.
            n_window = max(3, Nt // 10)
            early = float(np.mean(mag[:n_window]))
            late = float(np.mean(mag[-n_window:]))
            if early > 0.0:
                print(f"  {s}: |J^(1)| early/late mean = "
                      f"{early:.3e} / {late:.3e}  ratio = "
                      f"{late / early:.3e}")


if __name__ == "__main__":
    main()
