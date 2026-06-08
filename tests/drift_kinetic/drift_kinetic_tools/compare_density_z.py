#!/usr/bin/env python3
"""Compare ln|delta n_c(t)| between two drift_kinetic runs on one figure.

Mirrors the right panel of `drift_kinetic_density_z.py`'s ``delta_n.png``,
but for *two* runs at once. Per run, the script computes the modulus of
the mode-1 density Fourier coefficient

    delta n_c(t) = (2/Lz) * integral_0^Lz <n_ions>_{x,y}/n0 * sin(k z) dz,

then plots ``ln|delta n_c(t)|`` against ``t/T_theory`` (T from the first
run). Legend labels carry the per-species Np from each config so the
comparison reads at a glance, e.g. "Np = 10000" vs "Np = 2000".

Output: <run_a out_dir>/processed/<out_subdir>/<filename>.
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
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)

# Re-use the helpers from the single-run script. They all read `const`,
# so we must call them after each `init_constants(...)`.
from drift_kinetic_density_z import (
    candidate_dirs,
    extract_plasma_params,
    find_sine_perturbation,
    iter_density_diagnostics,
    load_frame,
    locate_dir,
    solve_ion_sound_dispersion,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_a", help="config.json of the primary run")
    parser.add_argument("config_b", help="config.json of the comparison run")
    parser.add_argument("--n0", type=float, default=1.0,
                        help="Background density used to normalise the "
                             "Fourier mode (default 1.0).")
    parser.add_argument("--out-subdir", default="compare_density_z",
                        help="Subdirectory under run-A's <out_dir>/processed.")
    parser.add_argument("--filename", default="compare_density_z.png",
                        help="Output PNG name.")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--t-max-T", type=float, default=None,
                        help="Upper x-axis limit of the right plot in units "
                             "of T (e.g. 2.75 means draw up to 2.75 T). "
                             "Default: full data span.")
    parser.add_argument("--species", default="ions",
                        help="Species whose density mode-1 modulus is "
                             "compared (default: ions).")
    parser.add_argument("--theory-delta-n", type=float, nargs="+",
                        default=[0.0276, 0.023],
                        help="One or more initial loading amplitudes "
                             "delta_n_0 used to anchor the theoretical "
                             "Landau envelopes delta_n_0 * exp(-Gamma t). "
                             "Each value produces a separate curve. Pass a "
                             "single non-positive value to skip the theory.")
    return parser.parse_args()


def _ions_row_for_species(species_name):
    """Locate the density-diagnostic frames for `species_name` (or the
    first available species, if it is missing). Returns (dir_path, steps)
    or (None, None)."""
    available = list(iter_density_diagnostics())
    chosen = next(((sp, ex) for sp, ex in available if sp == species_name),
                  None)
    if chosen is None and available:
        chosen = available[0]
        print(f"[warn] species '{species_name}' not in config; "
              f"falling back to '{chosen[0]}'")
    if chosen is None:
        return None, None, None
    sp, explicit = chosen
    dir_path, steps = locate_dir(sp, explicit)
    return sp, dir_path, steps


def _species_param(name, key, default=None):
    for s in const.config.get("Particles", []):
        if s.get("sort_name") == name:
            return s.get(key, default)
    return default


def load_run(config_path, species_name, n0):
    """Snapshot everything needed for one run into a plain dict so we can
    re-init `const` for the next run without losing the result."""
    init_constants(config_path)

    # Same in_dir / out_dir fallback that density_z.py applies for output
    # trees copied from another machine.
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if not os.path.isdir(const.in_dir):
        const.in_dir = config_dir
        const.out_dir = os.path.join(config_dir, "processed")

    sp, dir_path, steps = _ions_row_for_species(species_name)
    if dir_path is None:
        raise RuntimeError(
            f"{config_path}: no density diagnostic frames "
            f"(looked for {candidate_dirs(species_name, None)})")

    Np = _species_param(sp, "Np")
    z = (np.arange(const.Nz) + 0.5) * const.dz
    modes_extra = (2, 3, 4, 5, 6)
    sin_z_modes = {m: np.sin(m * 2.0 * np.pi * z / const.Lz)
                   for m in (1, *modes_extra)}

    # Match the density_z normalisation: average over (x, y) per frame,
    # then mode-m coefficient = 2 * mean_z[(<n>_{xy}/n0) * sin(m kz)].
    times = []
    mods = []
    coeffs = []
    mods_extra = {m: [] for m in modes_extra}
    profile0 = None
    for idx, name in steps:
        data = load_frame(dir_path, name)
        if data is None:
            continue
        profile = data.mean(axis=(1, 2))                # shape (Nz,)
        if profile0 is None:
            profile0 = profile.copy()
        norm = profile / n0
        coeff = 2.0 * float(np.mean(norm * sin_z_modes[1]))
        times.append(idx * const.dts)
        mods.append(abs(coeff))
        coeffs.append(coeff)
        for m in modes_extra:
            mods_extra[m].append(
                abs(2.0 * float(np.mean(norm * sin_z_modes[m]))))

    plasma = extract_plasma_params()
    _, wn_z = find_sine_perturbation()
    T_theory = None
    Gamma = None
    if plasma is not None and wn_z:
        k_wave = 2.0 * np.pi * wn_z / const.Lz
        omega_r, Gamma = solve_ion_sound_dispersion(k_wave, plasma)
        T_theory = 2.0 * np.pi / omega_r

    return {
        "config_path": config_path,
        "species": sp,
        "Np": Np,
        "times": np.array(times, dtype=float),
        "mod": np.array(mods, dtype=float),
        "coeff": np.array(coeffs, dtype=float),
        "T_theory": T_theory,
        "Gamma": Gamma,
        "out_dir": const.out_dir,
        "z": z,
        "Lz": float(const.Lz),
        "profile0": profile0,
        "n0": float(n0),
        "mods_extra": {m: np.array(v, dtype=float)
                       for m, v in mods_extra.items()},
    }


def main():
    args = parse_args()
    run_a = load_run(args.config_a, args.species, args.n0)
    out_dir_a = run_a["out_dir"]                       # capture before B
    run_b = load_run(args.config_b, args.species, args.n0)

    if run_a["T_theory"] is None:
        raise RuntimeError("Could not derive T_theory for run A; aborting")
    T_norm = run_a["T_theory"]

    def label(run):
        np_val = run["Np"]
        if np_val is None:
            return os.path.basename(os.path.dirname(run["config_path"]))
        # Render as scientific TeX, e.g. 10000 -> 10^4, 2000 -> 2\cdot10^3.
        try:
            n = int(np_val)
        except (TypeError, ValueError):
            return rf"$N_{{ppc}} = {np_val}$"
        if n == 0:
            return r"$N_{ppc} = 0$"
        exp = len(str(abs(n))) - 1
        base = n / 10 ** exp
        if abs(base - round(base)) < 1e-9:
            base_i = int(round(base))
            if base_i == 1:
                tex = rf"10^{{{exp}}}"
            else:
                tex = rf"{base_i}\cdot10^{{{exp}}}"
        else:
            tex = rf"{base:g}\cdot10^{{{exp}}}"
        return rf"$N_{{ppc}} = {tex}$"

    fig, (ax_prof, ax) = plt.subplots(1, 2, figsize=(15.0, 7.0))

    # --- Left panel: initial density profile from run A (mirrors
    # delta_n.png left panel in drift_kinetic_density_z.py). ------------
    ylim_lo, ylim_hi = 0.94, 1.06
    amp_hi, amp_lo = 1.03, 0.97
    panel_bbox = dict(facecolor="white", edgecolor="none",
                      alpha=0.75, boxstyle="round,pad=0.25")

    def draw_amplitude_lines(axx, x_label_pos, ha="left"):
        for level in (amp_hi, amp_lo):
            axx.axhline(level, color="tab:gray", linewidth=1.0,
                        linestyle="--", alpha=0.8)
            axx.text(x_label_pos, level + 0.002, f"{level:.2f}",
                     color="tab:gray", fontsize=ticksize,
                     ha=ha, va="bottom")

    if run_a["profile0"] is not None:
        ax_prof.plot(run_a["z"], run_a["profile0"] / run_a["n0"],
                     color="red", marker="o", linestyle="-",
                     linewidth=3.0, markersize=9.0, label=r"$n_i/n_0$")

    ax_prof.set_xlim(0.0, run_a["Lz"])
    ax_prof.set_ylim(ylim_lo, ylim_hi)
    ax_prof.set_xlabel(r"$z~(c/\omega_{pe})$", fontsize=labelsize + 4)
    ax_prof.set_ylabel(rf"$\langle \frac{{n}}{{n_{{0}}}} \rangle_{{x,y}}(z)$",
                       fontsize=labelsize + 4)
    ax_prof.tick_params(labelsize=ticksize + 4)
    ax_prof.grid(True, alpha=0.3)
    ax_prof.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    draw_amplitude_lines(ax_prof, x_label_pos=0.01 * run_a["Lz"], ha="left")
    ax_prof.text(0.97, 0.97, "(а)", transform=ax_prof.transAxes,
                 ha="right", va="top", fontsize=labelsize + 4, bbox=panel_bbox)
    ax_prof.legend(loc="lower left", fontsize=labelsize + 4)
    ax_prof.set_box_aspect(1)

    eps = 1e-30                          # guard log(0) at exact zeros
    for run, color in ((run_a, "red"), (run_b, "blue")):
        t_over_T = run["times"] / T_norm
        ax.plot(t_over_T, np.log(np.maximum(run["mod"], eps)),
                color=color, marker="o", markersize=4.0,
                linewidth=2.0, label=label(run))

    # Theoretical Landau envelopes from run A: ln(delta_n_0) - Gamma t,
    # one per requested initial amplitude. Same anchoring choice as
    # density_z.py's delta_n.png (envelope tied to the initial loading
    # amplitude, not to a data peak). Colors are paired with the data
    # curves above so reader can tell which theory belongs to which run.
    Gamma_a = run_a["Gamma"]
    deltas = [d for d in args.theory_delta_n if d > 0.0]
    theory_colors = ("darkred", "darkblue", "darkgreen", "darkorange")
    theory_used_colors = []
    if deltas and Gamma_a is not None:
        t_max = max(
            float(run_a["times"][-1]) if len(run_a["times"]) else 0.0,
            float(run_b["times"][-1]) if len(run_b["times"]) else 0.0,
        )
        t_th = np.linspace(0.0, t_max, 1200)
        for k, delta_n0 in enumerate(deltas):
            color = theory_colors[k % len(theory_colors)]
            theory_used_colors.append(color)
            ln_env = np.log(delta_n0) - Gamma_a * t_th
            ax.plot(t_th / T_norm, ln_env, color=color, linewidth=1.0,
                    linestyle="--", alpha=0.9,
                    label="_nolegend_")

    x_max = max(
        float(run_a["times"][-1]) / T_norm if len(run_a["times"]) else 0.0,
        float(run_b["times"][-1]) / T_norm if len(run_b["times"]) else 0.0,
    )
    if args.t_max_T is not None and args.t_max_T > 0.0:
        x_max = args.t_max_T
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel(r"$t/T$", fontsize=labelsize + 4)
    ax.set_ylabel(r"$\ln(|\delta n_k/n_0\,(t)|)$", fontsize=labelsize + 4)
    ax.tick_params(labelsize=ticksize + 4)
    ax.grid(True, alpha=0.3)
    handles, labels_ = ax.get_legend_handles_labels()
    if theory_used_colors:
        proxies = tuple(
            Line2D([0], [0], color=c, linewidth=1.5, linestyle="--", alpha=0.9)
            for c in theory_used_colors
        )
        handles.append(proxies)
        labels_.append(r"$\delta n \propto e^{-\Gamma t}$")
    ax.legend(handles, labels_, loc="lower left", fontsize=labelsize + 4,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    ax.text(0.97, 0.97, "(б)", transform=ax.transAxes,
            ha="right", va="top", fontsize=labelsize + 4, bbox=panel_bbox)
    ax.set_box_aspect(1)

    out_dir = os.path.join(out_dir_a, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.filename)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison figure written to {out_path}")
    print(f"  A: Np = {run_a['Np']}, T = {run_a['T_theory']:.4g}, "
          f"Gamma = {run_a['Gamma']:.4g}")
    print(f"  B: Np = {run_b['Np']}, T = {run_b['T_theory']}, "
          f"Gamma = {run_b['Gamma']}")

    # Separate figure: |delta n_m(t)| comparison for modes m = 2, 3, 4
    # between the two runs, no theory overlay. Same x-axis convention
    # (t / T from run A), same color pairing (A = red, B = blue).
    modes_extra = (2, 3, 4, 5, 6)
    fig_m, axes_m = plt.subplots(1, len(modes_extra),
                                 figsize=(7.0 * len(modes_extra), 7.0))
    for ax_m, m in zip(axes_m, modes_extra):
        for run, color in ((run_a, "red"), (run_b, "blue")):
            t_over_T = run["times"] / T_norm
            mod_m = run["mods_extra"].get(m)
            if mod_m is None or len(mod_m) == 0:
                continue
            ax_m.plot(t_over_T, np.log(np.maximum(mod_m, eps)),
                      color=color, marker="o", markersize=4.0,
                      linewidth=2.0, label=label(run))
        ax_m.set_xlim(0.0, x_max)
        ax_m.set_xlabel(r"$t/T$", fontsize=labelsize + 4)
        ax_m.set_ylabel(rf"$\ln(|\delta n_{m}/n_0\,(t)|)$",
                        fontsize=labelsize + 4)
        ax_m.tick_params(labelsize=ticksize + 4)
        ax_m.grid(True, alpha=0.3)
        ax_m.legend(loc="lower left", fontsize=labelsize + 4)
        ax_m.text(0.97, 0.97, rf"$m = {m}$", transform=ax_m.transAxes,
                  ha="right", va="top", fontsize=labelsize + 4, bbox=panel_bbox)
        ax_m.set_box_aspect(1)

    modes_path = os.path.join(
        out_dir,
        os.path.splitext(args.filename)[0] + "_modes234.png")
    fig_m.tight_layout(pad=0.6)
    fig_m.savefig(modes_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_m)
    print(f"Modes-2,3,4 figure written to {modes_path}")

    # Per-run deviation from theory at local maxima of |delta n_c(t)|.
    # Selection of which maxima to use is asymmetric: run A (red, larger
    # Np) typically has a clean last peak — drop it and use the prior
    # three; run B (blue, smaller Np) is noisier at the tail, so the last
    # three (including the very last) are used. The reported number is the
    # largest relative deviation (in %) among the chosen maxima, computed
    # against delta_n_0 * exp(-Gamma t) at the same t.
    def local_maxima(mod):
        out = []
        for k in range(1, len(mod) - 1):
            v0, v1, v2 = mod[k - 1], mod[k], mod[k + 1]
            if not (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(v2)):
                continue
            if v1 > v0 and v1 > v2:
                out.append(k)
        return out

    def deviation_at_maxima(run, delta_n0, gamma, indices):
        rows = []
        worst = 0.0
        for k in indices:
            t = float(run["times"][k])
            data = float(run["mod"][k])
            theory = delta_n0 * float(np.exp(-gamma * t))
            rel = (data - theory) / theory * 100.0 if theory != 0.0 \
                else float("nan")
            rows.append((t, data, theory, rel))
            if np.isfinite(rel) and abs(rel) > abs(worst):
                worst = rel
        return worst, rows

    # ------------------------------------------------------------------
    # Separate figure: decomposition delta_n_c(t) = A e^{-Gamma t} cos(omega_s t)
    # + B(t), where B(t) ~ B_0 exp(-(k v_i t)^2 / 2) is the ballistic
    # ("free-streaming") tail. We isolate B(t) by a one-period sliding
    # average of the *signed* coefficient (the cosine averages to zero),
    # and then plot ln|delta_n_c - B| against the original ln|delta_n_c|
    # alongside the theoretical Landau line. If the hypothesis holds, the
    # "bump" around 2T in ln|delta_n_c| disappears after subtracting B.
    def sliding_period_avg(times, values, period):
        """Trapezoidal sliding average over a window of length `period`,
        centered on each sample. Returns NaN where the centered window
        would extend past the data range — those points are excluded
        from later plots."""
        out = np.full_like(values, np.nan, dtype=float)
        half = 0.5 * period
        t_lo, t_hi = times[0], times[-1]
        for i, t in enumerate(times):
            if t - half < t_lo - 1e-12 or t + half > t_hi + 1e-12:
                continue
            lo = np.searchsorted(times, t - half, side="left")
            hi = np.searchsorted(times, t + half, side="right")
            tw = times[lo:hi]
            vw = values[lo:hi]
            if len(tw) >= 2 and tw[-1] - tw[0] > 0.0:
                out[i] = np.trapezoid(vw, tw) / (tw[-1] - tw[0])
        return out

    def extrema_decomposition(times, signed):
        """Pair each local maximum with the next local minimum (a
        half-period later) and return the midpoint-time arrays of
        B = 1/2 (delta_n(t_+) + delta_n(t_-)) and
        |A e^{-Gamma t}| = 1/2 |delta_n(t_+) - delta_n(t_-)|."""
        idx_max, idx_min = [], []
        for k in range(1, len(signed) - 1):
            v0, v1, v2 = signed[k - 1], signed[k], signed[k + 1]
            if v1 > v0 and v1 > v2:
                idx_max.append(k)
            elif v1 < v0 and v1 < v2:
                idx_min.append(k)
        t_mid, B_mid, A_mid = [], [], []
        j = 0
        for km in idx_max:
            while j < len(idx_min) and idx_min[j] <= km:
                j += 1
            if j >= len(idx_min):
                break
            kn = idx_min[j]
            t_p, t_m = times[km], times[kn]
            v_p, v_m = signed[km], signed[kn]
            t_mid.append(0.5 * (t_p + t_m))
            B_mid.append(0.5 * (v_p + v_m))
            A_mid.append(0.5 * abs(v_p - v_m))
        return (np.array(t_mid), np.array(B_mid), np.array(A_mid))

    fig_d, axes_d = plt.subplots(2, 2, figsize=(15.0, 14.0))
    delta_for_run = {}
    if len(deltas) >= 1:
        delta_for_run[id(run_a)] = deltas[0]
    if len(deltas) >= 2:
        delta_for_run[id(run_b)] = deltas[1]

    for row, (run, color, panel_letter) in enumerate((
            (run_a, "red", ("(а)", "(б)")),
            (run_b, "blue", ("(в)", "(г)")))):

        t = run["times"]
        signed = run.get("coeff")
        modulus = run["mod"]
        T_run = run["T_theory"] if run["T_theory"] is not None else T_norm
        Gamma_run = run["Gamma"]
        if signed is None or len(signed) == 0 or len(t) == 0:
            continue
        t_over_T = t / T_norm

        B_avg = sliding_period_avg(t, signed, T_run)
        osc = signed - B_avg                              # pure oscillation

        t_ex, B_ex, A_ex = extrema_decomposition(t, signed)

        # --- Left panel: signed delta_n_c(t) and extracted B(t) ----------
        ax_l = axes_d[row, 0]
        ax_l.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        ax_l.plot(t_over_T, signed, color=color, linewidth=1.5, alpha=0.85,
                  label=rf"$\delta n_c(t)$, {label(run)}")
        valid = np.isfinite(B_avg)
        if np.any(valid):
            ax_l.plot(t_over_T[valid], B_avg[valid],
                      color="black", linewidth=2.2,
                      label=r"$\overline{\delta n_c}(t)$ (sliding $T$)")
        if len(t_ex) > 0:
            ax_l.plot(t_ex / T_norm, B_ex, color="black", marker="s",
                      linestyle="", markersize=6.0, markerfacecolor="white",
                      label=r"$\frac{1}{2}(\delta n_c(t_+)+\delta n_c(t_-))$")
        ax_l.set_xlim(0.0, x_max)
        ax_l.set_xlabel(r"$t/T$", fontsize=labelsize + 4)
        ax_l.set_ylabel(r"$\delta n_c / n_0$", fontsize=labelsize + 4)
        ax_l.tick_params(labelsize=ticksize + 4)
        ax_l.grid(True, alpha=0.3)
        ax_l.legend(loc="upper right", fontsize=labelsize)
        ax_l.text(0.03, 0.97, panel_letter[0], transform=ax_l.transAxes,
                  ha="left", va="top", fontsize=labelsize + 4, bbox=panel_bbox)
        ax_l.set_box_aspect(1)

        # --- Right panel: ln|.| of original, of oscillating part, theory --
        ax_r = axes_d[row, 1]
        ax_r.plot(t_over_T, np.log(np.maximum(modulus, eps)),
                  color=color, linewidth=1.5, alpha=0.55,
                  label=r"$\ln|\delta n_c|$ (raw)")
        valid_osc = np.isfinite(osc)
        if np.any(valid_osc):
            ax_r.plot(t_over_T[valid_osc],
                      np.log(np.maximum(np.abs(osc[valid_osc]), eps)),
                      color=color, marker="o", markersize=4.0,
                      linewidth=2.0,
                      label=r"$\ln|\delta n_c - \overline{\delta n_c}|$")
        if len(t_ex) > 0:
            ax_r.plot(t_ex / T_norm, np.log(np.maximum(A_ex, eps)),
                      color="black", marker="s", linestyle="",
                      markersize=6.0, markerfacecolor="white",
                      label=r"$\ln\frac{1}{2}|\delta n_c(t_+)-\delta n_c(t_-)|$")
        if Gamma_run is not None:
            delta_n0 = delta_for_run.get(id(run))
            if delta_n0 is None and len(deltas) >= 1:
                delta_n0 = deltas[0]
            if delta_n0 is not None:
                t_th = np.linspace(0.0, t[-1], 600)
                ax_r.plot(t_th / T_norm,
                          np.log(delta_n0) - Gamma_run * t_th,
                          color="black", linewidth=1.2, linestyle="--",
                          label=r"$\ln(\delta n_0) - \Gamma t$")
        ax_r.set_xlim(0.0, x_max)
        ax_r.set_xlabel(r"$t/T$", fontsize=labelsize + 4)
        ax_r.set_ylabel(r"$\ln(|\cdot|/n_0)$", fontsize=labelsize + 4)
        ax_r.tick_params(labelsize=ticksize + 4)
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(loc="lower left", fontsize=labelsize)
        ax_r.text(0.03, 0.97, panel_letter[1], transform=ax_r.transAxes,
                  ha="left", va="top", fontsize=labelsize + 4, bbox=panel_bbox)
        ax_r.set_box_aspect(1)

    decomp_path = os.path.join(
        out_dir,
        os.path.splitext(args.filename)[0] + "_decomposition.png")
    fig_d.tight_layout(pad=0.6)
    fig_d.savefig(decomp_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_d)
    print(f"Decomposition figure written to {decomp_path}")

    if Gamma_a is not None and len(deltas) >= 2:
        runs_cfg = (
            ("A (red,  10000)", run_a, deltas[0], slice(-4, None)),
            ("B (blue, 2000)",  run_b, deltas[1], slice(-3, None)),
        )
        for tag, run, delta_n0, sel in runs_cfg:
            maxs = local_maxima(run["mod"])
            picked = maxs[sel]
            print(f"  {tag}: maxima used = {len(picked)} / {len(maxs)} "
                  f"(delta_n_0 = {delta_n0:g})")
            worst, rows = deviation_at_maxima(run, delta_n0, Gamma_a, picked)
            for t, data, theory, rel in rows:
                print(f"    t/T = {t / T_norm:7.3f}  data = {data:.4g}  "
                      f"theory = {theory:.4g}  rel = {rel:+.3f}%")
            print(f"    max |rel| = {abs(worst):.3f}%  (signed = {worst:+.3f}%)")


if __name__ == "__main__":
    main()
