#!/usr/bin/env python3
"""Compare ln|delta n_c(t)| across drift_kinetic runs on one figure.

Mirrors the right panel of `drift_kinetic_density_z.py`'s ``delta_n.png``,
but for an *arbitrary* number of runs at once (one primary run A plus one or
more comparison runs). Per run, the script computes the modulus of the
mode-1 density Fourier coefficient

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
from matplotlib.ticker import MaxNLocator

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

# Distinct colors for an arbitrary number of comparison runs; run A (the
# primary run, which sets T and Gamma) is always first (red).
RUN_COLORS = ["red", "blue", "green", "gold", "darkorange", "purple",
              "saddlebrown", "magenta", "teal", "navy"]


def run_color(i):
    return RUN_COLORS[i % len(RUN_COLORS)]


# Legend font size: reduced relative to the axis labels (labelsize + 4),
# matching the smaller-legend look of drift_kinetic_exb_ex1.py.
def legend_fs():
    return ticksize + 4


def exb_ticks(ax):
    # Axis-tick format from drift_kinetic_exb_ex1.py: ticks on all four sides,
    # pointing inward, plus minor ticks. Tick-label size is kept (ticksize + 4).
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, bottom=True, left=True, right=True,
                   labelsize=ticksize + 4)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_a", help="config.json of the primary run")
    parser.add_argument("configs_b", nargs="+",
                        help="config.json of one or more comparison runs")
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
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        metavar=("YMIN", "YMAX"),
                        help="y-axis limits of the comparison panel (the "
                             "ln|delta n_c| plot). The two values are sorted, "
                             "e.g. --ylim -3 -5 sets [-5, -3].")
    parser.add_argument("--species", default="ions",
                        help="Species whose density mode-1 modulus is "
                             "compared (default: ions).")
    parser.add_argument("--theory-delta-n", type=float, nargs="+",
                        default=[0.02843, 0.03],
                        help="One or more initial loading amplitudes "
                             "delta_n_0 used to anchor the theoretical "
                             "Landau envelopes delta_n_0 * exp(-Gamma t). "
                             "Each value produces a separate curve. Pass a "
                             "single non-positive value to skip the theory.")
    parser.add_argument("--fft-window", default="hann",
                        choices=("none", "hann"),
                        help="Window applied before the FFT of the signed "
                             "delta_n_c(t) time series (default: hann). A "
                             "window suppresses spectral leakage so two close "
                             "beating peaks stand out; pass 'none' for the "
                             "narrowest possible peaks.")
    parser.add_argument("--fft-pad", type=int, default=4,
                        help="Zero-padding factor for the FFT (default: 4). "
                             "Only interpolates the spectrum; the true "
                             "frequency resolution stays 2*pi/T_total.")
    parser.add_argument("--fft-xmax", type=float, default=3.0,
                        help="Upper limit of the FFT panel x-axis in units of "
                             "omega_r (default: 3.0).")
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
    out_dir_a = run_a["out_dir"]            # capture before others reset const
    runs = [run_a]
    for cb in args.configs_b:
        runs.append(load_run(cb, args.species, args.n0))

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

    # Per-run legend labels: prefer the N_ppc label, but if several runs share
    # the same N_ppc (e.g. comparing different tests at fixed Np) fall back to
    # the test directory names so the legend stays unambiguous.
    base_labels = [label(r) for r in runs]
    if len(set(base_labels)) != len(base_labels):
        base_labels = [os.path.basename(os.path.dirname(r["config_path"]))
                       for r in runs]
    for r, lbl in zip(runs, base_labels):
        r["plot_label"] = lbl

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
                     color="black", marker="o", linestyle="-",
                     linewidth=3.0, markersize=9.0, label=r"$n_i/n_0$")

    ax_prof.set_xlim(0.0, run_a["Lz"])
    ax_prof.set_ylim(ylim_lo, ylim_hi)
    ax_prof.set_xlabel(r"$z,\ c/\omega_{pe}$", fontsize=labelsize + 4)
    ax_prof.set_ylabel(rf"$\langle \frac{{n}}{{n_{{0}}}} \rangle_{{x,y}}(z)$",
                       fontsize=labelsize + 4)
    ax_prof.grid(True, alpha=0.3)
    ax_prof.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    draw_amplitude_lines(ax_prof, x_label_pos=0.01 * run_a["Lz"], ha="left")
    ax_prof.text(0.03, 0.97, "(a)", transform=ax_prof.transAxes,
                 ha="left", va="top", fontsize=labelsize + 4, bbox=panel_bbox)
    ax_prof.legend(loc="upper right", fontsize=legend_fs())
    exb_ticks(ax_prof)
    ax_prof.set_box_aspect(1)

    eps = 1e-30                          # guard log(0) at exact zeros
    # Order the data curves by descending particle number so the run with
    # the largest Np is drawn first (top of the legend) and gets the first
    # color (red), then blue, green, yellow for decreasing Np.
    def np_key(run):
        try:
            return float(run["Np"])
        except (TypeError, ValueError):
            return float("-inf")

    order = sorted(range(len(runs)), key=lambda i: np_key(runs[i]),
                   reverse=True)
    # Draw so the first curve (red) sits on top, then blue, then green:
    # higher zorder for earlier ranks. Legend order still follows the plot
    # order (red, blue, green from top to bottom).
    for rank, i in enumerate(order):
        run = runs[i]
        color = run_color(rank)
        t_over_T = run["times"] / T_norm
        ax.plot(t_over_T, np.log(np.maximum(run["mod"], eps)),
                color=color, marker="o", markersize=4.0,
                linewidth=2.0, label=run["plot_label"],
                zorder=5 + (len(order) - rank))

    # Theoretical Landau envelopes from run A: ln(delta_n_0) - Gamma t,
    # one per requested initial amplitude. Same anchoring choice as
    # density_z.py's delta_n.png (envelope tied to the initial loading
    # amplitude, not to a data peak). Colors are paired with the data
    # curves above so reader can tell which theory belongs to which run.
    # A single theoretical Landau envelope ln(delta_n_0) - Gamma t, with
    # Gamma from the primary run A and delta_n_0 the first requested initial
    # amplitude (default: run A's measured t=0 mode amplitude).
    Gamma_a = run_a["Gamma"]
    deltas = [d for d in args.theory_delta_n if d > 0.0]
    if not deltas and len(run_a["mod"]) > 0 and run_a["mod"][0] > 0.0:
        deltas = [float(run_a["mod"][0])]
    if deltas and Gamma_a is not None:
        t_max = max(
            (float(r["times"][-1]) if len(r["times"]) else 0.0) for r in runs)
        t_th = np.linspace(0.0, t_max, 1200)
        ax.plot(t_th / T_norm, np.log(deltas[0]) - Gamma_a * t_th,
                color="black", linewidth=1.5, linestyle="--", alpha=0.9,
                label=r"$\delta n \propto e^{-\Gamma t}$")

    x_max = max(
        (float(r["times"][-1]) / T_norm if len(r["times"]) else 0.0)
        for r in runs)
    if args.t_max_T is not None and args.t_max_T > 0.0:
        x_max = args.t_max_T
    ax.set_xlim(0.0, x_max)
    # Vertical dashed guides at t = 0.5 T, 1 T, 1.5 T, ... across the whole
    # range (same dashed style as the amplitude lines on panel (a)), unlabeled.
    k = 1
    while 0.5 * k <= x_max + 1e-9:
        ax.axvline(0.5 * k, color="tab:gray", linewidth=1.0,
                   linestyle="--", alpha=0.8, zorder=0)
        k += 1
    ax.set_xlabel(r"$t/T$", fontsize=labelsize + 4)
    ax.set_ylabel(r"$\ln(|\delta n_k/n_0\,(t)|)$", fontsize=labelsize + 4)
    if args.ylim is not None:
        ax.set_ylim(*sorted(args.ylim))
    # Horizontal grid only; the vertical structure is the 0.5 T dashed guides.
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=legend_fs())
    ax.text(0.03, 0.97, "(b)", transform=ax.transAxes,
            ha="left", va="top", fontsize=labelsize + 4, bbox=panel_bbox)
    exb_ticks(ax)
    # Cap the number of major ticks so the (large) tick labels don't crowd
    # into each other on either axis.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_box_aspect(1)

    out_dir = os.path.join(out_dir_a, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.filename)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison figure written to {out_path}")
    for i, run in enumerate(runs):
        tag = chr(ord("A") + i)
        print(f"  {tag}: {run['plot_label']}, Np = {run['Np']}, "
              f"T = {run['T_theory']}, Gamma = {run['Gamma']}")

    # Separate figure: |delta n_m(t)| comparison for modes m = 2, 3, 4
    # between the two runs, no theory overlay. Same x-axis convention
    # (t / T from run A), same color pairing (A = red, B = blue).
    modes_extra = (2, 3, 4, 5, 6)
    fig_m, axes_m = plt.subplots(1, len(modes_extra),
                                 figsize=(7.0 * len(modes_extra), 7.0))
    for ax_m, m in zip(axes_m, modes_extra):
        for i, run in enumerate(runs):
            color = run_color(i)
            t_over_T = run["times"] / T_norm
            mod_m = run["mods_extra"].get(m)
            if mod_m is None or len(mod_m) == 0:
                continue
            ax_m.plot(t_over_T, np.log(np.maximum(mod_m, eps)),
                      color=color, marker="o", markersize=4.0,
                      linewidth=2.0, label=run["plot_label"])
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

    n_runs = len(runs)
    fig_d, axes_d = plt.subplots(n_runs, 2, figsize=(15.0, 7.0 * n_runs),
                                 squeeze=False)
    delta_for_run = {}
    for i, run in enumerate(runs):
        if deltas:
            delta_for_run[id(run)] = deltas[i] if i < len(deltas) else deltas[-1]

    for row, run in enumerate(runs):
        color = run_color(row)
        panel_letter = (f"({chr(ord('a') + 2 * row)})",
                        f"({chr(ord('a') + 2 * row + 1)})")

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
                  label=rf"$\delta n_c(t)$, {run['plot_label']}")
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

    # ------------------------------------------------------------------
    # Separate figure: FFT of the signed mode-1 time series delta_n_c(t).
    # Diagnostic for the envelope modulation seen in ln|delta n_c|:
    #   * a single peak at omega_r  -> the wiggles are |cos| cusps / noise,
    #     the single-exponential Landau fit is clean;
    #   * two close peaks           -> genuine beating between two modes.
    # For the two-peak case the script prints the split Delta_omega and the
    # implied beat period 2*pi/Delta_omega, and compares it with the beat
    # period read off the time series itself (spacing of the |delta n_c|
    # envelope minima), i.e. Delta_t_beat and 2*pi/Delta_t_beat. Two peaks
    # only count as real beating if Delta_omega exceeds both the Landau
    # half-width Gamma and the frequency resolution 2*pi/T_total.
    omega_r_a = 2.0 * np.pi / T_norm

    def compute_spectrum(times, signed, window, pad):
        """Magnitude spectrum of the DC-removed, uniformly-resampled signal.
        Returns (omega, |FFT|) or (None, None) if the series is too short."""
        t = np.asarray(times, dtype=float)
        y = np.asarray(signed, dtype=float)
        good = np.isfinite(t) & np.isfinite(y)
        t, y = t[good], y[good]
        if len(t) < 8:
            return None, None
        dt = float(np.median(np.diff(t)))
        if dt <= 0.0:
            return None, None
        # Resample onto a strictly uniform grid (the diagnostic cadence is
        # uniform, but interpolation guards against any missing frames).
        n = int(np.floor((t[-1] - t[0]) / dt)) + 1
        t_u = t[0] + dt * np.arange(n)
        y_u = np.interp(t_u, t, y)
        y_u = y_u - np.mean(y_u)             # drop DC (ballistic tail / offset)
        if window == "hann" and len(y_u) > 1:
            y_u = y_u * np.hanning(len(y_u))
        nfft = 1
        while nfft < max(pad, 1) * len(y_u):
            nfft *= 2
        spec = np.abs(np.fft.rfft(y_u, n=nfft))
        omega = 2.0 * np.pi * np.fft.rfftfreq(nfft, d=dt)
        return omega, spec

    def find_spectral_peaks(omega, power, w_lo, w_hi, rel_thresh, min_sep):
        """Local maxima of `power` in (w_lo, w_hi) above rel_thresh of the
        in-band maximum, greedily merged when closer than `min_sep`. Returned
        sorted by descending power as a list of (omega, power)."""
        band = (omega >= w_lo) & (omega <= w_hi)
        idx = np.where(band)[0]
        if len(idx) == 0:
            return []
        pmax = float(np.max(power[idx]))
        if pmax <= 0.0:
            return []
        pk = []
        for j in idx:
            if j == 0 or j == len(power) - 1:
                continue
            if (power[j] >= power[j - 1] and power[j] >= power[j + 1]
                    and power[j] >= rel_thresh * pmax):
                pk.append((float(omega[j]), float(power[j])))
        pk.sort(key=lambda p: p[1], reverse=True)
        merged = []
        for w, p in pk:
            if all(abs(w - w0) > min_sep for w0, _ in merged):
                merged.append((w, p))
        return merged

    def measure_beat_period(times, mod):
        """Beat period from the time series: the spacing of the minima of the
        |delta n_c| peak-height envelope (the |cos| peaks are at every ~T/2;
        their slow modulation is the beat). Returns Delta_t_beat or None."""
        idx_pk = local_maxima(mod)           # |cos| peaks, ~ every T/2
        if len(idx_pk) < 5:
            return None
        tp = np.array([times[k] for k in idx_pk], dtype=float)
        hp = np.array([mod[k] for k in idx_pk], dtype=float)
        nodes = [j for j in range(1, len(hp) - 1)
                 if hp[j] < hp[j - 1] and hp[j] < hp[j + 1]]
        if len(nodes) < 2:
            return None
        return float(np.median(np.diff(tp[nodes])))

    fig_f, ax_f = plt.subplots(figsize=(8.5, 7.0))
    w_lo_plot, w_hi_plot = 0.0, args.fft_xmax * omega_r_a
    ax_f.axvline(1.0, color="black", linestyle="--", linewidth=1.5,
                 alpha=0.9, label=r"$\omega_r$ (theory)")
    if Gamma_a is not None and Gamma_a > 0.0:
        ax_f.axvspan((omega_r_a - Gamma_a) / omega_r_a,
                     (omega_r_a + Gamma_a) / omega_r_a,
                     color="tab:gray", alpha=0.2,
                     label=r"$\omega_r \pm \Gamma$")

    print("FFT diagnostics (signed delta_n_c, peak search in "
          f"[0.3, {args.fft_xmax:.1f}] * omega_r):")
    for rank, i in enumerate(order):
        run = runs[i]
        color = run_color(rank)
        omega, power = compute_spectrum(
            run["times"], run["coeff"], args.fft_window, args.fft_pad)
        if omega is None:
            print(f"  {run['plot_label']}: spectrum unavailable "
                  "(series too short)")
            continue
        inband = (omega >= w_lo_plot) & (omega <= w_hi_plot)
        pmax_band = float(np.max(power[inband])) if np.any(inband) else 0.0
        if pmax_band <= 0.0:
            continue
        ax_f.plot(omega / omega_r_a, power / pmax_band, color=color,
                  linewidth=2.0, label=run["plot_label"])

        T_total = float(run["times"][-1] - run["times"][0])
        dw_res = 2.0 * np.pi / T_total if T_total > 0.0 else float("inf")
        Gamma_run = run["Gamma"] if run["Gamma"] is not None else 0.0
        min_sep = max(dw_res, Gamma_run)
        peaks = find_spectral_peaks(omega, power, 0.3 * omega_r_a, w_hi_plot,
                                    rel_thresh=0.25, min_sep=min_sep)
        for wp, pp in peaks[:2]:
            ax_f.plot(wp / omega_r_a, pp / pmax_band, marker="v",
                      color=color, markersize=11.0, linestyle="",
                      markeredgecolor="black")

        print(f"  {run['plot_label']}:")
        print(f"    omega_r(theory) = {omega_r_a:.6g}, Gamma = {Gamma_run:.3g}"
              f", resolution 2pi/T_tot = {dw_res:.3g}")
        if not peaks:
            print("    no significant peak found")
            continue
        for j, (wp, pp) in enumerate(peaks[:2]):
            print(f"    peak[{j}]: omega = {wp:.6g} "
                  f"(= {wp / omega_r_a:.4f} omega_r), "
                  f"rel.power = {pp / pmax_band:.3f}")
        if len(peaks) >= 2:
            w0, w1 = sorted([peaks[0][0], peaks[1][0]])
            dW = w1 - w0
            T_beat_fft = 2.0 * np.pi / dW if dW > 0.0 else float("inf")
            resolvable = dW > max(dw_res, Gamma_run)
            tag = ("resolvable -> genuine beating" if resolvable
                   else "NOT resolvable (< max(Gamma, 2pi/T_tot)) "
                        "-> treat as single peak")
            print(f"    -> TWO peaks: Delta_omega = {dW:.6g}  ({tag})")
            print(f"       T_beat = 2pi/Delta_omega = {T_beat_fft:.6g} "
                  f"(= {T_beat_fft / T_norm:.3f} T)")
            dt_beat = measure_beat_period(run["times"], run["mod"])
            if dt_beat is not None and dt_beat > 0.0:
                print(f"       time-domain Delta_t_beat = {dt_beat:.6g} "
                      f"(= {dt_beat / T_norm:.3f} T) -> "
                      f"2pi/Delta_t_beat = {2.0 * np.pi / dt_beat:.6g} "
                      f"(compare with Delta_omega = {dW:.6g})")
            else:
                print("       time-domain beat: too few envelope minima "
                      "to measure")
        else:
            print("    -> SINGLE peak at omega_r: envelope wiggles are |cos| "
                  "cusps / noise; single-exponential fit is clean")

    ax_f.set_xlim(w_lo_plot / omega_r_a, w_hi_plot / omega_r_a)
    ax_f.set_xlabel(r"$\omega / \omega_r$", fontsize=labelsize + 4)
    ax_f.set_ylabel(r"$|\widehat{\delta n_c}(\omega)|$ (norm.)",
                    fontsize=labelsize + 4)
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(loc="upper right", fontsize=legend_fs())
    ax_f.text(0.03, 0.97, "(c)", transform=ax_f.transAxes, ha="left",
              va="top", fontsize=labelsize + 4, bbox=panel_bbox)
    exb_ticks(ax_f)
    ax_f.set_box_aspect(1)
    fft_path = os.path.join(
        out_dir, os.path.splitext(args.filename)[0] + "_fft.png")
    fig_f.tight_layout(pad=0.6)
    fig_f.savefig(fft_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_f)
    print(f"FFT figure written to {fft_path}")

    if Gamma_a is not None and deltas:
        for i, run in enumerate(runs):
            delta_n0 = deltas[i] if i < len(deltas) else deltas[-1]
            tag = chr(ord("A") + i)
            maxs = local_maxima(run["mod"])
            # Primary run usually has a clean last peak — keep the last four;
            # comparison runs are noisier at the tail — use the last three.
            picked = maxs[-4:] if i == 0 else maxs[-3:]
            print(f"  {tag}: {run['plot_label']}, maxima used = {len(picked)} "
                  f"/ {len(maxs)} (delta_n_0 = {delta_n0:g})")
            worst, rows = deviation_at_maxima(run, delta_n0, Gamma_a, picked)
            for t, data, theory, rel in rows:
                print(f"    t/T = {t / T_norm:7.3f}  data = {data:.4g}  "
                      f"theory = {theory:.4g}  rel = {rel:+.3f}%")
            print(f"    max |rel| = {abs(worst):.3f}%  (signed = {worst:+.3f}%)")


if __name__ == "__main__":
    main()
