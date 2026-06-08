#!/usr/bin/env python3
"""Plot the (x, y)-averaged particle density as a function of z for every
species with a density diagnostic, all curves overlaid on one figure.

For each common diagnostic timestep the script reads the 3D density frame
(DistributionMoment, moment="density"), averages it over the x and y axes
and draws <n>(z) for both electrons and ions on the same axes. One PNG per
timestep is written under <out_dir>/processed/<out_subdir>/.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from math import gcd

import numpy as np

TOOLS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS_DIR)

from lib.constants import const, init_constants
from lib.plot import bbox, labelsize, ticksize

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--species", nargs="+", default=None,
                        help="Particle sorts to plot (default: every species "
                             "with a density diagnostic in the config)")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out-subdir", default="density_z",
                        help="Subdirectory under <out_dir>/processed for PNGs")
    parser.add_argument("--anim-fps", type=int, default=10,
                        help="Frames per second for the assembled gif "
                             "(default: 10)")
    parser.add_argument("--anim-name", default="density_z.mp4",
                        help="Filename of the assembled mp4 inside <out-subdir>")
    parser.add_argument("--theory-period", type=float, default=8500,
                        help="Period T in units of 1/w_pe used for the "
                             "theoretical curve "
                             "n0 + sin(2 pi z/Lz) * [alpha "
                             "+ delta_n * cos(2 pi t/T + phi_0)] "
                             "(default: 19744.2).")
    parser.add_argument("--theory-n0", type=float, default=1.000000,
                        help="Background density n0 in the theoretical curve "
                             "(default: 1.000000, external fit).")
    parser.add_argument("--theory-alpha", type=float, default=0.015616,
                        help="DC (time-independent) mode-1 offset alpha in "
                             "the model n0 + sin(2 pi z/Lz) * [alpha + "
                             "delta_n * cos(2 pi t/T + phi_0)]. Default: "
                             "0.015616 (external fit).")
    parser.add_argument("--theory-delta-n", type=float, default=0.03,
                        help="Oscillation amplitude delta_n in the antinode. "
                             "Default: 0.083856 (external fit). Pass NaN to "
                             "fall back to in-script fitting / config / 0.")
    parser.add_argument("--theory-phase", type=float, default=0.0,
                        help="Initial phase phi_0 of the standing wave, "
                             "radians. Default: 0.036044 (= 2.065 deg, "
                             "external fit). Pass NaN to fall back to "
                             "in-script fitting.")
    parser.add_argument("--theory-fit-tmax", type=float, default=None,
                        help="Restrict the lstsq fit of A and phi_0 to "
                             "t <= TMAX (in 1/w_pe). Default: one period T. "
                             "Use a large value (or 0 for unlimited) to fit "
                             "the entire run.")
    return parser.parse_args()


def iter_density_diagnostics():
    seen = set()
    for diag in const.config.get("Diagnostics", []):
        if diag.get("diagnostic") != "DistributionMoment":
            continue
        if diag.get("moment") != "density":
            continue
        species = diag.get("particles")
        if species is None or species in seen:
            continue
        seen.add(species)
        yield species, diag.get("out_dir")


def candidate_dirs(species, explicit):
    if explicit is not None:
        return [explicit]
    return [
        os.path.join(species, "density"),
        f"{species}_density",
    ]


def expected_size_bytes():
    return const.Nx * const.Ny * const.Nz * 4


def list_timesteps(dir_path):
    entries = []
    for name in os.listdir(dir_path):
        full = os.path.join(dir_path, name)
        if os.path.isfile(full) and name.isdigit():
            entries.append((int(name), name))
    entries.sort()
    return entries


def locate_dir(species, explicit):
    expected = expected_size_bytes()
    for sub in candidate_dirs(species, explicit):
        dir_path = os.path.join(const.in_dir, sub)
        if not os.path.isdir(dir_path):
            continue
        steps = [
            (idx, name) for (idx, name) in list_timesteps(dir_path)
            if os.path.getsize(os.path.join(dir_path, name)) == expected
        ]
        if steps:
            return dir_path, steps
    return None, None


def load_frame(dir_path, name):
    path = os.path.join(dir_path, name)
    if os.path.getsize(path) != expected_size_bytes():
        return None
    raw = np.fromfile(path, dtype=np.float32,
                      count=const.Nx * const.Ny * const.Nz)
    return raw.reshape(const.Nz, const.Ny, const.Nx)


def find_sine_amplitude_z():
    for preset in const.config.get("Presets", []):
        if preset.get("command") != "SetParticles":
            continue
        coord = preset.get("coordinate", {})
        if coord.get("name") == "CoordinateInBoxSineDensity":
            amp = coord.get("amplitude")
            if amp is not None and len(amp) >= 3:
                return float(amp[2])
    return None


SPECIES_STYLE = {
    "electrons": {"color": "blue", "marker": "o", "linestyle": "-",
                  "linewidth": 3.0, "markersize": 10.0,
                  "label": r"$n_e$"},
    "ions":      {"color": "red",  "marker": "o", "linestyle": "-",
                  "linewidth": 3.0, "markersize": 10.0,
                  "label": r"$n_i$"},
}


def collect_rows(species_filter):
    rows = []
    available = list(iter_density_diagnostics())
    if species_filter is not None:
        wanted = set(species_filter)
        available_names = {sp for sp, _ in available}
        for name in sorted(wanted - available_names):
            print(f"[skip] no density diagnostic for species '{name}' in config")
        available = [(sp, ex) for sp, ex in available if sp in wanted]

    for species, explicit in available:
        dir_path, steps = locate_dir(species, explicit)
        if dir_path is None:
            print(f"[skip] no frames for species '{species}' "
                  f"(looked for {candidate_dirs(species, explicit)})")
            continue
        rows.append({"species": species, "dir": dir_path, "timesteps": steps})
    return rows


def main():
    args = parse_args()
    init_constants(args.config)

    # config['OutputDirectory'] may point at a remote path if the run was
    # produced on another machine; fall back to the directory holding
    # config.json so the script works on copied output trees.
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if not os.path.isdir(const.in_dir):
        const.in_dir = config_dir
        const.out_dir = os.path.join(config_dir, "processed")

    rows = collect_rows(args.species)
    if not rows and const.in_dir != config_dir:
        const.in_dir = config_dir
        const.out_dir = os.path.join(config_dir, "processed")
        rows = collect_rows(args.species)
    if not rows:
        print("No density diagnostics found for the requested species.")
        return

    common = set(idx for idx, _ in rows[0]["timesteps"])
    for row in rows[1:]:
        common &= set(idx for idx, _ in row["timesteps"])
    common = sorted(common)
    if not common:
        print("No timesteps common to all requested density diagnostics.")
        return

    names_per_row = []
    for row in rows:
        name_by_idx = dict(row["timesteps"])
        names_per_row.append([name_by_idx[idx] for idx in common])

    z = (np.arange(const.Nz) + 0.5) * const.dz

    def profile_ext(data):
        return data.mean(axis=(1, 2))

    n0 = args.theory_n0
    T_theory = args.theory_period
    alpha = args.theory_alpha

    # Antinode of sin(2*pi*z/Lz) lies at z = Lz/4 (sin = 1). Pick the cell
    # whose center is closest to that location.
    z_anti = const.Lz / 4.0
    i_anti = int(np.argmin(np.abs(z - z_anti)))
    sin_at_anti = float(np.sin(2.0 * np.pi * z[i_anti] / const.Lz))

    # Second antinode at z = 3*Lz/4 (sin = -1). The fit below uses the
    # single-cell time series at this location.
    z_anti2 = 3.0 * const.Lz / 4.0
    i_anti2 = int(np.argmin(np.abs(z - z_anti2)))
    sin_at_anti2 = float(np.sin(2.0 * np.pi * z[i_anti2] / const.Lz))

    # One pre-pass: collect vmax for axis limits, the antinode time series
    # at both z = Lz/4 and z = 3Lz/4 (for the right panel display and for
    # the antinode-LSQ fit) and the spatial mode-1 amplitude per frame
    # per species (kept as a secondary estimator).
    Nf = len(common)
    profiles_pre = {row["species"]: np.full((Nf, const.Nz), np.nan)
                    for row in rows}
    series_pre = {row["species"]: np.full(Nf, np.nan) for row in rows}
    series2_pre = {row["species"]: np.full(Nf, np.nan) for row in rows}
    mode1_pre = {row["species"]: np.full(Nf, np.nan) for row in rows}
    times_pre = np.array([idx * const.dts for idx in common], dtype=float)
    sin_z_full = np.sin(2.0 * np.pi * z / const.Lz)
    sin_z_norm = float(np.sum(sin_z_full * sin_z_full))  # = Nz/2 for full periods
    vmax = 0.0
    for row, names in zip(rows, names_per_row):
        for k, name in enumerate(names):
            data = load_frame(row["dir"], name)
            if data is None:
                continue
            profile = profile_ext(data)
            vmax = max(vmax, float(np.max(np.abs(profile))))
            profiles_pre[row["species"]][k, :] = profile
            series_pre[row["species"]][k] = float(profile[i_anti])
            series2_pre[row["species"]][k] = float(profile[i_anti2])
            mode1_pre[row["species"]][k] = float(
                np.sum((profile - n0) * sin_z_full) / sin_z_norm)
    if vmax <= 0.0:
        vmax = 1.0

    # Average mode-1 amplitude across species. Then both estimate the
    # amplitude/phase via the LSQ on cos+sin basis (relies on T_theory
    # being approximately right) and a robust "peak" estimator that needs
    # no period assumption.
    omega = 2.0 * np.pi / T_theory
    all_mode1 = np.array([mode1_pre[row["species"]] for row in rows])
    a_avg = np.nanmean(all_mode1, axis=0)
    fit_mask = np.isfinite(a_avg)

    # Fit window: by default the entire run, but capped at T_theory if
    # the run is much longer than that (so decay doesn't bias L2).
    if args.theory_fit_tmax is None:
        fit_tmax = T_theory
    elif args.theory_fit_tmax <= 0.0:
        fit_tmax = float(np.inf)
    else:
        fit_tmax = args.theory_fit_tmax

    win_mask = fit_mask & (times_pre <= fit_tmax)

    A_lsq = 0.0
    phi_lsq = 0.0
    if win_mask.sum() >= 2:
        y_win = a_avg[win_mask]
        t_win = times_pre[win_mask]
        basis = np.column_stack([np.cos(omega * t_win), np.sin(omega * t_win)])
        coeffs, *_ = np.linalg.lstsq(basis, y_win, rcond=None)
        c_cos, c_sin = float(coeffs[0]), float(coeffs[1])
        A_lsq = float(np.hypot(c_cos, c_sin))
        phi_lsq = float(-np.arctan2(c_sin, c_cos))

    # Robust peak-based estimator (no T assumption).
    A_peak = 0.0
    phi_peak = 0.0
    a_at_t0 = float("nan")
    if fit_mask.any():
        a_finite = a_avg[fit_mask]
        t_finite = times_pre[fit_mask]
        peak_idx = int(np.argmax(np.abs(a_finite)))
        A_peak = float(np.abs(a_finite[peak_idx]))
        a_at_t0 = float(a_finite[0])
        # Whichever sign |a| has at t = peak fixes (omega*t_peak + phi_0)
        # up to a multiple of pi:
        phase_at_peak = 0.0 if a_finite[peak_idx] > 0 else np.pi
        phi_peak = phase_at_peak - omega * float(t_finite[peak_idx])
        phi_peak = float((phi_peak + np.pi) % (2.0 * np.pi) - np.pi)

    # Zero-crossing estimator: in theory |a(t)| = 0 at t = T/4 (when the
    # wave passes through n0 everywhere). The observed time of minimum
    # |a(t)| pins phi_0 via  omega*t_min + phi_0 = pi/2, then the
    # amplitude is recovered from a(0) = A * cos(phi_0).
    A_zero = 0.0
    phi_zero = 0.0
    t_min_dev = float("nan")
    if fit_mask.any():
        a_finite = a_avg[fit_mask]
        t_finite = times_pre[fit_mask]
        i_min = int(np.argmin(np.abs(a_finite)))
        t_min_dev = float(t_finite[i_min])
        phi_z = np.pi / 2.0 - omega * t_min_dev
        a0 = float(a_finite[0])
        cos_phi = float(np.cos(phi_z))
        if abs(cos_phi) > 1e-6:
            A_signed = a0 / cos_phi
        else:
            A_signed = a0
        # Make A positive, absorb sign into phase.
        if A_signed < 0.0:
            A_signed = -A_signed
            phi_z += np.pi
        A_zero = float(A_signed)
        phi_zero = float((phi_z + np.pi) % (2.0 * np.pi) - np.pi)

    t_span = float(times_pre[fit_mask][-1] - times_pre[fit_mask][0]
                   if fit_mask.any() else 0.0)

    # Antinode LSQ at z = 3 Lz/4: use the single-cell time series there
    # and fit y(t) - n0 = sin_at_anti2 * A * cos(omega t + phi_0) for the
    # given period T. Works even on records shorter than half a period
    # because T is fixed and we have all the points along the temporal
    # curve at one z. Uses every frame, no time-window restriction.
    all_series2 = np.array([series2_pre[row["species"]] for row in rows])
    y_avg2 = np.nanmean(all_series2, axis=0)
    anti2_mask = np.isfinite(y_avg2)

    A_anti = 0.0
    phi_anti = 0.0
    if anti2_mask.sum() >= 2 and abs(sin_at_anti2) > 1e-6:
        y_a2 = y_avg2[anti2_mask] - n0
        t_a2 = times_pre[anti2_mask]
        basis = np.column_stack([np.cos(omega * t_a2), np.sin(omega * t_a2)])
        coeffs, *_ = np.linalg.lstsq(basis, y_a2, rcond=None)
        c_cos, c_sin = float(coeffs[0]), float(coeffs[1])
        # Model:  y = sin_at_anti2 * A * cos(omega t + phi_0)
        #           = (sin_at_anti2 * A) * [cos(phi_0) cos(omega t)
        #                                   - sin(phi_0) sin(omega t)]
        # So c_cos = sin_at_anti2 * A * cos(phi_0)
        #    c_sin = -sin_at_anti2 * A * sin(phi_0)
        # Recover A * cos(phi_0) and A * sin(phi_0), then A and phi_0.
        ac = float(c_cos / sin_at_anti2)        # A cos(phi_0)
        as_ = float(-c_sin / sin_at_anti2)      # A sin(phi_0)
        A_anti = float(np.hypot(ac, as_))
        phi_anti = float(np.arctan2(as_, ac))

    # Default: antinode LSQ at z = 3 Lz/4 (uses all frames, given period).
    # Falls back to zero-crossing, then global LSQ, then peak.
    if A_anti > 0.0:
        A_fit = A_anti
        phi_fit = phi_anti
        method = "antinode LSQ @ z=3Lz/4"
    elif A_zero > 0.0:
        A_fit = A_zero
        phi_fit = phi_zero
        method = "zero-crossing of |a(t)|"
    elif t_span >= 0.5 * T_theory and A_lsq > 0.0:
        A_fit = A_lsq
        phi_fit = phi_lsq
        method = "lstsq"
    else:
        A_fit = A_peak
        phi_fit = phi_peak
        method = "peak"

    config_delta = find_sine_amplitude_z()
    if args.theory_delta_n is not None and not np.isnan(args.theory_delta_n):
        delta_n = args.theory_delta_n
        amp_src = "CLI"
    elif A_fit > 0.0:
        delta_n = A_fit
        amp_src = "fit"
    elif config_delta is not None:
        delta_n = config_delta
        amp_src = "config"
    else:
        delta_n = 0.0
        amp_src = "fallback (0.0)"
        print("[warn] could not infer delta_n from data or config; using 0.0")

    if args.theory_phase is not None and not np.isnan(args.theory_phase):
        phi0 = args.theory_phase
        phase_src = "CLI"
    else:
        phi0 = phi_fit
        phase_src = "fit"

    print(f"Theory parameters:")
    print(f"  T (period)       = {T_theory:.6g}  [1/w_pe]")
    print(f"  n0               = {n0:.6g}")
    print(f"  alpha (DC mode)  = {alpha:+.6g}")
    print(f"  config delta_n_z = {config_delta}")
    print(f"  data span        = t in [{times_pre[fit_mask][0]:.3g}, "
          f"{times_pre[fit_mask][-1]:.3g}]   "
          f"({t_span / T_theory:.3f} T)")
    print(f"  a(0)  (mode m=1) = {a_at_t0:+.6g}")
    print(f"  fit window       = t <= "
          f"{fit_tmax if np.isfinite(fit_tmax) else 'inf'} "
          f"({int(win_mask.sum())} of {Nf} frames)")
    print(f"  estimator        = {method}")
    print(f"  lsq    A         = {A_lsq:+.6g}, "
          f"phi_0 = {phi_lsq:+.6f} rad "
          f"({np.degrees(phi_lsq):+.3f} deg)")
    print(f"  peak   A         = {A_peak:+.6g}, "
          f"phi_0 = {phi_peak:+.6f} rad "
          f"({np.degrees(phi_peak):+.3f} deg)")
    print(f"  zero   t_min     = {t_min_dev:.4g}  "
          f"(theory: T/4 = {T_theory / 4.0:.4g}; "
          f"shift: {(t_min_dev - T_theory / 4.0):+.4g})")
    print(f"  zero   A         = {A_zero:+.6g}, "
          f"phi_0 = {phi_zero:+.6f} rad "
          f"({np.degrees(phi_zero):+.3f} deg)")
    print(f"  anti2  A         = {A_anti:+.6g}, "
          f"phi_0 = {phi_anti:+.6f} rad "
          f"({np.degrees(phi_anti):+.3f} deg)   "
          f"[LSQ @ z={z[i_anti2]:.3g}, all {int(anti2_mask.sum())} frames]")
    print(f"  used delta_n     = {delta_n:+.6g}  ({amp_src})")
    print(f"  used phi_0       = {phi0:+.6f} rad "
          f"({np.degrees(phi0):+.3f} deg, {phase_src})")

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    z_theory = np.linspace(0.0, const.Lz, 200)

    # i_anti2 / sin_at_anti2 / z_anti2 were defined above (used for the
    # antinode-LSQ fit); reuse here for the right panel display.

    t_max_phys = common[-1] * const.dts

    fig, (ax_z, ax_t2, ax_amp) = plt.subplots(1, 3, figsize=(19, 6.5))

    lines = {}
    lines_t2 = {}
    lines_amp = {}
    series_t2 = {}
    series_amp = {}
    for row in rows:
        style = SPECIES_STYLE.get(row["species"],
                                  {"marker": "o", "linestyle": "-",
                                   "label": row["species"]})
        lines[row["species"]] = ax_z.plot([], [], **style)[0]
        lines_t2[row["species"]] = ax_t2.plot([], [], **style)[0]
        lines_amp[row["species"]] = ax_amp.plot([], [], **style)[0]
        series_t2[row["species"]] = []
        series_amp[row["species"]] = []

    ylim_lo, ylim_hi = 0.94, 1.06
    amp_hi, amp_lo = 1.03, 0.97

    def draw_amplitude_lines(ax, x_label_pos, ha="left"):
        """Dashed horizontal lines at amp_hi / amp_lo with the level text
        placed just above the line, anchored at x_label_pos in data coords."""
        for level in (amp_hi, amp_lo):
            ax.axhline(level, color="tab:gray", linewidth=1.0,
                       linestyle="--", alpha=0.8)
            ax.text(x_label_pos, level + 0.002, f"{level:.2f}",
                    color="tab:gray", fontsize=ticksize,
                    ha=ha, va="bottom")

    ax_z.set_xlim(0.0, const.Lz)
    ax_z.set_ylim(ylim_lo, ylim_hi)
    ax_z.set_xlabel(r"$z~(c/\omega_{pe})$", fontsize=labelsize)
    ax_z.set_ylabel(rf"$\langle \frac{{n}}{{n_{{0}}}} \rangle_{{x,y}}(z)$", fontsize=labelsize)
    ax_z.tick_params(labelsize=ticksize)
    ax_z.grid(True, alpha=0.3)
    ax_z.axvline(z[i_anti2], color="grey", linewidth=0.6, linestyle=":")
    ax_z.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    draw_amplitude_lines(ax_z, x_label_pos=0.01 * const.Lz, ha="left")
    ax_z.set_box_aspect(1)
    ax_z.set_title(r"Профиль плотности", fontsize=labelsize, bbox=bbox)

    t_over_T_max = t_max_phys / T_theory
    ax_t2.set_xlim(0.0, t_over_T_max)
    ax_t2.set_ylim(ylim_lo, ylim_hi)
    ax_t2.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_t2.set_ylabel(
        rf"$\langle \frac{{n}}{{n_{{0}}}} \rangle_{{x,y}}(t)$",
        fontsize=labelsize)
    ax_t2.tick_params(labelsize=ticksize)
    ax_t2.grid(True, alpha=0.3)
    ax_t2.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    draw_amplitude_lines(ax_t2, x_label_pos=0.01 * t_over_T_max, ha="left")

    # Vertical dashed lines at every multiple of T/4 up to the end of the
    # run; the label is the reduced fraction (T/4, T/2, 3T/4, T, 5T/4, ...)
    # rendered via \frac{}{} and pinned to the top inside the panel, just
    # right of the line.
    def quarter_period_label(n_quarters: int) -> str:
        g = gcd(n_quarters, 4)
        num = n_quarters // g
        den = 4 // g
        num_str = "T" if num == 1 else f"{num}T"
        if den == 1:
            return f"${num_str}$"
        return rf"$\frac{{{num_str}}}{{{den}}}$"

    n_q = 1
    label_y = ylim_hi - 0.003
    x_offset = 0.01 * t_over_T_max
    while n_q / 4.0 <= t_over_T_max + 1e-12:
        x_pos = n_q / 4.0
        ax_t2.axvline(x_pos, color="tab:gray", linewidth=1.0,
                      linestyle="--", alpha=0.7)
        ax_t2.text(x_pos + x_offset, label_y,
                   quarter_period_label(n_q),
                   color="tab:gray", fontsize=ticksize,
                   ha="left", va="top")
        n_q += 1

    ax_t2.set_title(r"Плотность в точке $z = 145c/\omega_{pe}$",
                    fontsize=labelsize, bbox=bbox)
    ax_t2.set_box_aspect(1)
    ax_z.legend(loc="upper right", fontsize=labelsize)

    # Amplitude panel: L2 norm of the density perturbation along z, normalized
    # so that for n/n0 = 1 + A*sin(kz - omega t) the panel reads exactly |A|.
    #   delta_n(t) = sqrt( (2/Lz) * integral ( <n>_{x,y}/n0 - 1 )^2 dz )
    # The integral is approximated by the cell-centered sum
    #   integral f(z) dz  ≈  sum_iz f(z_iz) * dz
    # so   (2/Lz) * integral  =  2 * mean_iz ( . )   since Nz*dz = Lz.
    # Y-range is the same as the antinode panel so the comparison with the
    # naive "amplitude inferred from a single z-cell" is direct.
    ax_amp.set_xlim(0.0, t_over_T_max)
    # Headroom for the reference line at delta_n: never let it sit outside
    # the axes, otherwise the label would escape into the figure area and
    # break the layout. Floor at 0.06 to keep the panel comparable to the
    # antinode one when delta_n is small.
    amp_ylim_hi = max(0.06, 1.2 * float(delta_n) if delta_n > 0.0 else 0.06)
    ax_amp.set_ylim(0.0, amp_ylim_hi)
    ax_amp.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_amp.set_ylabel(
        r"$\delta n(t) = \sqrt{\frac{2}{L_z}\int "
        r"\left(\langle n\rangle_{x,y}/n_0 - 1\right)^2 dz}$",
        fontsize=labelsize)
    ax_amp.tick_params(labelsize=ticksize)
    ax_amp.grid(True, alpha=0.3)
    # Reference level at the initial loading amplitude, if known. Clipped to
    # the axes box so a too-large delta_n never spills outside the panel.
    if delta_n > 0.0 and delta_n <= amp_ylim_hi:
        ax_amp.axhline(delta_n, color="tab:gray", linewidth=1.0,
                       linestyle="--", alpha=0.8)
        ax_amp.text(0.01 * t_over_T_max, delta_n + 0.001,
                    rf"$\delta n_0 = {delta_n:.3f}$",
                    color="tab:gray", fontsize=ticksize,
                    ha="left", va="bottom", clip_on=True)
    # Mark the same quarter-period grid as on the middle panel, with labels
    # (T/4, T/2, 3T/4, T, 5T/4, …) pinned just below the top of the panel.
    n_q = 1
    amp_label_y = amp_ylim_hi - 0.003 * (amp_ylim_hi / 0.06)
    while n_q / 4.0 <= t_over_T_max + 1e-12:
        x_pos = n_q / 4.0
        ax_amp.axvline(x_pos, color="tab:gray", linewidth=1.0,
                       linestyle="--", alpha=0.7)
        ax_amp.text(x_pos + x_offset, amp_label_y,
                    quarter_period_label(n_q),
                    color="tab:gray", fontsize=ticksize,
                    ha="left", va="top")
        n_q += 1
    ax_amp.set_title(r"Амплитуда возмущения", fontsize=labelsize, bbox=bbox)
    ax_amp.set_box_aspect(1)

    # Lay out the figure once on a "fully populated" state (suptitle and
    # all axis labels at their final widths), then lock the panel
    # positions via subplots_adjust. After this point the spacing never
    # changes from frame to frame, so the panels stay where they are and
    # the figure margins don't breathe.
    suptitle = fig.suptitle(rf"$t = {0.000:.3f}\,T$",
                            bbox=bbox, fontsize=labelsize)
    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.95))
    fig.canvas.draw()
    sp = fig.subplotpars
    fig.subplots_adjust(left=sp.left, right=sp.right,
                        bottom=sp.bottom, top=sp.top,
                        wspace=sp.wspace, hspace=sp.hspace)

    times_t = []

    def render(k):
        idx = common[k]
        t_phys = idx * const.dts
        times_t.append(t_phys / T_theory)
        for row, names in zip(rows, names_per_row):
            data = load_frame(row["dir"], names[k])
            if data is None:
                continue
            profile = profile_ext(data)
            lines[row["species"]].set_data(z, profile)
            series_t2[row["species"]].append(float(profile[i_anti2]) / n0)
            lines_t2[row["species"]].set_data(
                times_t, series_t2[row["species"]])
            # delta_n via L2 norm of (n/n0 - 1) along z. Uniform cell-centered
            # grid → (2/Lz) * integral = 2 * mean over z-cells. Calibrated so
            # a pure mode-1 sine reads its amplitude exactly.
            dev = profile / n0 - 1.0
            amp_val = float(np.sqrt(2.0 * np.mean(dev * dev)))
            series_amp[row["species"]].append(amp_val)
            lines_amp[row["species"]].set_data(
                times_t, series_amp[row["species"]])
        suptitle.set_text(rf"$t = {t_phys / T_theory:.3f}\,T$")
        figname = os.path.join(out_dir, f"{idx:04d}.png")
        print(f"Processing {figname} (idx={idx})")
        fig.savefig(figname, dpi=args.dpi)
        return [*lines.values(), *lines_t2.values(),
                *lines_amp.values(), suptitle]

    anim = FuncAnimation(fig, render, frames=len(common), blit=False)
    anim_path = os.path.join(out_dir, args.anim_name)
    anim.save(anim_path, writer=FFMpegWriter(fps=args.anim_fps), dpi=args.dpi)

    plt.close(fig)
    print(f"Frames written to {out_dir}")
    print(f"Animation written to {anim_path}")

    # Dump n(t) per species for every z cell into individual CSV files.
    # File name and CSV header carry the z value of the cell.
    species_order = [row["species"] for row in rows]
    csv_dir = os.path.join(out_dir, "n_vs_t")
    os.makedirs(csv_dir, exist_ok=True)
    iz_width = max(2, len(str(const.Nz - 1)))
    for iz in range(const.Nz):
        z_val = float(z[iz])
        csv_path = os.path.join(
            csv_dir, f"n_vs_t_iz{iz:0{iz_width}d}_z={z_val:.6g}.csv")
        header = (f"# z = {z_val:.6g} (c/w_pe), cell iz = {iz}\n"
                  "t," + ",".join(species_order))
        data_cols = [times_pre] + [
            profiles_pre[sp][:, iz] for sp in species_order]
        np.savetxt(csv_path, np.column_stack(data_cols),
                   delimiter=",", header=header, comments="",
                   fmt="%.6g")
    print(f"CSV (n vs t per z): {const.Nz} files in {csv_dir}")


if __name__ == "__main__":
    main()
