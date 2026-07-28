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

from lib.constants import const, init_constants, mec2
from lib.plot import bbox, labelsize, ticksize

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

plt.rc("text", usetex=False)  # Cyrillic panel labels are incompatible with usetex


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
    parser.add_argument("--theory-period", type=float, default=None,
                        help="Period T in units of 1/w_pe. Default: computed "
                             "from the complex root of the full kinetic "
                             "ion-acoustic dispersion relation eps(k,w)=0 "
                             "(plasma dispersion function for both electrons "
                             "and ions, so finite T_i is included) using the "
                             "plasma parameters in the config. Pass a value to "
                             "override the computed one.")
    parser.add_argument("--theory-gamma", type=float, default=None,
                        help="Landau damping rate Gamma in units of w_pe. "
                             "Default: computed from the full kinetic "
                             "dispersion relation eps(k,w)=0, i.e. "
                             "Gamma = -Im(omega). Pass a value to override.")
    parser.add_argument("--theory-n0", type=float, default=1.000000,
                        help="Background density n0 in the theoretical curve "
                             "(default: 1.000000, external fit).")
    parser.add_argument("--theory-alpha", type=float, default=0.015616,
                        help="DC (time-independent) mode-1 offset alpha in "
                             "the model n0 + sin(2 pi z/Lz) * [alpha + "
                             "delta_n * cos(2 pi t/T + phi_0)]. Default: "
                             "0.015616 (external fit).")
    parser.add_argument("--theory-delta-n", type=float, default=0.02,#0.0275,
                        help="Oscillation amplitude delta_n in the antinode. "
                             "Default: 0.083856 (external fit). Pass NaN to "
                             "fall back to in-script fitting / config / 0.")
    parser.add_argument("--theory-level", type=float, default=None,
                        help="Override the amplitude used for the theoretical "
                             "envelope/oscillation curves (delta n_0 in "
                             "delta n_0 * exp(-Gamma t) and the |cos| overlay). "
                             "Does not change axes scaling or other "
                             "data-derived quantities. Default: use --theory-delta-n.")
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


SINE_COORD_NAMES = (
    "CoordinateInBoxSineDensity",
    "CoordinateInBoxDisplacedSine",
    "CoordinateInBoxQuietSine",
)


def find_sine_perturbation():
    """Return (amplitude_z, wave_number_z) of the z-density perturbation
    declared in the SetParticles coordinate generator. Recognizes the sine,
    displaced-sine and quiet-sine loaders. Returns (None, None) if absent."""
    for preset in const.config.get("Presets", []):
        if preset.get("command") != "SetParticles":
            continue
        coord = preset.get("coordinate", {})
        if coord.get("name") not in SINE_COORD_NAMES:
            continue
        amp = coord.get("amplitude")
        wn = coord.get("wave_number")
        amp_z = float(amp[2]) if amp is not None and len(amp) >= 3 else None
        wn_z = float(wn[2]) if wn is not None and len(wn) >= 3 else None
        return amp_z, wn_z
    return None, None


def extract_plasma_params():
    """Pull the parameters needed for the ion-acoustic dispersion relation
    from config['Particles']. The electron sort is the one with q < 0, the
    ion sort the one with q > 0. Returns a dict or None if either is missing.

    Code units: w_pe = 1, c = 1, masses in electron-mass units, temperatures
    in keV (mec2 = 511 keV). Derived: v_Te = sqrt(T_e/(m_e*mec2)) [c],
    c_s = sqrt(T_e/(m_i*mec2)) [c], r_De = v_Te/w_pe [c/w_pe]."""
    sorts = const.config.get("Particles", [])
    electron = ion = None
    for s in sorts:
        q = float(s.get("q", 0.0))
        if q < 0.0 and electron is None:
            electron = s
        elif q > 0.0 and ion is None:
            ion = s
    if electron is None or ion is None:
        return None

    def temperature(s):
        if "T" in s:
            return float(s["T"])
        comps = [s.get("Tx"), s.get("Ty"), s.get("Tz")]
        comps = [float(c) for c in comps if c is not None]
        return sum(comps) / len(comps) if comps else 0.0

    T_e = temperature(electron)
    T_i = temperature(ion)
    m_e = float(electron["m"])
    m_i = float(ion["m"])
    n_e = float(electron.get("n", 1.0))
    n_i = float(ion.get("n", 1.0))
    q_e = float(electron["q"])
    q_i = float(ion["q"])

    v_Te = np.sqrt(T_e / (m_e * mec2))
    v_Ti = np.sqrt(T_i / (m_i * mec2)) if T_i > 0.0 else 0.0
    c_s = np.sqrt(T_e / (m_i * mec2))
    r_De = v_Te                           # = v_Te / w_pe, w_pe = 1
    # Plasma frequencies squared, normalized so that w_pe^2 = 1.
    omega_pe2 = n_e * q_e ** 2 / m_e
    omega_pi2 = (n_i * q_i ** 2 / m_i) / omega_pe2
    # Ion Debye length r_Di = v_Ti / w_pi (in c/w_pe, w_pe = 1). Infinite for
    # cold ions, in which case the ion term reverts to the cold fluid limit.
    r_Di = v_Ti / np.sqrt(omega_pi2) if v_Ti > 0.0 else float("inf")
    return {
        "T_e": T_e, "T_i": T_i, "m_e": m_e, "m_i": m_i,
        "v_Te": v_Te, "v_Ti": v_Ti, "c_s": c_s,
        "r_De": r_De, "r_Di": r_Di,
        "omega_pi2": omega_pi2,
    }


def _faddeeva(z, N=42):
    """Faddeeva function w(z) = exp(-z^2) erfc(-i z) via Weideman's rational
    approximation (J. A. C. Weideman, SIAM J. Numer. Anal. 31, 1994). Accurate
    to ~1e-13 for N = 42 and valid in the whole complex plane: the rational
    series is used directly in the upper half-plane (Im z >= 0) and the
    reflection w(z) = 2 exp(-z^2) - w(-z) continues it to the lower half-plane
    (damped modes, Im z < 0). Pure numpy, so no scipy dependency."""
    z = complex(z)
    if z.imag < 0.0:
        return 2.0 * np.exp(-z * z) - _faddeeva(-z, N)
    a = _weideman_coeffs(N)
    L = np.sqrt(N / np.sqrt(2.0))
    Z = (L + 1j * z) / (L - 1j * z)
    p = np.polyval(a, Z)
    return 2.0 * p / (L - 1j * z) ** 2 + (1.0 / np.sqrt(np.pi)) / (L - 1j * z)


_WEIDEMAN_CACHE = {}


def _weideman_coeffs(N):
    """Cached Taylor coefficients of Weideman's rational approximation."""
    a = _WEIDEMAN_CACHE.get(N)
    if a is None:
        M = 2 * N
        k = np.arange(-M + 1, M)
        L = np.sqrt(N / np.sqrt(2.0))
        theta = k * np.pi / M
        t = L * np.tan(theta / 2.0)
        f = np.concatenate(([0.0], np.exp(-t ** 2) * (L ** 2 + t ** 2)))
        a = np.flipud((np.real(np.fft.fft(np.fft.fftshift(f))) / (2 * M))[1:N + 1])
        _WEIDEMAN_CACHE[N] = a
    return a


def plasma_dispersion_Z(zeta):
    """Plasma dispersion function Z(zeta) = i sqrt(pi) w(zeta) (Fried & Conte),
    the analytic continuation of (1/sqrt(pi)) int exp(-x^2)/(x - zeta) dx onto
    the Landau contour. Z'(zeta) = -2 (1 + zeta Z(zeta))."""
    return 1j * np.sqrt(np.pi) * _faddeeva(zeta)


def kinetic_dielectric(omega, k, params):
    """Electrostatic dielectric function of an unmagnetized two-species
    Maxwellian plasma (electrons + ions):

        eps(k, w) = 1 + sum_s 1/(k^2 r_Ds^2) [1 + zeta_s Z(zeta_s)],
        zeta_s = w / (sqrt(2) k v_Ts).

    Landau damping of *both* species enters through Z. In the fast-wave limit
    zeta_s >> 1 the species term -> -w_ps^2/w^2, recovering the cold fluid
    dispersion; the finite-T corrections (ion pressure ~ 3 T_i and ion Landau
    damping) are what the old cold-ion relation (eq. 3.19) dropped. Cold ions
    (r_Di = inf) fall back to the exact -w_pi^2/w^2 term."""
    v_Te = params["v_Te"]
    r_De = params["r_De"]
    root2 = np.sqrt(2.0)

    zeta_e = omega / (root2 * k * v_Te)
    eps = 1.0 + (1.0 + zeta_e * plasma_dispersion_Z(zeta_e)) / (k * r_De) ** 2

    v_Ti = params.get("v_Ti", 0.0)
    r_Di = params.get("r_Di", float("inf"))
    if v_Ti and np.isfinite(r_Di):
        zeta_i = omega / (root2 * k * v_Ti)
        eps = eps + (1.0 + zeta_i * plasma_dispersion_Z(zeta_i)) / (k * r_Di) ** 2
    else:
        eps = eps - params["omega_pi2"] / omega ** 2   # cold-ion limit
    return eps


def solve_ion_sound_dispersion(k, params):
    """Complex root of the *full kinetic* ion-acoustic dispersion relation
    eps(k, w) = 0 (see kinetic_dielectric), keeping the plasma dispersion
    function for both electrons and ions. This supersedes the old cold-ion
    cubic (eq. 3.19): it accounts for finite ion temperature both in the real
    frequency (ion-pressure shift ~ 3 T_i) and in the damping (ion Landau
    damping), so runs that differ only in T_i now yield different theory.

    The old cold-ion cubic still provides the initial guess; a complex Newton
    iteration on the full eps then refines it. Returns (omega_r, gamma) =
    (Re w, -Im w) of the ion-acoustic branch; gamma > 0 is a damped wave
    (amplitude ~ exp(-gamma t))."""
    r_De = params["r_De"]
    v_Te = params["v_Te"]
    c_s = params["c_s"]
    omega_pi2 = params["omega_pi2"]

    krde2 = (k * r_De) ** 2
    A = 1.0 + 1.0 / krde2
    B = np.sqrt(np.pi / 2.0) / (k ** 3 * r_De ** 2 * v_Te)
    roots = np.roots([1j * B, A, 0.0, -omega_pi2])

    omega_s_approx = k * c_s / np.sqrt(1.0 + krde2)
    positive = [w for w in roots if w.real > 0.0] or list(roots)
    w = min(positive, key=lambda w: abs(w.real - omega_s_approx))

    # Complex Newton on the full kinetic eps, seeded with the cold-ion root.
    for _ in range(200):
        f = kinetic_dielectric(w, k, params)
        h = 1e-6 * abs(w) + 1e-12
        fp = (kinetic_dielectric(w + h, k, params) - f) / h
        if fp == 0:
            break
        step = f / fp
        w -= step
        if abs(step) <= 1e-13 * abs(w):
            break
    return float(abs(w.real)), float(-w.imag)


SPECIES_STYLE = {
    "electrons": {"color": "blue", "marker": "o", "linestyle": "-",
                  "linewidth": 3.0, "markersize": 5.0,
                  "label": r"$n_e$"},
    "ions":      {"color": "red",  "marker": "o", "linestyle": "-",
                  "linewidth": 3.0, "markersize": 5.0,
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
    alpha = args.theory_alpha

    # --- Ion-acoustic dispersion (full kinetic eps(k,w)=0) --------------
    # The wave period T and the Landau decrement Gamma are taken from the
    # complex root of the full kinetic dielectric (plasma dispersion function
    # for electrons and ions, so finite T_i enters), not the approximate
    # (3.20)/(3.21). CLI --theory-period / --theory-gamma override them.
    amp_cfg, wn_cfg = find_sine_perturbation()
    plasma = extract_plasma_params()
    k_wave = T_disp = Gamma_disp = omega_s_apx = None
    if plasma is not None and wn_cfg:
        k_wave = 2.0 * np.pi * wn_cfg / const.Lz
        omega_r, Gamma_disp = solve_ion_sound_dispersion(k_wave, plasma)
        T_disp = 2.0 * np.pi / omega_r
        krde2 = (k_wave * plasma["r_De"]) ** 2
        omega_s_apx = k_wave * plasma["c_s"] / np.sqrt(1.0 + krde2)

    if args.theory_period is not None:
        T_theory = args.theory_period
        period_src = "CLI override"
    elif T_disp is not None:
        T_theory = T_disp
        period_src = "full kinetic dispersion"
    else:
        T_theory = 8500.0
        period_src = "fallback"
        print("[warn] could not compute T from dispersion; using 8500")

    if args.theory_gamma is not None:
        Gamma = args.theory_gamma
        gamma_src = "CLI override"
    elif Gamma_disp is not None:
        Gamma = Gamma_disp
        gamma_src = "full kinetic dispersion"
    else:
        Gamma = 0.0
        gamma_src = "none (damping overlay disabled)"

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

    config_delta = amp_cfg
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
    if plasma is not None:
        print(f"  --- ion-acoustic dispersion (eq. 3.19), exact root ---")
        print(f"  T_e              = {plasma['T_e']:.6g}  [keV]")
        print(f"  m_i / m_e        = {plasma['m_i'] / plasma['m_e']:.6g}")
        print(f"  v_Te             = {plasma['v_Te']:.6g}  [c]")
        print(f"  c_s              = {plasma['c_s']:.6g}  [c]")
        print(f"  r_De             = {plasma['r_De']:.6g}  [c/w_pe]")
        print(f"  w_pi^2 / w_pe^2  = {plasma['omega_pi2']:.6g}")
    if k_wave is not None:
        print(f"  k                = {k_wave:.6g}  [w_pe/c]")
        print(f"  k * r_De         = {k_wave * plasma['r_De']:.6g}")
        print(f"  w_s (approx 3.20)= {omega_s_apx:.6g}  [w_pe]  "
              f"-> T = {2.0 * np.pi / omega_s_apx:.6g}")
    if T_disp is not None:
        print(f"  w_r (exact 3.19) = {2.0 * np.pi / T_disp:.6g}  [w_pe]")
        print(f"  Gamma  (exact)   = {Gamma_disp:+.6g}  [w_pe]  "
              f"(Gamma/w_r = {Gamma_disp * T_disp / (2.0 * np.pi):+.4g})")
    print(f"  T (period)       = {T_theory:.6g}  [1/w_pe]   ({period_src})")
    print(f"  Gamma (decrement)= {Gamma:+.6g}  [w_pe]   ({gamma_src})")
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
    cos_z_full = np.sin(2.0 * np.pi * z / const.Lz)

    # i_anti2 / sin_at_anti2 / z_anti2 were defined above (used for the
    # antinode-LSQ fit); reuse here for the right panel display.

    t_max_phys = common[-1] * const.dts

    # Visual constants reused by the standalone delta_n figure (drawn first,
    # below) and the per-frame animation panels (drawn second).
    ylim_lo, ylim_hi = 0.94, 1.06
    amp_hi, amp_lo = 1.03, 0.97
    t_over_T_max = t_max_phys / T_theory

    def draw_amplitude_lines(ax, x_label_pos, ha="left"):
        """Dashed horizontal lines at amp_hi / amp_lo with the level text
        placed just above the line, anchored at x_label_pos in data coords."""
        for level in (amp_hi, amp_lo):
            ax.axhline(level, color="tab:gray", linewidth=1.0,
                       linestyle="--", alpha=0.8)
            ax.text(x_label_pos, level + 0.002, f"{level:.2f}",
                    color="tab:gray", fontsize=ticksize,
                    ha=ha, va="bottom")

    # ----- Standalone amplitude figure (delta_n.png) --------------------
    # Drawn FIRST, before the per-frame animation, so the headline figure
    # appears on disk immediately rather than after the long render loop.
    # Self-contained: uses only the pre-pass profiles, no animation state.
    dn_dir = os.path.join(out_dir, "delta_n")
    os.makedirs(dn_dir, exist_ok=True)

    mod_series = {}
    for row in rows:
        prof = profiles_pre[row["species"]]                  # (Nf, Nz)
        coeff = 2.0 * np.nanmean((prof / n0) * cos_z_full[None, :], axis=1)
        mod_series[row["species"]] = np.abs(coeff)

    panel_bbox = dict(facecolor="white", edgecolor="none",
                      alpha=0.75, boxstyle="round,pad=0.25")

    fig2, (ax2_prof, ax2) = plt.subplots(1, 2, figsize=(15.0, 7.0))

    # --- Left panel (а): initial density profile, ions only -------------
    ions_row = next((row for row in rows if row["species"] == "ions"), None)
    if ions_row is not None:
        ion_style = dict(SPECIES_STYLE.get("ions", {}))
        ion_style["label"] = r"$n_i/n_0$"
        ion_style["color"] = "red"
        prof0 = profiles_pre[ions_row["species"]][0, :] / n0
        ax2_prof.plot(z, prof0, **ion_style)

    ax2_prof.set_xlim(0.0, const.Lz)
    ax2_prof.set_ylim(ylim_lo, ylim_hi)
    ax2_prof.set_xlabel(r"$z~(c/\omega_{pe})$", fontsize=labelsize + 4)
    ax2_prof.set_ylabel(rf"$\langle \frac{{n}}{{n_{{0}}}} \rangle_{{x,y}}(z)$",
                        fontsize=labelsize + 4)
    ax2_prof.tick_params(labelsize=ticksize + 4)
    ax2_prof.grid(True, alpha=0.3)
    ax2_prof.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    draw_amplitude_lines(ax2_prof, x_label_pos=0.01 * const.Lz, ha="left")
    ax2_prof.text(0.03, 0.97, "(а)", transform=ax2_prof.transAxes,
                  ha="left", va="top", fontsize=labelsize + 4, bbox=panel_bbox)
    ax2_prof.legend(loc="upper right", fontsize=labelsize + 4)
    ax2_prof.set_box_aspect(1)

    # --- Right panel (б): modulus perturbation, ions only ---------------
    t_over_T = times_pre / T_theory
    if ions_row is not None:
        ion_mod_style = dict(SPECIES_STYLE.get("ions", {}))
        ion_mod_style["label"] = r"$\delta n_i/n_0$"
        ion_mod_style["color"] = "red"
        ax2.plot(t_over_T, mod_series[ions_row["species"]], **ion_mod_style)

    ax2.set_xlim(0.0, t_over_T_max)
    ax2.set_ylim(0.0, 0.04)
    ax2.set_xlabel(r"$t/T$", fontsize=labelsize + 4)
    ax2.set_ylabel(r"$|\delta n_c/n_0\,(t)|$", fontsize=labelsize + 4)
    ax2.tick_params(labelsize=ticksize + 4)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.03, 0.97, "(б)", transform=ax2.transAxes,
             ha="left", va="top", fontsize=labelsize + 4, bbox=panel_bbox)

    # Quarter-period vertical lines, kept but without the T/4, T/2, ... text.
    n_q = 1
    while n_q / 4.0 <= t_over_T_max + 1e-12:
        ax2.axvline(n_q / 4.0, color="tab:gray", linewidth=1.0,
                    linestyle="--", alpha=0.7)
        n_q += 1

    # Envelope delta_n_0 * exp(-Gamma t) anchored at t = 0 with the initial
    # loading amplitude delta_n (CLI / config / fit, decided above).
    if ions_row is not None and delta_n > 0.0:
        mod_ion = mod_series[ions_row["species"]]
        t_th = np.linspace(0.0, t_max_phys, 1200)
        env = delta_n * np.exp(-Gamma * t_th)
        ax2.plot(t_th / T_theory, env, color="black", linewidth=2.2,
                 linestyle="--", alpha=0.9,
                 label=rf"${delta_n:.3f}\,e^{{-\Gamma t}}$")

        mins, maxs = [], []
        for k in range(1, len(mod_ion) - 1):
            v0, v1, v2_ = mod_ion[k - 1], mod_ion[k], mod_ion[k + 1]
            if not (np.isfinite(v0) and np.isfinite(v1)
                    and np.isfinite(v2_)):
                continue
            if v1 < v0 and v1 < v2_:
                mins.append(k)
            elif v1 > v0 and v1 > v2_:
                maxs.append(k)

        print("Delta_n diagnostics (ions):")
        if len(mins) >= 2:
            t_min1 = float(times_pre[mins[0]])
            t_min2 = float(times_pre[mins[1]])
            dt = t_min2 - t_min1
            half_T = 0.5 * T_theory
            rel = (dt - half_T) / half_T * 100.0
            print(f"  min1 t = {t_min1:.6g}, min2 t = {t_min2:.6g}, "
                  f"dt = {dt:.6g}; T/2 = {half_T:.6g}; "
                  f"rel diff = {rel:+.3f}%")
        else:
            print(f"  found only {len(mins)} local minima — "
                  f"need >= 2 to compare with T/2")

        if maxs:
            k_last = maxs[-1]
            t_last = float(times_pre[k_last])
            A_data = float(mod_ion[k_last])
            A_theory = delta_n * float(np.exp(-Gamma * t_last))
            rel = (A_data - A_theory) / A_theory * 100.0 \
                if A_theory != 0.0 else float("nan")
            print(f"  last max t = {t_last:.6g}, "
                  f"data = {A_data:.6g}, theory = {A_theory:.6g}; "
                  f"rel diff = {rel:+.3f}%")
        else:
            print("  no local maxima found in the ion modulus series")

    ax2.legend(loc="upper right", fontsize=labelsize + 4)
    ax2.set_box_aspect(1)
    fig2.tight_layout(pad=0.6)
    dn_path = os.path.join(dn_dir, "delta_n.png")
    fig2.savefig(dn_path, dpi=args.dpi)
    plt.close(fig2)
    print(f"Amplitude figure written to {dn_path}")

    fig, (ax_z, ax_cos, ax_amp) = plt.subplots(1, 3, figsize=(19, 6.5))

    lines = {}
    lines_cos = {}
    lines_amp = {}
    series_cos = {}
    series_amp = {}
    for row in rows:
        style = SPECIES_STYLE.get(row["species"],
                                  {"marker": "o", "linestyle": "-",
                                   "label": row["species"]})
        lines[row["species"]] = ax_z.plot([], [], **style)[0]
        lines_cos[row["species"]] = ax_cos.plot([], [], **style)[0]
        lines_amp[row["species"]] = ax_amp.plot([], [], **style)[0]
        series_cos[row["species"]] = []
        series_amp[row["species"]] = []

    ax_z.set_xlim(0.0, const.Lz)
    ax_z.set_ylim(ylim_lo, ylim_hi)
    ax_z.set_xlabel(r"$z~(c/\omega_{pe})$", fontsize=labelsize)
    ax_z.set_ylabel(rf"$\langle \frac{{n}}{{n_{{0}}}} \rangle_{{x,y}}(z)$", fontsize=labelsize)
    ax_z.tick_params(labelsize=ticksize)
    ax_z.grid(True, alpha=0.3)
    ax_z.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    draw_amplitude_lines(ax_z, x_label_pos=0.01 * const.Lz, ha="left")
    ax_z.set_box_aspect(1)
    ax_z.set_title(r"Профиль плотности", fontsize=labelsize, bbox=bbox)

    # Middle panel: MODULUS of the mode-1 density perturbation amplitude.
    # The signed mode-1 coefficient is
    #   delta n_c(t) = (2/Lz) * integral_0^Lz <n>_{x,y}/n0 * sin(kz) dz,
    # which on the uniform cell-centered grid is 2 * mean_z[(<n>/n0)*sin(kz)];
    # the panel plots its absolute value |delta n_c(t)|.
    # Overlaid in black: the theoretical Landau-damping curves from the EXACT
    # dispersion relation (eq. 3.19) -
    #   envelope    delta n_0 * exp(-Gamma t),
    #   oscillation delta n_0 * |cos(omega t + phi_0)| * exp(-Gamma t),
    # with omega = 2 pi / T and Gamma both from the exact complex root.
    cos_ylim_hi = max(0.06, 1.2 * float(delta_n) if delta_n > 0.0 else 0.06)
    ax_cos.set_xlim(0.0, t_over_T_max)
    ax_cos.set_ylim(0.0, cos_ylim_hi)
    ax_cos.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_cos.set_ylabel(
        r"$|\delta n_c(t)|,\quad \delta n_c=\frac{2}{L_z}\int_0^{L_z}"
        r"\frac{\langle n\rangle_{x,y}}{n_0}\sin(kz)\,dz$",
        fontsize=labelsize)
    ax_cos.tick_params(labelsize=ticksize)
    ax_cos.grid(True, alpha=0.3)

    # Theoretical damping overlay (eq. 3.19). Drawn on a dense time grid so
    # the |cos| cusps are smooth; omega and Gamma are the exact-dispersion
    # values, delta_n is the initial loading amplitude.
    theory_level = args.theory_level if args.theory_level is not None else delta_n
    if theory_level > 0.0 and len(times_pre):
        t_th = np.linspace(0.0, t_max_phys, 1200)
        env = theory_level * np.exp(-Gamma * t_th)
        osc = env * np.abs(np.cos(omega * t_th + phi0))
        ax_cos.plot(t_th / T_theory, osc, color="black", linewidth=1.4,
                    linestyle="-", alpha=0.65,
                    label=r"$\delta n_0\,|\cos(\omega t+\phi_0)|\,"
                          r"e^{-\Gamma t}$")
        ax_cos.plot(t_th / T_theory, env, color="black", linewidth=2.2,
                    linestyle="--", alpha=0.9,
                    label=rf"$\delta n_0\,e^{{-\Gamma t}},\ \ "
                          rf"\Gamma = {Gamma:.3e}\,\omega_{{pe}}$")
        ax_cos.legend(loc="upper right", fontsize=ticksize)

    if delta_n > 0.0 and delta_n <= cos_ylim_hi:
        ax_cos.axhline(delta_n, color="tab:gray", linewidth=1.0,
                       linestyle="--", alpha=0.8)
        ax_cos.text(0.01 * t_over_T_max, delta_n + 0.001,
                    rf"$\delta n_0 = {delta_n:.3f}$",
                    color="tab:gray", fontsize=ticksize,
                    ha="left", va="bottom", clip_on=True)

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
    label_y = cos_ylim_hi - 0.003 * (cos_ylim_hi / 0.06)
    x_offset = 0.01 * t_over_T_max
    while n_q / 4.0 <= t_over_T_max + 1e-12:
        x_pos = n_q / 4.0
        ax_cos.axvline(x_pos, color="tab:gray", linewidth=1.0,
                       linestyle="--", alpha=0.7)
        ax_cos.text(x_pos + x_offset, label_y,
                    quarter_period_label(n_q),
                    color="tab:gray", fontsize=ticksize,
                    ha="left", va="top")
        n_q += 1

    ax_cos.set_title(r"Модуль возмущения плотности",
                     fontsize=labelsize, bbox=bbox)
    ax_cos.set_box_aspect(1)
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
    # Same theoretical Landau overlay as on the middle panel: oscillation
    # |cos| and the bare envelope. delta_n is the loading amplitude; omega,
    # Gamma come from the exact dispersion (eq. 3.19).
    if theory_level > 0.0 and len(times_pre):
        t_th = np.linspace(0.0, t_max_phys, 1200)
        env = theory_level * np.exp(-Gamma * t_th)
        ax_amp.plot(t_th / T_theory, env, color="black", linewidth=2.2,
                    linestyle="--", alpha=0.9,
                    label=rf"$\delta n_0\,e^{{-\Gamma t}},\ \ "
                          rf"\Gamma = {Gamma:.3e}\,\omega_{{pe}}$")
        ax_amp.legend(loc="upper right", fontsize=ticksize)

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
            # Modulus of the mode-1 Fourier coefficient of the normalized
            # density profile: delta n_c = (2/Lz)*integral (<n>/n0)*sin(kz) dz.
            # Uniform cell-centered grid -> 2 * mean over z-cells; |.| taken.
            cos_val = abs(float(2.0 * np.mean((profile / n0) * cos_z_full)))
            series_cos[row["species"]].append(cos_val)
            lines_cos[row["species"]].set_data(
                times_t, series_cos[row["species"]])
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
        return [*lines.values(), *lines_cos.values(),
                *lines_amp.values(), suptitle]

    anim = FuncAnimation(fig, render, frames=len(common), blit=False)
    anim_path = os.path.join(out_dir, args.anim_name)
    anim.save(anim_path, writer=FFMpegWriter(fps=args.anim_fps), dpi=args.dpi)

    plt.close(fig)
    print(f"Frames written to {out_dir}")
    print(f"Animation written to {anim_path}")

    # The delta_n.png figure is rendered earlier (before the animation).

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
