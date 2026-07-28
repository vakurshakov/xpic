#!/usr/bin/env python3
"""Overlay the *full initial-value* (Landau) solution for the ion-acoustic
density perturbation on the measured |delta n_c(t)|.

Motivation
----------
The plain Landau decrement Gamma (the pole of eps(k, w) = 0) describes only the
pure eigen-mode  A e^{-Gamma t} cos(omega_s t).  In a PIC run the density is
launched from a *given* initial perturbation of the distribution function, which
projects not only onto that eigen-mode but also onto the ballistic
(free-streaming) continuum  e^{-i k v t}.  The interference of the two makes the
maxima/minima of |delta n_c(t)| oscillate *around* delta n_0 e^{-Gamma t} instead
of lying on it — exactly the effect seen in Terekhov, Timofeev, Lotov, Vestnik
NSU 5(2), 2010, eqs. (13)-(14), Fig. 6.

This script reconstructs the density analog of their eq. (14) and inverts it
numerically, then plots it on top of the PIC data so the wiggles can be compared
directly with theory (not just with the bare exponential envelope).

Theory reproduced here (single Fourier mode k, linear Vlasov-Poisson, Laplace in
time on the Landau contour).  For the initial perturbation loaded by the code —
a density modulation  n_s(z) = n0 (1 + a_n sin k z)  of a Maxwellian drifting at
u_s (both taken from the config: a_n from CoordinateInBox*Sine.amplitude_z, u_s
from momentum[_paired].velocity_z; for the standing-wave tests u_s = 0, i.e. an
*unshifted* Maxwellian) — the complex mode-1 amplitude of the ion density obeys

    c_i(w) = -R_i(w) + (chi_i(w)/eps(w)) [ R_i(w) - R_e(w) ],           (*)
    R_s(w) = a_n P_s(w) - u_s Q_s(w),
    P_s(w) = Z(zeta_s) / (sqrt(2) k v_Ts),                 zeta_s = w/(sqrt(2) k v_Ts),
    Q_s(w) = Z'(zeta_s) / (2 k v_Ts^2),                    Z'(z) = -2 (1 + z Z(z)),
    chi_s(w) = (1/(k r_Ds)^2)(1 + zeta_s Z(zeta_s)),       eps = 1 + chi_e + chi_i,

and the observable (the signed coefficient the density_z tools plot,
delta n_c = 2 <n/n0 * sin k z>_z) is

    delta n_c^theory(t) = -Im[ (1/2pi) \int_L c_i(w) e^{-i w t} dw ],

with L the horizontal Landau contour Im w = sigma > 0 above all poles.  The
integral is done by a direct DFT over a dense w-grid; because sigma > 0 the plasma
dispersion function is evaluated only in the upper half-plane, where Weideman's
rational approximation is directly valid (no lower-half reflection needed).

The full curve is additionally split into
  * pole part   -> residues of (*) at eps(w_j)=0  (the damped standing mode), and
  * ballistic   -> full - pole,
so the plot shows where the eigen-mode ends and the free-streaming tail begins.

Usage:
    drift_kinetic_density_bal.py <config.json> [--species ions] [options]
Works for both the standing-wave loaders (ex19-style, u_s = 0) and the
traveling-wave / shifted-Maxwell loaders (ex31-style, u_s != 0).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

TOOLS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from lib.constants import const, init_constants
from lib.plot import bbox, labelsize, ticksize

import matplotlib.pyplot as plt

# Re-use the exact machinery already validated in the single-run tool: config
# parsing, frame IO, the plasma-dispersion function and the ion-sound root
# finder. They all read the module-level `const`, so call after init_constants.
from drift_kinetic_density_z import (
    _weideman_coeffs,
    extract_plasma_params,
    find_sine_perturbation,
    iter_density_diagnostics,
    load_frame,
    locate_dir,
    solve_ion_sound_dispersion,
)

plt.rc("text", usetex=False)  # Cyrillic panel labels are incompatible with usetex


# --------------------------------------------------------------------------- #
# Vectorised plasma dispersion function on the upper half-plane (Im w = sigma
# > 0), i.e. Im(zeta) > 0.  Weideman's rational series is valid there directly,
# so — unlike the scalar `_faddeeva` in density_z, which recurses for Im < 0 —
# this operates on whole arrays with no branch.
# --------------------------------------------------------------------------- #
def faddeeva_upper(z, N=42):
    """Faddeeva w(z) = exp(-z^2) erfc(-i z) for arrays with Im z >= 0."""
    z = np.asarray(z, dtype=complex)
    a = _weideman_coeffs(N)
    L = np.sqrt(N / np.sqrt(2.0))
    Z = (L + 1j * z) / (L - 1j * z)
    p = np.polyval(a, Z.ravel()).reshape(Z.shape)
    return (2.0 * p / (L - 1j * z) ** 2
            + (1.0 / np.sqrt(np.pi)) / (L - 1j * z))


def Z_func(zeta):
    """Z(zeta) = i sqrt(pi) w(zeta) (Fried & Conte). Im(zeta) >= 0 assumed."""
    return 1j * np.sqrt(np.pi) * faddeeva_upper(zeta)


def Zp_func(zeta, Zval=None):
    """Z'(zeta) = -2 (1 + zeta Z(zeta))."""
    if Zval is None:
        Zval = Z_func(zeta)
    return -2.0 * (1.0 + zeta * Zval)


# --------------------------------------------------------------------------- #
# Initial condition read from the config: density amplitude a_n (shared by the
# quiet-start pair) and the per-species parallel drift u_s.  For the shifted
# Maxwellian loaders these two numbers fully specify the linear initial
# perturbation g_s(v) = -i n0 (a_n F_s(v) - u_s F_s'(v)); the J / n diagnostics
# are not needed (they only cross-check the loaded moments).
# --------------------------------------------------------------------------- #
def find_drift_velocities():
    """Return {species_name: u_parallel} from the SetParticles momentum blocks.
    'momentum' belongs to the 'particles' species, 'momentum_paired' to
    'paired_with'.  Missing / cold loaders give u = 0."""
    drifts = {}
    for preset in const.config.get("Presets", []):
        if preset.get("command") != "SetParticles":
            continue
        p_name = preset.get("particles")
        q_name = preset.get("paired_with")
        for key, name in (("momentum", p_name), ("momentum_paired", q_name)):
            block = preset.get(key)
            if block is None or name is None:
                continue
            vel = block.get("velocity")
            if vel is not None and len(vel) >= 3:
                drifts[name] = float(vel[2])
    return drifts


def species_names():
    """(electron_name, ion_name) by charge sign, as extract_plasma_params does."""
    e_name = i_name = None
    for s in const.config.get("Particles", []):
        q = float(s.get("q", 0.0))
        if q < 0.0 and e_name is None:
            e_name = s.get("sort_name")
        elif q > 0.0 and i_name is None:
            i_name = s.get("sort_name")
    return e_name, i_name


# --------------------------------------------------------------------------- #
# Full initial-value solution (eq. * above) and its inverse transform.
# --------------------------------------------------------------------------- #
def chi_and_eps(omega, k, params):
    """Return (chi_e, chi_i, eps) on the array `omega` (Im omega > 0).
    chi_s = (1/(k r_Ds)^2)(1 + zeta_s Z(zeta_s)); cold ions fall back to the
    fluid term -w_pi^2/w^2 (then there is no ion ballistic tail)."""
    root2 = np.sqrt(2.0)
    v_Te, r_De = params["v_Te"], params["r_De"]
    zeta_e = omega / (root2 * k * v_Te)
    chi_e = (1.0 + zeta_e * Z_func(zeta_e)) / (k * r_De) ** 2

    v_Ti = params.get("v_Ti", 0.0)
    r_Di = params.get("r_Di", float("inf"))
    if v_Ti and np.isfinite(r_Di):
        zeta_i = omega / (root2 * k * v_Ti)
        chi_i = (1.0 + zeta_i * Z_func(zeta_i)) / (k * r_Di) ** 2
    else:
        chi_i = -params["omega_pi2"] / omega ** 2
    return chi_e, chi_i, 1.0 + chi_e + chi_i


def R_s(omega, k, v_Ts, a_n, u_s):
    """R_s(w) = a_n P_s - u_s Q_s for one species (needs finite v_Ts)."""
    root2 = np.sqrt(2.0)
    zeta = omega / (root2 * k * v_Ts)
    Zv = Z_func(zeta)
    P = Zv / (root2 * k * v_Ts)
    Q = Zp_func(zeta, Zv) / (2.0 * k * v_Ts ** 2)
    return a_n * P - u_s * Q


def c_i_of_omega(omega, k, params, a_n, u_e, u_i):
    """Complex mode-1 ion-density amplitude in Laplace space, eq. (*)."""
    chi_e, chi_i, eps = chi_and_eps(omega, k, params)
    Ri = R_s(omega, k, params["v_Ti"], a_n, u_i)
    Re = R_s(omega, k, params["v_Te"], a_n, u_e)
    return -Ri + (chi_i / eps) * (Ri - Re)


def build_omega_grid(k, params, omega_r, Gamma, t_max):
    """Horizontal Landau contour w = xi + i sigma.  sigma sits above the pole
    (Im w_pole = -Gamma < 0); the xi-range must cover the electron ballistic
    width k v_Te and the mode, and dxi must resolve both the pole/Lorentzian
    width ~ (Gamma + sigma) and the longest time (dxi * t_max <~ 0.3)."""
    kvTe = k * params["v_Te"]
    kvTi = k * params.get("v_Ti", 0.0)
    Xi = max(10.0 * kvTe, 15.0 * omega_r, 15.0 * kvTi, 5.0 * max(Gamma, 0.0))
    scales = [0.3 / max(t_max, 1e-30)]
    if Gamma > 0.0:
        scales.append(0.1 * Gamma)
    if kvTi > 0.0:
        scales.append(0.1 * kvTi)
    dxi = min(scales)
    n = int(2.0 * Xi / dxi)
    if n > 4_000_000:                       # cap cost; coarsen if needed
        n = 4_000_000
        dxi = 2.0 * Xi / n
    xi = np.linspace(-Xi, Xi, n)
    sigma = 0.3 * Gamma if Gamma > 0.0 else 5.0 * dxi
    sigma = max(sigma, 3.0 * dxi)
    return xi, sigma, dxi


def invert_bromwich(xi, sigma, dxi, c_omega, times, chunk=64):
    """delta n_i(t) = (1/2pi) e^{sigma t} \int c_i(xi + i sigma) e^{-i xi t} dxi,
    a direct DFT over the dense xi-grid, chunked over the (few hundred) output
    times.  Returns the complex array delta n_i(t)."""
    weight = c_omega * dxi / (2.0 * np.pi)
    out = np.empty(times.shape, dtype=complex)
    for a in range(0, len(times), chunk):
        tb = times[a:a + chunk]
        phase = np.exp(-1j * np.outer(tb, xi))          # (chunk, Nxi)
        out[a:a + chunk] = np.exp(sigma * tb) * (phase @ weight)
    return out


def pole_part(times, k, params, a_n, u_e, u_i, omega_r, Gamma):
    """Residue (eigen-mode) contribution of eq. (*): sum over the two ion-sound
    poles w_j = +/- omega_r - i Gamma of  -i * e^{-i w_j t} *
    chi_i (R_i - R_e) / eps'(w_j).  eps' is taken by a complex finite
    difference.  This is the pure  A e^{-Gamma t} cos(omega_r t + phi)  part."""
    def residue(w0):
        # refine the root once more on the full eps (Newton), then eps'.
        w = complex(w0)
        for _ in range(60):
            _, _, e = chi_and_eps(np.array([w]), k, params)
            h = 1e-6 * abs(w) + 1e-12
            _, _, e2 = chi_and_eps(np.array([w + h]), k, params)
            ep = (e2[0] - e[0]) / h
            if ep == 0:
                break
            step = e[0] / ep
            w -= step
            if abs(step) <= 1e-13 * abs(w):
                break
        chi_e, chi_i, _ = chi_and_eps(np.array([w]), k, params)
        Ri = R_s(np.array([w]), k, params["v_Ti"], a_n, u_i)[0]
        Re = R_s(np.array([w]), k, params["v_Te"], a_n, u_e)[0]
        h = 1e-6 * abs(w) + 1e-12
        _, _, e = chi_and_eps(np.array([w]), k, params)
        _, _, e2 = chi_and_eps(np.array([w + h]), k, params)
        eps_p = (e2[0] - e[0]) / h
        amp = chi_i[0] * (Ri - Re) / eps_p
        return w, amp

    w1, A1 = residue(+omega_r - 1j * Gamma)
    w2, A2 = residue(-omega_r - 1j * Gamma)
    # Closing the contour downward (t > 0) picks -2 pi i * sum(res); the 1/2pi
    # of the inverse transform leaves -i * sum.
    dn = -1j * (A1 * np.exp(-1j * w1 * times) + A2 * np.exp(-1j * w2 * times))
    return dn


# --------------------------------------------------------------------------- #
# Measured signed mode-1 coefficient from the PIC density frames.
# --------------------------------------------------------------------------- #
def measured_delta_nc(species, explicit, n0, wn, Lz, Nz, dz):
    dir_path, steps = locate_dir(species, explicit)
    if dir_path is None:
        return None, None
    z = (np.arange(Nz) + 0.5) * dz
    sin_kz = np.sin(2.0 * np.pi * wn * z / Lz)
    times, coeff = [], []
    for idx, name in steps:
        data = load_frame(dir_path, name)
        if data is None:
            continue
        profile = data.mean(axis=(1, 2))                # <n>_{x,y}(z)
        coeff.append(2.0 * float(np.mean((profile / n0) * sin_kz)))
        times.append(idx * const.dts)
    if not times:
        return None, None
    return np.array(times, dtype=float), np.array(coeff, dtype=float)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="Path to config.json of the run")
    p.add_argument("--species", default="ions",
                   help="Species whose density mode-1 is modelled (default: ions)")
    p.add_argument("--n0", type=float, default=1.0,
                   help="Background density for normalisation (default 1.0)")
    p.add_argument("--out-subdir", default="density_bal",
                   help="Subdirectory under <out_dir>/processed for the figure")
    p.add_argument("--filename", default="density_bal.png")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--a-n", type=float, default=None,
                   help="Override the loaded density amplitude a_n "
                        "(default: CoordinateInBox*Sine.amplitude_z from config)")
    p.add_argument("--sigma", type=float, default=None,
                   help="Override the Landau-contour offset Im(w)=sigma")
    p.add_argument("--t-max-T", type=float, default=None,
                   help="Upper x-axis limit in units of T (default: full span)")
    return p.parse_args()


def main():
    args = parse_args()
    init_constants(args.config)

    # in_dir / out_dir fallback for output trees copied from another machine
    # (same convention as drift_kinetic_density_z.py).
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if not os.path.isdir(const.in_dir):
        const.in_dir = config_dir
        const.out_dir = os.path.join(config_dir, "processed")

    plasma = extract_plasma_params()
    if plasma is None:
        print("Could not extract electron+ion plasma parameters from config.")
        return
    if not (plasma.get("v_Ti", 0.0) > 0.0):
        print("[warn] cold ions (T_i = 0): no ion ballistic tail; the full "
              "solution reduces to the eigen-mode. Use a run with T_i > 0.")

    amp_cfg, wn = find_sine_perturbation()
    if not wn:
        print("No sine density perturbation (wave_number) found in config.")
        return
    a_n = args.a_n if args.a_n is not None else (amp_cfg or 0.0)
    if a_n == 0.0:
        print("[warn] density amplitude a_n = 0; nothing to model.")

    drifts = find_drift_velocities()
    e_name, i_name = species_names()
    u_e = drifts.get(e_name, 0.0)
    u_i = drifts.get(i_name, 0.0)

    k = 2.0 * np.pi * wn / const.Lz
    omega_r, Gamma = solve_ion_sound_dispersion(k, plasma)
    T_theory = 2.0 * np.pi / omega_r

    # --- measured signal ------------------------------------------------------
    available = {sp: ex for sp, ex in iter_density_diagnostics()}
    if args.species not in available:
        print(f"[skip] no density diagnostic for '{args.species}'; "
              f"available: {sorted(available)}")
        return
    t_meas, dnc_meas = measured_delta_nc(
        args.species, available[args.species], args.n0, wn,
        const.Lz, const.Nz, const.dz)
    if t_meas is None:
        print(f"No density frames found for '{args.species}'.")
        return
    t_max = float(t_meas[-1])

    # --- theory: full IVP solution + pole/ballistic split --------------------
    xi, sigma, dxi = build_omega_grid(k, plasma, omega_r, Gamma, t_max)
    if args.sigma is not None:
        sigma = args.sigma
    omega = xi + 1j * sigma
    c_omega = c_i_of_omega(omega, k, plasma, a_n, u_e, u_i)

    # Dense time grid for smooth theory curves, plus the measured times.
    t_dense = np.linspace(0.0, t_max, 2000)
    dn_full_dense = invert_bromwich(xi, sigma, dxi, c_omega, t_dense)
    dn_pole_dense = pole_part(t_dense, k, plasma, a_n, u_e, u_i, omega_r, Gamma)

    a_full = -np.imag(dn_full_dense)          # signed delta n_c^theory(t)
    a_pole = -np.imag(dn_pole_dense)
    a_ball = a_full - a_pole                   # free-streaming remainder

    # Analytic ion free-streaming Gaussian reference: |a_n| exp(-(k v_Ti t)^2/2).
    kvTi = k * plasma["v_Ti"] if plasma.get("v_Ti", 0.0) > 0.0 else 0.0
    gauss = (abs(a_n) * np.exp(-0.5 * (kvTi * t_dense) ** 2)
             if kvTi > 0.0 else None)

    # Console summary (mirrors density_z's diagnostic prints).
    print("Ballistic / full-IVP theory:")
    print(f"  species          = {args.species}")
    print(f"  k                = {k:.6g}  [w_pe/c],  k r_De = {k*plasma['r_De']:.4g}")
    print(f"  omega_r          = {omega_r:.6g}  [w_pe],  T = {T_theory:.6g}")
    print(f"  Gamma (Landau)   = {Gamma:+.6g}  [w_pe],  Gamma/omega_r = "
          f"{Gamma/omega_r:+.4g}")
    print(f"  a_n (density)    = {a_n:+.6g}   (src: "
          f"{'CLI' if args.a_n is not None else 'config'})")
    print(f"  u_e, u_i (drift) = {u_e:+.4g}, {u_i:+.4g}  [c]")
    if kvTi > 0.0:
        print(f"  k v_Ti           = {kvTi:.4g}  -> ion phase-mixing time "
              f"1/(k v_Ti) = {1.0/kvTi:.4g}  (= {1.0/kvTi/T_theory:.3f} T)")
    print(f"  contour: sigma   = {sigma:.4g}, dxi = {dxi:.4g}, "
          f"Xi = {xi[-1]:.4g}, N = {len(xi)}")
    print(f"  a_theory(0)      = {a_full[0]:+.6g}  (expect a_n = {a_n:+.6g})")
    print(f"  a_meas(0)        = {dnc_meas[0]:+.6g}")

    # --- figure ---------------------------------------------------------------
    t_hi = args.t_max_T if args.t_max_T else t_max / T_theory
    panel_bbox = dict(facecolor="white", edgecolor="none",
                      alpha=0.75, boxstyle="round,pad=0.25")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15.0, 7.0))

    # (a) signed delta n_c(t): data vs full theory vs pole-only.
    ax0.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax0.plot(t_meas / T_theory, dnc_meas, color="red", lw=1.6,
             label=r"PIC $\delta n_c(t)$")
    ax0.plot(t_dense / T_theory, a_full, color="black", lw=2.0,
             label=r"теория (полная, ур. 14)")
    ax0.plot(t_dense / T_theory, a_pole, color="tab:blue", lw=1.4, ls="--",
             label=r"собственная мода $A e^{-\Gamma t}\cos(\omega_r t+\phi)$")
    ax0.set_xlim(0.0, t_hi)
    ax0.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax0.set_ylabel(r"$\delta n_c(t)=\frac{2}{L_z}\int\frac{\langle n\rangle_{x,y}}"
                   r"{n_0}\sin(kz)\,dz$", fontsize=labelsize - 4)
    ax0.tick_params(labelsize=ticksize)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=ticksize)
    ax0.text(0.03, 0.97, "(а)", transform=ax0.transAxes, ha="left", va="top",
             fontsize=labelsize, bbox=panel_bbox)
    ax0.set_box_aspect(1)

    # (b) |delta n_c(t)|: data vs full theory vs envelopes.
    eps = 1e-30
    ax1.plot(t_meas / T_theory, np.abs(dnc_meas), color="red", lw=1.6,
             label=r"PIC $|\delta n_c|$")
    ax1.plot(t_dense / T_theory, np.abs(a_full), color="black", lw=2.0,
             label=r"теория (полная)")
    env = abs(a_n) * np.exp(-Gamma * t_dense)
    ax1.plot(t_dense / T_theory, env, color="tab:blue", lw=2.0, ls="--",
             label=rf"$\delta n_0 e^{{-\Gamma t}},\ \Gamma={Gamma:.2e}$")
    if gauss is not None:
        ax1.plot(t_dense / T_theory, gauss, color="tab:green", lw=1.6, ls=":",
                 label=r"баллистик $\delta n_0 e^{-(k v_{Ti} t)^2/2}$")
    ax1.set_xlim(0.0, t_hi)
    ax1.set_ylim(0.0, 1.25 * max(abs(a_n), float(np.max(np.abs(dnc_meas)))))
    ax1.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax1.set_ylabel(r"$|\delta n_c(t)|$", fontsize=labelsize)
    ax1.tick_params(labelsize=ticksize)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=ticksize)
    ax1.text(0.03, 0.97, "(б)", transform=ax1.transAxes, ha="left", va="top",
             fontsize=labelsize, bbox=panel_bbox)
    ax1.set_box_aspect(1)

    fig.suptitle(r"Начальная задача Ландау: полное решение $\delta n_c(t)$ "
                 "vs PIC", fontsize=labelsize, bbox=bbox)
    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.96))

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.filename)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Figure written to {out_path}")

    # CSV dump of the theory curves for downstream use.
    csv_path = os.path.join(out_dir,
                            os.path.splitext(args.filename)[0] + "_theory.csv")
    np.savetxt(csv_path,
               np.column_stack([t_dense, a_full, a_pole, a_ball]),
               delimiter=",",
               header="t,delta_nc_full,delta_nc_pole,delta_nc_ballistic",
               comments="", fmt="%.6g")
    print(f"Theory CSV written to {csv_path}")


if __name__ == "__main__":
    main()
