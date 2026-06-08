#!/usr/bin/env python3
"""
Solve dispersion equation (3.19) exactly as an algebraic equation:

    1 - omega_pi^2/omega^2
      + 1/(k^2 r_De^2) * (1 + i*sqrt(pi/2)*omega/(k*v_Te)) = 0.

The config is assumed to use standard PIC normalization me=e=c=1.
Temperatures in config are interpreted as keV, so
    T_code = T_keV / (m_e c^2) = T_keV / 510.99895.

Densities in config are interpreted as n/n0, with n0=1e13 cm^-3 by default.
In normalized code units:
    omega_ps^2 = n_s * q_s^2 / m_s,
    v_Te       = sqrt(T_e/m_e),
    r_De       = sqrt(T_e/(n_e*q_e^2)).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


MEC2_KEV = 510.99895000

# SI constants only for optional physical conversion.
EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
M_E_SI = 9.1093837015e-31
C_SI = 299792458.0


def plasma_frequency_unit_si(n0_cm3: float) -> float:
    """Angular electron plasma frequency for n0 in cm^-3, SI rad/s."""
    n0_m3 = n0_cm3 * 1.0e6
    return math.sqrt(n0_m3 * E_CHARGE**2 / (EPS0 * M_E_SI))


def get_species(config: dict, name: str | None, charge_sign: int) -> dict:
    particles = config["Particles"]
    if name is not None:
        for p in particles:
            if p.get("sort_name") == name:
                return p
        raise KeyError(f"Species '{name}' not found in config['Particles'].")

    # Fallback by charge sign.
    if charge_sign < 0:
        candidates = [p for p in particles if p.get("q", 0.0) < 0.0]
    else:
        candidates = [p for p in particles if p.get("q", 0.0) > 0.0]

    if not candidates:
        raise KeyError(f"No species with charge sign {charge_sign:+d} found.")
    return candidates[0]


def get_mode_vector(config: dict) -> np.ndarray:
    """
    Extract mode vector from the first SetParticles preset with coordinate.wave_number.
    For the provided test this is [0, 0, 1].
    """
    for preset in config.get("Presets", []):
        if preset.get("command") == "SetParticles":
            coord = preset.get("coordinate", {})
            if "wave_number" in coord:
                return np.asarray(coord["wave_number"], dtype=float)
    raise KeyError("Could not find Presets[*].coordinate.wave_number.")


def solve_from_config(
    config_path: str,
    n0_cm3: float = 1.0e13,
    electron_name: str = "electrons",
    ion_name: str = "ions",
    wave_number_is_mode: bool = True,
) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    electron = get_species(config, electron_name, charge_sign=-1)
    ion = get_species(config, ion_name, charge_sign=+1)

    # Temperatures: keV -> normalized energy in units of me*c^2.
    Te = float(electron["T"]) / MEC2_KEV
    Ti = float(ion["T"]) / MEC2_KEV  # not used in Eq. (3.19), kept for diagnostics.

    ne = float(electron["n"])
    ni = float(ion["n"])
    qe = abs(float(electron["q"]))
    qi = abs(float(ion["q"]))
    me = float(electron["m"])
    mi = float(ion["m"])

    mode = get_mode_vector(config)
    L = np.asarray(
        [
            float(config["Geometry"]["x"]),
            float(config["Geometry"]["y"]),
            float(config["Geometry"]["z"]),
        ],
        dtype=float,
    )

    if wave_number_is_mode:
        # Integer mode numbers in a periodic box: k_i = 2*pi*m_i/L_i.
        kvec = 2.0 * math.pi * mode / L
    else:
        # Direct interpretation: wave_number is already k in code units.
        kvec = mode

    k = float(np.linalg.norm(kvec))
    if k <= 0.0:
        raise ValueError("k must be nonzero.")

    # Normalized me=e=c=1 formulas.
    omega_pi = math.sqrt(ni * qi**2 / mi)
    omega_pe = math.sqrt(ne * qe**2 / me)
    vTe = math.sqrt(Te / me)
    rDe = math.sqrt(Te / (ne * qe**2))

    A = 1.0 / (k**2 * rDe**2)
    B = math.sqrt(math.pi / 2.0) / (k * vTe)

    # Eq. (3.19) multiplied by omega^2:
    #     i*A*B*omega^3 + (1 + A)*omega^2 - omega_pi^2 = 0.
    coeff = np.array([1j * A * B, 1.0 + A, 0.0, -omega_pi**2], dtype=complex)
    roots = np.roots(coeff)

    # Ion-acoustic branch: positive real frequency with damping Im(omega)<0.
    acoustic_candidates = [w for w in roots if (w.real > 0.0 and w.imag < 0.0)]
    if acoustic_candidates:
        omega_acoustic = min(acoustic_candidates, key=lambda w: abs(w.imag / w.real))
    else:
        omega_acoustic = max(roots, key=lambda w: w.real)

    # Small-damping estimate from Eqs. (3.20)-(3.21), for comparison only.
    cs = math.sqrt(Te / mi)
    omega_s_approx = k * cs / math.sqrt(1.0 + k**2 * rDe**2)
    gamma_approx = (
        omega_s_approx
        * math.sqrt(math.pi / 8.0 * me / mi)
        * (1.0 + k**2 * rDe**2) ** (-1.5)
    )

    omega_unit_si = plasma_frequency_unit_si(n0_cm3)
    length_unit_si = C_SI / omega_unit_si

    return {
        "Te_code": Te,
        "Ti_code": Ti,
        "ne_code": ne,
        "ni_code": ni,
        "me_code": me,
        "mi_code": mi,
        "kvec_code": kvec,
        "k_code": k,
        "vTe_code": vTe,
        "rDe_code": rDe,
        "omega_pe_code": omega_pe,
        "omega_pi_code": omega_pi,
        "A": A,
        "B": B,
        "cubic_coefficients": coeff,
        "roots_code": roots,
        "omega_acoustic_code": omega_acoustic,
        "omega_s_approx_code": omega_s_approx,
        "gamma_approx_code": gamma_approx,
        "omega_unit_si_rad_s": omega_unit_si,
        "length_unit_si_m": length_unit_si,
        "roots_si_rad_s": roots * omega_unit_si,
        "omega_acoustic_si_rad_s": omega_acoustic * omega_unit_si,
        "kvec_si_m_inv": kvec / length_unit_si,
        "k_si_m_inv": k / length_unit_si,
        "dt_code": float(config["Geometry"].get("dt", float("nan"))),
        "t_end_code": float(config["Geometry"].get("t", float("nan"))),
        "diagnose_period_code": float(config["Geometry"].get("diagnose_period", float("nan"))),
    }


def complex_to_pair_string(z: complex, scale: float = 1.0) -> str:
    """Format complex number multiplied by scale as 'Re +/- Im j'."""
    zs = z * scale
    return f"{zs.real:.16e} {zs.imag:+.16e}j"


def print_plasma_units(res: dict) -> None:
    """
    Print frequencies and characteristic times in electron-plasma units.

    Frequencies are normalized to the local electron plasma frequency

        Omega = omega / omega_pe,

    where omega_pe = sqrt(n_e q_e^2 / m_e) in code units.

    Times are printed in units of 1/omega_pe, i.e.

        theta = t * omega_pe.

    If n_e = 1, q_e = 1, m_e = 1, these values coincide with the usual
    code-unit frequencies and code-unit times.
    """
    omega_pe = float(res["omega_pe_code"])
    if omega_pe <= 0.0:
        raise ValueError("omega_pe_code must be positive.")

    w = complex(res["omega_acoustic_code"])
    omega_r = w.real
    gamma = -w.imag

    omega_r_pe = omega_r / omega_pe
    gamma_pe = gamma / omega_pe

    period_code = math.inf if omega_r == 0.0 else 2.0 * math.pi / abs(omega_r)
    damping_time_code = math.inf if gamma == 0.0 else 1.0 / gamma

    # Times measured in units of 1/omega_pe: theta = t_code * omega_pe.
    period_pe_inv = period_code * omega_pe
    damping_time_pe_inv = damping_time_code * omega_pe

    dt_code = float(res.get("dt_code", float("nan")))
    t_end_code = float(res.get("t_end_code", float("nan")))
    diagnose_period_code = float(res.get("diagnose_period_code", float("nan")))

    print("=== Electron-plasma units ===")
    print("Frequencies are in units of omega_pe; times are in units of 1/omega_pe.")
    print(f"omega_pe(code) = {omega_pe:.16e}")
    print()

    print("Exact roots normalized to omega_pe:")
    for j, root in enumerate(res["roots_code"]):
        print(f"root[{j}]/omega_pe = {complex_to_pair_string(root, 1.0 / omega_pe)}")
    print()

    print("Ion-acoustic branch normalized to omega_pe:")
    print(f"omega/omega_pe = {complex_to_pair_string(w, 1.0 / omega_pe)}")
    print(f"omega_r/omega_pe = {omega_r_pe:.16e}")
    print(f"Gamma/omega_pe   = {gamma_pe:.16e}")
    print(f"T_wave * omega_pe      = {period_pe_inv:.16e}")
    print(f"tau_damp * omega_pe    = {damping_time_pe_inv:.16e}")
    print()

    print("Config times in plasma units:")
    if math.isfinite(dt_code):
        print(f"dt * omega_pe              = {dt_code * omega_pe:.16e}")
    if math.isfinite(diagnose_period_code):
        print(f"diagnose_period * omega_pe = {diagnose_period_code * omega_pe:.16e}")
    if math.isfinite(t_end_code):
        print(f"t_end * omega_pe           = {t_end_code * omega_pe:.16e}")
    print()

    print("Resolution and run length:")
    if math.isfinite(dt_code) and dt_code > 0.0:
        print(f"steps per wave period  = {period_code / dt_code:.16e}")
        print(f"steps per damping time = {damping_time_code / dt_code:.16e}")
        print(f"omega_r * dt           = {omega_r * dt_code:.16e}")
        print(f"Gamma * dt             = {gamma * dt_code:.16e}")
    if math.isfinite(t_end_code) and t_end_code > 0.0:
        print(f"wave periods in run    = {t_end_code / period_code:.16e}")
        print(f"damping e-folds in run = {t_end_code / damping_time_code:.16e}")
        print(f"amplitude factor       = exp(-Gamma*t_end) = {math.exp(-gamma * t_end_code):.16e}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.json")
    parser.add_argument("--n0-cm3", type=float, default=1.0e13)
    parser.add_argument("--electron-name", default="electrons")
    parser.add_argument("--ion-name", default="ions")
    parser.add_argument(
        "--wave-number-is-k",
        action="store_true",
        help="Interpret config wave_number as k directly, not as an integer periodic mode.",
    )
    args = parser.parse_args()

    res = solve_from_config(
        args.config,
        n0_cm3=args.n0_cm3,
        electron_name=args.electron_name,
        ion_name=args.ion_name,
        wave_number_is_mode=not args.wave_number_is_k,
    )

    print("=== Input converted to me=e=c=1 code units ===")
    print(f"Te = {res['Te_code']:.16e}")
    print(f"Ti = {res['Ti_code']:.16e}   # not used in Eq. (3.19)")
    print(f"ne = {res['ne_code']:.16e}, ni = {res['ni_code']:.16e}")
    print(f"me = {res['me_code']:.16e}, mi = {res['mi_code']:.16e}")
    print(f"kvec = {res['kvec_code']}")
    print(f"k = {res['k_code']:.16e}")
    print(f"vTe = {res['vTe_code']:.16e}")
    print(f"r_De = {res['rDe_code']:.16e}")
    print(f"omega_pi = {res['omega_pi_code']:.16e}")
    print(f"k*r_De = {res['k_code'] * res['rDe_code']:.16e}")
    print()

    print("=== Exact roots of Eq. (3.19) in code units ===")
    for j, w in enumerate(res["roots_code"]):
        print(f"root[{j}] = {w.real:.16e} {w.imag:+.16e}j")

    w = res["omega_acoustic_code"]
    print()
    print("=== Ion-acoustic branch ===")
    print(f"omega = {w.real:.16e} {w.imag:+.16e}j")
    print(f"omega_r = {w.real:.16e}")
    print(f"Gamma = {-w.imag:.16e}   # for omega = omega_r - i*Gamma")
    print()

    print("=== Small-damping estimate, Eqs. (3.20)-(3.21), comparison only ===")
    print(f"omega_s = {res['omega_s_approx_code']:.16e}")
    print(f"Gamma   = {res['gamma_approx_code']:.16e}")
    print()

    print("=== Optional SI conversion for n0 = %.6e cm^-3 ===" % args.n0_cm3)
    print(f"omega_pe0 = {res['omega_unit_si_rad_s']:.16e} rad/s")
    print(f"c/omega_pe0 = {res['length_unit_si_m']:.16e} m")
    wsi = res["omega_acoustic_si_rad_s"]
    print(f"ion-acoustic omega = {wsi.real:.16e} {wsi.imag:+.16e}j rad/s")
    print(f"k = {res['k_si_m_inv']:.16e} m^-1")
    print()

    print_plasma_units(res)


if __name__ == "__main__":
    main()
