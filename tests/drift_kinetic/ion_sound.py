#!/usr/bin/env python3
"""Ion-acoustic (ion-sound) wave: kinetic dispersion relation, the linear
response of density and parallel velocity to a given electric field, and the
exact linearized Vlasov-Poisson evolution of the loaded initial condition.

The unperturbed distribution of every species is a one-dimensional Maxwellian

    F_0s(v) = n_s / (sqrt(2 pi) v_Ts) * exp(-v^2 / (2 v_Ts^2)),
    v_Ts    = sqrt(T_s / m_s).

Step 1.  Solve the longitudinal dispersion relation

    eps_L(omega, k) = 1 + sum_s 1/(k^2 lambda_Ds^2) [1 + zeta_s Z(zeta_s)] = 0,
    zeta_s = omega / (sqrt(2) k v_Ts),   lambda_Ds^2 = T_s / (n_s q_s^2),

for the ion-acoustic root  omega = omega_s - i Gamma_s  (Gamma_s > 0).  Z is the
plasma dispersion function, i.e. the Landau-CONTINUED integral

    Z(zeta) = 1/sqrt(pi) int_L du exp(-u^2)/(u - zeta),

which for Im(zeta) < 0 differs from the plain integral over the real u axis:

    Z(zeta) = int_R(...) + 2 i sqrt(pi) exp(-zeta^2).            (*)

Step 2.  For a real initial field  E_z(z, 0) = E_0 cos(k z)  the linear kinetic
response of the mode e^{i k z - i omega t} is

    f_s(v) = -i (q_s / m_s) E_0 dF_0s/dv / (omega - k v),

    n_hat_s = int f_s dv = -i (q_s E_0 / m_s) n_s / (k v_Ts^2) [1 + zeta_s Z(zeta_s)],

    n_s(z, 0) = n_s + A_s cos(k z) + B_s sin(k z),   n_hat_s = A_s - i B_s,

so that  A_s = Re(n_hat_s),  B_s = -Im(n_hat_s).  The velocity integral MUST be
taken along the Landau contour, i.e. with the continued Z of (*): only then does
the result satisfy Poisson's equation identically,

    sum_s q_s n_hat_s / (i k) = -i E_0 k (eps_L - 1) / (i k) = E_0   (eps_L = 0),

Using the plain real-axis integral instead breaks this identity and changes the
electron phase by O(2 sqrt(pi) zeta_e) ~ 25 % for typical ion-sound parameters.

Step 3.  The parallel-velocity perturbation follows from the exact (first-moment)
continuity equation  -i omega n_hat + i k n_s u_hat = 0,  u_hat = omega n_hat / (k n_s):

    du_par_s(z, 0) = 1/(k n_s) [ (omega_r A_s - gamma B_s) cos(k z)
                                +(omega_r B_s + gamma A_s) sin(k z) ].

Caveats of the model.
  * Linear theory: the printed amplitudes are meaningful only while C_n,s / n_s
    << 1; the script warns above 5 %.
  * Non-relativistic Maxwellian in v, while the PIC loader draws a Gaussian
    MOMENTUM and converts v = p / sqrt(m^2 + p^2).  The mismatch is O(v_T^2/c^2)
    (~4 % of the electron susceptibility at T_e = 20 keV).
  * 1D electrostatic: matches the drift-kinetic solver only for k || B with
    uniform B (no mirror force, no perpendicular dynamics).

Normalization.  The code uses the standard PIC units  m_e = e = c = 1, eps0 = 1.
Temperatures are given in keV and converted with  T_code = T_keV / (m_e c^2).
Densities are given in units of  n0 = 1e13 cm^-3.  Masses are given in units of
m_e, charges in units of e.  The field E_0 and the wave number k are given
directly in code units.
"""

import argparse
import math
import os
import sys

import numpy as np


MEC2_KEV = 510.99895000  # electron rest energy, keV

# SI constants, only for the optional physical conversion.
EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
M_E_SI = 9.1093837015e-31
C_SI = 299792458.0


# --------------------------------------------------------------------------- #
# Plasma dispersion function Z(zeta) = i sqrt(pi) w(zeta),  w = Faddeeva func. #
# --------------------------------------------------------------------------- #
def _cef(x: np.ndarray, n: int = 64) -> np.ndarray:
    """Faddeeva function w(x) = exp(-x^2) erfc(-i x) for Im(x) >= 0.

    Weideman's rational approximation (SIAM J. Numer. Anal. 31 (1994) 1497),
    accurate to ~machine precision for moderate |x| with n = 64.
    """
    m = 2 * n
    m2 = 2 * m
    k = np.arange(-m + 1, m)
    ll = math.sqrt(n / math.sqrt(2.0))
    theta = k * math.pi / m
    t = ll * np.tan(theta / 2.0)
    f = np.exp(-t**2) * (ll**2 + t**2)
    f = np.append(0.0, f)
    a = np.fft.fft(np.fft.fftshift(f)).real / m2
    a = np.flipud(a[1:n + 1])
    z = (ll + 1j * x) / (ll - 1j * x)
    p = np.polyval(a, z)
    return 2.0 * p / (ll - 1j * x)**2 + (1.0 / math.sqrt(math.pi)) / (ll - 1j * x)


def faddeeva(z: complex) -> complex:
    """Faddeeva function valid in the whole complex plane (scalar)."""
    z = complex(z)
    if z.imag >= 0.0:
        return complex(_cef(np.array([z]))[0])
    # Lower half-plane via  w(z) = 2 exp(-z^2) - w(-z).
    return 2.0 * np.exp(-z**2) - complex(_cef(np.array([-z]))[0])


def plasma_Z(zeta: complex) -> complex:
    """Plasma dispersion function Z(zeta)."""
    return 1j * math.sqrt(math.pi) * faddeeva(zeta)


def plasma_Z_prime(zeta: complex, z_val: complex | None = None) -> complex:
    """Derivative Z'(zeta) = -2 (1 + zeta Z(zeta))."""
    if z_val is None:
        z_val = plasma_Z(zeta)
    return -2.0 * (1.0 + zeta * z_val)


# --------------------------------------------------------------------------- #
# Dispersion relation                                                         #
# --------------------------------------------------------------------------- #
class Species:
    def __init__(self, name, n, q, m, T_code):
        self.name = name
        self.n = float(n)
        self.q = float(q)
        self.m = float(m)
        self.T = float(T_code)
        self.vT = math.sqrt(self.T / self.m)              # thermal velocity
        self.lamD = math.sqrt(self.T / (self.n * self.q**2))  # Debye length
        self.wp = math.sqrt(self.n * self.q**2 / self.m)  # plasma frequency

    def zeta(self, omega, k):
        return omega / (math.sqrt(2.0) * k * self.vT)


def epsilon(omega, k, species):
    """Longitudinal dielectric function eps_L(omega, k)."""
    total = 1.0 + 0.0j
    for s in species:
        zeta = s.zeta(omega, k)
        total += (1.0 / (k**2 * s.lamD**2)) * (1.0 + zeta * plasma_Z(zeta))
    return total


def epsilon_prime(omega, k, species):
    """d eps_L / d omega, analytic."""
    total = 0.0j
    for s in species:
        zeta = s.zeta(omega, k)
        zval = plasma_Z(zeta)
        zp = plasma_Z_prime(zeta, zval)
        dzeta_domega = 1.0 / (math.sqrt(2.0) * k * s.vT)
        total += (1.0 / (k**2 * s.lamD**2)) * dzeta_domega * (zval + zeta * zp)
    return total


def solve_dispersion(k, species, omega_guess, tol=1e-12, max_iter=200):
    """Newton iteration for the complex ion-acoustic root."""
    omega = complex(omega_guess)
    for _ in range(max_iter):
        f = epsilon(omega, k, species)
        fp = epsilon_prime(omega, k, species)
        if fp == 0:
            break
        step = f / fp
        omega -= step
        if abs(step) <= tol * max(1.0, abs(omega)):
            break
    return omega


def acoustic_initial_guess(k, electron, ion):
    """Fluid ion-acoustic frequency and Landau damping, used as a seed."""
    cs = math.sqrt(electron.T / ion.m)
    krd2 = (k * electron.lamD)**2
    omega_s = k * cs / math.sqrt(1.0 + krd2)
    gamma = omega_s * math.sqrt(math.pi / 8.0) * (
        math.sqrt(electron.m / ion.m) * (1.0 + krd2)**(-1.5)
    )
    return omega_s, gamma


# --------------------------------------------------------------------------- #
# Density and velocity perturbation amplitudes                                #
# --------------------------------------------------------------------------- #
def density_amplitudes(species, omega, k, E0):
    """Return A_s, B_s for n_s(z) = n_s + A_s cos(kz) + B_s sin(kz).

    A_s - i B_s = -i (q E0 / m) * n / (k vT^2) * (1 + zeta Z(zeta)).
    """
    zeta = species.zeta(omega, k)
    n_hat = -1j * (species.q * E0 / species.m) * species.n / (k * species.vT**2) \
        * (1.0 + zeta * plasma_Z(zeta))
    A = n_hat.real
    B = -n_hat.imag
    return A, B


def velocity_amplitudes(species, omega, k, A, B):
    """Return U_cos, U_sin for du_par_s(z) = U_cos cos(kz) + U_sin sin(kz)."""
    omega_r = omega.real
    gamma = -omega.imag
    U_cos = (omega_r * A - gamma * B) / (k * species.n)
    U_sin = (omega_r * B + gamma * A) / (k * species.n)
    return U_cos, U_sin


def density_sine(A, B):
    """Single-sinusoid form:  A cos(kz) + B sin(kz) = C sin(kz + phi),
    with  C = sqrt(A^2 + B^2),  phi = atan2(A, B)."""
    C = math.hypot(A, B)
    phi = math.atan2(A, B)
    return C, phi


def velocity_sine(species, omega, k, C_n, phi_n):
    """Single-sinusoid form of the parallel velocity perturbation:
        du_par_s = C_u sin(kz + phi_u),
        C_u   = sqrt(omega_r^2 + gamma^2)/k * C_n / n_s,
        phi_u = phi_n - atan2(gamma, omega_r).
    """
    omega_r = omega.real
    gamma = -omega.imag
    C_u = math.hypot(omega_r, gamma) / k * C_n / species.n
    phi_u = phi_n - math.atan2(gamma, omega_r)
    return C_u, phi_u


def field_from_density(species, omega, k, C_n_target):
    """Inverse problem: field amplitude E0 that produces a density
    perturbation of amplitude C_n_target for the given species.

    Since A_s, B_s are linear in E0, C_n(E0) = E0 * C_n(E0=1)."""
    A1, B1 = density_amplitudes(species, omega, k, 1.0)
    C1 = math.hypot(A1, B1)
    if C1 == 0.0:
        raise ValueError("Zero density response; cannot invert.")
    return C_n_target / C1


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def plasma_frequency_unit_si(n0_cm3):
    """Electron plasma frequency (rad/s) at reference density n0."""
    n0_m3 = n0_cm3 * 1.0e6
    return math.sqrt(n0_m3 * E_CHARGE**2 / (EPS0 * M_E_SI))


def run_theory(args):
    # Temperatures keV -> code units (m_e c^2).
    Te = args.Te / MEC2_KEV
    Ti = args.Ti / MEC2_KEV

    electron = Species("electrons", args.ne, args.qe, args.me, Te)
    ion = Species("ions", args.ni, args.qi, args.mi, Ti)
    species = [electron, ion]

    k = args.k if args.k is not None else 2.0 * math.pi * args.mode / args.Lz
    if k <= 0.0:
        raise ValueError("k must be positive.")

    # ---- Step 1: dispersion relation ------------------------------------- #
    omega_s_guess, gamma_guess = acoustic_initial_guess(k, electron, ion)
    omega = solve_dispersion(k, species, omega_s_guess - 1j * gamma_guess)
    resid = epsilon(omega, k, species)
    omega_r = omega.real
    Gamma = -omega.imag

    wpe = electron.wp  # electron plasma frequency in code units

    # ---- Field amplitude: forward (given E0) or inverse (given dn) -------- #
    inverse = args.dn is not None
    if inverse:
        target_species = ion if args.dn_species == "ions" else electron
        E0 = field_from_density(target_species, omega, k, args.dn)
    else:
        E0 = args.E0 if args.E0 is not None else 1.0e-3

    # ---- Step 2: density amplitudes -------------------------------------- #
    dens = {}
    for s in species:
        A, B = density_amplitudes(s, omega, k, E0)
        dens[s.name] = (A, B)

    # ---- Step 3: velocity amplitudes ------------------------------------- #
    vel = {}
    for s in species:
        A, B = dens[s.name]
        vel[s.name] = velocity_amplitudes(s, omega, k, A, B)

    # ---- Physical unit factors ------------------------------------------- #
    omega_unit = plasma_frequency_unit_si(args.n0_cm3)       # rad/s
    length_unit = C_SI / omega_unit                          # m
    field_unit = M_E_SI * C_SI * omega_unit / E_CHARGE       # V/m
    vel_unit = C_SI                                          # m/s

    # ===================================================================== #
    # Output                                                                #
    # ===================================================================== #
    print("=" * 70)
    print("INPUT PARAMETERS")
    print("=" * 70)
    print(f"  m_e = {args.me:.6g} m_e ,  m_i = {args.mi:.6g} m_e "
          f"(mass ratio m_i/m_e = {args.mi/args.me:.6g})")
    print(f"  T_e = {args.Te:.6g} keV ,  T_i = {args.Ti:.6g} keV")
    print(f"  n_e = {args.ne:.6g} n0 ,  n_i = {args.ni:.6g} n0    (n0 = {args.n0_cm3:.3e} cm^-3)")
    print(f"  q_e = {args.qe:+.6g} e ,  q_i = {args.qi:+.6g} e")
    if args.k is not None:
        print(f"  k   = {k:.6e} (code, given directly)")
    else:
        print(f"  L_z = {args.Lz:.6g} c/omega_p ,  mode = {args.mode}  ->  "
              f"k = 2*pi*mode/L_z = {k:.6e} (code)")
    if inverse:
        print(f"  MODE: inverse  (given C_n[{args.dn_species}] = {args.dn:.6e} n0"
              f"  ->  solved E0 = {E0:.6e} code)")
    else:
        print(f"  MODE: forward  (given E0 = {E0:.6e} code)")
    print()

    print("=" * 70)
    print("DERIVED PLASMA PARAMETERS (code units m_e=e=c=1)")
    print("=" * 70)
    print(f"  T_e = {Te:.6e} ,  T_i = {Ti:.6e}   [m_e c^2]")
    for s in species:
        print(f"  {s.name:9s}: v_T = {s.vT:.6e} ,  lambda_D = {s.lamD:.6e} ,"
              f"  omega_p = {s.wp:.6e}")
    print(f"  k lambda_De = {k*electron.lamD:.6e}")
    print(f"  c_s = sqrt(Te/mi) = {math.sqrt(electron.T/ion.m):.6e}")
    print()

    print("=" * 70)
    print("STEP 1 - ION-ACOUSTIC DISPERSION ROOT  omega = omega_s - i Gamma_s")
    print("=" * 70)
    print("  -- code units --")
    print(f"    omega_s = {omega_r:.6e}")
    print(f"    Gamma_s = {Gamma:.6e}")
    print(f"    |eps_L(omega)| residual = {abs(resid):.3e}")
    print("  -- physical units --")
    print(f"    omega_s = {omega_r*omega_unit:.6e} rad/s")
    print(f"    Gamma_s = {Gamma*omega_unit:.6e} rad/s")
    print(f"    f_s     = {omega_r*omega_unit/(2*math.pi):.6e} Hz")
    print(f"    k       = {k/length_unit:.6e} m^-1")
    print("  -- fluid seed for comparison --")
    print(f"    omega_s(fluid) = {omega_s_guess:.6e} , Gamma(Landau) = {gamma_guess:.6e} (code)")
    print()

    print("=" * 70)
    print("STEP 2 - DENSITY PERTURBATION")
    print("  cos/sin form :  dn_s(z,0) = A_s cos(kz) + B_s sin(kz)")
    print("  sine   form  :  dn_s(z,0) = C_n,s sin(kz + phi_n,s)")
    print("=" * 70)
    for s in species:
        A, B = dens[s.name]
        C_n, phi_n = density_sine(A, B)
        print(f"  {s.name}:")
        print(f"    A_s = {A:+.6e} ,  B_s = {B:+.6e}   [n0]")
        print(f"    a_s = A_s/n_s = {A/s.n:+.6e} ,  b_s = B_s/n_s = {B/s.n:+.6e}")
        print(f"    sine form :  amplitude C_n,s = {C_n:.6e} [n0] ,  "
              f"phase phi_n,s = {phi_n:+.6e} rad ({math.degrees(phi_n):+.3f} deg)")
        print(f"    MAX density perturbation |dn_s|_max = C_n,s = {C_n:.6e} [n0]  "
              f"( {C_n*args.n0_cm3:.6e} cm^-3 )")
        print(f"    relative amplitude C_n,s/n_s = {C_n/s.n:.6e}")
    print()

    print("=" * 70)
    print("STEP 3 - PARALLEL VELOCITY")
    print("  cos/sin form :  du_par_s(z,0) = U_cos cos(kz) + U_sin sin(kz)")
    print("  sine   form  :  du_par_s(z,0) = C_u,s sin(kz + phi_u,s)")
    print("=" * 70)
    for s in species:
        A, B = dens[s.name]
        Uc, Us = vel[s.name]
        C_n, phi_n = density_sine(A, B)
        C_u, phi_u = velocity_sine(s, omega, k, C_n, phi_n)
        print(f"  {s.name}:")
        print(f"    U_cos = {Uc:+.6e} ,  U_sin = {Us:+.6e}   [c]")
        print(f"    sine form :  amplitude C_u,s = {C_u:.6e} [c] ,  "
              f"phase phi_u,s = {phi_u:+.6e} rad ({math.degrees(phi_u):+.3f} deg)")
        print(f"    MAX velocity perturbation |du_s|_max = C_u,s = {C_u:.6e} [c]  "
              f"( {C_u*vel_unit:.6e} m/s )")
    print()

    print("=" * 70)
    print("UNIT CONVERSION FACTORS  (n0 = %.3e cm^-3)" % args.n0_cm3)
    print("=" * 70)
    print(f"  omega_pe0 (n0)      = {omega_unit:.6e} rad/s")
    print(f"  length  c/omega_pe0 = {length_unit:.6e} m")
    print(f"  field   E unit      = {field_unit:.6e} V/m   (E0_phys = {E0*field_unit:.6e} V/m)")
    print(f"  velocity unit c     = {vel_unit:.6e} m/s")
    print()

    # ---- Optional: theory-only |dn(t)| over --T periods ------------------ #
    if getattr(args, "T", None):
        # Loaded fluid IC = eigenmode moments: cn_hat = (A - iB)/n_s (relative),
        # u_hat = U_cos - i U_sin (velocity amplitude).
        cn_hat = {s.name: complex(dens[s.name][0], -dens[s.name][1]) / s.n
                  for s in species}
        u_hat = {s.name: complex(vel[s.name][0], -vel[s.name][1])
                 for s in species}
        out_path = args.out or "ion_sound_theory_dn.png"
        plot_theory_amplitude(species, cn_hat, u_hat, omega, k,
                              args.T, out_path, args.dpi,
                              exact_ic=args.exact_ic, E0=E0)
        print(f"Theory amplitude figure ({args.T:g} periods) written to {out_path}")
        print()


# --------------------------------------------------------------------------- #
# Model mode: compare the exact kinetic n_s(z, t) against a finished run       #
# --------------------------------------------------------------------------- #
def species_from_config(config):
    """Build (electron, ion) Species from config['Particles'] (T[keV]->code)."""
    electron = ion = None
    for s in config.get("Particles", []):
        q = float(s.get("q", 0.0))
        sp = Species(s.get("sort_name", ""), float(s.get("n", 1.0)), q,
                     float(s["m"]), float(s.get("T", 0.0)) / MEC2_KEV)
        if q < 0.0 and electron is None:
            electron = sp
        elif q > 0.0 and ion is None:
            ion = sp
    return electron, ion


def preset_for_species(config, name):
    for preset in config.get("Presets", []):
        if preset.get("command") == "SetParticles" and \
                preset.get("particles") == name:
            return preset
    return None


def loaded_perturbation(config, name):
    """Return (a_n, phi_n, C_u, phi_u) of the loaded z-perturbation for a sort.

    a_n, phi_n from the coordinate loader (amplitude_z / phase_z), C_u, phi_u
    from the momentum loader (velocity_z / phase_z). Handles two layouts:
      * a standalone SetParticles preset (particles == name);
      * a paired loader, where this sort is the `paired_with` target of another
        preset -- it then shares that preset's `coordinate` and takes its
        `momentum_paired` block. Missing keys -> 0."""
    def z_of(block, key):
        vec = block.get(key)
        return float(vec[2]) if vec is not None and len(vec) >= 3 else 0.0

    preset = preset_for_species(config, name)
    if preset is not None:
        coord, mom = preset.get("coordinate", {}), preset.get("momentum", {})
        return (z_of(coord, "amplitude"), z_of(coord, "phase"),
                z_of(mom, "velocity"), z_of(mom, "phase"))

    for pr in config.get("Presets", []):
        if pr.get("command") == "SetParticles" and pr.get("paired_with") == name:
            coord = pr.get("coordinate", {})            # shared coordinate
            mom = pr.get("momentum_paired", {})         # paired momentum
            return (z_of(coord, "amplitude"), z_of(coord, "phase"),
                    z_of(mom, "velocity"), z_of(mom, "phase"))
    return 0.0, 0.0, 0.0, 0.0


def field_amplitude_from_config(config):
    """E0 from the SetElectricField -> SetCosineField amplitude_z (or None)."""
    for preset in config.get("Presets", []):
        if preset.get("command") != "SetElectricField":
            continue
        amp = preset.get("setter", {}).get("amplitude")
        if amp is not None and len(amp) >= 3:
            return float(amp[2])
    return None


def wave_number_from_config(config):
    """Integer z-mode number of the density perturbation (default 1)."""
    for preset in config.get("Presets", []):
        if preset.get("command") != "SetParticles":
            continue
        wn = preset.get("coordinate", {}).get("wave_number")
        if wn is not None and len(wn) >= 3 and wn[2] != 0.0:
            return float(wn[2])
    return 1.0


def solve_vlasov_poisson(species, cn_hat, u_hat, k, t_max, n_record=400,
                         exact_ic=False, omega0=None, E0=0.0):
    """Exact linearized Vlasov-Poisson evolution of one Fourier mode e^{ikz}.

    Advances the first-harmonic perturbed distribution of every species,

        d/dt f_s(v,t) = -i k v f_s - (q_s/m_s) E(t) dF0s/dv,
        i k E(t) = sum_s q_s int f_s dv                       (Poisson, eps0=1),

    from the loaded initial condition. With `exact_ic=False` (default) the fluid
    (shifted-Maxwellian) IC is used,

        f_s(v,0) = cn_hat_s F0s(v) - u_hat_s dF0s/dv,

    matching only the loaded density (cn_hat_s) and bulk-velocity (u_hat_s)
    moments. With `exact_ic=True` the true kinetic eigenmode is loaded instead,

        f_s(v,0) = -i (q_s E0 / m_s) dF0s/dv / (omega0 - k v),

    (omega0 = omega_s - i Gamma_s), which starts as a pure eigenmode -> |dn(t)|
    decays as e^{-Gamma t} with no ballistic transient.

    RK4 on per-species velocity grids; captures the collective (Landau) mode AND
    the ballistic phase-mixing exactly and self-consistently.

    Returns (t_rec, nhat) with nhat[name] the complex first-harmonic density
    amplitude sampled at `n_record` uniform times t_rec in [0, t_max].
    """
    wp_tot = math.sqrt(sum(s.wp ** 2 for s in species))
    v_max = 7.0 * max(s.vT for s in species)
    # RK4 stability: resolve the plasma oscillation and the advection; also keep
    # enough steps to resolve the wave over the whole window.
    dt = min(0.9 / wp_tot, 0.5 / (k * v_max), t_max / (4 * n_record))
    n_steps = max(int(math.ceil(t_max / dt)), 4 * n_record)
    dt = t_max / n_steps

    grids, dF0, f = {}, {}, {}
    for s in species:
        # Velocity resolution set so the recurrence time 2*pi/(k dv) > 2 t_max.
        nv = int(np.clip(14.0 * s.vT * k * 2.0 * t_max / (2.0 * math.pi),
                         1500, 8000))
        # The exact-kinetic IC has a sharp resonant feature of width Gamma/k near
        # v = omega_s/k; when it lies inside the grid (electrons) refine dv so a
        # few cells span the width, otherwise the IC (and the mode) is corrupted.
        if exact_ic and omega0 is not None:
            v_ph = omega0.real / k
            width = abs(omega0.imag) / k
            if abs(v_ph) < 7.0 * s.vT and width > 0.0:
                nv = int(np.clip(max(nv, 14.0 * s.vT / (width / 3.0)),
                                 1500, 40000))
        v = np.linspace(-7.0 * s.vT, 7.0 * s.vT, nv)
        F0 = s.n / math.sqrt(2.0 * math.pi) / s.vT * np.exp(-v ** 2 / (2.0 * s.vT ** 2))
        grids[s.name] = v
        dF0[s.name] = -v / s.vT ** 2 * F0
        if exact_ic:
            # True kinetic eigenmode: -i (q E0 / m) dF0/dv / (omega0* - k v).
            # The conjugate root omega0* = omega_s + i Gamma is required to match
            # this solver's time convention (free streaming ~ e^{-i k v t}); it
            # yields the clean damped eigenmode |dn(t)| = |dn(0)| e^{-Gamma t}.
            # The denominator never vanishes (Im != 0).
            f[s.name] = -1j * (s.q * E0 / s.m) * dF0[s.name] \
                / (np.conj(omega0) - k * v)
        else:
            f[s.name] = cn_hat[s.name] * F0 - u_hat[s.name] * dF0[s.name]

    def rhs(state):
        rho = sum(s.q * np.trapezoid(state[s.name], grids[s.name]) for s in species)
        E = rho / (1j * k)
        return {s.name: -1j * k * grids[s.name] * state[s.name]
                - (s.q / s.m) * E * dF0[s.name] for s in species}

    stride = max(1, n_steps // n_record)
    t_rec = [0.0]
    nhat = {s.name: [complex(np.trapezoid(f[s.name], grids[s.name]))]
            for s in species}
    for step in range(1, n_steps + 1):
        k1 = rhs(f)
        k2 = rhs({n: f[n] + 0.5 * dt * k1[n] for n in f})
        k3 = rhs({n: f[n] + 0.5 * dt * k2[n] for n in f})
        k4 = rhs({n: f[n] + dt * k3[n] for n in f})
        f = {n: f[n] + dt / 6.0 * (k1[n] + 2 * k2[n] + 2 * k3[n] + k4[n]) for n in f}
        if step % stride == 0 or step == n_steps:
            t_rec.append(step * dt)
            for s in species:
                nhat[s.name].append(complex(np.trapezoid(f[s.name], grids[s.name])))
    return np.array(t_rec), {n: np.array(v) for n, v in nhat.items()}


def plot_theory_amplitude(species, cn_hat, u_hat, omega0, k, n_periods,
                          out_path, dpi, exact_ic=False, E0=0.0):
    """Theory-only figure of |dn_s(t)|/n0 over `n_periods` wave periods, from the
    exact linearized Vlasov-Poisson evolution. No simulation data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = 2.0 * math.pi / omega0.real
    Gamma = -omega0.imag
    t_max = n_periods * T
    t, nhat = solve_vlasov_poisson(species, cn_hat, u_hat, k, t_max,
                                   exact_ic=exact_ic, omega0=omega0, E0=E0)

    colors = {"ions": "red", "electrons": "blue"}
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    for s in species:
        ax.plot(t / T, np.abs(nhat[s.name]) / s.n,
                color=colors.get(s.name, "black"), linewidth=2.0,
                label=rf"$|\delta n_{{{s.name[0]}}}(t)|$ (theory)")

    a0 = abs(cn_hat.get("ions", next(iter(cn_hat.values()))))
    ax.plot(t / T, a0 * np.exp(-Gamma * t), color="black", linestyle="--",
            linewidth=1.5, alpha=0.7,
            label=r"$|\delta n(0)|\,e^{-\Gamma t}$ (pure Landau)")

    ax.set_xlim(0.0, n_periods)
    ax.set_ylim(0.0, None)
    ax.set_xlabel(r"$t/T$", fontsize=14)
    ax.set_ylabel(r"$|\delta n(t)|/n_0$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_title(r"Theory: perturbation amplitude (exact Vlasov-Poisson)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def field_view_dir(config, field="E"):
    """Output sub-directory of the FieldView diagnostic for `field` (or None)."""
    for d in config.get("Diagnostics", []):
        if str(d.get("diagnostic", "")).startswith("FieldView") \
                and d.get("field") == field:
            return d.get("out_dir")
    return None


def prepare_theory(testname):
    """Load output/<testname>/config.json, extract plasma parameters and the
    loaded initial condition, solve the dispersion root, and build the complex
    first-harmonic amplitudes cn_hat, u_hat. Returns a context dict shared by the
    density (`--model`) and field (`--model_electric`) comparison modes.

    Sets up sys.path and lib.constants' global `const`. Species without their own
    perturbation inherit the DENSITY (not the velocity) from one that has it."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(tests_dir, "..", ".."))
    for p in (os.path.join(repo_dir, "tools"),
              os.path.join(tests_dir, "drift_kinetic_tools")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from lib.constants import const, init_constants

    config_path = os.path.join(tests_dir, "output", testname, "config.json")
    if not os.path.isfile(config_path):
        raise SystemExit(f"config not found: {config_path}")
    init_constants(config_path)

    config_dir = os.path.dirname(config_path)
    if not os.path.isdir(const.in_dir):
        const.in_dir = config_dir
        const.out_dir = os.path.join(config_dir, "processed")

    config = const.config
    electron, ion = species_from_config(config)
    if electron is None or ion is None:
        raise SystemExit("config must define one electron (q<0) and one ion (q>0) sort.")
    species = [electron, ion]

    wn = wave_number_from_config(config)
    k = 2.0 * math.pi * wn / const.Lz

    omega_s_guess, gamma_guess = acoustic_initial_guess(k, electron, ion)
    omega0 = solve_dispersion(k, species, omega_s_guess - 1j * gamma_guess)

    E0 = field_amplitude_from_config(config)
    ic = {s.name: list(loaded_perturbation(config, s.name)) for s in species}
    # Fallback for configs that perturb only one species: inherit the DENSITY
    # (amplitude + phase), NOT the velocity, from a species that has it.
    donor = next((nm for nm in ic if ic[nm][0] != 0.0), None)
    for nm in ic:
        if ic[nm][0] == 0.0 and donor is not None:
            ic[nm][0], ic[nm][1] = ic[donor][0], ic[donor][1]
            print(f"[info] no density perturbation for '{nm}' in config; using "
                  f"'{donor}' (dn_{nm[0]} = dn_{donor[0]}; velocity left at 0)")
    cn_hat = {name: -1j * a_n * np.exp(1j * phi_n)
              for name, (a_n, phi_n, _, _) in ic.items()}
    u_hat = {name: -1j * C_u * np.exp(1j * phi_u)
             for name, (_, _, C_u, phi_u) in ic.items()}
    if E0 is None:  # fallback: Poisson  ik E0 = sum_s q_s cn_hat_s
        E0 = float((sum(s.q * cn_hat[s.name] for s in species) / (1j * k)).real)

    return dict(const=const, config=config, config_path=config_path,
                config_dir=config_dir, species=species, electron=electron,
                ion=ion, wn=wn, k=k, omega0=omega0, cn_hat=cn_hat, u_hat=u_hat,
                E0=E0, ic=ic, T_wave=2.0 * math.pi / omega0.real)


def run_model(args):
    ctx = prepare_theory(args.model)
    const, config = ctx["const"], ctx["config"]
    species, electron, ion = ctx["species"], ctx["electron"], ctx["ion"]
    k, omega0, wn = ctx["k"], ctx["omega0"], ctx["wn"]
    cn_hat, u_hat, E0, ic = ctx["cn_hat"], ctx["u_hat"], ctx["E0"], ctx["ic"]
    omega_r, Gamma, T_wave = omega0.real, -omega0.imag, ctx["T_wave"]
    config_path, config_dir = ctx["config_path"], ctx["config_dir"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from lib.plot import bbox, labelsize, ticksize
    import drift_kinetic_density_z as dz

    # ---- Load simulation density frames (reuse density_z machinery) -------- #
    rows = dz.collect_rows(args.species)
    if not rows and const.in_dir != config_dir:
        const.in_dir = config_dir
        const.out_dir = os.path.join(config_dir, "processed")
        rows = dz.collect_rows(args.species)
    if not rows:
        raise SystemExit("No density diagnostics found for the requested species.")

    common = set(idx for idx, _ in rows[0]["timesteps"])
    for row in rows[1:]:
        common &= set(idx for idx, _ in row["timesteps"])
    common = sorted(common)
    if not common:
        raise SystemExit("No timesteps common to all requested density diagnostics.")

    names_per_row = []
    for row in rows:
        name_by_idx = dict(row["timesteps"])
        names_per_row.append([name_by_idx[idx] for idx in common])

    n0 = 1.0
    z = (np.arange(const.Nz) + 0.5) * const.dz
    times = np.array([idx * const.dts for idx in common], dtype=float)
    t_max = float(times[-1]) if times.size else T_wave

    def profile_ext(data):
        return data.mean(axis=(1, 2))

    # ---- Exact linearized Vlasov-Poisson theory over the run window -------- #
    t_grid, nhat = solve_vlasov_poisson(species, cn_hat, u_hat, k,
                                        max(t_max, T_wave * 1e-3),
                                        exact_ic=args.exact_ic, omega0=omega0, E0=E0)
    theory = {s.name: np.abs(nhat[s.name]) / s.n for s in species}

    # ---- Summary ---------------------------------------------------------- #
    print("=" * 70)
    print(f"MODEL: {args.model}")
    print("=" * 70)
    print(f"  config      = {config_path}")
    print(f"  m_i/m_e = {ion.m/electron.m:.6g} ,  T_e = {electron.T*MEC2_KEV:.6g} keV ,"
          f"  T_i = {ion.T*MEC2_KEV:.6g} keV")
    print(f"  k = {k:.6e} (mode {wn:.0f}, Lz = {const.Lz:.6g})")
    print(f"  omega_s = {omega_r:.6e} ,  Gamma_s = {Gamma:.6e} ,  T = {T_wave:.6g} [1/w_pe]")
    print(f"  E0 = {E0:.6e}")
    for s in species:
        a_n, phi_n, C_u, phi_u = ic[s.name]
        print(f"  {s.name:9s}: a_n = {a_n:.6e}, phi_n = {phi_n:+.6f} ,"
              f"  C_u = {C_u:.6e}, phi_u = {phi_u:+.6f}")
        print(f"             theory dn(0) = {theory[s.name][0]:.6e}")
    print(f"  frames = {len(common)} , t in [0, {t_max:.6g}] [1/w_pe]")
    print()

    # ---- Figure: left = density profile, right = perturbation amplitude ---- #
    style = dict(dz.SPECIES_STYLE)
    # Left panel y-range scales with the loaded density amplitude dn/n0:
    # [1 - 2 dn/n0, 1 + 2 dn/n0], with dashed reference lines at 1 +- dn/n0.
    a_n_max = max((ic[s.name][0] for s in species), default=0.0) or 0.03
    ylim_lo, ylim_hi = 1.0 - 2.0 * a_n_max, 1.0 + 2.0 * a_n_max
    amp_hi, amp_lo = 1.0 + a_n_max, 1.0 - a_n_max
    panel_bbox = dict(facecolor="white", edgecolor="none", alpha=0.75,
                      boxstyle="round,pad=0.25")

    fig, (ax_z, ax_amp) = plt.subplots(1, 2, figsize=(15.0, 7.0))

    def draw_amplitude_lines(ax, x0):
        for level in (amp_hi, amp_lo):
            ax.axhline(level, color="tab:gray", linewidth=1.0,
                       linestyle="--", alpha=0.8)
            ax.text(x0, level + 0.002, f"{level:.2f}", color="tab:gray",
                    fontsize=ticksize, ha="left", va="bottom")

    lines_z = {}
    for row in rows:
        st = style.get(row["species"], {"marker": "o", "linestyle": "-",
                                        "label": row["species"]})
        lines_z[row["species"]] = ax_z.plot([], [], **st)[0]
    ax_z.set_xlim(0.0, const.Lz)
    ax_z.set_ylim(ylim_lo, ylim_hi)
    ax_z.set_xlabel(r"$z~(c/\omega_{pe})$", fontsize=labelsize)
    ax_z.set_ylabel(r"$\langle n/n_0 \rangle_{x,y}(z)$", fontsize=labelsize)
    ax_z.tick_params(labelsize=ticksize)
    ax_z.grid(True, alpha=0.3)
    ax_z.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    draw_amplitude_lines(ax_z, 0.01 * const.Lz)
    ax_z.legend(loc="upper right", fontsize=labelsize)
    ax_z.set_box_aspect(1)
    ax_z.set_title(r"Профиль плотности", fontsize=labelsize, bbox=bbox)

    # Right panel: simulation dn(t) for every species (accumulated), plus the
    # exact kinetic theory (ballistic + collective) for ions only.
    lines_amp = {}
    series_amp = {}
    for row in rows:
        st = dict(style.get(row["species"], {"marker": "o", "linestyle": "-"}))
        st["label"] = rf"$\delta n_{{{row['species'][0]}}}$ (model)"
        lines_amp[row["species"]] = ax_amp.plot([], [], **st)[0]
        series_amp[row["species"]] = []

    dn_theory_i = theory[ion.name]
    ax_amp.plot(t_grid / T_wave, dn_theory_i, color="black", linewidth=2.0,
                linestyle="-", alpha=0.9,
                label=r"$\delta n_i$ (theory, ballistic+collective)")

    amp_hi_lim = max(0.06, 1.25 * float(np.nanmax(dn_theory_i)))
    ax_amp.set_xlim(0.0, t_max / T_wave)
    ax_amp.set_ylim(0.0, amp_hi_lim)
    ax_amp.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_amp.set_ylabel(
        r"$\delta n(t) = \sqrt{\frac{2}{L_z}\int"
        r"\left(\langle n\rangle_{x,y}/n_0 - 1\right)^2 dz}$", fontsize=labelsize)
    ax_amp.tick_params(labelsize=ticksize)
    ax_amp.grid(True, alpha=0.3)
    ax_amp.set_title(r"Амплитуда возмущения", fontsize=labelsize, bbox=bbox)
    ax_amp.set_box_aspect(1)
    ax_amp.legend(loc="upper right", fontsize=ticksize)

    suptitle = fig.suptitle(rf"$t = {0.0:.3f}\,T$", bbox=bbox, fontsize=labelsize)
    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.95))
    fig.canvas.draw()
    sp = fig.subplotpars
    fig.subplots_adjust(left=sp.left, right=sp.right, bottom=sp.bottom,
                        top=sp.top, wspace=sp.wspace, hspace=sp.hspace)

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    times_acc = []

    def render(kf):
        idx = common[kf]
        t_phys = idx * const.dts
        times_acc.append(t_phys / T_wave)
        for row, names in zip(rows, names_per_row):
            data = dz.load_frame(row["dir"], names[kf])
            if data is None:
                continue
            profile = profile_ext(data)
            lines_z[row["species"]].set_data(z, profile / n0)
            dev = profile / n0 - 1.0
            series_amp[row["species"]].append(float(np.sqrt(2.0 * np.mean(dev * dev))))
            lines_amp[row["species"]].set_data(times_acc, series_amp[row["species"]])
        suptitle.set_text(rf"$t = {t_phys / T_wave:.3f}\,T$")
        return [*lines_z.values(), *lines_amp.values(), suptitle]

    anim = FuncAnimation(fig, render, frames=len(common), blit=False)
    anim_path = os.path.join(out_dir, args.anim_name)
    try:
        anim.save(anim_path, writer=FFMpegWriter(fps=args.anim_fps), dpi=args.dpi)
        print(f"Animation written to {anim_path}")
    except Exception as exc:  # ffmpeg missing / unavailable: keep the PNG
        print(f"[warn] could not write animation ({exc}); rendering final frame only")
        render(len(common) - 1)

    png_path = os.path.join(out_dir, "ion_sound_model.png")
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Final figure written to {png_path}")


def run_model_electric(args):
    ctx = prepare_theory(args.model_electric)
    const, config = ctx["const"], ctx["config"]
    species = ctx["species"]
    k, omega0 = ctx["k"], ctx["omega0"]
    cn_hat, u_hat, E0, ic = ctx["cn_hat"], ctx["u_hat"], ctx["E0"], ctx["ic"]
    omega_r, Gamma, T_wave = omega0.real, -omega0.imag, ctx["T_wave"]
    config_dir = ctx["config_dir"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from lib.plot import bbox, labelsize, ticksize
    plt.rc("text", usetex=False)  # Cyrillic panel titles are incompatible with usetex

    # ---- Locate the FieldView E frames ------------------------------------ #
    edir_name = field_view_dir(config, "E")
    if edir_name is None:
        raise SystemExit("config has no FieldView diagnostic for field 'E'.")
    epath = os.path.join(const.in_dir, edir_name)
    if not os.path.isdir(epath):
        epath = os.path.join(config_dir, edir_name)
    if not os.path.isdir(epath):
        raise SystemExit(f"E field frames not found (looked in '{edir_name}').")

    frames = sorted((int(n), n) for n in os.listdir(epath) if n.isdigit())
    if not frames:
        raise SystemExit(f"no E frames in {epath}.")
    idxs = [i for i, _ in frames]
    names = [n for _, n in frames]
    times = np.array([i * const.dts for i in idxs], dtype=float)
    t_max = float(times[-1])

    # The FieldView frame is a plane (or volume) of the 3-component E vector.
    ncomp = 3
    ncells = os.path.getsize(os.path.join(epath, names[0])) // 4 // ncomp
    nx_plane = ncells // const.Nz  # cells per z-row (x, and y for a volume dump)

    def load_Ez(name):
        raw = np.fromfile(os.path.join(epath, name), dtype=np.float32,
                          count=ncells * ncomp)
        return raw.reshape(-1, ncomp)[:, 2]  # E_z of every cell in the frame

    def Ez_profile(name):
        # Average E_z over the transverse cells -> E_z(z). z is the slow axis.
        return load_Ez(name).reshape(const.Nz, nx_plane).mean(axis=1)

    # Simulation amplitude, same recipe as the density: average over the
    # transverse cells first (filters PIC noise / higher transverse structure),
    # then the L2 metric  dE(t) = sqrt( (2/Lz) int <E_z>_x^2 dz ) = |E_z_hat|.
    profiles = [Ez_profile(n) for n in names]
    dE_sim = np.array([np.sqrt(2.0 * np.mean(pz ** 2)) for pz in profiles])

    # ---- Theory: E_hat(t) = (sum_s q_s n_hat_s) / (i k),  |E| = |sum|/k ----- #
    t_grid, nhat = solve_vlasov_poisson(species, cn_hat, u_hat, k,
                                        max(t_max, T_wave * 1e-3))
    rho_hat = sum(s.q * nhat[s.name] for s in species)
    dE_theory = np.abs(rho_hat) / k

    # ---- Summary ---------------------------------------------------------- #
    print("=" * 70)
    print(f"MODEL_ELECTRIC: {args.model_electric}")
    print("=" * 70)
    print(f"  k = {k:.6e} (mode {ctx['wn']:.0f}, Lz = {const.Lz:.6g})")
    print(f"  omega_s = {omega_r:.6e} ,  Gamma_s = {Gamma:.6e} ,  T = {T_wave:.6g}")
    print(f"  E0 (config/Poisson) = {E0:.6e}")
    for s in species:
        a_n, phi_n, C_u, phi_u = ic[s.name]
        print(f"  {s.name:9s}: a_n = {a_n:.6e}, phi_n = {phi_n:+.6f} ,"
              f"  C_u = {C_u:.6e}, phi_u = {phi_u:+.6f}")
    print(f"  E frames = {len(names)} in '{edir_name}' , t in [0, {t_max:.6g}]")
    print(f"  |E|(0): theory = {dE_theory[0]:.6e} , model = {dE_sim[0]:.6e}")
    print()

    # ---- Figure: left = E_z(z) profile, right = |E|(t) amplitude ---------- #
    z = (np.arange(const.Nz) + 0.5) * const.dz
    e_lim = max(1e-30, 1.3 * float(max(np.max(np.abs(pz)) for pz in profiles)))
    amp_hi_lim = max(float(np.nanmax(dE_theory)), float(np.nanmax(dE_sim))) * 1.25

    fig, (ax_z, ax_amp) = plt.subplots(1, 2, figsize=(15.0, 7.0))

    (line_z,) = ax_z.plot([], [], color="purple", marker="o", linestyle="-",
                          linewidth=2.5, markersize=4.0, label=r"$E_z$")
    ax_z.set_xlim(0.0, const.Lz)
    ax_z.set_ylim(-e_lim, e_lim)
    ax_z.set_xlabel(r"$z~(c/\omega_{pe})$", fontsize=labelsize)
    ax_z.set_ylabel(r"$\langle E_z \rangle_{x}(z)$", fontsize=labelsize)
    ax_z.tick_params(labelsize=ticksize)
    ax_z.grid(True, alpha=0.3)
    ax_z.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax_z.legend(loc="upper right", fontsize=labelsize)
    ax_z.set_box_aspect(1)
    ax_z.set_title(r"Профиль поля $E_z$", fontsize=labelsize, bbox=bbox)

    (line_amp,) = ax_amp.plot([], [], color="purple", marker="o", linestyle="-",
                              linewidth=2.5, markersize=4.0, label=r"$|E_z|$ (model)")
    ax_amp.plot(t_grid / T_wave, dE_theory, color="black", linewidth=2.0,
                label=r"$|E_z|$ (theory, Vlasov-Poisson)")
    ax_amp.set_xlim(0.0, t_max / T_wave)
    ax_amp.set_ylim(0.0, amp_hi_lim)
    ax_amp.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_amp.set_ylabel(
        r"$|E_z|(t) = \sqrt{\frac{2}{L_z}\int \langle E_z\rangle_x^2\,dz}$",
        fontsize=labelsize)
    ax_amp.tick_params(labelsize=ticksize)
    ax_amp.grid(True, alpha=0.3)
    ax_amp.set_title(r"Амплитуда поля", fontsize=labelsize, bbox=bbox)
    ax_amp.set_box_aspect(1)
    ax_amp.legend(loc="upper right", fontsize=ticksize)

    suptitle = fig.suptitle(rf"$t = {0.0:.3f}\,T$", bbox=bbox, fontsize=labelsize)
    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.95))
    fig.canvas.draw()
    sp = fig.subplotpars
    fig.subplots_adjust(left=sp.left, right=sp.right, bottom=sp.bottom,
                        top=sp.top, wspace=sp.wspace, hspace=sp.hspace)

    out_dir = os.path.join(const.out_dir, args.out_subdir + "_E")
    os.makedirs(out_dir, exist_ok=True)
    t_acc, amp_acc = [], []

    def render(kf):
        t_acc.append(times[kf] / T_wave)
        amp_acc.append(dE_sim[kf])
        line_z.set_data(z, Ez_profile(names[kf]))
        line_amp.set_data(t_acc, amp_acc)
        suptitle.set_text(rf"$t = {times[kf] / T_wave:.3f}\,T$")
        return [line_z, line_amp, suptitle]

    anim = FuncAnimation(fig, render, frames=len(names), blit=False)
    anim_path = os.path.join(out_dir, "ion_sound_model_E.mp4")
    try:
        anim.save(anim_path, writer=FFMpegWriter(fps=args.anim_fps), dpi=args.dpi)
        print(f"Animation written to {anim_path}")
    except Exception as exc:
        print(f"[warn] could not write animation ({exc}); rendering final frame only")
        render(len(names) - 1)

    png_path = os.path.join(out_dir, "ion_sound_model_E.png")
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Final figure written to {png_path}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Ion-sound kinetic theory (--theory) and comparison against "
                    "a finished simulation run (--model <testname>).")
    p.add_argument("--theory", action="store_true",
                   help="theory mode (default): print dispersion root and initial "
                        "density/velocity amplitudes for the CLI parameters")
    p.add_argument("--model", default=None,
                   help="model mode: test name (e.g. drift_kinetic_eigen_sound_ex1); "
                        "reads output/<name>/config.json and compares the exact "
                        "kinetic n_s(z,t) against the run's density diagnostics")
    p.add_argument("--model_electric", default=None,
                   help="field-comparison mode: test name; compares the exact "
                        "theory |E_z|(t) (first harmonic) against the run's E "
                        "FieldView frames (same L2 amplitude metric as density)")
    # --- theory-mode parameters ---
    p.add_argument("--me", type=float, default=1.0, help="electron mass [m_e]")
    p.add_argument("--mi", type=float, default=100.0, help="ion mass [m_e]")
    p.add_argument("--Te", type=float, default=5.0, help="electron temperature [keV]")
    p.add_argument("--Ti", type=float, default=0.01, help="ion temperature [keV]")
    p.add_argument("--ne", type=float, default=1.0, help="electron density [n0]")
    p.add_argument("--ni", type=float, default=1.0, help="ion density [n0]")
    p.add_argument("--qe", type=float, default=-1.0, help="electron charge [e]")
    p.add_argument("--qi", type=float, default=+1.0, help="ion charge [e]")
    p.add_argument("--Lz", type=float, default=200.0,
                   help="box length along k in units of c/omega_p (code length units)")
    p.add_argument("--mode", type=int, default=1,
                   help="integer mode number: k = 2*pi*mode/Lz")
    p.add_argument("--k", type=float, default=None,
                   help="wave number k in code units (overrides --Lz/--mode)")
    p.add_argument("--E0", type=float, default=None,
                   help="initial field amplitude E0 in code units (forward problem)")
    p.add_argument("--dn", type=float, default=None,
                   help="target density-perturbation amplitude C_n [n0] (inverse "
                        "problem: solve for E0); mutually exclusive with --E0")
    p.add_argument("--dn-species", default="ions", choices=["ions", "electrons"],
                   help="species whose density amplitude --dn refers to")
    p.add_argument("--n0-cm3", type=float, default=1.0e13,
                   help="reference density n0 [cm^-3]")
    p.add_argument("--T", type=float, default=None,
                   help="theory mode: if set, additionally plot the exact kinetic "
                        "|dn(t)| (ballistic + collective) over this many wave "
                        "periods, without any simulation data")
    p.add_argument("--exact-ic", action="store_true",
                   help="load the exact kinetic eigenmode f~(v) = -i(q E0/m) "
                        "dF0/dv / (omega0 - k v) as the theory initial condition "
                        "(pure e^{-Gamma t}, no ballistic transient) instead of "
                        "the fluid shifted-Maxwellian IC")
    p.add_argument("--out", default=None,
                   help="theory mode: output PNG path for the --T figure "
                        "(default: ion_sound_theory_dn.png)")
    # --- model-mode parameters ---
    p.add_argument("--species", nargs="+", default=None,
                   help="model mode: sorts to plot (default: every density diagnostic)")
    p.add_argument("--dpi", type=int, default=120, help="model mode: figure DPI")
    p.add_argument("--out-subdir", default="ion_sound_model",
                   help="model mode: subdir under <out_dir>/processed for the output")
    p.add_argument("--anim-fps", type=int, default=10, help="model mode: mp4 fps")
    p.add_argument("--anim-name", default="ion_sound_model.mp4",
                   help="model mode: animation filename")
    return p


def main():
    p = build_parser()
    args = p.parse_args()

    if args.model_electric is not None:
        run_model_electric(args)
        return

    if args.model is not None:
        run_model(args)
        return

    if args.E0 is not None and args.dn is not None:
        p.error("--E0 and --dn are mutually exclusive.")
    run_theory(args)


if __name__ == "__main__":
    main()
