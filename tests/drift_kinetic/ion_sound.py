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
  * KineticIonSoundMomentsQuiet samples this non-relativistic Gaussian directly
    in velocity space.  Older momentum-space Maxwell loaders instead convert
    v = p / sqrt(m^2 + p^2), producing an O(v_T^2/c^2) model mismatch.
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
import re
import sys

import numpy as np


# NumPy >= 2.0 renamed trapz -> trapezoid; keep both working.
_trapz = getattr(np, "trapezoid", None) or np.trapz

MEC2_KEV = 510.99895000  # electron rest energy, keV

# SI constants, only for the optional physical conversion.
EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
M_E_SI = 9.1093837015e-31
C_SI = 299792458.0


# --------------------------------------------------------------------------- #
# Plasma dispersion function Z(zeta) = i sqrt(pi) w(zeta),  w = Faddeeva func. #
# --------------------------------------------------------------------------- #
def _cef_coefficients(n: int):
    """Weideman coefficients (a, L); cached, they depend only on `n`."""
    m = 2 * n
    m2 = 2 * m
    k = np.arange(-m + 1, m)
    ll = math.sqrt(n / math.sqrt(2.0))
    theta = k * math.pi / m
    t = ll * np.tan(theta / 2.0)
    f = np.exp(-t**2) * (ll**2 + t**2)
    f = np.append(0.0, f)
    a = np.fft.fft(np.fft.fftshift(f)).real / m2
    return np.flipud(a[1:n + 1]), ll


def _cef(x: np.ndarray, n: int = 64) -> np.ndarray:
    """Faddeeva function w(x) = exp(-x^2) erfc(-i x) for Im(x) >= 0.

    Weideman's rational approximation (SIAM J. Numer. Anal. 31 (1994) 1497),
    accurate to ~machine precision for moderate |x| with n = 64.
    """
    a, ll = _cef_coefficients(n)
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


def solve_dispersion(k, species, omega_guess, tol=1e-12, max_iter=200,
                     coupling=1.0):
    """Newton iteration for the complex ion-acoustic root.

    ``coupling`` multiplies the susceptibility sum eps-1.  Its default is the
    continuum Vlasov-Poisson dispersion.  The drift-kinetic S1/S2/Yee spatial
    discretization uses coupling=sinc(k*dz/2)^4.
    """
    omega = complex(omega_guess)
    for _ in range(max_iter):
        f = 1.0 + coupling * (epsilon(omega, k, species) - 1.0)
        fp = coupling * epsilon_prime(omega, k, species)
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
    Te = args.Te / args.mec2_kev
    Ti = args.Ti / args.mec2_kev

    electron = Species("electrons", args.ne, args.qe, args.me, Te)
    ion = Species("ions", args.ni, args.qi, args.mi, Ti)
    species = [electron, ion]

    k = args.k if args.k is not None else 2.0 * math.pi * args.mode / args.Lz
    if k <= 0.0:
        raise ValueError("k must be positive.")
    if args.grid_dz is not None and args.T is not None:
        raise ValueError(
            "--grid-dz cannot be combined with --T: the optional IVP plot "
            "uses the continuum Vlasov-Poisson operator, not the "
            "semi-discrete DK operator")

    shape_s1 = 1.0
    coupling = 1.0
    field_grid_phase = 0.0
    if args.grid_dz is not None:
        if args.grid_dz <= 0.0:
            raise ValueError("--grid-dz must be positive")
        half_cell_phase = 0.5 * k * args.grid_dz
        sinc = math.sin(half_cell_phase) / half_cell_phase \
            if half_cell_phase != 0.0 else 1.0
        shape_s1 = sinc**2
        coupling = shape_s1**2
        field_grid_phase = half_cell_phase

    # ---- Step 1: dispersion relation ------------------------------------- #
    omega_s_guess, gamma_guess = acoustic_initial_guess(k, electron, ion)
    omega = solve_dispersion(k, species,
                             omega_s_guess - 1j * gamma_guess,
                             coupling=coupling)
    resid = 1.0 + coupling * (epsilon(omega, k, species) - 1.0)
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
    E_grid = E0 / shape_s1

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
    if args.grid_dz is not None:
        print(f"  DK GRID: dz = {args.grid_dz:.6g}, S1 = {shape_s1:.12g}, "
              f"coupling S1^2 = {coupling:.12g}")
    if inverse:
        print(f"  MODE: inverse  (given C_n[{args.dn_species}] = {args.dn:.6e} n0"
              f"  ->  solved E_force = {E0:.6e} code)")
    else:
        print(f"  MODE: forward  (given E_force = {E0:.6e} code)")
    if args.grid_dz is not None:
        print(f"  Yee grid amplitude E_grid = E_force/S1 = {E_grid:.6e}")
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
    residual_name = "eps_h" if args.grid_dz is not None else "eps_L"
    print(f"    |{residual_name}(omega)| residual = {abs(resid):.3e}")
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

    print("=" * 70)
    print("COPY-READY CODE CONSTANTS")
    print("=" * 70)
    print(f"  omega_real = {omega_r:.16e}")
    print(f"  gamma      = {Gamma:.16e}")
    print(f"  E_force    = {E0:.16e}")
    if args.grid_dz is not None:
        print(f"  E_grid     = {E_grid:.16e}")
        print(f"  grid phase = {field_grid_phase:.16e}  # +k*dz/2 for SetCosineField")
    for s in species:
        C_n, phi_n = density_sine(*dens[s.name])
        print(f"  {s.name:9s}: a_n = {C_n:.16e}, phi_n = {phi_n:.16e}")
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
    """Build species using the parallel temperature (Tz, or legacy T)."""
    electron = ion = None
    for s in config.get("Particles", []):
        q = float(s.get("q", 0.0))
        temperature_kev = float(s["Tz"] if "Tz" in s else s.get("T", 0.0))
        sp = Species(s.get("sort_name", ""), float(s.get("n", 1.0)), q,
                     float(s["m"]), temperature_kev / MEC2_KEV)
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
    from the momentum loader (velocity_z / phase_z).  For
    KineticIonSoundMomentsQuiet, the flux harmonic is derived directly from
    M1=omega*M0/k. Handles two layouts:
      * a standalone SetParticles preset (particles == name);
      * a paired loader, where this sort is the `paired_with` target of another
        preset -- it then shares that preset's `coordinate` and takes its
        `momentum_paired` block. Missing keys -> 0."""
    def z_of(block, key):
        vec = block.get(key)
        return float(vec[2]) if vec is not None and len(vec) >= 3 else 0.0

    def unpack(coord, mom):
        a_n = z_of(coord, "amplitude")
        phi_n = z_of(coord, "phase")
        if mom.get("name") != "KineticIonSoundMomentsQuiet":
            return (a_n, phi_n,
                    z_of(mom, "velocity"), z_of(mom, "phase"))

        mode = z_of(mom, "wave_number")
        Lz = float(config.get("Geometry", {}).get("z", 0.0))
        if mode == 0.0 or Lz <= 0.0:
            raise ValueError(
                "KineticIonSoundMomentsQuiet requires non-zero z mode and Lz")
        k = 2.0 * math.pi * mode / Lz
        omega = complex(float(mom["omega_real"]), -float(mom["gamma"]))
        density_hat = -1j * a_n * np.exp(1j * phi_n)
        velocity_hat = omega * density_hat / k
        return (a_n, phi_n, abs(velocity_hat),
                float(np.angle(1j * velocity_hat)))

    preset = preset_for_species(config, name)
    if preset is not None:
        coord, mom = preset.get("coordinate", {}), preset.get("momentum", {})
        return unpack(coord, mom)

    for pr in config.get("Presets", []):
        if pr.get("command") == "SetParticles" and pr.get("paired_with") == name:
            coord = pr.get("coordinate", {})            # shared coordinate
            mom = pr.get("momentum_paired", {})         # paired momentum
            return unpack(coord, mom)
    return 0.0, 0.0, 0.0, 0.0


def field_amplitude_from_config(config):
    """Physical E harmonic used by a kinetic loader, or its grid fallback."""
    for preset in config.get("Presets", []):
        if preset.get("command") != "SetParticles":
            continue
        momentum = preset.get("momentum", {})
        if momentum.get("name") == "KineticIonSoundMomentsQuiet" and \
                "force_electric_amplitude" in momentum:
            return float(momentum["force_electric_amplitude"])

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


def kinetic_loader_for_species(config, name):
    """Return a supported kinetic ion-sound momentum block, or None."""
    preset = preset_for_species(config, name)
    if preset is None:
        return None
    momentum = preset.get("momentum", {})
    if momentum.get("name") != "KineticIonSoundMomentsQuiet":
        return None
    return momentum


def distribution_diagnostic_for_species(config, name):
    """Return the 5-D drift-kinetic distribution diagnostic for a sort."""
    for diagnostic in config.get("Diagnostics", []):
        if diagnostic.get("diagnostic") == "DkDistributionFunction" and \
                diagnostic.get("particles") == name:
            return diagnostic
    return None


def density_diagnostic_for_species(config, name):
    """Return the scalar density diagnostic for a sort, or None."""
    for diagnostic in config.get("Diagnostics", []):
        if diagnostic.get("diagnostic") == "DistributionMoment" and \
                diagnostic.get("particles") == name and \
                diagnostic.get("moment") == "density":
            return diagnostic
    return None


def diagnostic_frame_path(const, config_dir, subdir, frame):
    """Find a diagnostic frame in the configured input or run directory."""
    filename = f"{frame:04d}"
    candidates = [os.path.join(const.in_dir, subdir, filename),
                  os.path.join(config_dir, subdir, filename)]
    return next((path for path in candidates if os.path.isfile(path)), None)


def load_kinetic_distribution_ic(config, const, config_dir, species, k,
                                 frame=0):
    """Read a realized initial condition from a 5-D distribution dump.

    The DkDistributionFunction histogram is integrated over mu_p and averaged
    over x,y.  Its velocity distribution in every z cell is treated as a
    conditional PDF and multiplied by the separately deposited density moment.
    This retains the realized velocity-space loading while making its zeroth
    moment identical to the density diagnostic used in the comparison plots.

    Returns a mapping accepted by solve_vlasov_poisson().  An empty mapping
    means that this config has no DkDistributionFunction diagnostics.  The
    momentum loader is deliberately irrelevant here: the dump is the realized
    initial condition, regardless of how the particles were generated.
    """
    result = {}
    diagnosed_species = [
        (s, distribution_diagnostic_for_species(config, s.name))
        for s in species
    ]
    diagnosed_species = [(s, diagnostic)
                          for s, diagnostic in diagnosed_species
                          if diagnostic is not None]
    if not diagnosed_species:
        return result

    z = (np.arange(const.Nz) + 0.5) * const.dz
    projector = np.exp(-1j * k * z)
    frame_times = []
    for s, diagnostic in diagnosed_species:
        vinfo = diagnostic["v_parallel"]
        muinfo = diagnostic["mu_p"]
        nv, nmu = int(vinfo["bins"]), int(muinfo["bins"])
        vmin, vmax = float(vinfo["min"]), float(vinfo["max"])
        mumin, mumax = float(muinfo["min"]), float(muinfo["max"])
        dv = (vmax - vmin) / nv
        dmu = (mumax - mumin) / nmu
        velocity = vmin + (np.arange(nv) + 0.5) * dv

        subdir = diagnostic.get(
            "out_dir", os.path.join(s.name, "distribution_function"))
        phase_path = diagnostic_frame_path(
            const, config_dir, subdir, frame)
        if phase_path is None:
            raise SystemExit(
                f"DkDistributionFunction frame {frame:04d} not found for "
                f"species '{s.name}' in '{subdir}'.")
        expected_size = const.Nz * const.Ny * const.Nx * nv * nmu
        if os.path.getsize(phase_path) != expected_size * 4:
            raise SystemExit(
                f"unexpected size of {phase_path}: "
                f"expected {expected_size * 4} bytes")

        # mmap avoids holding the ~100 MB electron frame and its float64 copy
        # simultaneously.  The reduced (Nz,Nv) marginal is small.
        phase_data = np.memmap(
            phase_path, dtype=np.float32, mode="r",
            shape=(const.Nz, const.Ny, const.Nx, nv, nmu))
        f_parallel = phase_data.sum(axis=4, dtype=np.float64).mean(
            axis=(1, 2)) * dmu
        del phase_data

        phase_density = np.sum(f_parallel, axis=1) * dv
        if np.any(phase_density <= 0.0):
            raise SystemExit(
                f"zero phase-space density in frame {frame:04d} for '{s.name}'")

        density_diagnostic = density_diagnostic_for_species(config, s.name)
        density_source = "phase-space marginal"
        if density_diagnostic is not None:
            density_subdir = density_diagnostic.get(
                "out_dir", os.path.join(s.name, "density"))
            density_path = diagnostic_frame_path(
                const, config_dir, density_subdir, frame)
            expected_density_size = const.Nz * const.Ny * const.Nx
            if density_path is not None and \
                    os.path.getsize(density_path) == expected_density_size * 4:
                density = np.fromfile(
                    density_path, dtype=np.float32,
                    count=expected_density_size).reshape(
                        const.Nz, const.Ny, const.Nx).mean(axis=(1, 2))
                f_parallel *= (density / phase_density)[:, None]
                density_source = density_path

        # The z average is the equilibrium actually represented by the PIC
        # markers in this dump.  Keeping it alongside the first harmonic lets
        # --model-pic linearize about the realized (cut off, regularized and
        # finite-marker) background instead of silently replacing it by the
        # analytic Maxwellian configured for the run.
        equilibrium = np.mean(f_parallel, axis=0)
        f_hat = 2.0 * np.mean(f_parallel * projector[:, None], axis=0)
        weights = np.full(nv, dv)
        density_hat = complex(np.dot(f_hat, weights))
        velocity_hat = complex(np.dot(velocity * f_hat, weights) / s.n)
        period = float(diagnostic.get(
            "diagnose_period", config.get("Geometry", {}).get(
                "diagnose_period", 1.0)))
        frame_times.append(frame * period)
        result[s.name] = {
            "v": velocity,
            "weights": weights,
            "f_hat": f_hat,
            "equilibrium": equilibrium,
            "density_hat": density_hat,
            "velocity_hat": velocity_hat,
            "phase_path": phase_path,
            "density_source": density_source,
            "frame": frame,
            "time": frame * period,
        }

    if frame_times and not np.allclose(frame_times, frame_times[0]):
        raise SystemExit(
            "kinetic distribution diagnostics use different frame times")
    return result


def solve_vlasov_poisson(species, cn_hat, u_hat, k, t_max, n_record=400,
                         exact_ic=False, omega0=None, E0=0.0,
                         initial_distribution=None, initial_field=None,
                         use_realized_equilibrium=False):
    """Exact linearized Vlasov-Poisson evolution of one Fourier mode e^{ikz}.

    Advances the first-harmonic perturbed distribution of every species,

        d/dt f_s(v,t) = -i k v f_s - (q_s/m_s) E(t) dF0s/dv,
        i k E(t) = sum_s q_s int f_s dv                       (Poisson, eps0=1).

    If `initial_field` is supplied, E is instead an independent state obeying

        dE/dt = -sum_s q_s int v f_s dv                       (Ampere),

    initialized from that saved complex field harmonic.  The Ampere form is
    equivalent when Gauss' law is exact and, unlike algebraic Poisson, retains
    the small realized Gauss-law mismatch of a finite-particle PIC dump.

    For the loaded initial condition, if `initial_distribution` contains a
    species, its saved velocity-bin centres, quadrature weights, and complex
    first spatial harmonic are used directly.  Otherwise, with
    `exact_ic=False` (default), the fluid (shifted-Maxwellian) IC is used,

        f_s(v,0) = cn_hat_s F0s(v) - u_hat_s dF0s/dv,

    matching only the loaded density (cn_hat_s) and bulk-velocity (u_hat_s)
    moments. With `exact_ic=True` a formal pole-shaped response is loaded,

        f_s(v,0) = -i (q_s E0 / m_s) dF0s/dv / (omega0* - k v).

    This regular real-axis response is useful diagnostically, but is not a
    true damped Vlasov eigenfunction and does not in general satisfy Gauss'
    law with E0.  The damped Landau root is a quasimode pole of the analytically
    continued response, not a discrete eigenvalue of the real-v operator.

    With `use_realized_equilibrium=True`, the force term is linearized about
    the z-averaged distribution saved in the same PIC dump as the initial
    harmonic.  The default keeps the historical analytic-Maxwellian model.

    RK4 on per-species velocity grids; captures the collective (Landau) mode AND
    the ballistic phase-mixing exactly and self-consistently.

    Returns (t_rec, nhat, ehat), sampled at `n_record` uniform times in
    [0, t_max].
    """
    use_ampere = initial_field is not None
    wp_tot = math.sqrt(sum(s.wp ** 2 for s in species))
    v_max = 7.0 * max(s.vT for s in species)
    # RK4 stability: resolve the plasma oscillation and the advection; also keep
    # enough steps to resolve the wave over the whole window.
    plasma_step = (0.25 if use_ampere else 0.9) / wp_tot
    dt = min(plasma_step, 0.5 / (k * v_max), t_max / (4 * n_record))
    n_steps = max(int(math.ceil(t_max / dt)), 4 * n_record)
    dt = t_max / n_steps

    initial_distribution = initial_distribution or {}
    grids, weights, dF0, f = {}, {}, {}, {}
    for s in species:
        saved = initial_distribution.get(s.name)
        if saved is not None:
            v = np.asarray(saved["v"], dtype=float)
            quadrature = np.asarray(saved["weights"], dtype=float)
            f_initial = np.asarray(saved["f_hat"], dtype=complex)
            if v.ndim != 1 or v.size < 2 or quadrature.shape != v.shape or \
                    f_initial.shape != v.shape:
                raise ValueError(
                    f"invalid saved distribution for '{s.name}'")
            if not np.all(np.diff(v) > 0.0) or np.any(quadrature <= 0.0):
                raise ValueError(
                    f"non-monotone velocity grid for '{s.name}'")
            equilibrium = saved.get("equilibrium") \
                if use_realized_equilibrium else None
            if equilibrium is None:
                F0 = s.n / math.sqrt(2.0 * math.pi) / s.vT * \
                    np.exp(-v ** 2 / (2.0 * s.vT ** 2))
            else:
                F0 = np.asarray(equilibrium, dtype=float)
                if F0.shape != v.shape or np.any(~np.isfinite(F0)) or \
                        np.any(F0 < 0.0):
                    raise ValueError(
                        f"invalid saved equilibrium for '{s.name}'")
            grids[s.name] = v
            weights[s.name] = quadrature
            dF0[s.name] = np.gradient(F0, v, edge_order=2)
            f[s.name] = f_initial.copy()
            continue

        # Velocity resolution set so the recurrence time 2*pi/(k dv) > 2 t_max.
        nv = int(np.clip(14.0 * s.vT * k * 2.0 * t_max / (2.0 * math.pi),
                         1500, 8000))
        # The pole-shaped IC has a sharp resonant feature of width Gamma/k near
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
        quadrature = np.full(v.size, v[1] - v[0])
        quadrature[[0, -1]] *= 0.5
        weights[s.name] = quadrature
        dF0[s.name] = -v / s.vT ** 2 * F0
        if exact_ic:
            # Regular upper-pole response.  It is a finite-time quasimode IC,
            # not an all-time damped eigenfunction on the real velocity axis.
            f[s.name] = -1j * (s.q * E0 / s.m) * dF0[s.name] \
                / (np.conj(omega0) - k * v)
        else:
            f[s.name] = cn_hat[s.name] * F0 - u_hat[s.name] * dF0[s.name]

    def poisson_field(state):
        rho = sum(s.q * np.dot(state[s.name], weights[s.name])
                  for s in species)
        return rho / (1j * k)

    def rhs(state, field=None):
        E = field if use_ampere else poisson_field(state)
        df = {s.name: -1j * k * grids[s.name] * state[s.name]
              - (s.q / s.m) * E * dF0[s.name] for s in species}
        if not use_ampere:
            return df, None
        dE = -sum(s.q * np.dot(grids[s.name] * state[s.name],
                               weights[s.name]) for s in species)
        return df, dE

    stride = max(1, n_steps // n_record)
    field = complex(initial_field) if use_ampere else poisson_field(f)
    t_rec = [0.0]
    nhat = {s.name: [complex(np.dot(f[s.name], weights[s.name]))]
            for s in species}
    ehat = [field]
    for step in range(1, n_steps + 1):
        k1f, k1e = rhs(f, field)
        state2 = {n: f[n] + 0.5 * dt * k1f[n] for n in f}
        field2 = field + 0.5 * dt * k1e if use_ampere else None
        k2f, k2e = rhs(state2, field2)
        state3 = {n: f[n] + 0.5 * dt * k2f[n] for n in f}
        field3 = field + 0.5 * dt * k2e if use_ampere else None
        k3f, k3e = rhs(state3, field3)
        state4 = {n: f[n] + dt * k3f[n] for n in f}
        field4 = field + dt * k3e if use_ampere else None
        k4f, k4e = rhs(state4, field4)
        f = {n: f[n] + dt / 6.0 *
             (k1f[n] + 2 * k2f[n] + 2 * k3f[n] + k4f[n]) for n in f}
        if use_ampere:
            field += dt / 6.0 * (k1e + 2 * k2e + 2 * k3e + k4e)
        else:
            field = poisson_field(f)
        if step % stride == 0 or step == n_steps:
            t_rec.append(step * dt)
            for s in species:
                nhat[s.name].append(complex(
                    np.dot(f[s.name], weights[s.name])))
            ehat.append(field)
    return (np.array(t_rec), {n: np.array(v) for n, v in nhat.items()},
            np.asarray(ehat))


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
    t, nhat, _ = solve_vlasov_poisson(
        species, cn_hat, u_hat, k, t_max,
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
    for d in config.get("Diagnostics", []) or []:
        if str(d.get("diagnostic", "")).startswith("FieldView") \
                and d.get("field") == field:
            return d.get("out_dir")
    return None


def measured_perturbation(const, species, k, frame, config_dir):
    """Measure the realized first-harmonic IC (cn_hat, u_hat) from the dumps.

    Returns (cn_hat, u_hat, t0) with the same conventions `prepare_theory` uses
    for the config-derived amplitudes: cn_hat_s is the RELATIVE density
    amplitude 2/Lz int (n_s/n_s - 1) e^{-ikz} dz, and u_hat_s the ABSOLUTE
    parallel-velocity amplitude 2/Lz int J_{z,s}/(q_s n_s) e^{-ikz} dz [c].

    Both moments are read from the SAME frame `frame`, and t0 = frame*dts is
    returned so the caller can start the theory there instead of at t = 0. The
    current is deposited over a step, so frame 0000 holds J == 0; the default
    caller passes frame 1. Starting the theory at t0 (rather than back-rotating
    to 0) keeps the IC exact -- no assumption about what happened before t0.

    The grid shape factor cancels: n_hat and J_hat carry the same deposition
    kernel as the frames we later compare against, and the evolution is linear.
    Raises SystemExit if a species has no density or no J frames."""
    Nx, Ny, Nz = const.Nx, const.Ny, const.Nz
    z = (np.arange(Nz) + 0.5) * const.dz
    kernel = np.exp(-1j * k * z)

    def read(sub, ncomp):
        for base in (const.in_dir, config_dir):
            path = os.path.join(base, sub, f"{frame:04d}")
            if not os.path.isfile(path):
                continue
            if os.path.getsize(path) != Nx * Ny * Nz * ncomp * 4:
                continue
            raw = np.fromfile(path, dtype=np.float32,
                              count=Nx * Ny * Nz * ncomp)
            return raw.reshape(Nz, Ny, Nx, ncomp) if ncomp > 1 \
                else raw.reshape(Nz, Ny, Nx)
        return None

    cn_hat, u_hat = {}, {}
    for s in species:
        dens = read(os.path.join(s.name, "density"), 1)
        if dens is None:
            raise SystemExit(
                f"--ic-from-dump: no 3D density frame {frame:04d} for '{s.name}'.")
        cur = read(os.path.join(s.name, "J"), 3)
        if cur is None:
            raise SystemExit(
                f"--ic-from-dump: no 3D '{s.name}/J' FieldView frame {frame:04d}; "
                "add that diagnostic to the config or drop --ic-from-dump.")
        dn = dens.mean(axis=(1, 2)) / s.n - 1.0
        jz = cur[..., 2].mean(axis=(1, 2))
        cn_hat[s.name] = complex(2.0 * np.mean(dn * kernel))
        u_hat[s.name] = complex(2.0 * np.mean(jz * kernel) / (s.q * s.n))
    return cn_hat, u_hat, frame * const.dts


def prepare_theory(testname, ic_frame=None):
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

    configured_moment_loader = None
    for s in species:
        candidate = kinetic_loader_for_species(config, s.name)
        if (candidate or {}).get("name") == "KineticIonSoundMomentsQuiet":
            configured_moment_loader = candidate
            break
    if configured_moment_loader is not None:
        omega0 = complex(float(configured_moment_loader["omega_real"]),
                         -float(configured_moment_loader["gamma"]))
    else:
        omega_s_guess, gamma_guess = acoustic_initial_guess(k, electron, ion)
        omega0 = solve_dispersion(
            k, species, omega_s_guess - 1j * gamma_guess)

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

    # A DkDistributionFunction dump stores the complete realized 5-D
    # distribution, independently of the particle loader.  Use its
    # velocity-space first harmonic rather than reducing it to density and bulk
    # velocity and inventing a shifted Maxwellian.  Frame zero is the
    # post-loader, pre-timestep state.  An explicit --ic-from-dump FRAME starts
    # from that later distribution when the corresponding 5-D frame exists.
    t0 = 0.0
    ic_measured = None
    kinetic_ic_frame = 0 if ic_frame is None else ic_frame
    initial_distribution = load_kinetic_distribution_ic(
        config, const, config_dir, species, k, kinetic_ic_frame)
    if initial_distribution:
        t0 = next(iter(initial_distribution.values()))["time"]
        ic_measured = kinetic_ic_frame
        for s in species:
            saved = initial_distribution.get(s.name)
            if saved is None:
                continue
            cn_hat[s.name] = saved["density_hat"] / s.n
            u_hat[s.name] = saved["velocity_hat"]
    elif ic_frame is not None:
        # Moment-only fallback for older MaxwellShiftedSine tests which have no
        # DkDistributionFunction initial condition.
        cn_hat, u_hat, t0 = measured_perturbation(const, species, k, ic_frame,
                                                  config_dir)
        ic_measured = ic_frame

    return dict(const=const, config=config, config_path=config_path,
                config_dir=config_dir, species=species, electron=electron,
                ion=ion, wn=wn, k=k, omega0=omega0, cn_hat=cn_hat, u_hat=u_hat,
                E0=E0, ic=ic, t0=t0, ic_measured=ic_measured,
                initial_distribution=initial_distribution,
                T_wave=2.0 * math.pi / omega0.real)


def plot_model_noise(times, series_amp, series_noise, series_harm, series_rest,
                     style, t_grid, T_wave, dn_theory_i, dn_exponential_i,
                     t_max, out_path, dpi, labelsize, ticksize, bbox):
    """Separate figure: signal vs noise (left) and mode coupling (right).

    The left-over signal is what the diagnostic profile keeps after the first
    z-harmonic reconstructed from `dn_1` is removed,

        delta n_noise(z, t) = delta n(z, t) - Re[dn_1(t) e^{i k z}],

    and it is measured with the same normalisation as a harmonic amplitude,
    i.e. sqrt(2 <delta n_noise^2>_z), so that a pure cos(k' z) mode of amplitude
    A shows up as A.

    LEFT panel: everything on a linear 0-100 % scale, each curve as a percentage
    of its own initial amplitude - the simulated harmonic and the noise of a
    sort are referred to |dn_1(t_0)| of that sort, the two theory curves to the
    theoretical |dn_{i,1}(t_0)|.  Both sorts therefore start at 100 %, and a
    noise curve reads directly "how many percent of the initial signal".

    RIGHT panel: the same residual split into the harmonics m*k that mode
    coupling would fill and the incoherent leftover, now as a percentage of the
    INSTANTANEOUS |dn_1(t)|.  This is the linearity test: quadratic coupling
    gives |dn_2|/|dn_1| ~ O(1) |dn_1|/n, so while that ratio stays under the
    ~10 % guide line the harmonics do not feed back on the studied mode and the
    run may be treated as linear.  Discrete-particle noise, in contrast, sits in
    the "rest" curve, is flat in absolute value and therefore only grows here
    because |dn_1(t)| decays - a large "rest" limits measurability, not
    linearity.
    """
    import matplotlib.pyplot as plt

    def percent(values, reference):
        arr = np.asarray(values, dtype=float)
        ref = np.asarray(reference, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = 100.0 * arr / ref
        return np.where(np.isfinite(out) & (ref > 0.0), out, np.nan)

    theory_ref = float(dn_theory_i[0]) if len(dn_theory_i) else float("nan")

    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(17.0, 7.5))

    # ---- Left: first harmonic and the whole residual, vs the initial value -- #
    ax_abs.plot(t_grid / T_wave, percent(dn_theory_i, theory_ref), color="black",
                linewidth=2.0, linestyle="-", alpha=0.9,
                label=r"$|\delta n_{i,1}|$")
    ax_abs.plot(t_grid / T_wave, percent(dn_exponential_i, theory_ref),
                color="black", linewidth=1.8, linestyle="--", alpha=0.75,
                label=r"$|\delta n_{i,1}(0)|e^{-\Gamma t}$")

    # Per-sort reference: the first simulated first-harmonic amplitude.
    refs = {}
    for name, amp in series_amp.items():
        finite = [a for a in amp if np.isfinite(a) and a > 0.0]
        refs[name] = float(finite[0]) if finite else float("nan")

    for name, amp in series_amp.items():
        st = dict(style.get(name, {"marker": "o", "linestyle": "-"}))
        st.pop("label", None)
        count = min(len(times), len(amp))
        ax_abs.plot(times[:count], percent(amp[:count], refs[name]),
                    label=rf"$|\delta n_{{{name[0]},1}}|$", **st)

    for name, noise in series_noise.items():
        st = dict(style.get(name, {"marker": "o", "linestyle": "-"}))
        st.pop("label", None)
        st["linestyle"] = ":"
        st["alpha"] = 0.8
        st["linewidth"] = 1.5
        st["markersize"] = 3.0
        count = min(len(times), len(noise))
        ax_abs.plot(times[:count],
                    percent(noise[:count], refs.get(name, float("nan"))),
                    label=rf"$|\delta n_{{{name[0]}}} - \delta n_{{{name[0]},1}}|$",
                    **st)

    ax_abs.set_xlim(0.0, t_max / T_wave)
    ax_abs.set_ylim(0.0, 100.0)
    ax_abs.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_abs.set_ylabel(r"% от начальной амплитуды $|\delta n_1(t_0)|$",
                      fontsize=labelsize)
    ax_abs.tick_params(labelsize=ticksize)
    ax_abs.grid(True, alpha=0.3)
    ax_abs.set_title(r"Первая гармоника и амплитуда шума", fontsize=labelsize,
                     bbox=bbox)
    ax_abs.set_box_aspect(1)
    ax_abs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=2, fontsize=ticksize)

    # ---- Right: coupled harmonics vs incoherent rest, vs |dn_1(t)| ---------- #
    linestyle_of = {2: "-", 3: "--"}
    for name, harmonics in series_harm.items():
        amp = np.asarray(series_amp.get(name, []), dtype=float)
        base = dict(style.get(name, {"marker": "o", "linestyle": "-"}))
        base.pop("label", None)
        base.pop("linestyle", None)
        for m in sorted(harmonics):
            st = dict(base)
            st["linestyle"] = linestyle_of.get(m, "-.")
            st["linewidth"] = 2.0
            st["markersize"] = 3.5
            st["alpha"] = 1.0 if m == 2 else 0.7
            count = min(len(times), len(harmonics[m]), amp.size)
            ax_rel.plot(times[:count],
                        percent(harmonics[m][:count], amp[:count]),
                        label=rf"$|\delta n_{{{name[0]},{m}}}|/"
                              rf"|\delta n_{{{name[0]},1}}|$", **st)

        rest = series_rest.get(name, [])
        st = dict(base)
        st["linestyle"] = ":"
        st["linewidth"] = 1.5
        st["markersize"] = 3.0
        st["alpha"] = 0.8
        count = min(len(times), len(rest), amp.size)
        ax_rel.plot(times[:count], percent(rest[:count], amp[:count]),
                    label=rf"шум$_{{{name[0]}}}/|\delta n_{{{name[0]},1}}|$", **st)

    ax_rel.axhline(10.0, color="tab:gray", linewidth=1.2, linestyle="--",
                   alpha=0.9)
    ax_rel.text(0.01 * t_max / T_wave, 11.0, "10 % — граница линейного режима",
                color="tab:gray", fontsize=ticksize, ha="left", va="bottom")

    ax_rel.set_xlim(0.0, t_max / T_wave)
    ax_rel.set_ylim(0.0, 100.0)
    ax_rel.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_rel.set_ylabel(r"% от текущей амплитуды $|\delta n_1(t)|$",
                      fontsize=labelsize)
    ax_rel.tick_params(labelsize=ticksize)
    ax_rel.grid(True, alpha=0.3)
    ax_rel.set_title(r"Связь гармоник и шум относительно моды",
                     fontsize=labelsize, bbox=bbox)
    ax_rel.set_box_aspect(1)
    ax_rel.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=2, fontsize=ticksize)

    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Noise figure written to {out_path}")


def run_model(args):
    ctx = prepare_theory(args.model, ic_frame=args.ic_from_dump)
    const, config = ctx["const"], ctx["config"]
    species, electron, ion = ctx["species"], ctx["electron"], ctx["ion"]
    k, omega0, wn = ctx["k"], ctx["omega0"], ctx["wn"]
    cn_hat, u_hat, E0, ic = ctx["cn_hat"], ctx["u_hat"], ctx["E0"], ctx["ic"]
    initial_distribution = ctx["initial_distribution"]
    t0 = ctx["t0"]
    omega_r, Gamma, T_wave = omega0.real, -omega0.imag, ctx["T_wave"]
    config_path, config_dir = ctx["config_path"], ctx["config_dir"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from lib.plot import bbox, labelsize, ticksize
    dz = _DensityZTools

    # lib.plot enables external LaTeX globally via lib.plot_utils.  The model
    # figures only need Matplotlib's built-in mathtext, and must also work on
    # systems where the `latex` executable is not installed.
    plt.rc("text", usetex=False)

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
    if args.model_tmax is not None:
        if args.model_tmax <= 0.0:
            raise SystemExit("--model-tmax must be positive.")
        t_limit = args.model_tmax * T_wave
        tolerance = 1.0e-12 * max(1.0, t_limit)
        common = [idx for idx in common if idx * const.dts <= t_limit + tolerance]
    if not common:
        raise SystemExit("No timesteps common to all requested density diagnostics "
                         "inside the requested --model-tmax interval.")

    names_per_row = []
    for row in rows:
        name_by_idx = dict(row["timesteps"])
        names_per_row.append([name_by_idx[idx] for idx in common])

    n0 = 1.0
    z = (np.arange(const.Nz) + 0.5) * const.dz
    first_harmonic_kernel = np.exp(-1j * k * z)
    # Kernels of the harmonics that quadratic/cubic mode coupling would fill.
    harmonic_kernels = {m: np.exp(-1j * m * k * z) for m in (2, 3)}
    times = np.array([idx * const.dts for idx in common], dtype=float)
    t_max = float(times[-1]) if times.size else T_wave

    def profile_ext(data):
        return data.mean(axis=(1, 2))

    # ---- Exact linearized Vlasov-Poisson theory over the run window -------- #
    # With --ic-from-dump the IC is the state at t0, so the theory clock starts
    # there and its time axis is shifted by t0 for plotting.
    t_grid, nhat, _ = solve_vlasov_poisson(
        species, cn_hat, u_hat, k,
        max(t_max - t0, T_wave * 1e-3),
        exact_ic=args.exact_ic, omega0=omega0, E0=E0,
        initial_distribution=initial_distribution)
    t_grid = t_grid + t0
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
        if initial_distribution.get(s.name) is not None:
            saved = initial_distribution[s.name]
            cn, uh = cn_hat[s.name], u_hat[s.name]
            def phase_of(c):
                return (np.angle(c) + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
            print(f"  {s.name:9s}: a_n = {a_n:.6e}, phi_n = {phi_n:+.6f}"
                  "   (config density)")
            print(f"  {'':9s}  a_n = {abs(cn):.6e}, phi_n = {phase_of(cn):+.6f} ,"
                  f"  C_u = {abs(uh):.6e}, phi_u = {phase_of(uh):+.6f}"
                  f"   (distribution frame {saved['frame']:04d})")
        else:
            print(f"  {s.name:9s}: a_n = {a_n:.6e}, phi_n = {phi_n:+.6f} ,"
                  f"  C_u = {C_u:.6e}, phi_u = {phi_u:+.6f}"
                  "   (config, requested)")
        if initial_distribution.get(s.name) is None and \
                ctx["ic_measured"] is not None:
            cn, uh = cn_hat[s.name], u_hat[s.name]
            # Inverse of cn_hat = -i a exp(i phi):  a = |cn_hat|, phi = arg + pi/2.
            def phase_of(c):
                return (np.angle(c) + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
            print(f"  {'':9s}  a_n = {abs(cn):.6e}, phi_n = {phase_of(cn):+.6f} ,"
                  f"  C_u = {abs(uh):.6e}, phi_u = {phase_of(uh):+.6f}"
                  f"   (frame {ctx['ic_measured']:04d}, realized)")
        print(f"             theory dn(t0) = {theory[s.name][0]:.6e}")
    if ctx["ic_measured"] is not None:
        source = "distribution" if initial_distribution else "moment"
        print(f"  theory started from {source} frame {ctx['ic_measured']:04d} "
              f"(t0 = {t0:.6g} = {t0 / T_wave:.4g} T)")
    print(f"  frames = {len(common)} , t in [0, {t_max:.6g}] [1/w_pe]")
    if args.model_tmax is not None:
        print(f"  drawing limit = {args.model_tmax:.6g} T")
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

    # Right panel: the first z-harmonic of simulation dn(t) for every species,
    # plus the exact kinetic theory and its pure exponential Landau envelope.
    lines_amp = {}
    series_amp = {}
    series_noise = {}
    series_harm = {}
    series_rest = {}
    for row in rows:
        st = dict(style.get(row["species"], {"marker": "o", "linestyle": "-"}))
        st["label"] = rf"$|\delta n_{{{row['species'][0]},1}}|$ (model)"
        lines_amp[row["species"]] = ax_amp.plot([], [], **st)[0]
        series_amp[row["species"]] = []
        series_noise[row["species"]] = []
        series_harm[row["species"]] = {m: [] for m in harmonic_kernels}
        series_rest[row["species"]] = []

    dn_theory_i = theory[ion.name]
    dn_exponential_i = abs(cn_hat[ion.name]) * np.exp(-Gamma * (t_grid - t0))
    ax_amp.plot(t_grid / T_wave, dn_theory_i, color="black", linewidth=2.0,
                linestyle="-", alpha=0.9,
                label=r"$|\delta n_{i,1}|$ (theory, ballistic+collective)")
    ax_amp.plot(t_grid / T_wave, dn_exponential_i, color="black", linewidth=1.8,
                linestyle="--", alpha=0.75,
                label=r"$|\delta n_{i,1}(0)|e^{-\Gamma t}$")

    amp_hi_lim = max(0.06, 1.25 * float(
        max(np.nanmax(dn_theory_i), np.nanmax(dn_exponential_i))))
    ax_amp.set_xlim(0.0, t_max / T_wave)
    ax_amp.set_ylim(0.0, amp_hi_lim)
    ax_amp.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_amp.set_ylabel(
        r"$|\delta n_1(t)| = \left|\frac{2}{L_z}\int"
        r"\left(\langle n\rangle_{x,y}/n_0 - 1\right)e^{-ikz}dz\right|$",
        fontsize=labelsize)
    ax_amp.tick_params(labelsize=ticksize)
    ax_amp.grid(True, alpha=0.3)
    ax_amp.set_title(r"Амплитуда первой гармоники", fontsize=labelsize, bbox=bbox)
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
            dn1 = 2.0 * np.mean(dev * first_harmonic_kernel)
            series_amp[row["species"]].append(float(abs(dn1)))
            # Everything the profile keeps once the first harmonic is removed;
            # normalised as sqrt(2)*rms so a pure mode of amplitude A reads A.
            residual = dev - np.real(dn1 * np.conj(first_harmonic_kernel))
            series_noise[row["species"]].append(
                float(np.sqrt(2.0 * np.mean(residual ** 2))))
            # Split that residual into the coupled harmonics m*k, which grow as
            # |dn_1|^m, and the leftover incoherent part (discrete-particle
            # noise), which does not follow the wave at all.
            rest = residual
            for m, kernel in harmonic_kernels.items():
                dnm = 2.0 * np.mean(dev * kernel)
                series_harm[row["species"]][m].append(float(abs(dnm)))
                rest = rest - np.real(dnm * np.conj(kernel))
            series_rest[row["species"]].append(
                float(np.sqrt(2.0 * np.mean(rest ** 2))))
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

    plot_model_noise(times_acc, series_amp, series_noise, series_harm,
                     series_rest, style, t_grid, T_wave,
                     dn_theory_i, dn_exponential_i, t_max,
                     os.path.join(out_dir, "ion_sound_model_noise.png"),
                     args.dpi, labelsize, ticksize, bbox)


def compare_label(testname):
    """Legend label: everything after the drift_kinetic_ prefix.

    drift_kinetic_ringdown_ex12 -> ringdown_ex12.  A name without that prefix
    is used as it is, so runs from other suites stay distinguishable."""
    name = os.path.basename(str(testname).rstrip("/"))
    prefix = "drift_kinetic_"
    return name[len(prefix):] if name.startswith(prefix) and \
        len(name) > len(prefix) else name


def compare_label_with_particles(testname, config):
    """Particle-count-only legend label for comparison plots."""
    counts = {
        str(item.get("sort_name")): item.get("Np")
        for item in config.get("Particles", [])
        if isinstance(item, dict) and item.get("sort_name") is not None
        and item.get("Np") is not None
    }
    electron_count = counts.get("electrons")
    ion_count = counts.get("ions")

    def format_count(value):
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return str(value)

    if electron_count is not None and ion_count is not None:
        if electron_count == ion_count:
            return f"Np={format_count(electron_count)}"
        return (f"Np_e={format_count(electron_count)}, "
                f"Np_i={format_count(ion_count)}")
    if len(counts) == 1:
        return f"Np={format_count(next(iter(counts.values())))}"
    return "Np=?"


def harmonic_series(ctx, dz, sort_name, model_tmax=None):
    """Complex first z-harmonic of the relative density of one sort.

    Returns (times, a1, noise), all sampled on the diagnostic frames:

        times   absolute code time of every frame [1/omega_pe],
        a1(t) = 2/Lz int (<n_s>_{x,y}/n_s - 1) e^{-ikz} dz          (COMPLEX),
        noise   mean |m-th z-harmonic| over the m that carry no signal: every
                m in [1, Nz/2] except the loaded mode and its first two
                multiples (the second harmonic can be driven nonlinearly).
                Those harmonics vanish in linear theory, so their level is the
                discrete-particle floor that also contaminates a1.

    Unlike `load_ion_harmonic` this keeps the PHASE of the harmonic, which is
    what makes the two ion-acoustic branches separable (see `fit_two_branch`).
    """
    const, config_dir = ctx["const"], ctx["config_dir"]
    species = next(s for s in ctx["species"] if s.name == sort_name)

    rows = dz.collect_rows([sort_name])
    if not rows and const.in_dir != config_dir:
        const.in_dir = config_dir
        const.out_dir = os.path.join(config_dir, "processed")
        rows = dz.collect_rows([sort_name])
    if not rows:
        raise SystemExit(f"No '{sort_name}' density diagnostic found in "
                         f"'{os.path.basename(config_dir)}'.")

    row = rows[0]
    steps = row["timesteps"]
    if model_tmax is not None:
        t_limit = model_tmax * ctx["T_wave"]
        tolerance = 1.0e-12 * max(1.0, t_limit)
        steps = [(idx, name) for idx, name in steps
                 if idx * const.dts <= t_limit + tolerance]
    if not steps:
        raise SystemExit(f"No '{sort_name}' density frames in the requested "
                         "--model-tmax interval.")

    z = (np.arange(const.Nz) + 0.5) * const.dz
    kernel = np.exp(-1j * ctx["k"] * z)
    wn = max(int(round(ctx["wn"])), 1)
    noise_modes = [m for m in range(1, const.Nz // 2 + 1)
                   if m % wn != 0 or m // wn > 2]

    times, harmonic, noise = [], [], []
    for idx, name in steps:
        data = dz.load_frame(row["dir"], name)
        if data is None:
            continue
        dn_relative = data.mean(axis=(1, 2)) / species.n - 1.0
        times.append(idx * const.dts)
        harmonic.append(complex(2.0 * np.mean(dn_relative * kernel)))
        spectrum = np.abs(np.fft.rfft(dn_relative)) * 2.0 / const.Nz
        noise.append(float(np.mean(spectrum[noise_modes])) if noise_modes
                     else math.nan)
    if not times:
        raise SystemExit(f"Could not read '{sort_name}' density frames.")
    return np.asarray(times), np.asarray(harmonic), np.asarray(noise)


def density_comparison_series(ctx, dz, sort_name, model_tmax=None):
    """Ion-density amplitudes used by multi-run model comparisons.

    All amplitudes use the convention in ``plot_model_noise``: sqrt(2)*rms
    for a real-space profile, so a pure sinusoid of amplitude A reads A.
    """
    const = ctx["const"]
    species = next(s for s in ctx["species"] if s.name == sort_name)
    rows = dz.collect_rows([sort_name])
    if not rows and const.in_dir != ctx["config_dir"]:
        const.in_dir = ctx["config_dir"]
        const.out_dir = os.path.join(ctx["config_dir"], "processed")
        rows = dz.collect_rows([sort_name])
    if not rows:
        raise SystemExit(f"No '{sort_name}' density diagnostic found.")
    steps = rows[0]["timesteps"]
    if model_tmax is not None:
        limit = model_tmax * ctx["T_wave"]
        steps = [(idx, name) for idx, name in steps
                 if idx * const.dts <= limit + 1.0e-12 * max(1.0, limit)]

    z = (np.arange(const.Nz) + 0.5) * const.dz
    kernel = np.exp(-1j * ctx["k"] * z)
    result = {name: [] for name in
              ("time", "first", "total", "residual")}
    for idx, name in steps:
        data = dz.load_frame(rows[0]["dir"], name)
        if data is None:
            continue
        profile = data.mean(axis=(1, 2)) / species.n - 1.0
        first = complex(2.0 * np.mean(profile * kernel))
        residual = profile - np.real(first * np.conj(kernel))
        result["time"].append(idx * const.dts / ctx["T_wave"])
        result["first"].append(abs(first))
        result["total"].append(float(np.sqrt(2.0 * np.mean(profile ** 2))))
        result["residual"].append(
            float(np.sqrt(2.0 * np.mean(residual ** 2))))
    if not result["time"]:
        raise SystemExit(f"Could not read '{sort_name}' density frames.")
    return {key: np.asarray(value, dtype=float)
            for key, value in result.items()}


def diagnostic_energy_series(ctx, model_tmax=None):
    """Read wE and per-sort wK columns from temporal/dk_diagnostic.txt."""
    const, config = ctx["const"], ctx["config"]
    candidates = [os.path.join(const.in_dir, "temporal", "dk_diagnostic.txt"),
                  os.path.join(ctx["config_dir"], "temporal",
                               "dk_diagnostic.txt")]
    path = next((candidate for candidate in candidates
                 if os.path.isfile(candidate)), None)
    if path is None:
        raise SystemExit(f"dk_diagnostic.txt not found for '{ctx['config_dir']}'.")
    with open(path, "r", encoding="utf-8") as stream:
        header = re.split(r"\s{2,}", stream.readline().strip())
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(header) or "Time" not in header:
        raise SystemExit(f"unexpected dk_diagnostic table: {path}")
    dt = float(config.get("Geometry", {}).get("dt", 1.0))
    time = data[:, header.index("Time")] * dt / ctx["T_wave"]
    if model_tmax is not None:
        keep = time <= model_tmax + 1.0e-12 * max(1.0, model_tmax)
        time, data = time[keep], data[keep]
    columns = {name: data[:, header.index(name)] for name in header
               if name == "wE" or name.startswith("wK_")}
    return time, columns


def electric_harmonic_series(ctx, first_frame=0, model_tmax=None):
    """Complex first harmonic of the saved longitudinal electric field.

    E_z is face centred in z, unlike density and DkDistributionFunction.  Use
    the face coordinates here so the complex phase passed to Vlasov--Ampere is
    consistent with the field that the PIC pusher actually saw.
    """
    const, config = ctx["const"], ctx["config"]
    config_dir, k = ctx["config_dir"], ctx["k"]
    edir_name = field_view_dir(config, "E")
    if edir_name is None:
        raise SystemExit("config has no FieldView diagnostic for field 'E'.")

    epath = os.path.join(const.in_dir, edir_name)
    if not os.path.isdir(epath):
        epath = os.path.join(config_dir, edir_name)
    if not os.path.isdir(epath):
        raise SystemExit(f"E field frames not found (looked in '{edir_name}').")

    frames = sorted((int(name), name) for name in os.listdir(epath)
                    if name.isdigit() and int(name) >= first_frame)
    if model_tmax is not None:
        t_limit = model_tmax * ctx["T_wave"]
        tolerance = 1.0e-12 * max(1.0, t_limit)
        frames = [(idx, name) for idx, name in frames
                  if idx * const.dts <= t_limit + tolerance]
    if not frames or frames[0][0] != first_frame:
        raise SystemExit(
            f"initial E frame {first_frame:04d} not found in '{epath}'.")

    first_path = os.path.join(epath, frames[0][1])
    ncomp = 3
    ncells = os.path.getsize(first_path) // 4 // ncomp
    if ncells <= 0 or ncells % const.Nz != 0:
        raise SystemExit(f"unexpected FieldView size: {first_path}")
    transverse_cells = ncells // const.Nz
    z_faces = (np.arange(const.Nz) + 0.5) * const.dz
    projector = np.exp(-1j * k * z_faces)

    times, harmonic = [], []
    for idx, name in frames:
        path = os.path.join(epath, name)
        raw = np.fromfile(path, dtype=np.float32, count=ncells * ncomp)
        if raw.size != ncells * ncomp:
            raise SystemExit(f"unexpected FieldView size: {path}")
        profile = raw.reshape(const.Nz, transverse_cells, ncomp)[..., 2].mean(
            axis=1)
        times.append(idx * const.dts)
        harmonic.append(complex(2.0 * np.mean(profile * projector)))
    return np.asarray(times), np.asarray(harmonic), edir_name


def electric_harmonic_ratio_series(testname, args, modes=(1, 2)):
    """Load |E_m|/|E_1| for selected longitudinal Fourier harmonics."""
    first_frame = 1 if args.ic_from_dump is None else args.ic_from_dump
    ctx = prepare_theory(testname, ic_frame=first_frame)
    const, config = ctx["const"], ctx["config"]
    config_dir = ctx["config_dir"]
    edir_name = field_view_dir(config, "E")
    if edir_name is None:
        raise SystemExit(
            f"'{testname}': config has no FieldView diagnostic for field 'E'.")

    epath = os.path.join(const.in_dir, edir_name)
    if not os.path.isdir(epath):
        epath = os.path.join(config_dir, edir_name)
    if not os.path.isdir(epath):
        raise SystemExit(f"'{testname}': E field frames not found in "
                         f"'{edir_name}'.")

    frames = sorted((int(name), name) for name in os.listdir(epath)
                    if name.isdigit() and int(name) >= first_frame)
    if args.model_tmax is not None:
        t_limit = args.model_tmax * ctx["T_wave"]
        tolerance = 1.0e-12 * max(1.0, t_limit)
        frames = [(idx, name) for idx, name in frames
                  if idx * const.dts <= t_limit + tolerance]
    if not frames or frames[0][0] != first_frame:
        raise SystemExit(f"'{testname}': initial E frame "
                         f"{first_frame:04d} not found in '{epath}'.")

    ncomp = 3
    first_path = os.path.join(epath, frames[0][1])
    ncells = os.path.getsize(first_path) // 4 // ncomp
    if ncells <= 0 or ncells % const.Nz != 0:
        raise SystemExit(f"unexpected FieldView size: {first_path}")
    transverse_cells = ncells // const.Nz
    z = (np.arange(const.Nz) + 0.5) * const.dz
    kernels = {m: np.exp(-1j * 2.0 * math.pi * m * z / const.Lz)
               for m in modes}
    amplitudes = {m: [] for m in modes}
    mode_energy_ratios = []
    times = []
    for idx, name in frames:
        path = os.path.join(epath, name)
        raw = np.fromfile(path, dtype=np.float32, count=ncells * ncomp)
        if raw.size != ncells * ncomp:
            raise SystemExit(f"unexpected FieldView size: {path}")
        profile = raw.reshape(const.Nz, transverse_cells, ncomp)[..., 2].mean(
            axis=1)
        times.append(idx * const.dts / ctx["T_wave"])
        for m, kernel in kernels.items():
            amplitudes[m].append(abs(2.0 * np.mean(profile * kernel)))
        power = np.abs(np.fft.rfft(profile)) ** 2
        first_mode_energy = 2.0 * power[1]
        other_mode_energy = 2.0 * float(np.sum(power[2:]))
        if const.Nz % 2 == 0 and power.size > 2:
            # The Nyquist coefficient has no distinct negative-frequency pair.
            other_mode_energy -= power[-1]
        energy_threshold = np.finfo(float).eps * max(
            first_mode_energy, other_mode_energy, 1.0)
        mode_energy_ratios.append(
            other_mode_energy / first_mode_energy
            if first_mode_energy > energy_threshold else math.nan)

    amplitudes = {m: np.asarray(values) for m, values in amplitudes.items()}
    denominator = amplitudes[1]
    threshold = np.finfo(float).eps * max(float(np.max(denominator)), 1.0)
    ratios = {m: np.divide(amplitudes[m], denominator,
                           out=np.full_like(denominator, np.nan),
                           where=denominator > threshold)
              for m in modes if m != 1}
    energy_time, energies = diagnostic_energy_series(ctx, args.model_tmax)
    return {"testname": testname,
            "label": compare_label_with_particles(testname, config),
            "time": np.asarray(times), "ratios": ratios,
            "mode_energy_ratio": np.asarray(mode_energy_ratios),
            "energy_time": energy_time, "energies": energies,
            "out_dir": const.out_dir}


def run_compare_electric(args):
    """Compare second/first electric harmonics in four stacked run panels."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(tests_dir, "..", ".."))
    tools_dir = os.path.join(repo_dir, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib.plot import bbox, labelsize, ticksize

    plt.rc("text", usetex=False)
    runs = [electric_harmonic_ratio_series(name, args)
            for name in [args.model_electric, *args.compare]]
    fig = plt.figure(figsize=(15.0, 9.0))
    grid = fig.add_gridspec(4, 2, width_ratios=(3.2, 1.0), hspace=0.0,
                            wspace=0.08)
    axes = []
    for row in range(4):
        axes.append(fig.add_subplot(
            grid[row, 0], sharex=None if row == 0 else axes[0]))
    legend_ax = fig.add_subplot(grid[:, 1])
    legend_ax.axis("off")
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "v", "P")
    handles = []
    for i, run in enumerate(runs):
        line, = axes[i].plot(run["time"], run["ratios"][2],
                            color=colors[i % len(colors)],
                            marker=markers[i % len(markers)], linewidth=2.2,
                            markersize=4.5, label=run["label"])
        handles.append(line)

    for i, ax in enumerate(axes):
        if i >= len(runs):
            ax.set_visible(False)
            continue
        ax.set_ylim(0.0, 1.5)
        ax.tick_params(labelsize=ticksize)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(r"$|E_{z,2}|/|E_{z,1}|$", fontsize=labelsize)
        if i < len(runs) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"$t/T$", fontsize=labelsize)
    axes[0].set_title(r"Вторая гармоника относительно первой",
                      fontsize=labelsize, bbox=bbox)
    legend_ax.legend(handles=handles,
                     labels=[run["label"] for run in runs],
                     loc="center", fontsize=labelsize, framealpha=0.9)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.09, top=0.94)

    out_dir = os.path.join(runs[0]["out_dir"], args.out_subdir + "_E")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "ion_sound_compare_E_ratios.png")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    energy_fig, energy_axis = plt.subplots(figsize=(9.5, 7.0))
    for i, run in enumerate(runs):
        if "wE" not in run["energies"]:
            raise SystemExit(
                f"wE column not found in dk_diagnostic for '{run['testname']}'.")
        energy_axis.plot(
            run["energy_time"], run["energies"]["wE"],
            color=colors[i % len(colors)], marker=markers[i % len(markers)],
            linewidth=2.2, markersize=4.5, label=run["label"])
    energy_axis.set_xlabel(r"$t/T$", fontsize=labelsize)
    energy_axis.set_ylabel(r"$w_E$", fontsize=labelsize)
    energy_axis.tick_params(labelsize=ticksize)
    energy_axis.grid(True, alpha=0.3)
    energy_axis.set_title(r"Энергия электрического поля",
                          fontsize=labelsize, bbox=bbox)
    energy_axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                       fontsize=ticksize, framealpha=0.9)
    energy_fig.tight_layout()
    energy_path = os.path.join(out_dir, "ion_sound_compare_wE.png")
    energy_fig.savefig(energy_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(energy_fig)

    mode_energy_fig, mode_energy_axis = plt.subplots(figsize=(9.5, 7.0))
    for i, run in enumerate(runs):
        mode_energy_axis.plot(
            run["time"], run["mode_energy_ratio"],
            color=colors[i % len(colors)], marker=markers[i % len(markers)],
            linewidth=2.2, markersize=4.5, label=run["label"])
    mode_energy_axis.set_xlabel(r"$t/T$", fontsize=labelsize)
    mode_energy_axis.set_ylabel(
        r"$\sum_{m>1}W_{E,m}/W_{E,1}$", fontsize=labelsize)
    mode_energy_axis.set_ylim(0.0, 1000.0)
    mode_energy_axis.tick_params(labelsize=ticksize)
    mode_energy_axis.grid(True, alpha=0.3)
    mode_energy_axis.set_title(
        r"Энергия остальных гармоник относительно первой",
        fontsize=labelsize, bbox=bbox)
    mode_energy_axis.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=ticksize, framealpha=0.9)
    mode_energy_fig.tight_layout()
    mode_energy_path = os.path.join(
        out_dir, "ion_sound_compare_E1_energy_ratio.png")
    mode_energy_fig.savefig(
        mode_energy_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(mode_energy_fig)

    print("=" * 70)
    print(f"COMPARE_ELECTRIC: {args.model_electric} + "
          f"{len(args.compare)} run(s)")
    print("=" * 70)
    for run in runs:
        print(f"  {run['label']:8s} {run['testname']}: "
              f"{run['time'].size} frames, t/T in "
              f"[{run['time'][0]:.6g}, {run['time'][-1]:.6g}]")
    print(f"Electric-harmonic comparison figure written to {png_path}")
    print(f"Electric-energy comparison figure written to {energy_path}")
    print(f"Electric mode-energy ratio figure written to {mode_energy_path}")


def run_model_pic(args):
    """Linear first-harmonic IVP from realized PIC f(z,v) and E(z).

    Only the selected spatial harmonic is evolved.  Its initial f_hat(v), the
    z-averaged realized equilibrium F0(v), and the matching electric-field
    harmonic are read from one PIC frame.  Linear Vlasov--Ampere retains the
    finite-marker Gauss mismatch without coupling to any other spatial mode.
    """
    ic_frame = 1 if args.ic_from_dump is None else args.ic_from_dump
    ctx = prepare_theory(args.model_pic, ic_frame=ic_frame)
    const, species = ctx["const"], ctx["species"]
    initial_distribution = ctx["initial_distribution"]
    t0, T_wave = ctx["t0"], ctx["T_wave"]

    missing = [s.name for s in species
               if s.name not in initial_distribution or
               "equilibrium" not in initial_distribution[s.name]]
    if missing:
        raise SystemExit(
            "--model-pic requires DkDistributionFunction in the selected "
            f"frame for every species; missing: {', '.join(missing)}")

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(tests_dir, "..", ".."))
    for path in (os.path.join(repo_dir, "tools"),
                 os.path.join(tests_dir, "drift_kinetic_tools")):
        if path not in sys.path:
            sys.path.insert(0, path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib.plot import bbox, labelsize, ticksize
    dz = _DensityZTools

    plt.rc("text", usetex=False)
    density_pic = {}
    for s in species:
        times, harmonic, noise = harmonic_series(
            ctx, dz, s.name, args.model_tmax)
        keep = times >= t0 - 1.0e-12 * max(1.0, t0)
        if not np.any(keep):
            raise SystemExit(
                f"no '{s.name}' density frames at or after {ic_frame:04d}")
        density_pic[s.name] = (times[keep], harmonic[keep], noise[keep])

    field_times, field_pic, field_dir = electric_harmonic_series(
        ctx, first_frame=ic_frame, model_tmax=args.model_tmax)
    initial_field = complex(field_pic[0])
    t_end = max([float(field_times[-1])] +
                [float(row[0][-1]) for row in density_pic.values()])
    duration = t_end - t0
    if duration <= 0.0:
        raise SystemExit("--model-pic needs at least two diagnostic times.")

    t_grid, nhat, E_theory = solve_vlasov_poisson(
        species, ctx["cn_hat"], ctx["u_hat"], ctx["k"], duration,
        initial_distribution=initial_distribution,
        initial_field=initial_field,
        use_realized_equilibrium=True)
    t_grid = t_grid + t0

    colors = {"electrons": "tab:blue", "ions": "tab:red"}
    markers = {"electrons": "o", "ions": "s"}
    fig, (ax_n, ax_e) = plt.subplots(1, 2, figsize=(15.0, 7.0))

    density_errors = {}
    for s in species:
        times, harmonic, _ = density_pic[s.name]
        measured = np.abs(harmonic) / s.n
        predicted = np.interp(times, t_grid, np.abs(nhat[s.name]) / s.n)
        color = colors.get(s.name, None)
        marker = markers.get(s.name, "o")
        ax_n.plot(times / T_wave, measured, color=color, marker=marker,
                  linewidth=1.8, markersize=4.0,
                  label=rf"$|\delta n_{{{s.name[0]},1}}|$ PIC")
        ax_n.plot(times / T_wave, predicted, color=color, linestyle="--",
                  linewidth=2.2,
                  label=rf"$|\delta n_{{{s.name[0]},1}}|$ model-pic linear")
        density_errors[s.name] = float(np.sqrt(
            np.mean(np.square(measured - predicted))))

    field_predicted = np.interp(field_times, t_grid, np.abs(E_theory))
    ax_e.plot(field_times / T_wave, np.abs(field_pic), color="purple",
              marker="o", linewidth=1.8, markersize=4.0,
              label=r"$|E_{z,1}|$ PIC")
    ax_e.plot(field_times / T_wave, field_predicted, color="black",
              linestyle="--", linewidth=2.2,
              label=r"$|E_{z,1}|$ model-pic linear")
    field_error = float(np.sqrt(np.mean(
        np.square(np.abs(field_pic) - field_predicted))))
    for ax in (ax_n, ax_e):
        ax.set_xlim(t0 / T_wave, t_end / T_wave)
        ax.tick_params(labelsize=ticksize)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=ticksize)
        ax.set_box_aspect(1)
    ax_n.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_n.set_ylabel(r"$|\delta n_1|/n_0$", fontsize=labelsize)
    ax_n.set_title(r"Плотность: PIC и линейная realized-$F_0$ теория",
                   fontsize=labelsize, bbox=bbox)
    ax_e.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_e.set_ylabel(r"$|E_{z,1}|$", fontsize=labelsize)
    ax_e.set_title(r"Поле: линейный Vlasov--Ampère, гармоника 1",
                   fontsize=labelsize, bbox=bbox)
    fig.tight_layout()

    out_dir = os.path.join(const.out_dir, "ion_sound_model_pic")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "ion_sound_model_pic.png")
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)

    rho0 = sum(s.q * initial_distribution[s.name]["density_hat"]
               for s in species)
    gauss_field = rho0 / (1j * ctx["k"])
    print("=" * 70)
    print(f"MODEL_PIC: {args.model_pic}")
    print("=" * 70)
    print(f"  IC frame = {ic_frame:04d}, t0 = {t0:.6g} "
          f"= {t0 / T_wave:.6g} T")
    print(f"  field source = {field_dir}/{ic_frame:04d}")
    print(f"  E_k(t0) PIC   = {initial_field.real:+.6e}"
          f"{initial_field.imag:+.6e}j")
    print(f"  E_k(t0) Gauss = {gauss_field.real:+.6e}"
          f"{gauss_field.imag:+.6e}j")
    print(f"  |Gauss mismatch| = {abs(initial_field - gauss_field):.6e}")
    for s in species:
        saved = initial_distribution[s.name]
        equilibrium_density = float(np.dot(
            saved["equilibrium"], saved["weights"]))
        print(f"  {s.name:9s}: integral F0_pic dv = "
              f"{equilibrium_density:.9e}, density RMSE = "
              f"{density_errors[s.name]:.6e}")
    print(f"  field RMSE = {field_error:.6e}")
    print(f"PIC-informed theory figure written to {png_path}")


def fit_two_branch(t, a, omega_guess, gamma_guess,
                   omega_range=(0.85, 1.20), gamma_range=(0.05, 4.0),
                   n_grid=41, n_passes=7):
    """Least-squares fit of both ion-acoustic branches to a COMPLEX signal:

        a(t) = A_p exp(-i w t - g t) + A_m exp(+i w t - g t).

    The dispersion relation has the root pair +-w_s - i Gamma, so anything that
    is not a pure travelling eigenmode carries both.  Their interference beats
    at 2 w, which is exactly the wobble seen on |a(t)| around the exponential.
    Fitting the complex harmonic rather than its modulus separates them, and
    since the model is LINEAR in A_p, A_m, zero-mean noise in `a` does not bias
    the fitted amplitudes -- a log-fit of |a| has no such property.

    (w, g) are found by successive grid refinement over `omega_range` *
    omega_guess and `gamma_range` * gamma_guess (no SciPy dependency); A_p, A_m
    follow from a 2x2 normal-equation solve at every candidate.

    Returns (w, g, A_p, A_m, relative_residual).
    """
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=complex)
    scale = float(np.linalg.norm(a))

    def solve(w, g):
        basis = np.exp(-g * t) * np.vstack([np.exp(-1j * w * t),
                                            np.exp(+1j * w * t)])
        gram = basis.conj() @ basis.T
        rhs = basis.conj() @ a
        try:
            coefficients = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:  # degenerate window: branches collinear
            coefficients = np.linalg.lstsq(basis.T, a, rcond=None)[0]
        return coefficients, float(np.linalg.norm(a - basis.T @ coefficients))

    lo_w, hi_w = (r * omega_guess for r in omega_range)
    lo_g, hi_g = (r * gamma_guess for r in gamma_range)
    best = None
    for _ in range(n_passes):
        for w in np.linspace(lo_w, hi_w, n_grid):
            for g in np.linspace(lo_g, hi_g, n_grid):
                coefficients, residual = solve(w, g)
                if best is None or residual < best[0]:
                    best = (residual, w, g, coefficients)
        # Re-bracket within one grid step of the current optimum.
        step_w = (hi_w - lo_w) / (n_grid - 1)
        step_g = (hi_g - lo_g) / (n_grid - 1)
        lo_w, hi_w = best[1] - step_w, best[1] + step_w
        lo_g, hi_g = max(best[2] - step_g, 0.0), best[2] + step_g

    residual, w, g, coefficients = best
    return (w, g, complex(coefficients[0]), complex(coefficients[1]),
            residual / scale if scale > 0.0 else math.nan)


def fourier_peak_frequency(times, signals, omega_band=None,
                           zero_padding=32):
    r"""Return the interpolated peak of the spatially averaged time spectrum.

    The spatially uniform component is removed at every time, a Hann window
    suppresses leakage, and the powers |FFT|^2 are averaged over all supplied
    points.  We deliberately do not subtract each point's temporal mean: on a
    short record that operation distorts the oscillation and shifts its peak.
    Zero padding samples the peak finely; a three-bin parabolic interpolation
    of log power removes the remaining FFT-bin quantisation.
    """
    times = np.asarray(times, dtype=float)
    signals = np.asarray(signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[:, None]
    if times.size < 4 or signals.shape[0] != times.size:
        return (math.nan, math.nan, math.nan, math.nan,
                np.empty(0), np.empty(0))
    steps = np.diff(times)
    time_step = float(np.mean(steps))
    if time_step <= 0.0 or not np.allclose(
            steps, time_step, rtol=1.0e-6, atol=1.0e-12 * time_step):
        return (math.nan, math.nan, math.nan, math.nan,
                np.empty(0), np.empty(0))

    window = np.hanning(times.size)[:, None]
    centred = signals - np.mean(signals, axis=1, keepdims=True)
    minimum_size = max(times.size * int(zero_padding), times.size)
    fft_size = 1 << (minimum_size - 1).bit_length()
    transform = np.fft.rfft(centred * window, n=fft_size, axis=0)
    power = np.mean(np.abs(transform) ** 2, axis=1)
    omega_grid = 2.0 * math.pi * np.fft.rfftfreq(fft_size, time_step)

    candidates = np.arange(1, omega_grid.size)
    if omega_band is not None:
        low, high = omega_band
        candidates = candidates[(omega_grid[candidates] >= low)
                                & (omega_grid[candidates] <= high)]
    if candidates.size == 0:
        return (math.nan, math.nan, math.nan, math.nan,
                omega_grid, power)
    peak_index = int(candidates[np.argmax(power[candidates])])

    offset = 0.0
    if 0 < peak_index < power.size - 1:
        floor = np.finfo(float).tiny
        left, centre, right = np.log(
            np.maximum(power[peak_index - 1:peak_index + 2], floor))
        denominator = left - 2.0 * centre + right
        if denominator != 0.0:
            offset = float(np.clip(
                0.5 * (left - right) / denominator, -0.5, 0.5))
    omega_step = float(omega_grid[1] - omega_grid[0])
    native_omega_resolution = 2.0 * math.pi / (times.size * time_step)
    omega = float(omega_grid[peak_index] + offset * omega_step)
    period = 2.0 * math.pi / omega if omega > 0.0 else math.nan
    return (omega, period, omega_step, native_omega_resolution,
            omega_grid, power)


def load_ion_harmonic(testname, args, dz):
    """Load the relative ion-density first harmonic for one finished run."""
    ctx = prepare_theory(testname, ic_frame=args.ic_from_dump)
    density = density_comparison_series(
        ctx, dz, ctx["ion"].name, args.model_tmax)
    energy_time, energies = diagnostic_energy_series(ctx, args.model_tmax)
    return {
        "testname": testname,
        "label": compare_label_with_particles(testname, ctx["config"]),
        "time": density["time"],
        "amplitude": density["first"],
        "total_amplitude": density["total"],
        "noise_residual": density["residual"],
        "energy_time": energy_time,
        "energies": energies,
        "T_wave": ctx["T_wave"],
        "Gamma": -ctx["omega0"].imag,
        "a0": abs(ctx["cn_hat"][ctx["ion"].name]),
        "t0": ctx["t0"],
        "out_dir": ctx["const"].out_dir,
    }


class _DensityZTools:
    """Small self-contained subset of the former density-z plot helper.

    ``ion_sound.py`` only needs diagnostic discovery, binary-frame loading and
    the default species styles.  Keeping those operations here avoids relying
    on the removed ``tests/drift_kinetic/drift_kinetic_tools`` directory.
    """

    SPECIES_STYLE = {
        "electrons": {
            "color": "blue", "marker": "o", "linestyle": "-",
            "linewidth": 3.0, "markersize": 5.0, "label": r"$n_e$",
        },
        "ions": {
            "color": "red", "marker": "o", "linestyle": "-",
            "linewidth": 3.0, "markersize": 5.0, "label": r"$n_i$",
        },
    }

    @staticmethod
    def _expected_size_bytes(const):
        return (const.Nx * const.Ny * const.Nz
                * np.dtype(np.float32).itemsize)

    @staticmethod
    def collect_rows(species_filter):
        from lib.constants import const

        available = []
        seen = set()
        for diagnostic in const.config.get("Diagnostics", []):
            if diagnostic.get("diagnostic") != "DistributionMoment" or \
                    diagnostic.get("moment") != "density":
                continue
            species = diagnostic.get("particles")
            if species is None or species in seen:
                continue
            seen.add(species)
            available.append((species, diagnostic.get("out_dir")))

        if species_filter is not None:
            wanted = set(species_filter)
            available_names = {species for species, _ in available}
            for name in sorted(wanted - available_names):
                print(f"[skip] no density diagnostic for species '{name}' "
                      "in config")
            available = [(species, explicit)
                         for species, explicit in available
                         if species in wanted]

        expected_size = _DensityZTools._expected_size_bytes(const)
        rows = []
        for species, explicit in available:
            candidates = ([explicit] if explicit is not None else [
                os.path.join(species, "density"),
                f"{species}_density",
            ])
            for candidate in candidates:
                directory = (candidate if os.path.isabs(candidate) else
                             os.path.join(const.in_dir, candidate))
                if not os.path.isdir(directory):
                    continue
                timesteps = sorted(
                    (int(name), name) for name in os.listdir(directory)
                    if name.isdigit()
                    and os.path.isfile(os.path.join(directory, name))
                    and os.path.getsize(os.path.join(directory, name))
                    == expected_size)
                if timesteps:
                    rows.append({
                        "species": species, "dir": directory,
                        "timesteps": timesteps,
                    })
                    break
            else:
                print(f"[skip] no density frames for species '{species}'")
        return rows

    @staticmethod
    def load_frame(directory, name):
        from lib.constants import const

        path = os.path.join(directory, name)
        if not os.path.isfile(path) or os.path.getsize(path) != \
                _DensityZTools._expected_size_bytes(const):
            return None
        raw = np.fromfile(
            path, dtype=np.float32, count=const.Nx * const.Ny * const.Nz)
        return raw.reshape(const.Nz, const.Ny, const.Nx)


def plot_density_harmonics_and_kinetic_energy(runs, out_dir, dpi):
    """Write the ion-only multi-run noise and kinetic-energy figures."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from lib.plot import bbox, labelsize, ticksize
    plt.rc("text", usetex=False)

    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "v", "P")
    outputs = {}
    def percent(values, reference):
        values = np.asarray(values, dtype=float)
        reference = np.asarray(reference, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = 100.0 * values / reference
        return np.where(np.isfinite(ratio) & (reference > 0.0), ratio, np.nan)

    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(17.0, 7.5))
    for index, run in enumerate(runs):
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        first = run["amplitude"]
        finite = first[np.isfinite(first) & (first > 0.0)]
        reference = float(finite[0]) if finite.size else math.nan
        common = dict(color=color, marker=marker, markersize=3.5)
        ax_abs.plot(run["time"], percent(first, reference), linewidth=2.0,
                    linestyle="-", label=run["label"], **common)
        ax_abs.plot(run["time"], percent(run["noise_residual"], reference),
                    linewidth=1.5, linestyle=":", alpha=0.8,
                    label="_nolegend_", **common)
    ax_abs.set_ylabel(r"% от $|\delta n_{i,1}(t_0)|$", fontsize=labelsize)
    ax_abs.set_title(r"Первая гармоника и полный остаток ионов",
                     fontsize=labelsize, bbox=bbox)
    x_max = max(float(run["time"][-1]) for run in runs)
    ax_abs.set_xlim(0.0, x_max)
    ax_abs.set_ylim(0.0, 100.0)
    ax_abs.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_abs.tick_params(labelsize=ticksize)
    ax_abs.grid(True, alpha=0.3)
    ax_abs.set_box_aspect(1)

    run_handles = [Line2D([], [], color=colors[i % len(colors)],
                          marker=markers[i % len(markers)], linewidth=1.8,
                          label=run["label"])
                   for i, run in enumerate(runs)]
    ax_abs.legend(handles=run_handles + [
        Line2D([], [], color="black", linestyle="-", label=r"$|\delta n_{i,1}|$"),
        Line2D([], [], color="black", linestyle=":", label="полный остаток")],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
        fontsize=ticksize)
    fig.tight_layout(pad=0.6)
    noise_path = os.path.join(out_dir, "ion_sound_compare_noise.png")
    fig.savefig(noise_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    outputs["noise"] = noise_path

    fig, axis = plt.subplots(figsize=(9.5, 7.0))
    for index, run in enumerate(runs):
        for species_name, linestyle in (("electrons", "-"), ("ions", "--")):
            key = f"wK_{species_name}"
            if key in run["energies"]:
                energy = run["energies"][key]
                initial_energy = float(energy[0])
                if initial_energy == 0.0:
                    raise SystemExit(
                        f"{key}(0) is zero for '{run['testname']}'; relative "
                        "kinetic-energy change is undefined.")
                relative_energy = (energy - initial_energy) / initial_energy
                axis.plot(run["energy_time"], relative_energy,
                          color=colors[index % len(colors)],
                          linestyle=linestyle, linewidth=1.8)
    axis.set_xlabel(r"$t/T$", fontsize=labelsize)
    axis.set_ylabel(r"$(w_K(t)-w_K(0))/w_K(0)$", fontsize=labelsize)
    axis.tick_params(labelsize=ticksize)
    axis.grid(True, alpha=0.3)
    axis.set_title("Кинетическая энергия", fontsize=labelsize, bbox=bbox)

    run_handles = [Line2D([], [], color=colors[i % len(colors)],
                          linewidth=1.8,
                          label=run["label"])
                   for i, run in enumerate(runs)]
    species_handles = [Line2D([], [], color="black", linestyle="-",
                              label="electrons"),
                       Line2D([], [], color="black", linestyle="--",
                              label="ions")]
    axis.legend(handles=run_handles + species_handles, loc="upper left",
                bbox_to_anchor=(1.02, 1.0), fontsize=ticksize, framealpha=0.9)
    fig.tight_layout()
    energy_path = os.path.join(out_dir, "ion_sound_compare_wK.png")
    fig.savefig(energy_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    outputs["wK"] = energy_path
    return outputs


def run_compare_article(args):
    """Publication-style three-panel ion-density comparison.

    ``--article_log`` keeps the profile panel linear and uses logarithmic
    y-axes for the two amplitude panels.
    """
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(tests_dir, "..", ".."))
    for path in (os.path.join(repo_dir, "tools"),
                 os.path.join(tests_dir, "drift_kinetic_tools")):
        if path not in sys.path:
            sys.path.insert(0, path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    dz = _DensityZTools

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"
    label_fs, tick_fs, legend_fs, panel_fs = 18, 15, 14, 18

    names = [args.model, *args.compare]
    article_tmax = (args.article_tmax if args.article_tmax is not None
                    else args.model_tmax)
    article_args = argparse.Namespace(**vars(args))
    article_args.model_tmax = article_tmax
    runs = [load_ion_harmonic(name, article_args, dz) for name in names]
    base = runs[0]
    colors = ("red", "blue", "green", "purple", "orange", "brown")

    def particle_label(run):
        run_ctx = prepare_theory(run["testname"], ic_frame=args.ic_from_dump)
        particles = run_ctx["config"].get("Particles", [])
        ion_info = next((item for item in particles
                         if item.get("sort_name") == run_ctx["ion"].name), {})
        count = ion_info.get("Np", "?")
        try:
            count = f"{float(count):g}"
        except (TypeError, ValueError):
            count = str(count)
        return rf"$N_{{\mathrm{{ppc}}}}={count}$"

    article_labels = [particle_label(run) for run in runs]

    # Reload the base context last: density_z uses module-level geometry, so
    # this also restores the correct grid before reading the two profiles.
    ctx = prepare_theory(args.model, ic_frame=args.ic_from_dump)
    const, ion = ctx["const"], ctx["ion"]
    rows = dz.collect_rows([ion.name])
    if not rows and const.in_dir != ctx["config_dir"]:
        const.in_dir = ctx["config_dir"]
        const.out_dir = os.path.join(ctx["config_dir"], "processed")
        rows = dz.collect_rows([ion.name])
    if not rows:
        raise SystemExit("--article: ion density diagnostic not found for "
                         f"'{args.model}'.")

    targets = (0.0, ctx["T_wave"])
    available = [(idx * const.dts, idx, name)
                 for idx, name in rows[0]["timesteps"]]
    selected = [min(available, key=lambda item: abs(item[0] - target))
                for target in targets]
    profiles = []
    for _, _, frame_name in selected:
        data = dz.load_frame(rows[0]["dir"], frame_name)
        if data is None:
            raise SystemExit(f"--article: cannot read ion frame {frame_name}.")
        profiles.append(data.mean(axis=(1, 2)) / ion.n)
    z = (np.arange(const.Nz) + 0.5) * const.dz

    # Temporal Fourier spectrum averaged over every longitudinal grid point.
    point_times, point_signals = [], []
    point_limit = (article_tmax * ctx["T_wave"]
                   if article_tmax is not None else None)
    for frame_time, _, frame_name in available:
        if point_limit is not None and frame_time > point_limit + \
                1.0e-12 * max(1.0, point_limit):
            continue
        data = dz.load_frame(rows[0]["dir"], frame_name)
        if data is None:
            continue
        profile = data.mean(axis=(1, 2)) / ion.n - 1.0
        point_times.append(frame_time)
        point_signals.append(profile)
    point_times = np.asarray(point_times, dtype=float)
    point_signals = np.asarray(point_signals, dtype=float)
    (fft_omega, fft_period, fft_omega_step, fft_native_resolution,
     fft_omega_grid, fft_power) = fourier_peak_frequency(
         point_times, point_signals,
         omega_band=(0.5 * ctx["omega0"].real,
                     1.5 * ctx["omega0"].real))
    fft_omega_error = 100.0 * (fft_omega / ctx["omega0"].real - 1.0)
    fft_period_error = 100.0 * (fft_period / ctx["T_wave"] - 1.0)

    x_max = (float(article_tmax) if article_tmax is not None else
             max(float(run["time"][-1]) for run in runs))
    theory_start = ctx["t0"] / ctx["T_wave"]
    theory_end = max(x_max * ctx["T_wave"] - ctx["t0"],
                     1.0e-3 * ctx["T_wave"])
    theory_time, theory_density, _ = solve_vlasov_poisson(
        ctx["species"], ctx["cn_hat"], ctx["u_hat"], ctx["k"], theory_end,
        exact_ic=args.exact_ic, omega0=ctx["omega0"], E0=ctx["E0"],
        initial_distribution=ctx["initial_distribution"])
    theory_time = (theory_time + ctx["t0"]) / ctx["T_wave"]
    exact_ion = np.abs(theory_density[ion.name]) / ion.n
    exact_at_base_frames = np.interp(base["time"], theory_time, exact_ion)
    first_harmonic_difference = base["amplitude"] - exact_at_base_frames
    exact_norm = float(np.linalg.norm(exact_at_base_frames))
    first_harmonic_l2_error = (
        100.0 * float(np.linalg.norm(first_harmonic_difference)) / exact_norm
        if exact_norm > 0.0 else math.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pointwise_error = 100.0 * np.abs(first_harmonic_difference) / \
            exact_at_base_frames
    finite_pointwise_error = pointwise_error[np.isfinite(pointwise_error)]
    first_harmonic_max_error = (
        float(np.max(finite_pointwise_error))
        if finite_pointwise_error.size else math.nan)
    exponential_time = np.linspace(theory_start, x_max, 500)
    exponential = base["a0"] * np.exp(
        -base["Gamma"] *
        (exponential_time * base["T_wave"] - base["t0"]))

    fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.2))
    ax_profile, ax_total, ax_noise = axes

    # (a) Base-run ion profiles at the requested two phases.
    initial_amplitude = float(base["amplitude"][0])
    ax_profile.plot(z, profiles[0], color="black", linewidth=2.2,
                    marker="o", markersize=4.0, linestyle="-", label=r"$t=0$")
    ax_profile.plot(z, profiles[1], color="black", linewidth=2.2,
                    marker="o", markersize=4.0, linestyle="--",
                    label=r"$t=T$")
    ax_profile.axhline(1.0 + initial_amplitude, color="0.5", linewidth=1.2,
                       linestyle="--")
    ax_profile.axhline(1.0 - initial_amplitude, color="0.5", linewidth=1.2,
                       linestyle="--")
    ax_profile.set_xlim(0.0, const.Lz)
    ax_profile.set_ylim(0.96, 1.04)
    ax_profile.set_xlabel(r"$z,\ c/\omega_{pe}$", fontsize=label_fs)
    ax_profile.set_ylabel(r"$n_i/n_0$", fontsize=label_fs)
    ax_profile.legend(loc="upper left", fontsize=legend_fs, framealpha=0.9)

    # (b) Full profile amplitude and the base-run Landau exponential.
    total_lines = [None] * len(runs)
    for index in reversed(range(len(runs))):
        run, article_label = runs[index], article_labels[index]
        line, = ax_total.plot(run["time"], run["total_amplitude"],
                              color=colors[index % len(colors)], linewidth=3.2,
                              label=article_label)
        total_lines[index] = line
    exponential_line, = ax_total.plot(
        exponential_time, exponential, color="black", linestyle="--",
        linewidth=2.0, label=r"$|\delta n_i(0)|e^{-\Gamma t}$")
    ax_total.set_xlabel(r"$t/T$", fontsize=label_fs)
    ax_total.set_ylabel(r"$|\delta n_i(t)|/n_0$", fontsize=label_fs)
    ax_total.legend(handles=[*reversed(total_lines), exponential_line],
                    loc="upper left", ncol=2, fontsize=legend_fs,
                    framealpha=0.9)

    # (c) First mode and the full non-first-mode residue, plus exact theory.
    noise_lines = [None] * len(runs)
    for index in reversed(range(len(runs))):
        run, article_label = runs[index], article_labels[index]
        color = colors[index % len(colors)]
        line, = ax_noise.plot(run["time"], run["amplitude"], color=color,
                              linewidth=3.2, linestyle="-",
                              label=article_label)
        noise_lines[index] = line
        ax_noise.plot(run["time"], run["noise_residual"], color=color,
                      linewidth=1.6, linestyle=":", label="_nolegend_")
    ax_noise.plot(theory_time, exact_ion, color="black", linewidth=2.0,
                  linestyle="--", label=r"$\mathrm{theory}$")
    ax_noise.set_xlabel(r"$t/T$", fontsize=label_fs)
    ax_noise.set_ylabel(r"$|\delta n_{i,1}(t)|/n_0$", fontsize=label_fs)
    residual_handle = Line2D([], [], color="black", linestyle=":",
                             linewidth=1.6, label=r"$\mathrm{residual}$")
    theory_handle = Line2D([], [], color="black", linestyle="--",
                           linewidth=2.0, label=r"$\mathrm{theory}$")
    ax_noise.legend(
                    handles=[*reversed(noise_lines), theory_handle,
                             residual_handle], loc="upper left", ncol=2,
                    fontsize=legend_fs, framealpha=0.9)

    for axis in (ax_total, ax_noise):
        axis.set_xlim(0.0, x_max)
        if args.article_log:
            axis.set_yscale("log")
            axis.set_ylim(2.0e-3, 6.0e-2)
        else:
            axis.set_ylim(0.0, 0.04)

    panel_box = dict(facecolor="white", edgecolor="none", alpha=0.6,
                     boxstyle="round,pad=0.2")
    for axis, panel in zip(axes, ("(a)", "(b)", "(c)")):
        axis.minorticks_on()
        axis.tick_params(axis="both", which="both", direction="in",
                         top=True, right=True, labelsize=tick_fs)
        axis.grid(True, alpha=0.25)
        axis.set_box_aspect(1)
        axis.text(0.97, 0.97, panel, transform=axis.transAxes,
                  ha="right", va="top", fontsize=panel_fs, bbox=panel_box)

    fig.tight_layout(w_pad=1.8)
    out_dir = os.path.join(base["out_dir"], args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    article_filename = ("ion_sound_compare_article_log.png"
                        if args.article_log else
                        "ion_sound_compare_article.png")
    article_path = os.path.join(out_dir, article_filename)
    fig.savefig(article_path, dpi=args.dpi, bbox_inches="tight",
                pad_inches=0.12)
    plt.close(fig)

    spectrum_path = os.path.join(
        out_dir, "ion_sound_article_fourier_spectrum.png")
    if fft_omega_grid.size and np.any(fft_power > 0.0):
        relative_omega = fft_omega_grid / ctx["omega0"].real
        displayed = (relative_omega >= 0.25) & (relative_omega <= 1.75)
        normalization = float(np.max(fft_power[displayed]))
        normalized_power = fft_power / normalization
        peak_power = float(np.interp(
            fft_omega, fft_omega_grid, normalized_power))

        spectrum_fig, spectrum_axis = plt.subplots(figsize=(6.6, 6.2))
        spectrum_axis.plot(relative_omega[displayed],
                           normalized_power[displayed], color="red",
                           linewidth=2.6,
                           label=r"$\langle|\mathcal{F}_t[\delta n_i]|^2"
                                 r"\rangle_z$")
        spectrum_axis.axvline(
            1.0, color="black", linestyle="--", linewidth=2.0,
            label=r"$\omega_s$")
        spectrum_axis.plot(
            fft_omega / ctx["omega0"].real, peak_power, marker="o",
            markersize=7.0, markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=1.8,
            linestyle="none", label=r"$\omega_{\mathrm{FFT}}$")
        spectrum_axis.set_xlim(0.25, 1.75)
        spectrum_axis.set_ylim(0.0, 1.04 * float(
            np.max(normalized_power[displayed])))
        spectrum_axis.set_xlabel(r"$\omega/\omega_s$", fontsize=label_fs)
        spectrum_axis.set_ylabel(r"$P(\omega)/P_{\max}$",
                                  fontsize=label_fs)
        spectrum_axis.minorticks_on()
        spectrum_axis.tick_params(
            axis="both", which="both", direction="in", top=True,
            right=True, labelsize=tick_fs)
        spectrum_axis.grid(True, alpha=0.25)
        spectrum_axis.legend(loc="upper left", fontsize=legend_fs,
                             framealpha=0.9)
        spectrum_axis.set_box_aspect(1)
        spectrum_fig.tight_layout(pad=0.6)
        spectrum_fig.savefig(spectrum_path, dpi=args.dpi,
                             bbox_inches="tight", pad_inches=0.12)
        plt.close(spectrum_fig)
    else:
        spectrum_path = None

    print("=" * 70)
    print(f"ARTICLE FOURIER AND KINETIC ACCURACY: {args.model}")
    print("=" * 70)
    print("  all-grid temporal Fourier maximum:")
    print(f"    time samples = {point_times.size}, grid points = {const.Nz}")
    print("    spectrum = <|FFT_t[dn_i(z,t) - <dn_i>_z]|^2>_z, Hann window")
    if np.isfinite(fft_omega):
        print(f"    omega_FFT = {fft_omega:.9e}, theory = "
              f"{ctx['omega0'].real:.9e}, error = "
              f"{fft_omega_error:+.4f} %")
        print(f"    period_FFT = {fft_period:.9e}, theory = "
              f"{ctx['T_wave']:.9e}, error = {fft_period_error:+.4f} %")
        print(f"    native angular-frequency resolution = "
              f"{fft_native_resolution:.9e}")
        print(f"    zero-padded angular-frequency sampling step = "
              f"{fft_omega_step:.9e}")
    else:
        print("    frequency unavailable: invalid or insufficient time samples")
    print("  first harmonic vs exact kinetic theory:")
    print(f"    relative L2 error       = {first_harmonic_l2_error:.6f} %")
    print(f"    maximum relative error = {first_harmonic_max_error:.6f} %")
    print(f"Article comparison figure written to {article_path}")
    if spectrum_path is not None:
        print(f"Fourier spectrum figure written to {spectrum_path}")


def run_compare(args):
    """Compare ion first-harmonic amplitudes from --model and --compare runs."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(tests_dir, "..", ".."))
    for path in (os.path.join(repo_dir, "tools"),
                 os.path.join(tests_dir, "drift_kinetic_tools")):
        if path not in sys.path:
            sys.path.insert(0, path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from lib.plot import bbox, labelsize, ticksize
    dz = _DensityZTools

    plt.rc("text", usetex=False)
    runs = [load_ion_harmonic(name, args, dz)
            for name in [args.model, *args.compare]]
    base = runs[0]

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "v", "P")
    for i, run in enumerate(runs):
        ax.plot(run["time"], run["amplitude"],
                color=colors[i % len(colors)], marker=markers[i % len(markers)],
                linestyle="-", linewidth=2.2, markersize=4.5,
                label=run["label"])

    x_max = max(float(run["time"][-1]) for run in runs)
    theory_start = base["t0"] / base["T_wave"]
    if x_max >= theory_start:
        t_theory_T = np.linspace(theory_start, x_max, 500)
        t_theory = t_theory_T * base["T_wave"]
        dn_exponential = base["a0"] * np.exp(
            -base["Gamma"] * (t_theory - base["t0"]))
        # Theory is deliberately excluded from the legend: it should contain
        # only the compact exN labels requested for the simulation runs.
        ax.plot(t_theory_T, dn_exponential, color="black", linestyle="--",
                linewidth=1.8, alpha=0.8, label="_nolegend_")
    else:
        dn_exponential = np.empty(0)

    all_maxima = [float(np.nanmax(run["amplitude"])) for run in runs]
    if dn_exponential.size:
        all_maxima.append(float(np.nanmax(dn_exponential)))
    amp_hi_lim = max(0.06, 1.25 * max(all_maxima))
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, amp_hi_lim)
    ax.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax.set_ylabel(
        r"$|\delta n_{i,1}(t)|/n_i = \left|\frac{2}{L_z}\int"
        r"\left(\langle n_i\rangle_{x,y}/n_i - 1\right)e^{-ikz}dz\right|$",
        fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    ax.grid(True, alpha=0.3)
    ax.set_title(r"Амплитуда первой гармоники ионов",
                 fontsize=labelsize, bbox=bbox)
    ax.set_box_aspect(1)
    # Legend outside, to the right: the test names are long and would otherwise
    # cover the curves.  tight_layout ignores an out-of-axes legend, so the
    # figure is saved with bbox_inches="tight" to keep it inside the PNG.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=ticksize,
              borderaxespad=0.0, framealpha=0.9)
    fig.tight_layout()

    out_dir = os.path.join(base["out_dir"], args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "ion_sound_compare.png")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    log_fig, log_ax = plt.subplots(figsize=(8.5, 7.0))
    for i, run in enumerate(runs):
        log_ax.plot(run["time"], run["amplitude"],
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], linestyle="-",
                    linewidth=2.2, markersize=4.5, label=run["label"])
    if dn_exponential.size:
        log_ax.plot(t_theory_T, dn_exponential, color="black",
                    linestyle="--", linewidth=1.8, alpha=0.8,
                    label="_nolegend_")
    log_ax.set_yscale("log")
    log_ax.set_xlim(0.0, x_max)
    log_ax.set_xlabel(r"$t/T$", fontsize=labelsize)
    log_ax.set_ylabel(r"$|\delta n_{i,1}(t)|/n_i$", fontsize=labelsize)
    log_ax.tick_params(labelsize=ticksize)
    log_ax.grid(True, which="both", alpha=0.3)
    log_ax.set_title(r"Амплитуда первой гармоники ионов (логарифмический масштаб)",
                     fontsize=labelsize, bbox=bbox)
    log_ax.set_box_aspect(1)
    log_ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=ticksize, borderaxespad=0.0, framealpha=0.9)
    log_fig.tight_layout()
    log_path = os.path.join(out_dir, "ion_sound_compare_log.png")
    log_fig.savefig(log_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(log_fig)

    # Full real-space perturbation amplitude.  Unlike the classical first
    # harmonic this includes every resolved z mode and the particle floor.
    total_fig, total_ax = plt.subplots(figsize=(8.5, 7.0))
    total_maxima = []
    for i, run in enumerate(runs):
        color = colors[i % len(colors)]
        total_ax.plot(run["time"], run["total_amplitude"], color=color,
                      marker=markers[i % len(markers)], linestyle="-",
                      linewidth=2.2, markersize=4.5, label=run["label"])
        theory_start = run["t0"] / run["T_wave"]
        theory_end = float(run["time"][-1])
        if theory_end >= theory_start:
            theory_time_T = np.linspace(theory_start, theory_end, 500)
            theory_amplitude = run["a0"] * np.exp(
                -run["Gamma"] *
                (theory_time_T * run["T_wave"] - run["t0"]))
            total_ax.plot(theory_time_T, theory_amplitude, color=color,
                          linestyle="--", linewidth=1.7, alpha=0.8,
                          label="_nolegend_")
            total_maxima.append(float(np.nanmax(theory_amplitude)))
        total_maxima.append(float(np.nanmax(run["total_amplitude"])))
    total_ax.set_xlim(0.0, x_max)
    total_ax.set_ylim(0.0, max(0.06, 1.25 * max(total_maxima)))
    total_ax.set_xlabel(r"$t/T$", fontsize=labelsize)
    total_ax.set_ylabel(
        r"$\sqrt{2\langle(n_i/n_{i0}-1)^2\rangle_z}$", fontsize=labelsize)
    total_ax.tick_params(labelsize=ticksize)
    total_ax.grid(True, alpha=0.3)
    total_ax.set_title(r"Полная амплитуда возмущения плотности ионов",
                       fontsize=labelsize, bbox=bbox)
    total_ax.set_box_aspect(1)
    total_handles = [Line2D([], [], color=colors[i % len(colors)],
                            marker=markers[i % len(markers)], linewidth=2.2,
                            label=run["label"])
                     for i, run in enumerate(runs)]
    total_handles.append(Line2D([], [], color="black", linestyle="--",
                                linewidth=1.7,
                                label=r"$|\delta n_i(0)|e^{-\Gamma t}$"))
    total_ax.legend(handles=total_handles, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), fontsize=ticksize,
                    borderaxespad=0.0, framealpha=0.9)
    total_fig.tight_layout()
    total_path = os.path.join(
        out_dir, "ion_sound_compare_total_amplitude.png")
    total_fig.savefig(total_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(total_fig)

    detail_paths = plot_density_harmonics_and_kinetic_energy(
        runs, out_dir, args.dpi)

    print("=" * 70)
    print(f"COMPARE: {args.model} + {len(args.compare)} run(s)")
    print("=" * 70)
    for run in runs:
        print(f"  {run['label']:8s} {run['testname']}: "
              f"{run['time'].size} frames, t/T in "
              f"[{run['time'][0]:.6g}, {run['time'][-1]:.6g}]")
    print(f"  theory: {base['label']} exponential, "
          f"Gamma = {base['Gamma']:.6e}")
    print(f"Comparison figure written to {png_path}")
    print(f"Log-scale comparison figure written to {log_path}")
    print(f"Total-amplitude comparison figure written to {total_path}")
    for detail_path in detail_paths.values():
        print(f"Comparison figure written to {detail_path}")


def run_model_adv(args):
    """Two-branch (+-omega) decomposition of the ion first harmonic.

    |dn_i,1(t)| measured in a PIC run does not decay as a clean e^{-Gamma t}:
    it wobbles with period T/2 around the exponential.  The cause is that BOTH
    ion-acoustic branches, +w_s - i Gamma and -w_s - i Gamma, are excited --
    partly by the residual mismatch of the loaded initial condition to the exact
    kinetic eigenmode, and dominantly at late times by discrete-particle noise,
    which is broadband and therefore feeds the counter-propagating branch too.
    Their interference beats at 2 w:

        |dn_1(t)| = e^{-Gamma t} sqrt(|A_p|^2 + |A_m|^2
                                      + 2 |A_p| |A_m| cos(2 w t + dphi)),

    with fringe visibility 2r/(1 + r^2) and peak/trough (1 + r)/(1 - r), where
    r = |A_m| / |A_p|.

    This mode fits that model to the COMPLEX harmonic (see `fit_two_branch`) and
    plots |A_p| e^{-Gamma t}: the envelope of the travelling wave alone, free of
    the beat and unbiased by noise.  Gamma comes out of the fit rather than out
    of a log-fit of the beating modulus.  The noise floor read off the harmonics
    that carry no signal is drawn alongside, because |A_m| is only meaningful
    while it stays above it.
    """
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(tests_dir, "..", ".."))
    for path in (os.path.join(repo_dir, "tools"),
                 os.path.join(tests_dir, "drift_kinetic_tools")):
        if path not in sys.path:
            sys.path.insert(0, path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib.plot import bbox, labelsize, ticksize
    dz = _DensityZTools

    plt.rc("text", usetex=False)  # Cyrillic panel titles are incompatible

    runs = []
    for testname in args.model_adv:
        ctx = prepare_theory(testname, ic_frame=args.ic_from_dump)
        ion, T_wave = ctx["ion"], ctx["T_wave"]
        omega_s, Gamma_theory = ctx["omega0"].real, -ctx["omega0"].imag
        times, harmonic, noise = harmonic_series(ctx, dz, ion.name,
                                                 args.model_tmax)
        omega, Gamma, A_p, A_m, residual = fit_two_branch(
            times, harmonic, omega_s, Gamma_theory)
        runs.append(dict(
            testname=testname, label=compare_label(testname),
            time=times / T_wave, amplitude=np.abs(harmonic), noise=noise,
            envelope=abs(A_p) * np.exp(-Gamma * times),
            model=np.abs(A_p * np.exp(-1j * omega * times - Gamma * times)
                         + A_m * np.exp(+1j * omega * times - Gamma * times)),
            T_wave=T_wave, omega_s=omega_s, Gamma_theory=Gamma_theory,
            omega=omega, Gamma=Gamma, A_p=A_p, A_m=A_m, residual=residual,
            out_dir=ctx["const"].out_dir))

    base = runs[0]
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "v", "P")

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    for i, run in enumerate(runs):
        color = colors[i % len(colors)]
        ax.plot(run["time"], run["amplitude"], color=color,
                marker=markers[i % len(markers)], linestyle="-", linewidth=1.6,
                markersize=3.5, alpha=0.85, label=run["label"])
        # One shared legend entry below explains the dashed envelopes; labelling
        # every run twice would double an already long legend.
        ax.plot(run["time"], run["envelope"], color=color, linestyle="--",
                linewidth=2.0, label="_nolegend_")
    ax.plot([], [], color="tab:gray", linestyle="--", linewidth=2.0,
            label=r"штрих: $|A_+|e^{-\Gamma t}$")
    ax.plot(base["time"], base["model"], color="black", linestyle="-",
            linewidth=1.6, alpha=0.8,
            label=r"$|A_+e^{-i\omega t}+A_-e^{+i\omega t}|e^{-\Gamma t}$")
    ax.plot(base["time"], base["noise"], color="tab:gray", linestyle=":",
            linewidth=1.8, label=rf"{base['label']}: шум по несигнальным $m$")

    drawn = np.concatenate([run["amplitude"] for run in runs])
    drawn = drawn[np.isfinite(drawn) & (drawn > 0.0)]
    ax.set_yscale("log")
    ax.set_xlim(0.0, max(float(run["time"][-1]) for run in runs))
    # The noise floor starts at ~0 (a quiet start loads the density exactly) and
    # saturates within a fraction of a period, so bound the axis by its median
    # rather than by that first, meaningless point.
    ax.set_ylim(0.3 * float(np.median(base["noise"])), 3.0 * float(drawn.max()))
    ax.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax.set_ylabel(
        r"$|\delta n_{i,1}(t)|/n_i = \left|\frac{2}{L_z}\int"
        r"\left(\langle n_i\rangle_{x,y}/n_i - 1\right)e^{-ikz}dz\right|$",
        fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(r"Разложение на встречные ветви", fontsize=labelsize, bbox=bbox)
    # Log scale leaves the band below the noise floor empty.
    ax.legend(loc="lower right", fontsize=ticksize, framealpha=0.9)
    fig.tight_layout()

    out_dir = os.path.join(base["out_dir"], args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "ion_sound_two_branch.png")
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)

    print("=" * 70)
    print(f"MODEL_ADV: two-branch fit of {len(runs)} run(s)")
    print("=" * 70)
    print("  a(t) = A_p exp(-i w t - g t) + A_m exp(+i w t - g t),  r = |A_m/A_p|")
    for run in runs:
        r = abs(run["A_m"]) / abs(run["A_p"]) if run["A_p"] else math.nan
        span = float(run["time"][-1])
        # Median, not mean: the floor is ~0 at t = 0 (quiet start) and only
        # saturates once the random thermal velocities have streamed.
        noise = float(np.median(run["noise"]))
        print(f"  {run['label']:8s} {run['testname']}")
        print(f"    frames = {run['time'].size} , t/T in "
              f"[{run['time'][0]:.4g}, {span:.4g}] , T = {run['T_wave']:.6g}")
        print(f"    w   = {run['omega']:.6e}  = {run['omega']/run['omega_s']:.4f} w_s")
        print(f"    G   = {run['Gamma']:.6e}  = "
              f"{run['Gamma']/run['Gamma_theory']:.4f} Gamma_theory")
        print(f"    |A_p| = {abs(run['A_p']):.6e} , |A_m| = {abs(run['A_m']):.6e}"
              f" , r = {r:.4f}")
        print(f"    beat: period = {math.pi/run['omega']/run['T_wave']:.4f} T , "
              f"visibility = {2.0*r/(1.0+r*r):.4f} , "
              f"peak/trough = {(1.0+r)/(1.0-r) if r < 1.0 else math.inf:.4f}")
        print(f"    relative residual = {run['residual']:.4f} , "
              f"noise floor = {noise:.6e}")
        if span < 1.0:
            print("    [warn] window shorter than one wave period: the two "
                  "branches are nearly collinear, r is unreliable.")
        if abs(run["A_m"]) < noise:
            print("    [warn] |A_m| is below the noise floor: the "
                  "counter-propagating branch is not resolved above the "
                  "discrete-particle noise.")
    print(f"Two-branch figure written to {png_path}")


def density_view_dir(config, species):
    """Output sub-directory of a species density diagnostic (or its default)."""
    for diagnostic in config.get("Diagnostics", []) or []:
        if diagnostic.get("diagnostic") == "DistributionMoment" and \
                diagnostic.get("particles") == species and \
                diagnostic.get("moment") == "density":
            return diagnostic.get("out_dir") or os.path.join(species, "density")
    return os.path.join(species, "density")


def resolve_output_dir(const, config_dir, subdir):
    """Resolve a diagnostic directory against configured and local run roots."""
    if os.path.isabs(subdir) and os.path.isdir(subdir):
        return subdir
    for root in (const.in_dir, config_dir):
        path = os.path.join(root, subdir)
        if os.path.isdir(path):
            return path
    return None


def numeric_frames(path, expected_bytes):
    """Return {step: filename} for complete numeric binary frames in `path`."""
    if path is None:
        return {}
    return {
        int(name): name
        for name in os.listdir(path)
        if name.isdigit()
        and os.path.isfile(os.path.join(path, name))
        and os.path.getsize(os.path.join(path, name)) == expected_bytes
    }


def load_current_temperature(testname, args):
    r"""Load the longitudinal resolved-flow temperature inferred from J.

    At every grid point we form the density-normalized bulk velocity

        u_z = J_z / (q n),

    then calculate its density-weighted central second moment,

        T_parallel,J = m <(u_z - <u_z>_n)^2>_n.

    This is the kinetic temperature of the spatially resolved bulk-velocity
    fluctuations (the coherent wave and grid-scale current noise).  A current
    density is only a first particle moment, so it cannot recover the full
    microscopic particle temperature m(<v_z^2>-<v_z>^2).
    """
    ctx = prepare_theory(testname)
    const, config = ctx["const"], ctx["config"]
    config_dir = ctx["config_dir"]
    shape = (const.Nz, const.Ny, const.Nx)
    scalar_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    vector_bytes = 3 * scalar_bytes
    series = {}

    for species in ctx["species"]:
        density_subdir = density_view_dir(config, species.name)
        current_subdir = field_view_dir(config, f"{species.name}/J") \
            or os.path.join(species.name, "J")
        density_dir = resolve_output_dir(const, config_dir, density_subdir)
        current_dir = resolve_output_dir(const, config_dir, current_subdir)
        if density_dir is None or current_dir is None:
            raise SystemExit(
                f"--compare-temp: need density and J diagnostics for "
                f"'{species.name}' in '{testname}' (density={density_subdir}, "
                f"J={current_subdir}).")

        density_frames = numeric_frames(density_dir, scalar_bytes)
        current_frames = numeric_frames(current_dir, vector_bytes)
        # J frame 0000 is zero because no step has yet been deposited.  It is
        # not a physical temperature sample and would spoil relative changes.
        common = sorted((set(density_frames) & set(current_frames)) - {0})
        if args.model_tmax is not None:
            t_limit = args.model_tmax * ctx["T_wave"]
            tolerance = 1.0e-12 * max(1.0, t_limit)
            common = [idx for idx in common
                      if idx * const.dts <= t_limit + tolerance]
        if not common:
            raise SystemExit(
                f"--compare-temp: no common nonzero density/J frames for "
                f"'{species.name}' in '{testname}'.")

        times, temperatures = [], []
        for idx in common:
            density = np.fromfile(
                os.path.join(density_dir, density_frames[idx]),
                dtype=np.float32, count=int(np.prod(shape))).reshape(shape)
            current = np.fromfile(
                os.path.join(current_dir, current_frames[idx]),
                dtype=np.float32, count=3 * int(np.prod(shape))).reshape(*shape, 3)

            valid = np.isfinite(density) & np.isfinite(current[..., 2]) \
                & (density > max(abs(species.n) * 1.0e-12, 1.0e-30))
            if not np.any(valid):
                continue
            n = density[valid].astype(np.float64)
            uz = current[..., 2][valid].astype(np.float64) / (species.q * n)
            weight = float(np.sum(n))
            mean_uz = float(np.sum(n * uz) / weight)
            variance = float(np.sum(n * (uz - mean_uz) ** 2) / weight)
            temperatures.append(species.m * variance * MEC2_KEV)
            times.append(idx * const.dts / ctx["T_wave"])

        if not times:
            raise SystemExit(
                f"--compare-temp: all density/J frames for '{species.name}' "
                f"in '{testname}' are empty or invalid.")
        series[species.name] = {
            "time": np.asarray(times),
            "temperature": np.asarray(temperatures),
            "configured_temperature": species.T * MEC2_KEV,
        }

    return {
        "testname": testname,
        "label": compare_label_with_particles(testname, config),
        "series": series,
        "out_dir": const.out_dir,
    }


def run_compare_temp(args):
    """Compare electron/ion longitudinal velocity variance inferred from J."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(tests_dir, "..", ".."))
    tools_dir = os.path.join(repo_dir, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib.plot import bbox, labelsize, ticksize

    plt.rc("text", usetex=False)
    runs = [load_current_temperature(name, args)
            for name in [args.model, *args.compare_temp]]

    fig, (ax_temp, ax_relative) = plt.subplots(
        2, 1, figsize=(10.0, 10.0), sharex=True)
    colors = plt.get_cmap("tab10").colors
    linestyles = {"electrons": "-", "ions": "--"}
    species_symbols = {"electrons": "e", "ions": "i"}
    positive_temperatures = []

    for run_idx, run in enumerate(runs):
        color = colors[run_idx % len(colors)]
        for species_name, values in run["series"].items():
            time = values["time"]
            temperature = values["temperature"]
            symbol = species_symbols.get(species_name, species_name)
            style = linestyles.get(species_name, "-")
            label = rf"{run['label']}, ${symbol}$"
            ax_temp.plot(time, temperature, color=color, linestyle=style,
                         linewidth=2.2, label=label)
            delta = temperature - temperature[0]
            ax_relative.plot(time, delta, color=color, linestyle=style,
                             linewidth=2.2, label=label)
            positive_temperatures.extend(
                temperature[np.isfinite(temperature) & (temperature > 0.0)])

    if positive_temperatures:
        spread = max(positive_temperatures) / min(positive_temperatures)
        if spread >= 20.0:
            ax_temp.set_yscale("log")
    ax_temp.set_ylabel(r"$T_{\parallel,J}$ [keV]", fontsize=labelsize)
    ax_temp.set_title(r"Продольная температура из $J_z$",
                      fontsize=labelsize, bbox=bbox)
    ax_temp.grid(True, alpha=0.3, which="both")
    ax_temp.tick_params(labelsize=ticksize)
    ax_temp.legend(loc="best", fontsize=ticksize, ncol=2)

    ax_relative.axhline(0.0, color="black", linewidth=0.9, alpha=0.6)
    all_deltas = [
        np.abs(values["temperature"] - values["temperature"][0])
        for run in runs for values in run["series"].values()
    ]
    delta_max = max(float(np.nanmax(delta)) for delta in all_deltas)
    if delta_max > 0.0:
        ax_relative.set_yscale("symlog", linthresh=max(delta_max * 3.0e-2,
                                                        1.0e-12))
    ax_relative.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_relative.set_ylabel(r"$T_{\parallel,J}(t)-T_{\parallel,J}(t_1)$ [keV]",
                           fontsize=labelsize)
    ax_relative.set_title(r"Изменение $T_{\parallel,J}$ (обмен энергией)",
                          fontsize=labelsize, bbox=bbox)
    ax_relative.grid(True, alpha=0.3)
    ax_relative.tick_params(labelsize=ticksize)

    x_max = max(float(values["time"][-1])
                for run in runs for values in run["series"].values())
    ax_relative.set_xlim(0.0, x_max)
    fig.tight_layout()

    out_dir = os.path.join(runs[0]["out_dir"], args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "ion_sound_compare_temp.png")
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)

    print("=" * 70)
    print(f"COMPARE_TEMP: {args.model} + {len(args.compare_temp)} run(s)")
    print("=" * 70)
    print("  T_parallel,J = m <(J_z/(q n) - <J_z/(q n)>_n)^2>_n")
    print("  This is resolved bulk-flow/noise energy, not the full particle "
          "second-moment temperature.")
    for run in runs:
        for species_name, values in run["series"].items():
            temperature = values["temperature"]
            change = temperature[-1] / temperature[0] - 1.0 \
                if temperature[0] != 0.0 else math.nan
            print(f"  {run['label']:8s} {species_name:9s}: "
                  f"{temperature.size} frames, "
                  f"T_J = {temperature[0]:.6e} -> {temperature[-1]:.6e} keV, "
                  f"change = {change:+.3%}, "
                  f"T_config = {values['configured_temperature']:.6e} keV")
    print(f"Temperature comparison figure written to {png_path}")


def plot_electric_mode_amplitudes(times, profiles, z, Lz, T_wave, out_path, dpi,
                                  modes=(1, 2, 3, 4, 5), theory=None):
    """Separate figure: |E_z,m(t)| of the first `modes` z harmonics on one axis.

    Every harmonic uses the same convention as the main comparison,
    |E_z,m| = |2/Lz int <E_z>_{x,y} e^{-i k_m z} dz|,  k_m = 2 pi m / Lz,
    so the m-th curve is directly comparable with the theory of that mode.
    Higher harmonics are normally orders of magnitude below the driven one
    (they only grow through nonlinearity and particle noise), hence the log
    ordinate.  `theory` is an optional (t, |E_1|, mode) triple, drawn dashed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib.plot import bbox, labelsize, ticksize
    plt.rc("text", usetex=False)  # the Cyrillic title is incompatible with usetex

    colors = ["purple", "tab:red", "tab:green", "tab:orange", "tab:blue",
              "tab:brown", "tab:pink"]

    fig, ax = plt.subplots(figsize=(11.0, 7.5))
    amplitudes = {}
    floor = np.inf
    ceiling = 0.0
    for i, m in enumerate(modes):
        kernel = np.exp(-1j * (2.0 * math.pi * m / Lz) * z)
        amp = np.array([abs(2.0 * np.mean(pz * kernel)) for pz in profiles])
        amplitudes[m] = amp
        positive = amp[amp > 0.0]
        if positive.size:
            floor = min(floor, float(positive.min()))
            ceiling = max(ceiling, float(positive.max()))
        ax.plot(times / T_wave, np.where(amp > 0.0, amp, np.nan),
                color=colors[i % len(colors)], linewidth=2.0,
                label=rf"$|E_{{z,{m}}}|$ (model)")

    if theory is not None:
        t_theory, dE_theory, theory_mode = theory
        ax.plot(t_theory / T_wave, dE_theory, color="black", linestyle="--",
                linewidth=2.0,
                label=rf"$|E_{{z,{theory_mode:g}}}|$ (theory)")
        positive = dE_theory[dE_theory > 0.0]
        if positive.size:
            floor = min(floor, float(positive.min()))
            ceiling = max(ceiling, float(positive.max()))

    if np.isfinite(floor) and ceiling > 0.0:
        ax.set_yscale("log")
        # Deep noise floors of the high harmonics would squash the driven mode;
        # keep at most six decades below its peak.
        ax.set_ylim(max(floor, ceiling * 1e-6) * 0.5, ceiling * 2.0)
    ax.set_xlim(float(times[0]) / T_wave, float(times[-1]) / T_wave)
    ax.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax.set_ylabel(
        r"$|E_{z,m}(t)| = \left|\frac{2}{L_z}\int"
        r"\langle E_z\rangle_{x,y}e^{-ik_m z}\,dz\right|$",
        fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(r"Амплитуды мод $m = 1\ldots%d$" % max(modes),
                 fontsize=labelsize, bbox=bbox)
    # Six overlapping curves leave no free corner; keep the legend off the axes.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, fontsize=0.8 * ticksize)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return amplitudes


def run_model_electric(args):
    # Frame 0000 contains the field prescribed by SetElectricField, before the
    # first self-consistent particle/field step.  For the electric comparison
    # the default IVP therefore starts from the realized frame 0001.
    electric_ic_frame = 1 if args.ic_from_dump is None else args.ic_from_dump
    ctx = prepare_theory(args.model_electric, ic_frame=electric_ic_frame)
    const, config = ctx["const"], ctx["config"]
    species = ctx["species"]
    k, omega0 = ctx["k"], ctx["omega0"]
    cn_hat, u_hat, E0, ic = ctx["cn_hat"], ctx["u_hat"], ctx["E0"], ctx["ic"]
    initial_distribution = ctx["initial_distribution"]
    t0 = ctx["t0"]
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

    frames = sorted((int(n), n) for n in os.listdir(epath)
                    if n.isdigit() and int(n) >= electric_ic_frame)
    if not frames:
        raise SystemExit(
            f"no E frames at or after {electric_ic_frame:04d} in {epath}.")
    if frames[0][0] != electric_ic_frame:
        raise SystemExit(
            f"initial E frame {electric_ic_frame:04d} not found in {epath}.")
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
    # transverse cells first, then retain only the requested z harmonic.
    # Unlike the former L2 metric, this excludes the DC component and every
    # other longitudinal mode from the comparison.
    # E_z lives on z faces, whereas density and the phase-space histogram live
    # at cell centres.  Their Fourier amplitudes must use their own staggered
    # coordinates; the modulus is unchanged, but the complex phase supplied to
    # the Vlasov--Ampere IVP is not.
    z = (np.arange(const.Nz) + 0.5) * const.dz
    first_harmonic_kernel = np.exp(-1j * k * z)
    profiles = [Ez_profile(n) for n in names]
    E_sim = np.array([2.0 * np.mean(pz * first_harmonic_kernel)
                      for pz in profiles])
    dE_sim = np.abs(E_sim)
    initial_field = complex(E_sim[0])

    # E is evolved from the saved first harmonic by longitudinal Ampere.  This
    # preserves the realized finite-particle Gauss-law mismatch instead of
    # replacing E(0001) immediately by rho(0001)/(ik).
    t_grid, nhat, E_theory = solve_vlasov_poisson(
        species, cn_hat, u_hat, k,
        max(t_max - t0, T_wave * 1e-3),
        initial_distribution=initial_distribution,
        initial_field=initial_field)
    t_grid = t_grid + t0
    dE_theory = np.abs(E_theory)

    # ---- Summary ---------------------------------------------------------- #
    print("=" * 70)
    print(f"MODEL_ELECTRIC: {args.model_electric}")
    print("=" * 70)
    print(f"  k = {k:.6e} (mode {ctx['wn']:.0f}, Lz = {const.Lz:.6g})")
    print(f"  omega_s = {omega_r:.6e} ,  Gamma_s = {Gamma:.6e} ,  T = {T_wave:.6g}")
    print(f"  E0 (configured frame 0000) = {E0:.6e}")
    for s in species:
        a_n, phi_n, C_u, phi_u = ic[s.name]
        print(f"  {s.name:9s}: a_n = {a_n:.6e}, phi_n = {phi_n:+.6f} ,"
              f"  C_u = {C_u:.6e}, phi_u = {phi_u:+.6f}")
    if ctx["ic_measured"] is not None:
        print(f"  IC measured from frame {ctx['ic_measured']:04d} "
              f"(t0 = {t0:.6g} = {t0 / T_wave:.4g} T)")
        for s in species:
            print(f"  {s.name:9s}: |cn_hat| = {abs(cn_hat[s.name]):.6e} , "
                  f"|u_hat| = {abs(u_hat[s.name]):.6e}   (realized)")
    print(f"  E frames = {len(names)} in '{edir_name}' , "
          f"t in [{times[0]:.6g}, {t_max:.6g}]")
    i0 = int(np.argmin(np.abs(times - t0)))
    print(f"  E_1(t0): theory/model = "
          f"{E_theory[0].real:+.6e}{E_theory[0].imag:+.6e}j")
    print(f"  |E_1|(t0): theory = {dE_theory[0]:.6e} , "
          f"model = {dE_sim[i0]:.6e}")
    print()

    # ---- Figure: left = E_z(z) profile, right = |E_1|(t) amplitude -------- #
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
                              linewidth=2.5, markersize=4.0,
                              zorder=3,
                              label=r"$|E_{z,1}|$ (model)")
    ax_amp.plot(t_grid / T_wave, dE_theory, color="black", linewidth=2.0,
                zorder=2,
                label=r"$|E_{z,1}|$ (theory, Vlasov--Ampere)")
    ax_amp.set_xlim(t0 / T_wave, t_max / T_wave)
    ax_amp.set_ylim(0.0, amp_hi_lim)
    ax_amp.set_xlabel(r"$t/T$", fontsize=labelsize)
    ax_amp.set_ylabel(
        r"$|E_{z,1}(t)| = \left|\frac{2}{L_z}\int"
        r"\langle E_z\rangle_{x,y}e^{-ikz}\,dz\right|$",
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

    # ---- Separate figure: harmonics m = 1..5 of E_z on one axis ------------ #
    modes = (1, 2, 3, 4, 5)
    theory_curve = (t_grid, dE_theory, ctx["wn"]) \
        if float(ctx["wn"]).is_integer() and ctx["wn"] in modes else None
    modes_path = os.path.join(out_dir, "ion_sound_model_E_modes.png")
    amplitudes = plot_electric_mode_amplitudes(
        times, profiles, z, const.Lz, T_wave, modes_path, args.dpi,
        modes=modes, theory=theory_curve)
    print("  z-harmonics of E_z  (first frame -> last frame):")
    for m in modes:
        amp = amplitudes[m]
        print(f"    m = {m}: {amp[0]:.6e} -> {amp[-1]:.6e} , "
              f"max = {amp.max():.6e}")
    print(f"Mode-amplitude figure written to {modes_path}")


# --------------------------------------------------------------------------- #
# Phase-space mode: f(z, v_parallel) from the 5-D DK histogram                 #
# --------------------------------------------------------------------------- #
def moment_matched_kinetic_parallel(species, z, v, loader,
                                    density_amplitude, density_phase, Lz):
    """Velocity-bin-averaged positive M0/M1/M2 ion-sound quasimode.

    This mirrors KineticIonSoundMomentsQuiet.  Its local Gaussian has density,
    particle flux, and parallel second moment fixed by the Landau frequency
    and the linear Vlasov moment hierarchy.  The exponentially tiny change
    caused by the C++ loader's |v|<c rejection is intentionally omitted.
    """
    mode = float(loader.get("wave_number", [0.0, 0.0, 1.0])[2])
    field_phase = float(loader.get(
        "field_phase", [0.0, 0.0, 0.0])[2])
    z_min = float(loader.get("min", [0.0, 0.0, 0.0])[2])
    k = 2.0 * math.pi * mode / Lz
    omega = complex(float(loader["omega_real"]), -float(loader["gamma"]))
    E_hat = float(loader["force_electric_amplitude"]) * \
        np.exp(1j * field_phase)
    density_hat = (-1j * species.n * density_amplitude *
                   np.exp(1j * density_phase))
    flux_hat = omega * density_hat / k
    second_hat = (omega * flux_hat -
                  1j * (species.q / species.m) * species.n * E_hat) / k

    spatial_phase = np.exp(1j * k * (z - z_min))
    n_target = species.n + np.real(density_hat * spatial_phase)
    flux = np.real(flux_hat * spatial_phase)
    second = species.n * species.vT**2 + np.real(
        second_hat * spatial_phase)
    bulk = flux / n_target
    variance = second / n_target - bulk**2
    if np.any(n_target <= 0.0) or np.any(variance <= 0.0):
        raise SystemExit(
            "moment-matched kinetic PDF has non-positive density or variance")

    dv = float(v[1] - v[0]) if v.size > 1 else 2.0 * species.vT
    samples_per_bin = 32
    offsets = ((np.arange(samples_per_bin) + 0.5) /
               samples_per_bin - 0.5) * dv
    velocity_samples = v[:, None] + offsets[None, :]
    normalized = ((velocity_samples[None, :, :] - bulk[:, None, None]) /
                  np.sqrt(variance)[:, None, None])
    conditional = np.exp(-0.5 * normalized**2) / (
        math.sqrt(2.0 * math.pi) * np.sqrt(variance)[:, None, None])
    f_parallel = n_target[:, None] * conditional.mean(axis=2)
    F0 = (np.exp(-velocity_samples**2 / (2.0 * species.vT**2)) /
          (math.sqrt(2.0 * math.pi) * species.vT)).mean(axis=1)
    return f_parallel, n_target, F0


def run_phase(args):
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(tests_dir, "..", ".."))
    tools_dir = os.path.join(repo_dir, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    from lib.constants import const, init_constants

    config_path = os.path.join(
        tests_dir, "output", args.phase, "config.json")
    if not os.path.isfile(config_path):
        raise SystemExit(f"config not found: {config_path}")
    init_constants(config_path)
    config_dir = os.path.dirname(config_path)
    if not os.path.isdir(const.in_dir):
        const.in_dir = config_dir
        const.out_dir = os.path.join(config_dir, "processed")

    electron, ion = species_from_config(const.config)
    all_species = [s for s in (electron, ion) if s is not None]
    requested = set(args.species or [s.name for s in all_species])
    species = [s for s in all_species if s.name in requested]
    unknown = requested - {s.name for s in species}
    if unknown:
        raise SystemExit("unknown --species: " + ", ".join(sorted(unknown)))
    if not species:
        raise SystemExit("no species selected for --phase")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(len(species), 3,
                             figsize=(16.0, 5.0 * len(species)),
                             squeeze=False)
    summaries = []
    parallel_marginals = []

    for row, sp in enumerate(species):
        diagnostic = distribution_diagnostic_for_species(const.config, sp.name)
        loader = kinetic_loader_for_species(const.config, sp.name)
        preset = preset_for_species(const.config, sp.name)
        if diagnostic is None:
            raise SystemExit(
                f"no DkDistributionFunction diagnostic for '{sp.name}'")
        if loader is None or preset is None:
            raise SystemExit(
                f"no supported kinetic ion-sound loader for '{sp.name}'")

        vinfo = diagnostic["v_parallel"]
        muinfo = diagnostic["mu_p"]
        nv, nmu = int(vinfo["bins"]), int(muinfo["bins"])
        vmin, vmax = float(vinfo["min"]), float(vinfo["max"])
        mumin, mumax = float(muinfo["min"]), float(muinfo["max"])
        dv, dmu = (vmax - vmin) / nv, (mumax - mumin) / nmu
        v = vmin + (np.arange(nv) + 0.5) * dv
        z = (np.arange(const.Nz) + 0.5) * const.dz

        subdir = diagnostic.get(
            "out_dir", os.path.join(sp.name, "distribution_function"))
        candidates = [os.path.join(const.in_dir, subdir,
                                   f"{args.phase_frame:04d}"),
                      os.path.join(config_dir, subdir,
                                   f"{args.phase_frame:04d}")]
        frame_path = next((path for path in candidates if os.path.isfile(path)),
                          None)
        if frame_path is None:
            raise SystemExit(
                f"phase-space frame {args.phase_frame:04d} not found for "
                f"'{sp.name}' in '{subdir}'")

        expected_size = const.Nz * const.Ny * const.Nx * nv * nmu
        if os.path.getsize(frame_path) != expected_size * 4:
            raise SystemExit(
                f"unexpected size of {frame_path}: expected {expected_size * 4} bytes")
        raw = np.fromfile(frame_path, dtype=np.float32, count=expected_size)
        data = raw.reshape(const.Nz, const.Ny, const.Nx, nv, nmu)
        f_model = data.sum(axis=4).mean(axis=(1, 2)) * dmu

        coord = preset.get("coordinate", {})
        amplitude = float(coord.get("amplitude", [0.0, 0.0, 0.0])[2])
        density_phase = float(coord.get("phase", [0.0, 0.0, 0.0])[2])
        theory_label = "moment-matched theory"
        f_theory, n_target, F0 = moment_matched_kinetic_parallel(
            sp, z, v, loader, amplitude, density_phase, const.Lz)

        baseline = n_target[:, None] * F0[None, :]
        expected_markers = f_theory * dv * const.Nx * const.Ny * \
            next(float(p["Np"]) for p in const.config["Particles"]
                 if p.get("sort_name") == sp.name) / sp.n
        mask = (expected_markers >= 5.0) & (baseline > 0.0)
        delta_model = np.full_like(f_model, np.nan, dtype=float)
        delta_theory = np.full_like(f_theory, np.nan, dtype=float)
        delta_model[mask] = f_model[mask] / baseline[mask] - 1.0
        delta_theory[mask] = f_theory[mask] / baseline[mask] - 1.0
        residual = delta_model - delta_theory

        finite_reference = np.concatenate([
            np.abs(delta_model[np.isfinite(delta_model)]),
            np.abs(delta_theory[np.isfinite(delta_theory)])])
        color_limit = max(0.05, float(np.nanpercentile(
            finite_reference, 99.0))) if finite_reference.size else 1.0
        residual_limit = max(0.02, float(np.nanpercentile(
            np.abs(residual[np.isfinite(residual)]), 99.0))) \
            if np.any(np.isfinite(residual)) else 1.0

        panels = [delta_model, delta_theory, residual]
        titles = ["PIC", theory_label, "PIC - theory"]
        limits = [color_limit, color_limit, residual_limit]
        for col, (panel, title, limit) in enumerate(zip(panels, titles, limits)):
            ax = axes[row, col]
            im = ax.imshow(panel.T, origin="lower", aspect="auto",
                           extent=(0.0, const.Lz, vmin / sp.vT, vmax / sp.vT),
                           cmap="RdBu_r", norm=TwoSlopeNorm(
                               vmin=-limit, vcenter=0.0, vmax=limit))
            ax.set_title(f"{sp.name}: {title}")
            ax.set_xlabel(r"$z\;(c/\omega_{pe})$")
            if col == 0:
                ax.set_ylabel(r"$v_\parallel/v_T$")
            fig.colorbar(im, ax=ax, pad=0.02,
                         label=r"$f_\parallel/[n(z)F_0]-1$")

        n_model = np.sum(f_model, axis=1) * dv
        n_theory = np.sum(f_theory, axis=1) * dv
        u_model = np.sum(f_model * v[None, :], axis=1) * dv / \
            np.maximum(n_model, 1.0e-300)
        u_theory = np.sum(f_theory * v[None, :], axis=1) * dv / \
            np.maximum(n_theory, 1.0e-300)
        weighted_l1 = np.sum(np.abs(f_model - f_theory)) * dv / \
            max(np.sum(np.abs(f_theory)) * dv, 1.0e-300)
        weighted_rms = math.sqrt(np.mean(
            ((f_model[mask] - f_theory[mask]) /
             np.maximum(f_theory[mask], 1.0e-300))**2)) if np.any(mask) else math.nan
        mode = float(loader.get("wave_number", [0.0, 0.0, 1.0])[2])
        field_phase = float(loader.get(
            "field_phase", [0.0, 0.0, 0.0])[2])
        k = 2.0 * math.pi * mode / const.Lz
        theta = k * z + field_phase
        phase_projector = np.exp(-1j * theta)[:, None]
        conditional_model = f_model / np.maximum(n_model[:, None], 1.0e-300)
        conditional_theory = f_theory / np.maximum(n_theory[:, None], 1.0e-300)
        relative_model = conditional_model / np.maximum(F0[None, :], 1.0e-300) - 1.0
        relative_theory = conditional_theory / np.maximum(F0[None, :], 1.0e-300) - 1.0
        harmonic_model = 2.0 * np.mean(
            relative_model * phase_projector, axis=0)
        harmonic_theory = 2.0 * np.mean(
            relative_theory * phase_projector, axis=0)
        summaries.append((sp.name,
                          float(np.max(np.abs(n_model - n_target) / sp.n)),
                          float(np.max(np.abs(n_theory - n_target) / sp.n)),
                          float(np.max(np.abs(u_model - u_theory))),
                          float(weighted_l1), float(weighted_rms)))
        # Sum over all spatial z cells (and divide by Nz so the result retains
        # the units of a velocity PDF).  f_model has already been integrated
        # over mu_p and averaged over x,y above.
        parallel_marginals.append({
            "species": sp,
            "v": v,
            "model": np.mean(f_model, axis=0),
            "theory": np.mean(f_theory, axis=0),
            "maxwell": sp.n * F0,
            "resonance": float(loader["omega_real"]) / k,
            "resonance_width": float(loader["gamma"]) / abs(k),
            "harmonic_model": harmonic_model,
            "harmonic_theory": harmonic_theory,
            "theory_label": theory_label,
        })

    fig.suptitle(
        f"Ion-sound phase space, {args.phase}, frame {args.phase_frame:04d}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_dir = os.path.join(const.out_dir, "ion_sound_phase")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"ion_sound_phase_{args.phase_frame:04d}.png")
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    # A separate one-dimensional view makes the parallel marginal and its
    # tails readable without the z-dependent colour scale of the phase plot.
    # Plot v_T f/n so electron and ion curves are dimensionless and directly
    # comparable despite their very different thermal speeds.
    marginal_fig, marginal_axes = plt.subplots(
        len(parallel_marginals), 2,
        figsize=(15.0, 4.2 * len(parallel_marginals)), squeeze=False)
    for row, item in enumerate(parallel_marginals):
        sp = item["species"]
        ax = marginal_axes[row, 0]
        velocity = item["v"] / sp.vT
        scale = sp.vT / sp.n
        pic = item["model"] * scale
        theory = item["theory"] * scale
        maxwell = item["maxwell"] * scale

        ax.semilogy(velocity, pic, drawstyle="steps-mid", linewidth=1.4,
                    label=r"PIC: $\sum_z f/N_z$")
        ax.semilogy(velocity, theory, linewidth=1.8,
                    label=item["theory_label"])
        ax.semilogy(velocity, maxwell, "--", linewidth=1.3,
                    label="Maxwellian")
        resonance_vt = item["resonance"] / sp.vT
        if velocity[0] <= resonance_vt <= velocity[-1]:
            ax.axvline(resonance_vt, color="0.35", linestyle=":",
                       linewidth=1.1, label=r"$v_{res}=\omega_r/k$")
        positive = np.concatenate((pic[pic > 0.0], theory[theory > 0.0]))
        if positive.size:
            ax.set_ylim(max(float(np.min(positive)) * 0.5, 1.0e-9),
                        max(float(np.max(positive)) * 1.35, 1.0e-8))
        ax.set_xlabel(r"$v_\parallel/v_T$")
        ax.set_ylabel(r"$v_T\langle f_\parallel\rangle_z/n_0$")
        ax.set_title(f"{sp.name}: parallel-velocity marginal")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")

        # Summing over a full wavelength cancels the O(delta) kinetic
        # perturbation.  The neighbouring panel therefore isolates its first
        # spatial Fourier harmonic and zooms into omega_r/k instead of trying
        # to find the resonance in the z-summed marginal.
        resonance_ax = marginal_axes[row, 1]
        resonance = item["resonance"]
        width = item["resonance_width"]
        if item["v"][0] <= resonance <= item["v"][-1]:
            dv = float(item["v"][1] - item["v"][0])
            half_window = max(8.0 * width, 2.5 * dv)
            lo = max(float(item["v"][0] - 0.5 * dv), resonance - half_window)
            hi = min(float(item["v"][-1] + 0.5 * dv), resonance + half_window)
            selected = (item["v"] >= lo) & (item["v"] <= hi)
            resonance_ax.plot(
                item["v"][selected] / sp.vT,
                np.abs(item["harmonic_theory"][selected]), "o-",
                linewidth=1.5, markersize=5,
                label=item["theory_label"] + ", bin-averaged")
            resonance_ax.plot(
                item["v"][selected] / sp.vT,
                np.abs(item["harmonic_model"][selected]), "s",
                markersize=5, label="PIC bins")
            resonance_ax.axvline(resonance / sp.vT, color="tab:red",
                                 linestyle=":", linewidth=1.1)
            resonance_ax.set_xlim(lo / sp.vT, hi / sp.vT)
            resonance_ax.set_ylabel(r"$|\widehat{\delta f/F_0}_1|$")
            resonance_ax.legend(loc="best", fontsize="small")
        else:
            resonance_ax.text(
                0.5, 0.5,
                r"Resonance is outside the loaded range" "\n" +
                rf"$v_{{res}}/v_T={resonance / sp.vT:.2f}$, " +
                rf"$|v_\parallel|_{{max}}/v_T={max(abs(item['v'])) / sp.vT:.2f}$",
                ha="center", va="center", transform=resonance_ax.transAxes)
            resonance_ax.set_yticks([])
        resonance_ax.set_xlabel(r"$v_\parallel/v_T$")
        resonance_ax.set_title(f"{sp.name}: resonant first z harmonic")
        resonance_ax.grid(True, alpha=0.25)

    marginal_fig.suptitle(
        f"Parallel distribution summed over z, {args.phase}, "
        f"frame {args.phase_frame:04d}")
    marginal_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    marginal_path = os.path.join(
        out_dir, f"ion_sound_vparallel_{args.phase_frame:04d}.png")
    marginal_fig.savefig(marginal_path, dpi=args.dpi)
    plt.close(marginal_fig)

    print("=" * 70)
    print(f"PHASE: {args.phase}, frame {args.phase_frame:04d}")
    print("=" * 70)
    for name, dn_model, dn_theory, du, l1, rms in summaries:
        print(f"  {name:9s}: max|n_model-n_target|/n0 = {dn_model:.6e}, "
              f"quadrature = {dn_theory:.6e}")
        print(f"             max|u_model-u_theory| = {du:.6e}, "
              f"L1 = {l1:.6e}, masked RMS = {rms:.6e}")
    print(f"Phase-space figure written to {out_path}")
    print(f"Parallel marginal figure written to {marginal_path}")


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
    p.add_argument("--model-pic", default=None, metavar="TEST",
                   help="PIC-informed linear model: evolve only the first "
                        "spatial harmonic with linear Vlasov--Ampere from "
                        "the realized DkDistributionFunction and E harmonic "
                        "in one saved frame. "
                        "Defaults to frame 0001; --ic-from-dump selects another")
    p.add_argument("-compare", "--compare", nargs="+", action="append",
                   default=None,
                   metavar="TEST",
                   help="comparison mode: add up to five finished tests to "
                        "--model or --model_electric (the option may be repeated). "
                        "With --model, draws one static "
                        "plot with only the ion first-harmonic amplitudes and "
                        "the base run's exponentially damped theory. Every run "
                        "is labelled by whatever follows 'drift_kinetic_' in "
                        "its test name. With --model_electric, draws two panels "
                        "with |E_z,2|/|E_z,1| and |E_z,3|/|E_z,1|")
    p.add_argument("--article", action="store_true",
                   help="with --model and --compare, write one publication-style "
                        "three-panel ion-density comparison figure")
    p.add_argument("--article_log", "--article-log", action="store_true",
                   help="same as --article, but use logarithmic y-axes in "
                        "panels (b) and (c)")
    p.add_argument("-T", dest="article_tmax", type=float, default=None,
                   metavar="PERIODS",
                   help="article modes: upper t/T limit of panels (b) and (c); "
                        "takes precedence over --model-tmax")
    p.add_argument("--compare-temp", nargs="+", action="append", default=None,
                   metavar="TEST",
                   help="temperature-comparison mode: add up to five finished "
                        "tests to --model. Infers u_z=J_z/(q*n) and compares "
                        "the density-weighted longitudinal velocity variance "
                        "of electrons and ions")
    p.add_argument("--model_adv", nargs="+", default=None, metavar="TEST",
                   help="two-branch mode: one or more test names; fits the "
                        "COMPLEX ion first harmonic to "
                        "A_p e^{-i w t} + A_m e^{+i w t}, both damped as "
                        "e^{-Gamma t}, and plots the travelling-wave envelope "
                        "|A_p| e^{-Gamma t} free of the 2 w beat that the two "
                        "branches produce on |dn_1|. Gamma comes from the "
                        "linear least-squares problem, so it is not biased by "
                        "the beat nor by zero-mean particle noise")
    p.add_argument("--model_electric", default=None,
                   help="field-comparison mode: test name; compares the exact "
                        "theory |E_z|(t) (first harmonic) against the run's E "
                        "FieldView frames. By default the Vlasov--Ampere IVP "
                        "starts from distribution_function/0001 and E/0001; "
                        "--ic-from-dump selects another starting frame. "
                        "Additionally writes a separate figure with the "
                        "z-harmonics m = 1..5 of E_z on one axis")
    p.add_argument("--phase", default=None,
                   help="phase-space mode: test name; read the 5-D "
                        "DkDistributionFunction and compare f(z,v_parallel) "
                        "with its configured kinetic loading theory")
    p.add_argument("--phase-frame", type=int, default=0, metavar="FRAME",
                   help="phase-space diagnostic frame to read (default: 0)")
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
    p.add_argument("--mec2-kev", type=float, default=MEC2_KEV,
                   help="electron rest energy used to nondimensionalize Te/Ti "
                        f"(default: {MEC2_KEV:g} keV; use 511 for xpic code constants)")
    p.add_argument("--grid-dz", type=float, default=None,
                   help="apply the drift-kinetic S1/S2/Yee spatial coupling "
                        "for this grid spacing; prints both particle-force "
                        "and Yee-grid electric amplitudes")
    p.add_argument("--T", type=float, default=None,
                   help="theory mode: if set, additionally plot the exact kinetic "
                        "|dn(t)| (ballistic + collective) over this many wave "
                        "periods, without any simulation data")
    p.add_argument("--exact-ic", action="store_true",
                   help="load the formal upper-pole response as the theory "
                        "initial condition instead of the fluid moment IC; "
                        "this is a diagnostic quasimode, not an exact damped "
                        "real-velocity eigenfunction")
    p.add_argument("--out", default=None,
                   help="theory mode: output PNG path for the --T figure "
                        "(default: ion_sound_theory_dn.png)")
    # --- model-mode parameters ---
    p.add_argument("--species", nargs="+", default=None,
                   help="model mode: sorts to plot (default: every density diagnostic)")
    p.add_argument("--ic-from-dump", type=int, nargs="?", const=1, default=None,
                   metavar="FRAME",
                   help="model modes: start the theory from the REALIZED initial "
                        "condition measured in dump FRAME (default 1) instead of "
                        "the amplitudes requested in config.json. Needs a 3D "
                        "'<sort>/J' FieldView for the velocity moment; frame 0 is "
                        "unusable because J is deposited over a step. The theory "
                        "clock then starts at t0 = FRAME*dts. --model-pic uses "
                        "frame 1 by default and requires a phase-space dump and "
                        "E FieldView from the same frame")
    p.add_argument("--model-tmax", type=float, default=None,
                   help="model mode: draw only frames with t/T not greater than "
                        "this positive number of ion-sound periods")
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

    if args.article and args.article_log:
        p.error("--article and --article_log are mutually exclusive.")
    article_mode = args.article or args.article_log
    if article_mode and (args.model is None or args.compare is None or
                         args.model_electric is not None):
        p.error("--article/--article_log requires density --model together "
                "with --compare.")
    if args.article_tmax is not None and not article_mode:
        p.error("-T is available only together with --article or --article_log.")
    if args.article_tmax is not None and args.article_tmax <= 0.0:
        p.error("-T must be positive.")

    if args.model_pic is not None:
        conflicts = (args.model is not None or args.model_electric is not None or
                     args.model_adv is not None or args.phase is not None or
                     args.compare is not None or args.compare_temp is not None)
        if conflicts:
            p.error("--model-pic cannot be combined with other model/compare modes.")
        if args.ic_from_dump is not None and args.ic_from_dump < 0:
            p.error("--ic-from-dump must be non-negative.")
        if args.model_tmax is not None and args.model_tmax <= 0.0:
            p.error("--model-tmax must be positive.")
        run_model_pic(args)
        return

    if args.phase is not None:
        if args.phase_frame < 0:
            p.error("--phase-frame must be non-negative.")
        if args.model is not None or args.model_electric is not None or \
                args.compare is not None or args.compare_temp is not None:
            p.error("--phase cannot be combined with model/compare modes.")
        run_phase(args)
        return

    if args.model_adv is not None:
        if args.model is not None or args.model_electric is not None or \
                args.compare is not None or args.compare_temp is not None:
            p.error("--model_adv cannot be combined with model/compare modes.")
        if len(args.model_adv) > 6:
            p.error("--model_adv accepts at most six tests.")
        if args.model_tmax is not None and args.model_tmax <= 0.0:
            p.error("--model-tmax must be positive.")
        run_model_adv(args)
        return

    if args.compare_temp is not None:
        args.compare_temp = [name for group in args.compare_temp for name in group]
        if args.model is None:
            p.error("--compare-temp requires --model.")
        if args.compare is not None:
            p.error("--compare-temp cannot be combined with --compare.")
        if args.model_electric is not None:
            p.error("--compare-temp cannot be used with --model_electric.")
        if len(args.compare_temp) > 5:
            p.error("--compare-temp accepts at most five additional tests.")
        if args.model_tmax is not None and args.model_tmax <= 0.0:
            p.error("--model-tmax must be positive.")
        run_compare_temp(args)
        return

    if args.compare is not None:
        args.compare = [name for group in args.compare for name in group]
        if args.model is None and args.model_electric is None:
            p.error("--compare requires --model or --model_electric.")
        if args.model is not None and args.model_electric is not None:
            p.error("--compare accepts only one base mode: --model or "
                    "--model_electric.")
        if len(args.compare) > 5:
            p.error("--compare accepts at most five additional tests.")
        if args.model_tmax is not None and args.model_tmax <= 0.0:
            p.error("--model-tmax must be positive.")
        if args.model_electric is not None:
            if article_mode:
                p.error("--article/--article_log supports density --model "
                        "--compare only.")
            if len(args.compare) > 3:
                p.error("electric --compare accepts at most three additional "
                        "tests (four stacked panels including the base test).")
            run_compare_electric(args)
        else:
            if article_mode:
                run_compare_article(args)
            else:
                run_compare(args)
        return

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
