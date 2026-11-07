#!/usr/bin/env python3
"""Fast magnetosonic wave of hybrid_ex2_fms against the linear theory.

Setup: k = k x, B0 = B0 z, uniform plasma, Bz(x, 0) = B0 (1 + eps cos(k x)),
E = 0, Te = Ti = T. Units: c = omega_pe = e = m_e = 1, T_code = T_keV / 511.

Theory.
  Hot magnetized ions (full Vlasov, all cyclotron harmonics, k_par = 0) and
  drift-kinetic electrons (E x B drift, mu conservation, no polarization
  drift) give the dielectric tensor of the X-mode
    eps_xx = 1 + chi_i,xx,
    eps_xy = -i D,  D = D_i + 1 / (omega B0),
    eps_yy = 1 + chi_i,yy - 2 n Te k^2 / (B0^2 omega^2),
  and the dispersion relation eps_xx (eps_yy - k^2/omega^2) - D^2 = 0. For
  omega << Omega_i and k rho_i << 1 it reduces to
    omega^2 = k^2 (vA^2 + cs^2) / (1 + vA^2),
    vA^2 = B0^2 / (n m_i),  cs^2 = 2 (Te + Ti) / m_i,
  where gamma = 2 for the perpendicular compression with conserved mu.

  The initial state is not an eigenmode: n and p are uniform, B is not.
  Linearized, B/n and p_perp/(n B) are frozen into the plasma (no k_par), so
  the zero-frequency pressure-balanced part keeps the initial values of both
  invariants, and the rest is a standing fast wave. With b = dBz/B0 and
  p = n (Te + Ti), mu conservation gives dp_perp / p = 2 (b - b(0)), so the
  momentum equation n mi du/dt = -d/dx (dp_perp + B0^2 b) yields for the mode
    d^2 b/dt^2 = -omega^2 (b - b_inf),  b_inf = eps 2p / (B0^2 + 2p),
  an oscillation around the shifted pressure balance b_inf, not around zero:
    dBz_k(t) / (eps B0) = beta / (1 + beta) + cos(omega t) / (1 + beta),
    beta = 2 n (Te + Ti) / B0^2.

The mode k of Bz (averaged over y, z) is fitted by A + C cos(omega t + phi)
and compared with the theory; the run fails the check if omega or the static
fraction deviate by more than the limits below.

Usage: hybrid_ex2_fms.py [--no-check] [output directory]
"""

import argparse
import json
import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(TEST_DIR, "output", "hybrid_ex2_fms")

sys.path.append(os.path.join(TEST_DIR, "../../tools"))

from lib.plot_utils import figure, subplot
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True

MEC2_KEV = 510.99895

OMEGA_LIMIT = 0.02   # |omega_pic / omega_theory - 1|
STATIC_LIMIT = 0.05  # |A_pic - A_theory| in units of eps B0

LABELSIZE = 22
TICKSIZE = 20
LEGENDSIZE = 18
PANELSIZE = 22


class Parameters:
    def __init__(self, config):
        geometry = config["Geometry"]
        sorts = {s["sort_name"]: s for s in config["Particles"]}
        field = next(p["setter"] for p in config["Presets"]
                     if p.get("command") == "SetField" and p["field"] == "B")

        self.dx = geometry["dx"]
        self.dt = geometry["dt"]
        self.period = geometry["diagnose_period"]
        self.shape = [round(geometry[a] / geometry["d" + a]) for a in "zyx"]
        self.k = 2 * math.pi / geometry["x"]

        self.B0 = field["value"][2]
        self.eps = field["modulation"]
        self.n = sorts["ions"]["n"]
        self.mi = sorts["ions"]["m"]
        self.Ti = sorts["ions"]["T"] / MEC2_KEV
        self.Te = sorts["electrons"]["T"] / MEC2_KEV

        self.vA2 = self.B0**2 / (self.n * self.mi)
        self.cs2 = 2 * (self.Te + self.Ti) / self.mi
        self.Omega_i = self.B0 / self.mi
        self.beta = 2 * self.n * (self.Te + self.Ti) / self.B0**2


def omega_fluid(p, k):
    return k * math.sqrt((p.vA2 + p.cs2) / (1 + p.vA2))


def bessel_i_scaled(order, x, terms=60):
    """exp(-x) I_n(x) by the power series, enough for x = (k rho_i)^2 << 1."""
    order = abs(order)
    term = (0.5 * x)**order / math.factorial(order)
    total = 0.0
    for m in range(terms):
        total += term
        term *= (0.5 * x)**2 / ((m + 1) * (m + 1 + order))
    return math.exp(-x) * total


def dispersion(p, k, omega, harmonics=20):
    lam = k**2 * p.Ti / (p.mi * p.Omega_i**2)  # (k rho_i)^2
    wp2 = p.n / p.mi
    exx, D, eyy = 1.0, 1 / (omega * p.B0), 1.0
    for m in range(-harmonics, harmonics + 1):
        I = bessel_i_scaled(m, lam)
        dI = bessel_i_scaled(1, lam) if m == 0 else 0.5 * (
            bessel_i_scaled(m - 1, lam) + bessel_i_scaled(m + 1, lam))
        r = wp2 / omega / (omega - m * p.Omega_i)
        exx -= r * m * m * I / lam
        D += r * m * (dI - I)
        eyy -= r * (m * m * I / lam + 2 * lam * (I - dI))
    eyy -= 2 * p.n * p.Te * k**2 / (p.B0 * omega)**2
    return exx * (eyy - (k / omega)**2) - D**2


def omega_kinetic(p, k):
    lo, hi = 0.5 * omega_fluid(p, k), min(1.5 * omega_fluid(p, k),
                                          0.99 * p.Omega_i)
    f_lo = dispersion(p, k, lo)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = dispersion(p, k, mid)
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def omega_discrete(p, omega):
    """Yee curls replace k by 2/dx sin(k dx/2), the implicit midpoint rule
    maps omega to 2/dt atan(omega dt/2)."""
    k = 2 / p.dx * math.sin(0.5 * p.k * p.dx)
    omega *= k / p.k
    return 2 / p.dt * math.atan(0.5 * omega * p.dt)


def load_mode(out, p):
    """Complex amplitude of exp(i k x) in Bz / (eps B0), Bz is averaged over
    y, z and taken at the staggered x nodes (i + 1/2) dx."""
    directory = os.path.join(out, "B")
    frames = sorted(f for f in os.listdir(directory) if f.isdigit())
    nz, ny, nx = p.shape
    x = (np.arange(nx) + 0.5) * p.dx
    phase = np.exp(-1j * p.k * x)

    t, mode, profiles = [], [], []
    for frame in frames:
        data = np.fromfile(os.path.join(directory, frame), dtype=np.float32)
        Bz = data.reshape(nz, ny, nx, 3)[..., 2].mean(axis=(0, 1)) - p.B0
        t.append(int(frame) * p.period)
        mode.append(2 * np.mean(Bz * phase) / (p.eps * p.B0))
        profiles.append(Bz / (p.eps * p.B0))
    return np.array(t), np.array(mode), np.array(profiles), x


def fit(t, y, omega0):
    """Least squares y = A + C cos(omega t + phi), scanned over omega."""
    def residual(omega):
        M = np.column_stack([np.ones_like(t), np.cos(omega * t),
                             np.sin(omega * t)])
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        return np.sum((M @ coef - y)**2), coef

    omegas = np.linspace(0.7 * omega0, 1.3 * omega0, 601)
    omega = omegas[np.argmin([residual(w)[0] for w in omegas])]
    lo, hi = omega - 1e-3 * omega0, omega + 1e-3 * omega0
    for _ in range(60):  # golden section
        a = hi - 0.618 * (hi - lo)
        b = lo + 0.618 * (hi - lo)
        if residual(a)[0] < residual(b)[0]:
            hi = b
        else:
            lo = a
    omega = 0.5 * (lo + hi)
    _, (A, c, s) = residual(omega)
    return omega, A, math.hypot(c, s), math.atan2(-s, c)


def style_axes(ax, panel):
    ax.minorticks_on()
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(axis="both", which="both", direction="in",
        top=True, bottom=True, left=True, right=True, labelsize=TICKSIZE)
    ax.text(0.02, 0.95, panel, transform=ax.transAxes,
        ha="left", va="top", fontsize=PANELSIZE, zorder=20,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6,
            boxstyle="round,pad=0.2"))


def check(name, value, limit):
    print(f"  {name:<34} = {value:.3e} (limit {limit:.1e})")
    if not np.isfinite(value) or value > limit:
        raise AssertionError(f"{name} is {value:.3e}, expected at most {limit:.1e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default=OUT_DIR)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.out, "config.json")) as f:
        p = Parameters(json.load(f))

    w_fluid = omega_fluid(p, p.k)
    w_kinetic = omega_kinetic(p, p.k)
    w_theory = omega_discrete(p, w_kinetic)
    A_theory = p.beta / (1 + p.beta)
    C_theory = 1 / (1 + p.beta)

    t, mode, profiles, x = load_mode(args.out, p)
    # cos(k x) initial profile: the mode is real, Im is noise.
    w_pic, A_pic, C_pic, phi_pic = fit(t, mode.real, w_theory)

    print("hybrid_ex2_fms: fast magnetosonic wave, hybrid model")
    print(f"  k = {p.k:.5e}, vA = {p.vA2**0.5:.4e}, cs = {p.cs2**0.5:.4e},"
          f" beta = {p.beta:.4f}, k rho_i = {p.k * (p.Ti / p.mi)**0.5 / p.Omega_i:.4f}")
    print(f"  omega fluid     = {w_fluid:.6e} = {w_fluid / p.Omega_i:.5f} Omega_i")
    print(f"  omega kinetic   = {w_kinetic:.6e} ({w_kinetic / w_fluid - 1:+.2e} to fluid)")
    print(f"  omega discrete  = {w_theory:.6e} ({w_theory / w_kinetic - 1:+.2e} to kinetic)")
    print(f"  omega PIC       = {w_pic:.6e} ({w_pic / w_theory - 1:+.2e} to discrete)")
    print(f"  static part A   = {A_pic:.4f} (theory {A_theory:.4f})")
    print(f"  wave part C     = {C_pic:.4f} (theory {C_theory:.4f}), phase {phi_pic:+.3f}")


    # Plots are in units of B0: the fit and the checks above use eps B0.
    eps = p.eps
    T = 2 * math.pi / w_theory
    theory = lambda time: eps * (A_theory + C_theory * np.cos(w_theory * time))
    title = (f"Fast magnetosonic wave: $\\omega_{{PIC}}/\\omega = "
             f"{w_pic / w_theory:.4f}$, $\\omega = {w_theory / p.Omega_i:.4f}"
             f"\\,\\Omega_i$, $\\beta = 2n(T_e + T_i)/B_0^2 = {p.beta:.3f}$")

    def plot_spectrum(ax):
        spectrum = np.abs(np.fft.rfft((mode.real - mode.real.mean())
                                      * np.hanning(len(t)), n=16 * len(t)))
        freq = 2 * math.pi * np.fft.rfftfreq(16 * len(t), d=t[1] - t[0])
        ax.plot(freq / p.Omega_i, spectrum / spectrum.max(), color="black", lw=1.5)
        for w, color, label in ((w_theory, "tab:red", "theory"),
                                (w_pic, "tab:blue", "fit")):
            ax.axvline(w / p.Omega_i, color=color, ls="--", lw=1.5, label=label)
        ax.set_xlim(0, 0.5)
        ax.set_xlabel(r"$\omega\,/\,\Omega_i$", fontsize=LABELSIZE)
        ax.set_ylabel("spectrum, a.u.", fontsize=LABELSIZE)
        ax.legend(loc="upper right", fontsize=LEGENDSIZE)
        style_axes(ax, "(b)")

    def plot_profiles(ax):
        for fraction, color in ((0, "black"), (0.25, "tab:blue"),
                                (0.5, "tab:red")):
            i = np.argmin(np.abs(t - fraction * T))
            ax.plot(x / p.dx, eps * profiles[i], color=color, lw=1.5,
                    label=f"$t = {t[i] / T:.2f}\\,T$")
            ax.plot(x / p.dx, theory(t[i]) * np.cos(p.k * x), color=color,
                    ls="--", lw=1.2)
        ax.set_xlabel(r"$x\,/\,\Delta x$", fontsize=LABELSIZE)
        ax.set_ylabel(r"$\delta B_z\,/\,B_0$", fontsize=LABELSIZE)
        ax.legend(loc="upper right", fontsize=LEGENDSIZE)
        style_axes(ax, "(c)")

    fig, gs = figure(ncols=3, nrows=1, figsize=(24, 7.5))

    ax = subplot(fig, gs, 0, 0)
    ax.plot(t / T, eps * mode.real, color="black", lw=1.5, label="PIC")
    ax.plot(t / T, theory(t), color="tab:red", ls="--", lw=2, label="theory")
    ax.plot(t / T, eps * mode.imag, color="0.6", lw=1, label="Im (noise)")
    ax.axhline(eps * A_theory, color="tab:red", ls=":", lw=1.2)
    ax.set_xlabel(r"$t\,/\,T_{\rm theory}$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$\delta B_{z,k}\,/\,B_0$", fontsize=LABELSIZE)
    ax.set_ylim(-1.0 * eps, 1.5 * eps)
    ax.legend(loc="upper center", ncol=3, fontsize=LEGENDSIZE)
    style_axes(ax, "(a)")

    plot_spectrum(subplot(fig, gs, 1, 0))
    plot_profiles(subplot(fig, gs, 2, 0))

    fig.suptitle(title, fontsize=LABELSIZE)
    fig.tight_layout(w_pad=2.0)
    for extension in ("png", "pdf"):
        image = os.path.join(args.out, f"hybrid_ex2_fms.{extension}")
        fig.savefig(image, dpi=150, bbox_inches="tight", pad_inches=0.12)
        print(f"  saved figure to {image}")
    plt.close(fig)

    # Animation: panel (a) shows the harmonic k of Bz(x, t) instead of its
    # amplitude history, on top of the raw profile averaged over y, z.
    fig, gs = figure(ncols=3, nrows=1, figsize=(24, 7.5))

    ax = subplot(fig, gs, 0, 0)
    xs = x / p.dx
    bound = 1.15 * eps * max(np.max(np.abs(profiles)), 1.0)
    raw_line, = ax.plot(xs, eps * profiles[0], color="0.6", lw=1, label="PIC")
    mode_line, = ax.plot(xs, eps * np.real(mode[0] * np.exp(1j * p.k * x)),
                         color="black", lw=2, label="PIC, harmonic $k$")
    theory_line, = ax.plot(xs, theory(t[0]) * np.cos(p.k * x),
                           color="tab:red", ls="--", lw=2, label="theory")
    time_text = ax.text(0.98, 0.95, "", transform=ax.transAxes, ha="right",
                        va="top", fontsize=LEGENDSIZE)
    ax.set_ylim(-bound, bound)
    ax.set_xlabel(r"$x\,/\,\Delta x$", fontsize=LABELSIZE)
    ax.set_ylabel(r"$\delta B_z\,/\,B_0$", fontsize=LABELSIZE)
    ax.legend(loc="lower left", fontsize=LEGENDSIZE)
    style_axes(ax, "(a)")

    plot_spectrum(subplot(fig, gs, 1, 0))
    plot_profiles(subplot(fig, gs, 2, 0))
    fig.suptitle(title, fontsize=LABELSIZE)
    fig.tight_layout(w_pad=2.0)

    def update(i):
        raw_line.set_ydata(eps * profiles[i])
        mode_line.set_ydata(eps * np.real(mode[i] * np.exp(1j * p.k * x)))
        theory_line.set_ydata(theory(t[i]) * np.cos(p.k * x))
        time_text.set_text(f"$t = {t[i] / T:.2f}\\,T$")
        return raw_line, mode_line, theory_line, time_text

    animation = FuncAnimation(fig, update, frames=len(t), blit=True)
    image = os.path.join(args.out, "hybrid_ex2_fms.mp4")
    animation.save(image, writer=FFMpegWriter(fps=6, bitrate=3000), dpi=80)
    print(f"  saved animation to {image}")
    plt.close(fig)

    if not args.no_check:
        check("|omega_PIC / omega - 1|", abs(w_pic / w_theory - 1), OMEGA_LIMIT)
        check("|A_PIC - A_theory|", abs(A_pic - A_theory), STATIC_LIMIT)


if __name__ == "__main__":
    main()
