#!/usr/bin/env python3
"""Check a hybrid fast magnetosonic wave against kinetic and discrete theory.

The initial magnetic cosine perturbation excites a standing FMS wave plus
an equilibrium shelf beta/(1+beta). Fit the first harmonic by A+C*cos(w*t+phi).
Yee spatial staggering and implicit-midpoint time dispersion are included.

The PNG/PDF shows the spectrum and full amplitude history. The MP4 and PNG
frames animate the spatial profile separately. Before plotting, check energy
and charge continuity from hybrid_diagnostic.txt. Accumulate per-step energy
increments to avoid subtracting rounded absolute energies.

Usage: hybrid_drift_kinetic_ex2_fms.py [--no-check] [--no-animation] [output directory]
Checks require at least two wave periods; --no-check permits unfinished runs.
"""

import argparse
import math
import os
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(TEST_DIR, "output", "hybrid_drift_kinetic_ex2_fms")

sys.path.insert(0, os.path.join(TEST_DIR, "../../tools"))
from lib.constants import init_constants, mec2
from lib.data_consistency import is_correct_timestep
from lib.data_format import FieldView
from lib.plot import FontSize, PlotLinear, annotate_x, figure, subplot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers

plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "cmr10"
plt.rcParams["axes.formatter.use_mathtext"] = True

OMEGA_LIMIT = 0.02   # |omega_pic / omega_theory - 1|
STATIC_LIMIT = 0.05  # |A_pic - A_theory| in units of eps B0
ENERGY_LIMIT = 1e-10  # max |sum(dE+dB+dK)| / WK0
CHARGE_LIMIT = 1e-12  # max L2 norm of drho/dt + div(J)

LEGENDSIZE = FontSize.ticks


def check(name, value, limit):
    print(f"  {name:<34} = {value:.3e} (limit {limit:.1e})")
    if not np.isfinite(value) or value > limit:
        raise AssertionError(f"{name} is {value:.3e}, expected at most {limit:.1e}")


def check_conservation(out):
    path = os.path.join(out, "temporal", "hybrid_diagnostic.txt")
    with open(path) as stream:
        header = re.split(r"\s{2,}", stream.readline().strip())
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    if data.shape[1] != len(header) or not np.all(np.isfinite(data)):
        raise ValueError(f"Invalid conservation diagnostic: {path}")
    table = dict(zip(header, data.T))
    WK0 = table["wK"][0]
    if WK0 <= 0:
        raise ValueError("Conservation diagnostics require positive initial particle energy")
    energy = np.cumsum(table["dE+dB+dK"]) / WK0
    charge = table["N2dQ_tot"]
    check("max |W-W0| / WK0", np.max(np.abs(energy)), ENERGY_LIMIT)
    check("max charge continuity residual", np.max(np.abs(charge)), CHARGE_LIMIT)


class Parameters:
    def __init__(self, constants):
        config = constants.config
        sorts = {s["sort_name"]: s for s in config["Particles"]}
        field = next(p["setter"] for p in config["Presets"]
                     if p.get("command") == "SetField" and p["field"] == "B")

        self.dx, self.dt = constants.dx, constants.dt
        self.stride, self.period = constants.Ndts, constants.Ndts * constants.dt
        self.last_step = constants.Nt
        self.shape = (constants.Nz, constants.Ny, constants.Nx)
        self.k = 2 * math.pi / constants.Lx

        self.B0 = field["value"][2]
        perturbation = next((preset["setter"] for preset in config["Presets"]
                             if preset.get("command") == "SetField" and preset.get("field") == "B"
                             and preset["setter"]["name"] == "SetGeneralCosineField"), None)
        self.eps = (perturbation["amplitude"][2] / self.B0 if perturbation is not None
                    else field["modulation"])
        if perturbation is not None and perturbation["wave_number"] != [1, 0, 0]:
            raise ValueError("Expected the fundamental cosine mode along x")
        if self.B0 <= 0 or self.eps <= 0 or self.stride < 1:
            raise ValueError("Expected positive B0, perturbation and diagnostic cadence")
        self.n = sorts["ions"]["n"]
        self.mi = sorts["ions"]["m"]
        self.Ti = sorts["ions"]["T"] / mec2
        self.Te = sorts["electrons"]["T"] / mec2

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
    if not np.isfinite(f_lo) or f_lo * dispersion(p, k, hi) >= 0:
        raise ValueError("The kinetic FMS root is not bracketed")
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
    frames = sorted((f for f in os.listdir(directory) if f.isdigit()), key=int)
    if not frames:
        raise ValueError("No magnetic snapshots")
    if [int(f) for f in frames] != list(range(len(frames))):
        raise ValueError("Magnetic snapshots must start at zero with no gaps")
    if int(frames[-1]) * p.stride > p.last_step:
        raise ValueError("Snapshots exceed configured duration; check for files from an older run")
    nz, ny, nx = p.shape
    x = (np.arange(nx) + 0.5) * p.dx
    phase = np.exp(-1j * p.k * x)

    view = FieldView()
    view.region = FieldView.Region(3, np.zeros(3, dtype=int), np.array([nx, ny, nz, 3]))
    view.path = lambda index: os.path.join(directory, frames[index])
    t, mode, profiles = [], [], []
    for index, frame in enumerate(frames):
        if not is_correct_timestep(index, view):
            if frame == frames[-1]:
                print(f"  skipping incomplete snapshot {frame}")
                break
            raise ValueError(f"Invalid snapshot size: {frame}")
        data = np.fromfile(view.path(index), dtype=np.float32)
        if not np.all(np.isfinite(data)):
            raise ValueError(f"Nonfinite magnetic snapshot: {frame}")
        Bz = data.reshape(nz, ny, nx, 3)[..., 2].mean(axis=(0, 1), dtype=float) - p.B0
        t.append(int(frame) * p.period)
        mode.append(2 * np.mean(Bz * phase) / (p.eps * p.B0))
        profiles.append(Bz / (p.eps * p.B0))
    if not t:
        raise ValueError("No complete magnetic snapshots")
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
    _, coefficients = residual(omega)
    return omega, coefficients[0]


def style_axes(ax, panel):
    PlotLinear(ax).draw_info()
    ax.minorticks_on()
    ax.grid(True, which="both", alpha=0.25)
    annotate_x(ax, panel, x=0.02, y=0.95, size=FontSize.label, ha="left")




def wave_signals(p, t, mode, profiles, x, omega):
    """Shelf-subtracted fields in units of B0, and positive amplitudes."""
    shelf = p.beta / (1 + p.beta)
    cosine = np.cos(p.k * x)
    model = p.eps * (profiles - shelf * cosine)
    first_mode = p.eps * (mode - shelf)
    first = np.real(first_mode[:, None] * np.exp(1j * p.k * x))
    theory_mode = p.eps / (1 + p.beta) * np.cos(omega * t)
    theory = theory_mode[:, None] * cosine
    return {
        "model": model, "first": first, "theory": theory,
        "model_amplitude": 0.5 * np.ptp(model, axis=1),
        "first_amplitude": np.abs(first_mode),
    }


def temporal_spectrum(t, signal):
    """Original temporal spectrum: DC removed, Hann window, zero padding."""
    if len(t) < 4:
        return np.array([0.0]), np.array([0.0])
    if not np.allclose(np.diff(t), t[1] - t[0]):
        raise ValueError("Temporal spectrum requires equally spaced snapshots")
    values = (signal - np.mean(signal)) * np.hanning(len(t))
    spectrum = np.abs(np.fft.rfft(values, n=16 * len(t)))
    maximum = np.max(spectrum)
    if maximum > 0:
        spectrum /= maximum
    frequency = 2 * np.pi * np.fft.rfftfreq(16 * len(t), d=t[1] - t[0])
    return frequency, spectrum


def plot_history(out, p, t, mode, signals, omega, fitted):
    period = 2 * np.pi / omega
    frequency, spectrum = temporal_spectrum(t, mode.real)
    fig, gs = figure(2, 1, figsize=(16, 7.5))
    spectrum_ax, amplitude_ax = [subplot(fig, gs, i) for i in range(2)]
    spectrum_ax.plot(frequency / p.Omega_i, spectrum, color="black")
    spectrum_ax.axvline(omega / p.Omega_i, color="tab:red", ls="--", label="Theory")
    if fitted is not None:
        label = "Numerical" if t[-1]-t[0] >= 2*period else "Numerical (preliminary)"
        spectrum_ax.axvline(fitted[0] / p.Omega_i, color="tab:blue", ls="-.", label=label)
    spectrum_ax.set(xlim=(0, 0.5), ylim=(0, 1.08), xlabel=r"$\omega/\Omega_i$",
                    ylabel="Temporal spectrum, a.u.")
    dense_t = np.linspace(t[0], t[-1], max(1000, 16*len(t)))
    amplitude_ax.plot(dense_t/period, p.eps/(1+p.beta)*np.abs(np.cos(omega*dense_t)),
                      color="tab:red", ls="--", label="Theory")
    for key, color, label in (("model_amplitude", "0.35", "Model, half peak-to-peak"),
                              ("first_amplitude", "tab:blue", "First harmonic")):
        amplitude_ax.plot(t/period, signals[key], color=color, marker=".", label=label)
    amplitude_ax.set(xlabel=r"$t/T_{\mathrm{theory}}$", ylabel=r"$|\delta B_{\mathrm{osc}}|/B_0$")
    for axis, panel in ((spectrum_ax, "(a)"), (amplitude_ax, "(b)")):
        style_axes(axis, panel)
        axis.legend(fontsize=LEGENDSIZE)
    fig.tight_layout()
    try:
        for extension in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"hybrid_drift_kinetic_ex2_fms.{extension}"), dpi=150)
    finally:
        plt.close(fig)


def animate_profile(out, p, t, x, signals, omega, fps, dpi):
    if not writers.is_available("ffmpeg"):
        raise ValueError("ffmpeg is required for MP4 output; use --no-animation")
    fig, gs = figure(1, 1, figsize=(8, 7.5))
    axis = subplot(fig, gs)
    lines = []
    for key, color, style, label in (("theory", "tab:red", "--", "Theory"),
                                     ("model", "0.35", "-", "Model, full profile"),
                                     ("first", "tab:blue", "-", "First harmonic")):
        line, = axis.plot(x/p.dx, signals[key][0], color=color, ls=style, label=label)
        lines.append((line, signals[key]))
    bound = max(1.15 * max(np.max(np.abs(values)) for _, values in lines), 1e-12)
    axis.set(ylim=(-bound, bound), xlabel=r"$x/\Delta x$",
             ylabel=r"$(\delta B_z-\delta B_{\mathrm{shelf}})/B_0$")
    style_axes(axis, "FMS")
    axis.legend(fontsize=LEGENDSIZE)
    title = fig.suptitle("", fontsize=FontSize.label)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    frames_dir = os.path.join(out, "processed", "fms", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    try:
        with writer.saving(fig, os.path.join(out, "hybrid_drift_kinetic_ex2_fms.mp4"), dpi=dpi):
            for i, time in enumerate(t):
                for line, values in lines:
                    line.set_ydata(values[i])
                step = round(time/p.dt)
                title.set_text(f"FMS: step {step}, t = {time*omega/(2*np.pi):.3f} T")
                fig.savefig(os.path.join(frames_dir, f"fms_step_{step:06d}.png"), dpi=dpi)
                writer.grab_frame()
    finally:
        plt.close(fig)
    print(f"  saved {len(t)} profile frames to {frames_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default=OUT_DIR)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--no-animation", action="store_true", help="Only save the static spectrum and amplitude PNG/PDF")
    args = parser.parse_args()
    if args.fps <= 0 or args.dpi <= 0:
        parser.error("fps and dpi must be positive")
    try:
        if not args.no_check:
            check_conservation(args.out)
        constants = init_constants(os.path.join(args.out, "config.json"))
        p = Parameters(constants)
        w_fluid = omega_fluid(p, p.k)
        w_kinetic = omega_kinetic(p, p.k)
        w_theory = omega_discrete(p, w_kinetic)
        t, mode, profiles, x = load_mode(args.out, p)
        period = 2 * np.pi / w_theory
        can_fit = len(t) >= 12 and t[-1]-t[0] >= period
        enough_data = can_fit and t[-1]-t[0] >= 2*period
        fitted = fit(t, mode.real, w_theory) if can_fit else None
        print("hybrid_drift_kinetic_ex2_fms: shelf-subtracted FMS profiles, spectrum and amplitudes")
        print(f"  omega fluid = {w_fluid:.6e}, kinetic = {w_kinetic:.6e}, discrete = {w_theory:.6e}")
        print(f"  theoretical shelf / (eps B0) = {p.beta/(1+p.beta):.6f}")
        if fitted is not None:
            print(f"  omega PIC = {fitted[0]:.6e} = {fitted[0]/p.Omega_i:.5f} Omega_i; fitted shelf = {fitted[1]:.6f}")
            if not enough_data:
                print("  numerical frequency is preliminary; acceptance checks need two periods")
        else:
            print("  insufficient data for a numerical frequency (need one period and 12 snapshots)")
        if not args.no_check:
            if not enough_data:
                print("FMS checks require two complete wave periods", file=sys.stderr)
                return 2
            check("|omega_PIC / omega - 1|", abs(fitted[0]/w_theory-1), OMEGA_LIMIT)
            check("|A_PIC - A_theory|", abs(fitted[1]-p.beta/(1+p.beta)), STATIC_LIMIT)
        signals = wave_signals(p, t, mode, profiles, x, w_theory)
        plot_history(args.out, p, t, mode, signals, w_theory, fitted)
        if not args.no_animation:
            animate_profile(args.out, p, t, x, signals, w_theory, args.fps, args.dpi)
        return 0
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"FMS analysis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
