#!/usr/bin/env python3
"""Сравнение первой гармоники плотности ионов с огибающей Ландау."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MEC2_KEV = 511


class Species:
    """Параметры сорта частиц для продольной диэлектрической функции."""
    def __init__(self, name, n, q, mass, temp):
        self.name, self.n, self.q, self.m, self.T = name, n, q, mass, temp
        self.vt = math.sqrt(temp / mass)
        self.ld = math.sqrt(temp / (n * q**2))


def plasma_z(z, order=64):
    """Плазменная дисперсионная функция."""
    def upper(x):
        m = 2 * order
        k = np.arange(-m + 1, m)
        length = math.sqrt(order / math.sqrt(2.0))
        theta = k * math.pi / m
        t = length * np.tan(theta / 2.0)
        f = np.append(0.0, np.exp(-t**2) * (length**2 + t**2))
        coeff = np.flipud((np.fft.fft(np.fft.fftshift(f)).real / (2 * m))[1:order + 1])
        q = (length + 1j * x) / (length - 1j * x)
        return (2 * np.polyval(coeff, q) / (length - 1j * x)**2
                + 1 / math.sqrt(math.pi) / (length - 1j * x))

    w = upper(z) if z.imag >= 0 else 2 * np.exp(-z**2) - upper(-z)
    return 1j * math.sqrt(math.pi) * w


def dielectric(omega, k, species):
    """Возвращает epsilon(omega, k) и её производную по omega."""
    value, derivative = 1.0 + 0j, 0j
    for s in species:
        z = omega / (math.sqrt(2.0) * k * s.vt)
        Z = plasma_z(z)
        factor = 1.0 / (k * s.ld)**2
        value += factor * (1.0 + z * Z)
        derivative += factor * (Z - 2 * z * (1 + z * Z)) / (math.sqrt(2) * k * s.vt)
    return value, derivative


def solve_dispersion(k, species):
    """Ищет ионно-звуковой корень epsilon(omega,k)=0 методом Ньютона."""
    electron, ion = species
    real = k * math.sqrt(electron.T / ion.m) / math.sqrt(1 + (k * electron.ld)**2)
    gamma = real * math.sqrt(math.pi / 8) * math.sqrt(electron.m / ion.m)
    omega = real - 1j * gamma
    for _ in range(100):
        value, derivative = dielectric(omega, k, species)
        step = value / derivative
        omega -= step
        if abs(step) < 1e-12 * max(1.0, abs(omega)):
            return omega
    raise RuntimeError("не удалось решить дисперсионное уравнение")


def temperature(p):
    """Температура в кэВ: старые конфиги используют T, новые — Tz."""
    return float(p.get("Tz", p.get("T")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path, help="каталог модели")
    parser.add_argument("-delta_n", type=float, help="начальный уровень огибающей")
    parser.add_argument("-T", type=float, help="предел графика в периодах волны")
    args = parser.parse_args()

    root = args.dir.resolve()
    with open(root / "config.json", encoding="utf-8") as stream:
        config = json.load(stream)
    geometry = config["Geometry"]
    particles = config["Particles"]
    electron = next(p for p in particles if p["q"] < 0)
    ion_cfg = next(p for p in particles if p["q"] > 0)
    species = [Species(p["sort_name"], p.get("n", 1.0), p["q"], p["m"],
                       temperature(p) / MEC2_KEV) for p in (electron, ion_cfg)]

    # wave_number по z в загрузчике — целый номер пространственной гармоники.
    preset = next(p for p in config.get("Presets", [])
                  if p.get("command") == "SetParticles"
                  and "wave_number" in p.get("coordinate", {}))
    mode = float(preset["coordinate"]["wave_number"][2])
    length, step = float(geometry["z"]), float(geometry["dz"])
    nx, ny, nz = (round(float(geometry[a]) / float(geometry["d" + a]))
                  for a in "xyz")
    k = 2.0 * math.pi * mode / length
    omega = solve_dispersion(k, species)
    omega_th, gamma_th = omega.real, -omega.imag
    period = 2.0 * math.pi / omega_th

    # Учитываем out_dir диагностики, если он отличается от стандартного.
    diagnostic = next((d for d in config.get("Diagnostics", [])
                       if d.get("diagnostic") == "DistributionMoment"
                       and d.get("moment") == "density"
                       and d.get("particles") == ion_cfg["sort_name"]), {})
    density_dir = root / diagnostic.get("out_dir", f'{ion_cfg["sort_name"]}/density')
    frames = sorted((int(p.name), p) for p in density_dir.iterdir()
                    if p.is_file() and p.name.isdigit())
    dt_out = float(geometry["diagnose_period"])
    t_end = args.T * period if args.T is not None else float(geometry["t"])
    z = (np.arange(nz) + 0.5) * step
    kernel = np.exp(-1j * k * z)
    time, amplitude = [], []
    for index, path in frames:
        t = index * dt_out
        if t > t_end:
            continue
        density = np.fromfile(path, np.float32)
        if density.size != nx * ny * nz:
            continue
        profile = density.reshape(nz, ny, nx).mean(axis=(1, 2))
        dn = profile / float(ion_cfg.get("n", 1.0)) - 1.0
        time.append(t)
        amplitude.append(abs(2.0 * np.mean(dn * kernel)))
    time, amplitude = np.asarray(time), np.asarray(amplitude)
    if time.size < 3:
        raise SystemExit("в заданном интервале недостаточно кадров плотности")

    # Модуль гармоники имеет максимум через pi/omega; первый максимум исключаем.
    maxima = np.flatnonzero((amplitude[1:-1] > amplitude[:-2]) &
                            (amplitude[1:-1] >= amplitude[2:])) + 1
    maxima = maxima[1:]
    if maxima.size < 2 or np.any(amplitude[maxima] <= 0.0):
        raise SystemExit("до T нужны как минимум три положительных максимума")
    # Частоту фитим по временам максимумов, декремент — по логарифму амплитуд.
    peak_t, peak_a = time[maxima], amplitude[maxima]
    half_period = np.polyfit(np.arange(maxima.size), peak_t, 1)[0]
    omega_fit = math.pi / half_period
    gamma_fit = -np.polyfit(peak_t, np.log(peak_a), 1)[0]

    configured_dn = abs(float(preset["coordinate"].get("amplitude", [0, 0, amplitude[0]])[2]))
    delta_n = args.delta_n if args.delta_n is not None else configured_dn
    theory_t = np.linspace(0.0, t_end, 500)
    plt.plot(time / period, amplitude, "o-", ms=3, label="test")
    plt.plot(theory_t / period, delta_n * np.exp(-gamma_th * theory_t), "k--",
             label=r"$\delta n_0 e^{-\Gamma t}$")
    plt.xlabel(r"$t/T$")
    plt.ylabel(r"$|\delta n_{i,1}|/n_i$")
    plt.xlim(0.0, t_end / period)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output = root / "ion_sound_compare.png"
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"omega_fit/omega_theory = {omega_fit / omega_th:.6g}")
    print(f"Gamma_fit/Gamma_theory = {gamma_fit / gamma_th:.6g}")
    print(output)


if __name__ == "__main__":
    main()
