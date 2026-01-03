#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


def load_trace(base_dir, test_name, omega_dt, kind):
    filename = f"omega_dt_{omega_dt:.4f}_{kind}.txt"
    path = os.path.join(base_dir, "output", test_name, "temporal", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Нет файла с траекторией: {path}")
    return np.loadtxt(path, skiprows=1)


def plot_trace(ax3d, ax_xz, ax_yz, data, label, linestyle, color):
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]

    ax3d.plot(x, y, z, linestyle=linestyle, color=color, linewidth=1.2, label=label)
    ax_xz.plot(x, z, linestyle=linestyle, color=color, linewidth=1.2, label=label)
    ax_yz.plot(y, z, linestyle=linestyle, color=color, linewidth=1.2, label=label)

    ax3d.scatter(x[-1], y[-1], z[-1], color=color, s=30, zorder=10)
    ax_xz.scatter(x[-1], z[-1], color=color, s=30, zorder=10)
    ax_yz.scatter(y[-1], z[-1], color=color, s=30, zorder=10)


def main():
    parser = argparse.ArgumentParser(
        description="Визуализация тестов drift_kinetic_grid_boris_*."
    )
    parser.add_argument(
        "--test",
        default="drift_kinetic_grid_boris_ex5",
        help="Имя теста (например, drift_kinetic_grid_boris_ex5).",
    )
    parser.add_argument(
        "--omega_dt",
        type=float,
        default=0.1,
        help="Значение omega_dt для выбора траекторий.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Путь к PNG. По умолчанию сохраняет в temporal директорию теста.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Показать график вместо сохранения в файл.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)

    try:
        data_analytical = load_trace(base_dir, args.test, args.omega_dt, "analytical")
        data_grid = load_trace(base_dir, args.test, args.omega_dt, "grid")
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    plot_trace(ax1, ax2, ax3, data_analytical, "аналитика", "--", "#1f77b4")
    plot_trace(ax1, ax2, ax3, data_grid, "сетка", "-", "#d62728")

    ax1.set_xlabel(r"$x\,[c/\omega_p]$")
    ax1.set_ylabel(r"$y\,[c/\omega_p]$")
    ax1.set_zlabel(r"$z\,[c/\omega_p]$")
    ax1.set_title(r"$\mathrm{3D\ траектория}$")
    ax1.legend(loc="upper left")

    ax2.set_xlabel(r"$x\,[c/\omega_p]$")
    ax2.set_ylabel(r"$z\,[c/\omega_p]$")
    ax2.set_title(r"$\mathrm{Проекция}\ (x, z)$")
    ax2.legend()

    ax3.set_xlabel(r"$y\,[c/\omega_p]$")
    ax3.set_ylabel(r"$z\,[c/\omega_p]$")
    ax3.set_title(r"$\mathrm{Проекция}\ (y, z)$")
    ax3.legend()

    fig.tight_layout()

    if args.show:
        plt.show()
        return 0

    if args.output is None:
        out_dir = os.path.join(base_dir, "output", args.test, "temporal")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"{args.test}_omega_dt_{args.omega_dt:.4f}_comparison.png"
        )
    else:
        out_path = args.output

    fig.savefig(out_path, dpi=150)
    print(f"Сохранено: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
