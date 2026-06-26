#!/usr/bin/env python3

import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.normcase(os.path.join(os.path.dirname(__file__), "../drift_kinetic"))
ROOT = os.path.normcase(os.path.join(os.path.dirname(__file__), "../../"))

OMEGA_DT = 0.1
PITCH_FRAC = [
    0.995,
    1.005,
    1.200,
]
PUSHERS = [
    "Boris",
    "DK",
]

COLORS = {
    "Boris": "black",
    "DK": "red",
}


def mark_endpoints(ax, xs, ys, color):
    ax.plot(xs[0], ys[0], 'o', color='white', markeredgecolor=color,
            markeredgewidth=1.5, markersize=7, zorder=5)
    ax.plot(xs[-1], ys[-1], 'o', color=color, markersize=7, zorder=5)


def mark_endpoints_3d(ax, xs, ys, zs, color):
    ax.scatter(xs[0], ys[0], zs[0], s=50, facecolors='white',
               edgecolors=color, linewidths=1.5, zorder=10)
    ax.scatter(xs[-1], ys[-1], zs[-1], s=50, color=color, zorder=10)


def load_trace(push, frac):
    return np.loadtxt(
        f"{DIR}/output/drift_kinetic_push_ex3/temporal/{push}_omega_dt_{OMEGA_DT:.4f}_pf_{frac:.3f}.txt",
        skiprows=1,
    )


def get_kinetic_energy(push, data):
    if push == "drift_kinetic":
        p_par = data[:, 4]
        p_perp = data[:, 5]
        return 0.5 * (p_par ** 2 + p_perp ** 2)

    velocity = data[:, 4:7]
    return 0.5 * np.sum(velocity ** 2, axis=1)


def get_relative_energy_error(push, data):
    energy = get_kinetic_energy(push, data)
    energy0 = energy[0]

    if np.isclose(energy0, 0.0):
        raise ValueError(
            f"Начальная энергия для {push} слишком мала, относительную ошибку вычислить нельзя."
        )

    return np.abs(energy - energy0) / abs(energy0)


def draw_trajectory(ax3d, ax_xz, ax_yz, push, data):
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    color = COLORS[push]

    ax3d.plot(x, y, z, label=push, linewidth=1, color=color)
    ax_xz.plot(x, z, label=push, linewidth=1, color=color)
    ax_yz.plot(y, z, label=push, linewidth=1, color=color)

    mark_endpoints_3d(ax3d, x, y, z, color)
    mark_endpoints(ax_xz, x, z, color)
    mark_endpoints(ax_yz, y, z, color)


def draw_energy_error(ax, push, frac, data):
    t = data[:, 0]
    rel_error = get_relative_energy_error(push, data)
    plot_error = np.maximum(rel_error, np.finfo(float).tiny)

    ax.plot(t, plot_error, label=push, linewidth=1)
    print(
        f"pitch_frac={frac:.3f}, push={push}: "
        f"max relative energy error = {np.max(rel_error):.6e}"
    )


for frac in PITCH_FRAC:
    subprocess.run(
        [
            f"{ROOT}/build/Release/tests/drift_kinetic/drift_kinetic_push_ex3.out",
            "-omega_dt",
            str(OMEGA_DT),
            "-pitch_frac",
            str(frac),
        ],
        check=True,
    )

    traces = {push: load_trace(push, frac) for push in PUSHERS}

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    #ax4 = fig.add_subplot(1, 4, 4)

    for push, data in traces.items():
        draw_trajectory(ax1, ax2, ax3, push, data)
        #draw_energy_error(ax4, push, frac, data)

    ax1.set_xlabel(r"x [c/$\omega_{pe}$]", fontsize=13, labelpad=8)
    ax1.set_ylabel(r"y [c/$\omega_{pe}$]", fontsize=13, labelpad=8)
    ax1.set_zlabel(r"z [c/$\omega_{pe}$]", fontsize=13, labelpad=8)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.legend(fontsize=11)

    ax2.set_xlabel(r"x [c/$\omega_{pe}$]", fontsize=13, labelpad=8)
    ax2.set_ylabel(r"z [c/$\omega_{pe}$]", fontsize=13, labelpad=8)
    #ax2.legend()

    #ax3.set_xlabel(r"y [c/$\omega_p$]")
    #ax3.set_ylabel(r"z [c/$\omega_p$]")
    #ax3.legend()

    #ax4.set_xlabel(r"t [1/$\omega_p$]")
    #ax4.set_ylabel(r"$|E(t)-E(0)| / |E(0)|$")
    #ax4.set_title("Относительная ошибка энергии")
    #ax4.grid(True, which="both", alpha=0.3)
    #ax4.legend()

    fig.tight_layout()

    fig.savefig(
        f"{DIR}/output/drift_kinetic_push_ex3/temporal/omega_dt_{OMEGA_DT:.4f}_pf_{frac:.3f}.png"
    )
