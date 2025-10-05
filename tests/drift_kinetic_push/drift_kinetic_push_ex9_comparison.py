#!/usr/bin/env python3

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

dir = os.path.dirname(__file__)
root = os.path.normcase(os.path.join(dir, "../../"))

omega_dt = 0.1

pitch_frac = [
    0.995,
    1.005,
    1.200,
]

def draw_trajectory(push, dt, frac):
    data = np.loadtxt(f"{dir}/output/drift_kinetic_push_ex9/temporal/{push}_omega_dt_{omega_dt:.1f}_pf_{frac:.3f}.txt", skiprows=1)

    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]

    ax1.plot(x, y, z, label=push, linewidth=1)
    ax2.plot(x, z, label=push, linewidth=1)
    ax3.plot(y, z, label=push, linewidth=1)

    ax1.scatter(x[0], y[0], z[0], s=50, zorder=10)
    ax2.scatter(x[0], z[0], s=50, zorder=10)
    ax3.scatter(y[0], z[0], s=50, zorder=10)

for frac in pitch_frac:
    os.system(f"{root}/build/Release/tests/drift_kinetic_push/drift_kinetic_push_ex9.out -omega_dt {omega_dt} -pitch_frac {frac}")

    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    draw_trajectory("boris", 0.1, frac)
    draw_trajectory("crank_nicolson", 0.1, frac)
    draw_trajectory("drift_kinetic", 0.1, frac)

    ax1.set_xlabel('x [c/ωpe]')
    ax1.set_ylabel('y [c/ωpe]')
    ax1.set_zlabel('z [c/ωpe]')
    ax1.set_title('3D trajectory')
    ax1.set_xlim(-0.2, 0.2)
    ax1.set_ylim(-0.2, 0.2)
    ax1.set_zlim(-6.0, 6.0)
    ax1.legend(loc="upper left")

    ax2.set_xlabel('x [c/ωpe]')
    ax2.set_ylabel('z [c/ωpe]')
    ax2.set_title('Projection (x, z)')
    ax2.set_xlim(-0.2, 0.2)
    ax2.set_ylim(-6.0, 6.0)
    ax2.legend()

    ax3.set_xlabel('y [c/ωpe]')
    ax3.set_ylabel('z [c/ωpe]')
    ax3.set_title('Projection (y, z)')
    ax3.set_xlim(-0.2, 0.2)
    ax3.set_ylim(-6.0, 6.0)
    ax3.legend()

    fig.tight_layout()

    # plt.show()
    fig.savefig(f"{dir}/output/drift_kinetic_push_ex9/temporal/omega_dt_{omega_dt:.1f}_pf_{frac:.3f}.png")
