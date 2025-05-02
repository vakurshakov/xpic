import sys
import struct
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)

from constants import *

def find_dprr(r, prr, paa):
    "Returns the integral from 0 to 'inf' of non-gyrotropic addition, `(prr - paa) / r`"
    return spi.cumulative_trapezoid((prr - paa) / (r + r0_tolerance), r, initial=0, dx=dr)

def draw_all(d, r, chi, b, n, j, prr, paa, dprr):
    "Utility to show the calculated parameter distributions visually."

    def _v(d: str): return "v_{_T}" if d == "maxw" else "V"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(r, b, label="$B/B_v$")
    axes[0].plot(r, (-1) * j, label="$-J_{\\psi} / e n_0 " f"{_v(d)}" "$")
    axes[0].plot(r, n, label="$n/n_0$")
    axes[0].plot(r, (chi - chi[0]) / (a * a), label="$\\chi / a^2 B_v$")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("\\rm a.u.")

    axes[1].plot(r, prr, label="$\\Pi_{rr}$")
    axes[1].plot(r, paa, label="$\\Pi_{\\psi \\psi}$")
    axes[1].plot(r, b * b / 2, label="$B^2 / 2$")
    axes[1].plot(r, b * b / 2 + dprr, label="$\\Delta \\Pi_{rr}$", linestyle="--")
    axes[1].set_ylabel(f"$m n_0 {_v(d)}^2$")

    balance = np.std(b * b / 2 + prr + dprr - 1)
    fig.suptitle("\\rm Kotelnikov equilibrium: ${" + f"a/\\rho_i = {a:.1f}, B_" "{in}" f"/B_v = {B_in:.2f}" + "}, \\sigma = " f"{balance:.2e}" "$", y=0.93)

    for axis in axes:
        axis.set_xlabel("$r,~c/\\omega_{pi}$")
        axis.set_xlim(r0_rmax)
        axis.legend(draggable=True)
        axis.grid()

    # plt.show()
    fig.savefig(f"{out_dir}/{d}_all.png")

def store_all(d, chi, b, n, j, prr, paa):
    "Utility to store the computed distribution functions into binaries."

    data = [chi + (0.5 * B_in * a * a), b, n, j, prr, paa]
    names = ["chi", "b", "n", "j", "prr", "paa"]
    params = [*r0_rmax, dr]

    for data, name in zip(data, names):
        with open(f"{out_dir}/{d}_{name}", "wb") as file:
            result = params + list(data)
            file.write(struct.pack('d' * len(result), *result))