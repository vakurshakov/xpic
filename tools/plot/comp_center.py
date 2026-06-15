#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from plot import *
from runner import *

fig, gs = figure(3, 1, figsize=(18, 6))

def plot(title, i):
    plot = PlotLinear(subplot(fig, gs, i, 0))
    plot.info.set_args(
        title=title,
        xlabel="$t / \\tau$"
    )
    return plot

def ts(d): return np.arange(0, len(d)) * const.dts / const.tau
def read(n): return np.load(f"{const.in_dir}/processed/collections/{n}_t.npy")

b = plot("\\rm Magnetic field, $B_z / B_v$", 0)
n = plot("\\rm Ion density, $n_i$", 1)
p = plot("\\rm Plasma beta, $\\beta$", 2)

def plot(s, l):
    init_constants(f"results/{s}/{s}.json")
    const.B0 = float(const.config["Presets"][0]["setter"]["value"][2])
    const.tau = read_scalar(const.config["StepPresets"][3]["tau"])

    dp = (2.35) * 2 * (read("pxi") + read("pxe")) / np.square(const.B0)
    db = read("bz") / const.B0
    dn = read("ni")
    t = ts(dp)

    b.axis.plot(t, db)
    n.axis.plot(t, dn)

    p.axis.plot(t, dp, label=f"\\rm {l}")

for sim, Np, tol in pool:
    dir = f"{sim}_dx{dx:.1f}_dt{dt:.1f}_Np{Np:d}_mi{mi_me:d}_tol{tol:.0e}"
    plot(dir, f"\\rm {sim}, $\\epsilon = 10^{{{int(np.log10(tol)):d}}}$")

for d in [b, n, p]:
    d.axis.grid()
    d.info.draw()

p.axis.legend(fontsize=0.6*FontSize.label)

out_dir = f"./results/other"
makedirs(out_dir)

fig.tight_layout(w_pad=1.3)
fig.savefig(f"{out_dir}/center.png")