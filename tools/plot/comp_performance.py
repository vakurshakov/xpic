#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from plot import *
from runner import *

def read_log(sim):
    fname = f"./results/{sim}/log-EachTimestep.txt"
    return np.loadtxt(fname, skiprows=1, usecols=1)

fig, gs = figure(2, 1, figsize=(18, 8))

def plot(title, i):
    plot = PlotLinear(subplot(fig, gs, i, 0))
    plot.info.set_args(title=title)
    return plot

tt = plot("\\rm timestep time, sec", 0)
tt.info.set_args(xlabel="$t / \\tau$")

ct = plot("\\rm calculated time, $t / \\tau$", 1)
ct.info.set_args(xlabel="\\rm time, hours")

def ts(d): return np.arange(len(d)) * const.dt / const.tau

def plot(dir, l):
    init_constants(f"results/{dir}/{dir}.json")
    const.B0 = float(const.config["Presets"][0]["setter"]["value"][2])
    const.tau = read_scalar(const.config["StepPresets"][3]["tau"])

    dt = sliding_average(read_log(dir), 10)
    t = ts(dt)

    tt.axis.plot(t, dt)#, label=f"\\rm {l}")
    ct.axis.plot(np.cumsum(dt) / (60 * 60), t, label=l)

for sim, Np, tol in pool:
    dir = f"{sim}_dx{dx:.1f}_dt{dt:.1f}_Np{Np:d}_mi{mi_me:d}_tol{tol:.0e}"
    plot(dir, f"\\rm {sim}, $\\epsilon = 10^{{{int(np.log10(tol)):d}}}$")

# tt.axis.set_ylim(20 * (-0.1), 20 * (1.1))
# ct.axis.set_ylim(200 * (-0.1), 200 * (1.1))

for d in [tt, ct]:
    d.axis.grid()
    d.info.draw()

ct.axis.legend(fontsize=0.65*FontSize.label, loc="lower right")

out_dir = f"./results/other"
makedirs(out_dir)

fig.tight_layout()
fig.savefig(f"{out_dir}/performance.png")