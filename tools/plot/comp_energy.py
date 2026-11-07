#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from plot import *
from runner import pool

ec_cols = {
    "ecsimcorr": (1, 2, 16),
    "eccapfim": (1, 2, 10),
}

d_w = 1000
d_s = 1
def avg(d): return sliding_average(d, d_w)
    
def read_energy(conf):
    cols = ec_cols[conf.split("_")[0]]
    max_cols = 0

    r = [[] for _ in cols]
    
    with open(f"{const.in_dir}/temporal/energy_conservation.txt") as f:
        for il, l in enumerate(f):
            if il == 0:
                continue
            i = 0 
            for j, d in enumerate(l.split()):
                if j in cols:
                    r[i].append(float(d))
                    i += 1
            max_cols = min(i, len(cols))

    for i, d in enumerate(r):
        if i >= max_cols:
            d.append(r[i][-1])
        d = avg(avg(d))

    return r

fig, gs = figure(3, 2, figsize=(20, 10))

def plot(title, i, j):
    plot = PlotLinear(subplot(fig, gs, i, j))
    plot.info.set_args(
        title=title,
        xlabel="$t / \\tau$"
    )
    return plot

fde = plot("$\\delta W_E$", 0, 0)
fdb = plot("$\\delta W_B$", 1, 0)
fsum = plot("$\\delta (W_E + W_B + K)$", 2, 0)
fide = plot("$\\int \\delta W_E dt$", 0, 1)
fidb = plot("$\\int \\delta W_B dt$", 1, 1)
fisum = plot("$\\int \\delta (W_E + W_B + K) dt$", 2, 1)

def draw(conf, l):
    init_constants(f"results/{conf}/{conf}.json")
    const.tau = read_scalar(const.config["StepPresets"][3]["tau"])
    const.pte = int(1.0 * const.tau / const.dt)

    de, db, sum = read_energy(conf)
    ln = len(de)
    ts = np.arange(ln) * const.dt / const.tau

    fde.axis.plot(ts, de, label=l)
    fdb.axis.plot(ts, db)
    fsum.axis.plot(ts, sum)
    fide.axis.plot(ts, np.cumsum(de) * const.dt)
    fidb.axis.plot(ts, np.cumsum(db) * const.dt)
    fisum.axis.plot(ts, np.cumsum(sum) * const.dt)

# draw("ecsimcorr_dx3.0_dt1.5_Np1000", "\\rm ecsimcorr")
# draw("eccapfim_dx3.0_dt1.5_Np1000", "\\rm eccapfim; $\\epsilon_a = 10^{-7}, \\epsilon_r = 10^{-5}$")

for Np, tol in pool:
    draw(f"eccapfim_dx3.0_dt1.5_Np{Np:d}_Ny4_tol{tol:.0e}", f"$N_p = {Np},~\\epsilon = 10^{{{int(np.log10(tol)):d}}}$") # \\rm eccapfim, 

for f in [fde, fdb, fsum, fide, fidb, fisum]:
    f.info.draw()

fde.axis.legend(fontsize=0.6*FontSize.ticks)

out_dir = f"./results/other"
makedirs(out_dir)

fig.tight_layout()
fig.savefig(f"{out_dir}/energy.png")