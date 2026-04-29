#!/usr/bin/env python3

from plot import *

config = f"{sys.argv[1]}_dx3.0_dt1.5_Np1000"
init_constants(f"results/{config}/{config}.json")
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])

ec_cols = {
    "ecsimcorr": (1, 2, 3, 7, 11, 12, 13, 14, 15, 16),
    "eccapfim": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
}

d_w = 100
d_s = 250
d_t = int(1.8 * const.tau / const.dt)

def avg(d): return sliding_average(d, d_w)[::d_s]
    
def read_energy(f, cols):
    d = np.loadtxt(f"{const.in_dir}/temporal/{f}.txt", skiprows=1, max_rows=d_t, usecols=cols)
    r = []
    for i in range(len(cols)):
        r.append(avg(d[:,i]))
    return r

e, b, k_i, k_e = read_energy("energy", (1, 2, 3, 4))
de, db, dk_i, dk_e, d, rm_i, rm_e, inj_i, inj_e, sum = read_energy("energy_conservation", ec_cols[sys.argv[1]])

fig, gs = figure(1, 2, figsize=(10, 12))

def plot(title, j):
    plot = PlotLinear(subplot(fig, gs, 0, j))
    plot.info.set_args(
        title=title,
        xlabel="$t / \\tau$"
    )
    return plot

en = plot("energy", 0)
ec = plot("$\\delta$(energy)", 1)

ts = np.arange(d_t, step=d_s) * const.dt / const.tau

en.axis.plot(ts, e, label="e", c="C0")
en.axis.plot(ts, b, label="b", c="C1")
en.axis.plot(ts, k_i, label="k_i", c="C2")
en.axis.plot(ts, k_e, label="k_e", c="C3")
en.axis.set_yscale("log")
en.axis.legend(fontsize=0.8*ticksize)
en.info.draw()

ec.axis.plot(ts, de, label="de", c="C0")
ec.axis.plot(ts, db, label="db", c="C1")
ec.axis.plot(ts, d, label="d", c="C4")
ec.axis.plot(ts, dk_i, label="dk_i", c="C2")
ec.axis.plot(ts, dk_e, label="dk_e", c="C3")
ec.axis.plot(ts, rm_i, label="rm_i", c="C2", ls="--")
ec.axis.plot(ts, rm_e, label="rm_e", c="C3", ls="--")
ec.axis.plot(ts, inj_i, label="inj_i", c="C2", ls="-.", alpha=0.8)
ec.axis.plot(ts, inj_e, label="inj_e", c="C3", ls="-.", alpha=0.8)
ec.axis.plot(ts, sum, label="sum")
# en.axis.set_yscale("log")
ec.axis.legend(fontsize=0.8*ticksize)
ec.info.draw()

out_dir = f"{const.out_dir}/other"
makedirs(out_dir)

fig.tight_layout()
fig.savefig(f"{out_dir}/energy.png")