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
d_t = int(2.7 * const.tau / const.dt)

def avg(d): return sliding_average(d, d_w)[::d_s]
    
def read_energy(f, cols):
    d = np.loadtxt(f"{const.in_dir}/temporal/{f}.txt", skiprows=1, max_rows=d_t, usecols=cols)
    r = []
    for i in range(len(cols)):
        r.append(avg(d[:,i]))
    return r

de, db, dk_i, dk_e, d, rm_i, rm_e, inj_i, inj_e, sum = read_energy("energy_conservation", ec_cols[sys.argv[1]])

fig, gs = figure(1, 2, figsize=(10, 12))

def plot(title, j):
    plot = PlotLinear(subplot(fig, gs, 0, j))
    plot.info.set_args(
        title=title,
        xlabel="$t / \\tau$"
    )
    return plot

ts = np.arange(d_t, step=d_s) * const.dt / const.tau

ec = plot("$\\delta W$", 0)
ec.axis.plot(ts, de, label="de", c="C0")
ec.axis.plot(ts, db, label="db", c="C1")
ec.axis.plot(ts, d, label="d", c="C4")
ec.axis.plot(ts, dk_i, label="dk_i", c="C2")
ec.axis.plot(ts, dk_e, label="dk_e", c="C3")
ec.axis.plot(ts, rm_i, label="rm_i", c="C2", ls="--")
ec.axis.plot(ts, rm_e, label="rm_e", c="C3", ls="--")
ec.axis.plot(ts, inj_i, label="inj_i", c="C2", ls="-.", alpha=0.8)
ec.axis.plot(ts, inj_e, label="inj_e", c="C3", ls="-.", alpha=0.8)
ec.axis.plot(ts, sum, label="sum", c="C5")
# en.axis.set_yscale("log")
ec.axis.legend(fontsize=0.8*FontSize.ticks)
ec.info.draw()

def draw(d, l, c):
    d = np.abs(np.cumsum(d)) * const.dt
    en.axis.plot(ts, d, label=l, c=c)
    en.axis.fill_between(ts, [0]*len(ts), d, color=c, alpha=0.05)

en = plot("$|\\int \\delta W dt|$", 1)
draw(de, "e", "C0")
draw(db, "b", "C1")
draw(dk_i + rm_i - inj_i, "k_i + rm_i - inj_i", "C2")
draw(dk_e + rm_e - inj_e, "k_e + rm_e - inj_e", "C3")
draw(sum, "sum", "C5")
en.axis.set_yscale("log")
en.axis.set_ylim((1e-7, 1e+1))
en.axis.legend(fontsize=0.8*FontSize.ticks)
en.info.draw()


out_dir = f"{const.out_dir}/other"
makedirs(out_dir)

fig.tight_layout()
fig.savefig(f"{out_dir}/energy.png")