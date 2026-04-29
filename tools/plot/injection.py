#!/usr/bin/env python3

from plot import *

config = f"{sys.argv[1]}_dx3.0_dt1.5_Np1000"
init_constants(f"results/{config}/{config}.json")
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])

logs = {
    "ecsimcorr": "logs/run.sh.o2244938",
    "eccapfim": "logs/run.sh.o2244941",
}

cols = {
    "ecsimcorr": (12, 13, 14, 15),
    "eccapfim": (6, 7, 8, 9),
}

d_w = 100
d_s = 250
d_t = int(1.8 * const.tau / const.dt)

def avg(d): return sliding_average(d, d_w)[::d_s]
    
def read_energy():
    cols=cols[sys.argv[1]]
    d = np.loadtxt(f"{const.in_dir}/temporal/energy_conservation.txt", skiprows=1, max_rows=d_t, usecols=cols)
    r = []
    for i in range(len(cols)):
        r.append(avg(d[:,i]))
    return r
        
def read_log():
    rm_ni = []
    rm_ne = []
    inj_n = []
    with open(f"{logs[sys.argv[1]]}") as f:
        for l in f:
            if "Particles have been removed from \"ions\"" in l:
                rm_ni.append(int(f.readline().split()[3][:-1]))
            elif "Particles have been removed from \"electrons\"" in l:
                rm_ne.append(int(f.readline().split()[3][:-1]))
            elif "Particles have been injected" in l:
                inj_n.append(int(f.readline().split()[3][:-1]))
            if len(inj_n) >= len(rm_i):
                break
    return avg(rm_ni), avg(rm_ne), avg(inj_n)

rm_i, rm_e, inj_i, inj_e = read_energy()
rm_ni, rm_ne, inj_n = read_log()

fig, gs = figure(1, 2, figsize=(10, 12))

def plot(title, j):
    plot = PlotLinear(subplot(fig, gs, 0, j))
    plot.info.set_args(
        title=title,
        xlabel="$t / \\tau$"
    )
    return plot

en = plot("energy", 0)
n = plot("counts", 1)

inj_n_th = 30 * 30 * 5 * 1000 / 60000
inj_i_th = inj_n_th / 1000 * (8 / 511) * 3 / 2
inj_e_th = inj_n_th / 1000 * (2 / 511) * 3 / 2

ts = np.arange(d_t, step=d_s) * const.dt / const.tau

en.axis.plot(ts, rm_i, label="rm_i")
en.axis.plot(ts, rm_e, label="rm_e")
en.axis.plot(ts, inj_i, label="inj_i")
en.axis.plot(ts, inj_e, label="inj_e")
en.axis.plot([0, len(rm_i) * const.dt / const.tau], [inj_i_th]*2, label="inj_i, th.", ls="--")
en.axis.plot([0, len(rm_i) * const.dt / const.tau], [inj_e_th]*2, label="inj_e, th.", ls="--")
en.axis.legend(fontsize=0.8*ticksize)
en.info.draw()

n.axis.plot(ts, rm_ni, label="rm_ni")
n.axis.plot(ts, rm_ne, label="rm_ne")
n.axis.plot(ts, inj_n, label="inj_n")
n.axis.plot([0, len(rm_ni) * const.dt / const.tau], [inj_n_th]*2, label="inj_n, th.", ls="--")
n.axis.legend(fontsize=0.8*ticksize)
n.info.draw()

out_dir = f"{const.out_dir}/other"
makedirs(out_dir)

fig.tight_layout()
fig.savefig(f"{out_dir}/injection.png")