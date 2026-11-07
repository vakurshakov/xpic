#!/usr/bin/env python3

from plot import *

config = f"{sys.argv[1]}_dx3.0_dt1.5_Np1000"
init_constants(f"results/{config}/{config}.json")
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])
const.T_i = read_scalar(const.config["Particles"][0]["T"]) / mec2
const.T_e = read_scalar(const.config["Particles"][1]["T"]) / mec2
const.Np = read_scalar(const.config["Particles"][0]["Np"])
const.inj_dx = (
    read_scalar(const.config["StepPresets"][3]["coordinate"]["max"][0]) - 
    read_scalar(const.config["StepPresets"][3]["coordinate"]["min"][0])) / const.dx
const.inj_dy = (
    read_scalar(const.config["StepPresets"][3]["coordinate"]["max"][1]) - 
    read_scalar(const.config["StepPresets"][3]["coordinate"]["min"][1])) / const.dy
const.inj_dz = (
    read_scalar(const.config["StepPresets"][3]["coordinate"]["max"][2]) - 
    read_scalar(const.config["StepPresets"][3]["coordinate"]["min"][2])) / const.dz

logs = {
    "ecsimcorr": "logs/run.sh.o2244938",
    "eccapfim": "logs/run.sh.o2244941",
}

en_cols = {
    "ecsimcorr": (12, 13, 14, 15),
    "eccapfim": (6, 7, 8, 9),
}

d_w = 100
d_s = 250
d_t = int(2.7 * const.tau / const.dt)

def avg(d): return sliding_average(d, d_w)[::d_s]
    
def read_energy():
    cols=en_cols[sys.argv[1]]
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
    n = len(rm_i)
    return avg(rm_ni)[:n], avg(rm_ne)[:n], avg(inj_n)[:n]

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

inj_n_th = const.inj_dx * const.inj_dy * const.inj_dz * const.Np / (const.tau / const.dt)
inj_i_th = inj_n_th / const.Np * const.T_i * 3 / 2
inj_e_th = inj_n_th / const.Np * const.T_e * 3 / 2

ts = np.arange(d_t, step=d_s) * const.dt / const.tau

en.axis.plot(ts, rm_i, label="rm_i")
en.axis.plot(ts, rm_e, label="rm_e")
en.axis.plot(ts, inj_i, label="inj_i")
en.axis.plot(ts, inj_e, label="inj_e")
en.axis.plot([0, ts[-1]], [inj_i_th]*2, label="inj_i, th.", ls="--")
en.axis.plot([0, ts[-1]], [inj_e_th]*2, label="inj_e, th.", ls="--")
en.axis.legend(fontsize=0.8*FontSize.ticks)
en.info.draw()

n.axis.plot(ts, rm_ni, label="rm_ni")
n.axis.plot(ts, rm_ne, label="rm_ne")
n.axis.plot(ts, inj_n, label="inj_n")
n.axis.plot([0, ts[-1]], [inj_n_th]*2, label="inj_n, th.", ls="--")
n.axis.legend(fontsize=0.8*FontSize.ticks)
n.info.draw()

out_dir = f"{const.out_dir}/other"
makedirs(out_dir)

fig.tight_layout()
fig.savefig(f"{out_dir}/injection.png")