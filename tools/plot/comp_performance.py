#!/usr/bin/env python3

from plot import *

const.dt = 1.5
const.tau = 60000 * const.dt

d_t = int(3 * const.tau / const.dt)

def read_log(sim):
    fname = f"./results/{sim}_dx3.0_dt1.5_Np1000/log-EachTimestep.txt"
    return np.loadtxt(fname, skiprows=1, usecols=1)

dt_ecsimcorr = read_log("ecsimcorr")
# dt_ecsimcorr[::4999] = dt_ecsimcorr[::4998]

dt_eccapfim = read_log("eccapfim")
# dt_eccapfim[::4999] = dt_eccapfim[::4998]

fig, gs = figure(2, 1, figsize=(18, 6))

def plot(title, i):
    plot = PlotLinear(subplot(fig, gs, i, 0))
    plot.info.set_args(
        title=title,
        xlabel="$t / \\tau$"
    )
    return plot

tt = plot("\\rm timestep time, sec", 0)
ct = plot("\\rm integral time, hours", 1)

def ts(d): return np.arange(len(d)) * const.dt / const.tau

ts0 = ts(dt_ecsimcorr)
ts1 = ts(dt_eccapfim)

tt.axis.plot(ts0, dt_ecsimcorr, label="\\rm ecsimcorr")
tt.axis.plot(ts1, dt_eccapfim, label="\\rm eccfim")
tt.axis.set_ylim(40 * (-0.1), 40 * (1.1))
tt.axis.grid()
tt.axis.legend(fontsize=0.8*FontSize.label, loc="upper right", framealpha=1)
tt.info.draw()

ct.axis.plot(ts0, np.cumsum(dt_ecsimcorr) / (60 * 60), label="\\rm ecsimcorr")
ct.axis.plot(ts1, np.cumsum(dt_eccapfim) / (60 * 60), label="\\rm eccfim")
ct.axis.set_ylim(500 * (-0.1), 500 * (1.1))
ct.axis.grid()
ct.axis.legend(fontsize=0.8*FontSize.label)
ct.info.draw()

out_dir = f"./results/other"
makedirs(out_dir)

fig.tight_layout()
fig.savefig(f"{out_dir}/performance.png")