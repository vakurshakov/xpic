#!/usr/bin/env python3

from plot import *

from scipy.optimize import fsolve

const.B0 = 0.2
const.dt = 1.5
const.dts = 500 * const.dt
const.tau = 60000 * const.dt
const.T_e = 2
const.T_i = 8

fig, gs = figure(3, 1, figsize=(18, 6))

def plot(title, i):
    plot = PlotLinear(subplot(fig, gs, i, 0))
    plot.info.set_args(
        title=title,
        xlabel="$t / \\tau$"
    )
    return plot

d_t = int(3 * const.tau / const.dt)
d_s = 500

# ts = np.arange(0, d_t, d_s) * const.dt / const.tau
def ts(d): return np.arange(0, len(d)) * const.dts / const.tau

def read(sim, n):
    return np.load(f"./results/{sim}/processed/collections/{n}_t.npy")#[:int(d_t / d_s)]

b = plot("\\rm Magnetic field, $B_z / B_v$", 0)
n = plot("\\rm Ion density, $n_i$", 1)
p = plot("\\rm Plasma beta, $\\beta$", 2)

def plot(s, l):
    px = 2 * (read(s, "pxi") + read(s, "pxe")) / np.square(const.B0)
    t = ts(px)

    b.axis.plot(t, read(s, "bz") / const.B0, label=f"\\rm {l}")
    n.axis.plot(t, read(s, "ni"), label=f"\\rm {l}")
    p.axis.plot(t, px, label=f"\\rm {l}")

plot("ecsimcorr_dx3.0_dt1.5_Np1000", "ecsimcorr")
plot("eccapfim_dx3.0_dt1.5_Np1000", "eccapfim, $\\epsilon_a = 10^{-7}, \\epsilon_r = 10^{-5}$")
plot("eccapfim_dx3.0_dt1.5_Np1000_tol1e-07", "eccapfim, $\\epsilon_a = \\epsilon_r = 10^{-7}$")

for d in [b, n, p]:
    d.axis.grid()
    d.info.draw()

p.axis.legend(fontsize=0.8*FontSize.label)

out_dir = f"./results/other"
makedirs(out_dir)

fig.tight_layout(w_pad=1.3)
fig.savefig(f"{out_dir}/center.png")