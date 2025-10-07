#!/usr/bin/env python3

from plot import *

fig, gs = figure(1, 3, figsize=(12, 17))

set_titlesize(34)
set_labelsize(30)
set_ticksize(28)

const.Nt = 2000
const.dt = 0.5

ts = np.arange(0, const.Nt + 1) * const.dt
const.Lt = const.Nt * const.dt

targs = dict(xlim=(0, const.Lt), xticks=np.linspace(0, const.Lt, 5), xlabel="$\\omega_{pe}t$")

energy = PlotLinear(axis=subplot(fig, gs, 0, 0), vmap=(26, 28))
energy.info.set_args(title="Electrons energy", ylabel="$K = \\sum\\limits_p m_p v_p^2 / 2$", **targs)

energy_cons = PlotLinear(axis=subplot(fig, gs, 0, 1))
energy_cons.info.set_args(title="Energy conservation", ylabel="$\\Delta(W_E + W_B + K)$", **targs)

charge = PlotLinear(axis=subplot(fig, gs, 0, 2), vmap=(0, 1e-13))
charge.info.set_args(title="Charge conservation", ylabel="$[ \\sum\\limits_g (\\Delta \\rho_g / \\Delta t ~+~ {\\rm div} {\\bf J}_g )^2 \\, ]^{1/2}$", **targs)

def draw_impl(plot, s, d, l, n=-1):
    plot.data = np.abs(np.loadtxt(f"{const.input_path}/../{s}/temporal/{d}.txt", skiprows=1)[:, n])
    if d == "energy":
        plot.plot_info["label"] = l
    plot.plot_info["linewidth"] = 2
    plot.draw(ts)
    plot.info.draw()

def draw(s, l):
    draw_impl(energy, s, "energy", l, 3)
    draw_impl(charge, s, "charge_conservation", l)
    draw_impl(energy_cons, s, "energy_conservation", l)

    energy.axis.legend(fontsize=0.75 * labelsize, loc="upper left", framealpha=1)
    energy_cons.axis.set_yscale("log")

draw("basic", "basic")
draw("ecsimcorr", "ecsimcorr")
draw("eccapfim_dt0.5", "eccapfim, $\\Delta t = 0.5$")
# draw("eccapfim", "eccapfim")

energy_cons.axis.plot([0, const.Lt], [1e-7]*2, linestyle="--", linewidth=2, color="black")

fig.tight_layout()

res_dir = f"{const.output_path}/../../processed/other"
makedirs(res_dir)

fig.savefig(f"{res_dir}/conserved_props.png")
