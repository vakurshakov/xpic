#!/usr/bin/env python3

from plot import *

fig, gs = figure(1, 3, figsize=(12, 17))

set_titlesize(34)
set_labelsize(30)
set_ticksize(28)

Lt = 1000

targs = dict(xlim=(0, Lt), xticks=np.linspace(0, Lt, 5), xlabel="$\\omega_{pe}t$")

energy = PlotLinear(axis=subplot(fig, gs, 0, 0))#, vmap=(26, 28))
energy.info.set_args(title="Electrons energy", ylabel="$K = \\sum\\limits_p m_p v_p^2 / 2$", **targs)

energy_cons = PlotLinear(axis=subplot(fig, gs, 0, 1))
energy_cons.info.set_args(title="Energy conservation", ylabel="$\\Delta(W_E + W_B + K)$", **targs)

charge = PlotLinear(axis=subplot(fig, gs, 0, 2))
charge.info.set_args(title="Charge conservation", ylabel="$[ \\sum\\limits_g (\\Delta \\rho_g / \\Delta t ~+~ {\\rm div} {\\bf J}_g )^2 \\, ]^{1/2}$", **targs)

def draw_impl(plot, s, d, l, dt, n=-1):
    plot.data = np.abs(np.loadtxt(f"{const.input_path}/../{s}/temporal/{d}.txt", skiprows=1)[:, n])
    if d == "energy":
        plot.plot_info["label"] = l

    plot.plot_info["linewidth"] = 2
    plot.draw(np.arange(len(plot.data)) * dt)
    plot.info.draw()

def draw(s, l, dt):
    # draw_impl(energy, s, "energy", l, dt, 1)
    draw_impl(energy, s, "energy", l, dt, 3)
    draw_impl(charge, s, "charge_conservation", l, dt)
    draw_impl(energy_cons, s, "energy_conservation", l, dt)

    energy.axis.legend(fontsize=0.55 * labelsize, loc="upper left", framealpha=1)
    energy_cons.axis.set_yscale("log")

draw("basic", "explicit", 0.5)
draw("ecsimcorr", "sim, $\\Delta t = 0.5, N_p = 50$", 0.5)
draw("eccapfim_dt0.5_np50", "fim, $\\Delta t = 0.5, N_p = 50$", 0.5)
draw("eccapfim_dt0.5_np100", "fim, $\\Delta t = 0.5, N_p = 100$", 0.5)
draw("eccapfim_dt0.25_np50", "fim, $\\Delta t = 0.25, N_p = 50$", 0.25)

energy_cons.axis.plot([0, Lt], [1e-7]*2, linestyle="--", linewidth=2, color="black")

fig.tight_layout()

res_dir = f"{const.output_path}/../../processed/other"
makedirs(res_dir)

fig.savefig(f"{res_dir}/conserved_props.png")
