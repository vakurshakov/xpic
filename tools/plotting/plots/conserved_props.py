#!/usr/bin/env python3

from plot import *

fig, gs = figure(1, 3, figsize=(12, 17))

set_titlesize(34)
set_labelsize(30)
set_ticksize(28)

const.input_path = "results/Langmuir_wave/"

Lt = 30

targs = dict(xlim=(0, Lt), xticks=np.linspace(0, Lt, 7), xlabel="$\\omega_{pe}t$")

energy = PlotLinear(axis=subplot(fig, gs, 0, 0))#, vmap=(26, 28))
# energy.info.set_args(title="Electrons energy", ylabel="$K = \\sum\\limits_p m_p v_p^2 / 2$", **targs)
energy.info.set_args(title="Electric energy", ylabel="$W_E = \\sum\\limits_g {\\bf E}_g^2 / 2$", **targs)

energy_cons = PlotLinear(axis=subplot(fig, gs, 0, 1))
energy_cons.info.set_args(title="Energy conservation", ylabel="$\\Delta(W_E + W_B + K)$", **targs)

charge = PlotLinear(axis=subplot(fig, gs, 0, 2))
charge.info.set_args(title="Charge conservation", ylabel="$[ \\sum\\limits_g (\\Delta \\rho_g / \\Delta t ~+~ {\\rm div} {\\bf J}_g )^2 \\, ]^{1/2}$", **targs)

def draw_impl(plot, s, d, l, dt, n=-1):
    plot.data = np.abs(np.loadtxt(f"{const.input_path}/{s}/temporal/{d}.txt", skiprows=1)[:, n])
    if d == "energy":
        plot.plot_info["label"] = l
        # plot.data = (plot.data / plot.data[0] - 1)

    plot.plot_info["linewidth"] = 2
    plot.draw(np.arange(len(plot.data)) * dt)
    plot.info.draw()

def draw(s, l, dt):
    draw_impl(energy, s, "energy", l, dt, 1)
    draw_impl(charge, s, "charge_conservation", l, dt)
    draw_impl(energy_cons, s, "energy_conservation", l, dt)

    energy_cons.axis.set_yscale("log")

ts = np.linspace(0, Lt, 300)

draw("basic_nz32_dx0.5_np1000", "explicit", 0.1)
draw("ecsimcorr_nz32_dx0.5_np1000", "semi im.", 0.1)
draw("eccapfim_nz32_dx0.5_np1000", "fully im.", 0.1)

wp = 0.994
energy.axis.plot(ts, 0.56 * (np.square(np.sin(wp * ts))), label=f"$A \\sin^2(\\omega t), ~ \\omega \\approx {wp:3}$", linewidth=2)
energy.axis.legend(fontsize=0.55 * labelsize, bbox_to_anchor=(0.68, 1.03), framealpha=1)

energy_cons.axis.plot([0, Lt], [1e-7]*2, linestyle="--", linewidth=2, color="black")

fig.tight_layout()

res_dir = f"{const.output_path}/../../processed/other"
makedirs(res_dir)

fig.savefig(f"{res_dir}/conserved_props.png")
