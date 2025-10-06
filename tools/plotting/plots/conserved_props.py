#!/usr/bin/env python3

from plot import *

fig, gs = figure(1, 2, figsize=(10, 15))

ts = np.arange(0, const.Nt + 1) * const.dt

targs = dict(xlim=(ts[0], ts[-1]), xticks=np.linspace(ts[0], ts[-1], 5), xlabel="$\\omega_{pe}t$")

energy = PlotLinear(axis=subplot(fig, gs, 0, 0))
energy.info.set_args(title="Energy conservation", **targs)
energy.data = np.loadtxt(f"{const.input_path}/temporal/energy_conservation.txt", skiprows=1)[:, -1]
energy.draw(ts)

charge = PlotLinear(axis=subplot(fig, gs, 0, 1))
charge.info.set_args(title="Charge conservation", ylabel="$\\Big[ \\sum\\limits_g \\Big(\\Delta \\rho_g / \\Delta t ~+~ {\\rm div} {\\bf J}_g \\Big)^2 \\, \\Big]^{1/2}$", **targs)
charge.data = np.loadtxt(f"{const.input_path}/temporal/charge_conservation.txt", skiprows=1)[:, -1]
charge.draw(ts)

fig.tight_layout()

res_dir = f"{const.output_path}/other"
makedirs(res_dir)

fig.savefig(f"{res_dir}/conserved_props.png")
