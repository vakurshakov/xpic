#!/usr/bin/python

import sys
import struct
from pathlib import Path

sys.path.append(str(Path(__file__).parents[5]))

from tools.plotting.plots.plot import *

const.Nt = 0

vmap_n = (0, 1)
vmap_j = 1e-3 * np.asarray((-1, +1))
vmap_p = (0, 0.02)
vmap_b = (0, 0.2)

rs = np.arange(0, const.boundaries['Z'][1])

n = gen_plot("$n$", "ions/density_planeZ_0001", 'Z', 0, 1, FieldView.Cartesian, vmap_n, unsigned_cmap)
ja = gen_plot("$J_{\\phi}$", "ions/current_planeZ_0001", 'Z', 1, 3, FieldView.Cylindrical, vmap_j, signed_cmap)
prr = gen_plot("$\\Pi_{rr}$", "ions/momentum_flux_diag_cyl_planeZ_0001", 'Z', 0, 3, FieldView.Cartesian, vmap_p, unsigned_cmap)
paa = gen_plot("$\\Pi_{\\phi\\phi}$", "ions/momentum_flux_diag_cyl_planeZ_0001", 'Z', 1, 3, FieldView.Cartesian, vmap_p, unsigned_cmap)
pzz = gen_plot("$\\Pi_{\\phi\\phi}$", "ions/momentum_flux_diag_cyl_planeZ_0001", 'Z', 2, 3, FieldView.Cartesian, vmap_p, unsigned_cmap)
bz = gen_plot("$B_z$", "B_planeZ_0001", 'Z', 2, 3, FieldView.Cylindrical, vmap_b, unsigned_cmap)

plots = (n, ja, prr, bz)

n_av = gen_linear("$n$", 'Z', vmap_n, linewidth=3)
ja_av = gen_linear("$J_{\\phi}$", 'Z', vmap_j, linewidth=3)
prr_av = gen_linear("\\rm Pressures", 'Z', vmap_p, linewidth=3)
paa_av = gen_linear("", 'Z', vmap_p, linewidth=3)
pzz_av = gen_linear("", 'Z', vmap_p, linewidth=3)
bz_av = gen_linear("$B_z$", 'Z', vmap_b, linewidth=3)

plots_av = (n_av, ja_av, prr_av, bz_av)

def get_theory(name):
    cache = Path(__file__).parent / f"../cache/a1.5_B0.01/maxw_{name}"
    with open(cache, "rb") as file:
        buff = file.read()
        return np.asarray(struct.unpack('d' * (len(buff) // 8), buff))

data = get_theory("n")

xmin = data[0]
xmax = data[1]
dx = data[2]

rth = np.arange(xmin, xmax + dx, dx) * np.sqrt(mi_me)

vi = np.sqrt(T_i / mi_me)

n_th = data[3:]
ja_th = get_theory("j")[3:] * vi
prr_th = get_theory("prr")[3:] * T_i
paa_th = get_theory("paa")[3:] * T_i
bz_th = get_theory("b")[3:] * np.sqrt(T_i)

plots_th = (n_th, ja_th, prr_th, bz_th)

def callback(t):
    for p, av, th in zip(plots, plots_av, plots_th):
        p.data = p.view.parse(t)
        p.draw()
        av.data = phi_averaged(p.data, const.rmap)

        ax = av.axis
        if p == prr:
            paa.data = paa.view.parse(t)
            pzz.data = pzz.view.parse(t)
            paa_av.data = phi_averaged(paa.data, const.rmap)
            pzz_av.data = phi_averaged(pzz.data, const.rmap)

            ax.plot(rs, prr_av.data, **prr_av.plot_info, label="$\\Pi_{rr}$")
            ax.plot(rth, prr_th, **prr_av.plot_info, linestyle="--", label="$\\Pi_{rr}^{th}$")

            ax.plot(rs, paa_av.data, **paa_av.plot_info, label="$\\Pi_{\\phi\\phi}$")
            ax.plot(rth, paa_th, **paa_av.plot_info, linestyle="--", label="$\\Pi_{\\phi\\phi}^{th}$")

            ax.plot(rs, pzz_av.data, **pzz_av.plot_info, label="$\\Pi_{zz}$")

            ax.legend(fontsize=ticksize)
        else:
            ax.plot(rs, av.data, **av.plot_info)
            ax.plot(rth, th, **av.plot_info, linestyle="--")

        av.draw_info()
        ax.set_xlim(0, 250)

    n_av.axis.set_ylim(-0.05, 1.05)


process_plots("comparison", lambda t: f"$t = {t * const.dt:.3f}$", plots + plots_av, callback)
