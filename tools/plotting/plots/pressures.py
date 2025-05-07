#!/usr/bin/env python3

from plot import *

vmap_i = (0, 0.05)
vmap_e = (0, 0.02 / 100)
cmap = unsigned_cmap

rs = np.arange(0, const.boundaries['Z'][1])

pr_i = gen_plot("$\\Pi_{rr}^i$",         "ions/momentum_flux_diag_cyl_planeZ_0001", 'Z', 0, 3, FieldView.Cartesian, vmap_i, cmap)
pa_i = gen_plot("$\\Pi_{\\phi\\phi}^i$", "ions/momentum_flux_diag_cyl_planeZ_0001", 'Z', 1, 3, FieldView.Cartesian, vmap_i, cmap)
pz_i = gen_plot("$\\Pi_{zz}^i$",         "ions/momentum_flux_diag_cyl_planeZ_0001", 'Z', 2, 3, FieldView.Cartesian, vmap_i, cmap)
pr_e = gen_plot("$\\Pi_{rr}^e$",         "electrons/momentum_flux_diag_cyl_planeZ_0001", 'Z', 0, 3, FieldView.Cartesian, vmap_e, cmap)
pa_e = gen_plot("$\\Pi_{\\phi\\phi}^e$", "electrons/momentum_flux_diag_cyl_planeZ_0001", 'Z', 1, 3, FieldView.Cartesian, vmap_e, cmap)
pz_e = gen_plot("$\\Pi_{zz}^e$",         "electrons/momentum_flux_diag_cyl_planeZ_0001", 'Z', 2, 3, FieldView.Cartesian, vmap_e, cmap)

p_s = (pr_i, pa_i, pz_i, pr_e, pa_e, pz_e)

pr_i_av = gen_linear("$\\Pi_{rr}^i(r)$",         'Z', vmap_i, linewidth=3)
pa_i_av = gen_linear("$\\Pi_{\\phi\\phi}^i(r)$", 'Z', vmap_i, linewidth=3)
pz_i_av = gen_linear("$\\Pi_{zz}^i(r)$",         'Z', vmap_i, linewidth=3)
pr_e_av = gen_linear("$\\Pi_{rr}^e(r)$",         'Z', vmap_e, linewidth=3)
pa_e_av = gen_linear("$\\Pi_{\\phi\\phi}^e(r)$", 'Z', vmap_e, linewidth=3)
pz_e_av = gen_linear("$\\Pi_{zz}^e(r)$",         'Z', vmap_e, linewidth=3)

p_s_av = (pr_i_av, pa_i_av, pz_i_av, pr_e_av, pa_e_av, pz_e_av)

const.nrows = 4
const.ncols = 3

def callback(t):
    for p, av in zip(p_s, p_s_av):
        p.data = p.view.parse(t)
        p.draw()

        av.data = phi_averaged(p.data, const.rmap)
        av.draw(rs)

process_plots("pressures", time_wpe, p_s + p_s_av, callback)
