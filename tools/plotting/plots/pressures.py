#!/usr/bin/env python3

from plot import *

# Example of the plot with 2D image and phi-averaged 1D plot

vmap_i = (0, +2e-7)
vmap_e = (0, +2e-3)
cmap = unsigned_cmap

rs = np.arange(0, const.boundaries['Z'][1])

pr_i = gen_plot("$\\Pi_{rr}^i$", "ions/mVrVr_PlaneZ_05", 'Z', '', 1, vmap_i, cmap)
pr_e = gen_plot("$\\Pi_{rr}^e$", "electrons/mVrVr_PlaneZ_05", 'Z', '', 1, vmap_e, cmap)
pa_i = gen_plot("$\\Pi_{\\phi\\phi}^i$", "ions/mVphiVphi_PlaneZ_05", 'Z', '', 1, vmap_i, cmap)
pa_e = gen_plot("$\\Pi_{\\phi\\phi}^e$", "electrons/mVphiVphi_PlaneZ_05", 'Z', '', 1, vmap_e, cmap)

p_s = (pr_i, pr_e, pa_i, pa_e)

pr_i_av = gen_linear("$\\bar{\\Pi}_{rr}^i$", 'Z', vmap_i, linewidth=3)
pr_e_av = gen_linear("$\\bar{\\Pi}_{rr}^e$", 'Z', vmap_e, linewidth=3)
pa_i_av = gen_linear("$\\bar{\\Pi}_{\\phi\\phi}^i$", 'Z', vmap_i, linewidth=3)
pa_e_av = gen_linear("$\\bar{\\Pi}_{\\phi\\phi}^e$", 'Z', vmap_e, linewidth=3)

p_s_av = (pr_i_av, pr_e_av, pa_i_av, pa_e_av)

def callback(t):
    for p, av in zip(p_s, p_s_av):
        p.data = p.view.parse(t)
        p.draw()

        av.data = phi_averaged(pr_i.data, const.rmap)
        p.draw(rs)

process_plots("pressures", time_wpe, p_s + p_s_av, callback)
