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

pr_i_avg = gen_linear("$\\bar{\\Pi}_{rr}^i$", 'Z', vmap_i, linewidth=3)
pr_e_avg = gen_linear("$\\bar{\\Pi}_{rr}^e$", 'Z', vmap_e, linewidth=3)
pa_i_avg = gen_linear("$\\bar{\\Pi}_{\\phi\\phi}^i$", 'Z', vmap_i, linewidth=3)
pa_e_avg = gen_linear("$\\bar{\\Pi}_{\\phi\\phi}^e$", 'Z', vmap_e, linewidth=3)

p_s_avg = (pr_i_avg, pr_e_avg, pa_i_avg, pa_e_avg)

def callback(t):
    for p in p_s:
        p.data = p.view.parse(t)
        p.draw()
    
    pr_i_avg.data = phi_averaged(pr_i.data, const.rmap)
    pa_i_avg.data = phi_averaged(pa_i.data, const.rmap)
    pr_e_avg.data = phi_averaged(pr_e.data, const.rmap)
    pa_e_avg.data = phi_averaged(pa_e.data, const.rmap)
    
    for p in p_s_avg:
        p.draw(rs)

process_plots("pressures", lambda t: f"$t = {t * const.dt:.3f}$", p_s + p_s_avg, callback)
