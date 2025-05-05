#!/usr/bin/env python3

from plot import *

vmap_i = 1e-3 * np.asarray((-1, +1))
vmap_e = 1e-3 * np.asarray((-1, +1))
cmap = unsigned_cmap

rs = np.arange(0, const.boundaries['Z'][1])

jai = gen_plot("$J^i_{\\phi}$", "ions/current_planeZ_0001", 'Z', 1, 3, FieldView.Cylindrical, vmap_i)
jae = gen_plot("$J^e_{\\phi}$", "electrons/current_planeZ_0001", 'Z', 1, 3, FieldView.Cylindrical, vmap_e)

js = (jai, jae)

jai_av = gen_linear("$J^i_{\\phi}$", 'Z', vmap_i, linewidth=3)
jae_av = gen_linear("$J^e_{\\phi}$", 'Z', vmap_e, linewidth=3)

js_av = (jai_av, jae_av)

def callback(t):
    for p in js:
        p.data = p.view.parse(t)
        p.draw()
    
    jai_av.data = phi_averaged(jai.data, const.rmap)
    jae_av.data = phi_averaged(jae.data, const.rmap)
    
    for p in js_av:
        p.draw(rs)
        p.axis.set_xlim(0, 150)

process_plots("currents", lambda t: f"$t = {t * const.dt:.3f}$", js + js_av, callback)
