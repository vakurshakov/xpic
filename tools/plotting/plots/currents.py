#!/usr/bin/env python3

from plot import *

vmap = np.array([-2e-2, +2e-2])

vmap_s = {
    "ions": 1e-3 * vmap,
    "electrons": vmap,
}

plots = []

for s in const.sorts:
    plots.append(gen_plot(f"$J_x^{s[0]}$", f"{s}/Vx_PlaneX_05", 'Z', 'x', 1, vmap_s.get(s)))
    plots.append(gen_plot(f"$J_y^{s[0]}$", f"{s}/Vy_PlaneX_05", 'Z', 'y', 1, vmap_s.get(s)))
    plots.append(gen_plot(f"$J_z^{s[0]}$", f"{s}/Vz_PlaneZ_05", 'X', 'z', 1, vmap_s.get(s)))

process_basic("currents", lambda t: f"$t = {t * const.dt:.3f}$", plots)
