#!/usr/bin/env python3

from plot import *

vmap = np.array([-1e-2, +1e-2])

vmap_s = {
    "ions": vmap,
    "electrons": vmap,
}

plots = []

for s in const.sorts:
    plots.append(gen_plot(f"$J_x^{s[0]}$", f"{s}/current_planeZ_0001", 'Z', 0, 3, FieldView.Cartesian, vmap_s.get(s)))
    plots.append(gen_plot(f"$J_y^{s[0]}$", f"{s}/current_planeZ_0001", 'Z', 1, 3, FieldView.Cartesian, vmap_s.get(s)))
    plots.append(gen_plot(f"$J_z^{s[0]}$", f"{s}/current_planeZ_0001", 'Z', 2, 3, FieldView.Cartesian, vmap_s.get(s)))

process_basic("currents", lambda t: f"$t = {t * const.dt:.3f}$", plots)
