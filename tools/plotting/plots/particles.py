#!/usr/bin/env python3

from plot import *

vmap_j = {
    "ions": np.array([-2e-5, +2e-5]),
    "electrons": np.array([-2e-2, +2e-2]),
}

vmap_n = 1 + 0.01 * vmap_j["electrons"]

for s in sorts:
    process_basic(f"info_{s}", lambda t: f"$t = {t * dts:.3f}$", (
        gen_plot(f"$n_{s[0]}$", f"{s}/Density_PlaneZ_05", 'Z', '', 1, vmap_n, unsigned_cmap),
        gen_plot(f"$J_x^{s[0]}$", f"{s}/Vx_PlaneZ_05", 'Z', 'x', 1, vmap_j.get(s)),
        gen_plot(f"$J_y^{s[0]}$", f"{s}/Vy_PlaneZ_05", 'Z', 'y', 1, vmap_j.get(s)),
        gen_plot(f"$J_z^{s[0]}$", f"{s}/Vz_PlaneZ_05", 'Z', 'z', 1, vmap_j.get(s)),

        gen_plot(f"$n_{s[0]}$", f"{s}/Density_PlaneX_05", 'X', '', 1, vmap_n, unsigned_cmap),
        gen_plot(f"$J_x^{s[0]}$", f"{s}/Vx_PlaneX_05", 'X', 'x', 1, vmap_j.get(s)),
        gen_plot(f"$J_y^{s[0]}$", f"{s}/Vy_PlaneX_05", 'X', 'y', 1, vmap_j.get(s)),
        gen_plot(f"$J_z^{s[0]}$", f"{s}/Vz_PlaneX_05", 'X', 'z', 1, vmap_j.get(s)),
    ))
