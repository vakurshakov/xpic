#!/usr/bin/env python3

from plot import *

vmap_e = np.array([-0.02, +0.02])
vmap_i = 0.001 * vmap_e

process_basic("currents", lambda t: f"$t = {t * const.dt:.3f}$", (
    gen_plot("$J_x^i$", "ions/Vx_PlaneX_05", 'Z', 'x', 1, vmap_i),
    gen_plot("$J_y^i$", "ions/Vy_PlaneX_05", 'Z', 'y', 1, vmap_i),
    gen_plot("$J_z^i$", "ions/Vz_PlaneZ_05", 'X', 'z', 1, vmap_i),

    gen_plot("$J_x^e$", "electrons/Vx_PlaneX_05", 'Z', 'x', 1, vmap_e),
    gen_plot("$J_y^e$", "electrons/Vy_PlaneZ_05", 'Z', 'y', 1, vmap_e),
    gen_plot("$J_z^e$", "electrons/Vz_PlaneZ_05", 'X', 'z', 1, vmap_e),
))
