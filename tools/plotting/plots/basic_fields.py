#!/usr/bin/env python3

from plot import *

# This is the most basic example of the plot

vmap = np.array([-0.02, +0.02])

process_basic("fields", lambda t: f"$t = {t * const.dt:.3f}$", (
    gen_plot("$E_x$", "E_planeX_05", 'X', 'x', 3, vmap),
    gen_plot("$E_y$", "E_planeX_05", 'X', 'y', 3, vmap),
    gen_plot("$E_z$", "E_planeX_05", 'X', 'z', 3, vmap),

    gen_plot("$E_x$", "E_planeZ_05", 'Z', 'x', 3, vmap),
    gen_plot("$E_y$", "E_planeZ_05", 'Z', 'y', 3, vmap),
    gen_plot("$E_z$", "E_planeZ_05", 'Z', 'z', 3, vmap),

    gen_plot("$B_x$", "B_planeX_05", 'X', 'x', 3, vmap),
    gen_plot("$B_y$", "B_planeX_05", 'X', 'y', 3, vmap),
    gen_plot("$B_z$", "B_planeZ_05", 'Z', 'z', 3, const.B0+vmap),

    gen_plot("$B_x$", "B_planeZ_05", 'Z', 'x', 3, vmap),
    gen_plot("$B_y$", "B_planeZ_05", 'Z', 'y', 3, vmap),
    gen_plot("$B_z$", "B_planeX_05", 'X', 'z', 3, const.B0+vmap),
))
