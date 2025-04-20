#!/usr/bin/env python3

from plot import *

# This is the most basic example of the plot

vmap = np.array([-0.02, +0.02])

process_basic("fields", lambda t: f"$t / \\tau = {t * dts:.3f}$", (
    gen_plot("$E_x$", "E_PlaneZ_05", 'Z', 'x', 3, vmap),
    gen_plot("$E_x$", "E_PlaneX_05", 'X', 'x', 3, vmap),
    gen_plot("$B_x$", "B_PlaneZ_05", 'Z', 'x', 3, vmap),
    gen_plot("$B_x$", "B_PlaneX_05", 'X', 'x', 3, vmap),

    gen_plot("$E_y$", "E_PlaneZ_05", 'Z', 'y', 3, vmap),
    gen_plot("$E_y$", "E_PlaneX_05", 'X', 'y', 3, vmap),
    gen_plot("$B_y$", "B_PlaneZ_05", 'Z', 'y', 3, vmap),
    gen_plot("$B_y$", "B_PlaneX_05", 'X', 'y', 3, vmap),

    gen_plot("$E_z$", "E_PlaneZ_05", 'Z', 'z', 3, vmap),
    gen_plot("$E_z$", "E_PlaneX_05", 'X', 'z', 3, vmap),
    gen_plot("$B_z$", "B_PlaneZ_05", 'Z', 'z', 3, B0+vmap),
    gen_plot("$B_z$", "B_PlaneX_05", 'X', 'z', 3, B0+vmap),
))
