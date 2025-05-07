#!/usr/bin/env python3

import os, sys

from plot import *

test_output = "basic/output/basic_ex1/"

const = Constants()
const.input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../tests/", test_output)
const.output_path = os.path.join(const.input_path, "processed")

const.ncols = 3
const.nrows = 3

with open(os.path.join(const.input_path, "config.json"), "r") as file:
    config = json.load(file)

const.init_from_config(config)
const.init_dependent()

vmap_e = np.array([-0.002, +0.002])
vmap_b = vmap_e * 0.1
vmap_n = vmap_e * 40 + 1

plane = 'Z'
plane_position = sys.argv[1] if len(sys.argv) == 2 else const.Nz // 2

def gen_plot_tests(title, path, comp, dof, vmap, cmap=signed_cmap):
    plot = gen_plot(title, path, plane, comp, dof, vmap, cmap, const=const)
    plot.view.region = FieldView.Region(dof, (0, 0, 0, 0), (const.Nx, const.Ny, const.Nz, dof))
    plot.view.plane_position = plane_position
    return plot

process_basic(f"PlaneZ_{plane_position}", time_wpe, (
    gen_plot_tests("$E_x$", 'E', 'x', 3, vmap_e),
    gen_plot_tests("$E_y$", 'E', 'y', 3, vmap_e),
    gen_plot_tests("$E_z$", 'E', 'z', 3, vmap_e),

    gen_plot_tests("$B_x$", 'B', 'x', 3, vmap_b),
    gen_plot_tests("$B_y$", 'B', 'y', 3, vmap_b),
    gen_plot_tests("$B_z$", 'B', 'z', 3, vmap_b),

    gen_plot_tests("$n_e$", "electrons/Density", '', 1, vmap_n),
), const)
