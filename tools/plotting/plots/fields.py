#!/usr/bin/env python3

import argparse

from plot import *

parser = argparse.ArgumentParser(prog="fields")

parser.add_argument("-c", "--cyl", action="store_true", help="draw fields in cylindrical components")
parser.add_argument("-v", "--vmap", action="store_const", default=0.02, help="colormap scale used for \'Plot\'")

args = parser.parse_args()

c1 = n1 = 'x'
c2 = n2 = 'y'
c3 = n3 = 'z'

if args.cyl:
    c1 = 'r'; n1 = 'r'
    c2 = 'phi'; n2 = '\\phi'

vmap = args.vmap * np.array([-1, +1])

plots = []

def add(name, path, comp, vmap):
    diag = find_diag(f"FieldView.{path}")
    if not diag is None:
        path = get_diag_path(diag)
        plane = get(diag, "region.plane")

        # Use this to directly attach the generated plot to `plots` 
        plots.append(gen_plot(name, path, plane, comp, 3, vmap))

add(f"$E_{n1}$", "E.X", c1, vmap)
add(f"$E_{n2}$", "E.X", c2, vmap)
add(f"$E_{n3}$", "E.X", c3, vmap)

add(f"$E_{n1}$", "E.Z", c1, vmap)
add(f"$E_{n2}$", "E.Z", c2, vmap)
add(f"$E_{n3}$", "E.Z", c3, vmap)

add(f"$B_{n1}$", "B.X", c1, vmap)
add(f"$B_{n2}$", "B.X", c2, vmap)
add(f"$B_{n3}$", "B.X", c3, const.B0+vmap)

add(f"$B_{n1}$", "B.Z", c1, vmap)
add(f"$B_{n2}$", "B.Z", c2, vmap)
add(f"$B_{n3}$", "B.Z", c3, const.B0+vmap)

process_basic("fields", lambda t: f"$t = {t * const.dt:.3f}$", plots)
