#!/usr/bin/env python3

import argparse

from plot import *

parser = argparse.ArgumentParser(prog="fields")

parser.add_argument("-c", "--cyl", action="store_true", help="draw fields in cylindrical components")
parser.add_argument("-v", "--vmap", action="store_const", default=0.02, help="colormap scale used for \'Plot\'")

args = parser.parse_args()

coords = FieldView.Cartesian if not args.cyl else FieldView.Cylinder

comps = {
    0: 'x' if coords == FieldView.Cartesian else 'r',
    1: 'y' if coords == FieldView.Cartesian else '\\phi',
    2: 'z',
}

vmap = args.vmap * np.array([-1, +1])

plots = []

def add(path, comp, vmap):
    diag = find_diag(f"FieldView.{path}")
    if not diag is None:
        name = path.split('.')[0]
        path = get_diag_path(diag)
        plane = get(diag, "region.plane")
        plots.append(gen_plot(f"${name}_{{ {comps.get(comp)} }}$", path, plane, comp, 3, coords, vmap))

add("E.X", 0, vmap)
add("E.X", 1, vmap)
add("E.X", 2, vmap)

add("E.Z", 0, vmap)
add("E.Z", 1, vmap)
add("E.Z", 2, vmap)

add("B.X", 0, vmap)
add("B.X", 1, vmap)
add("B.X", 2, const.B0+vmap)

add("B.Z", 0, vmap)
add("B.Z", 1, vmap)
add("B.Z", 2, const.B0+vmap)

process_basic("fields", lambda t: f"$t = {t * const.dt:.3f}$", plots)
