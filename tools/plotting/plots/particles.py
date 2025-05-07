#!/usr/bin/env python3

import argparse

from plot import *

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--cyl", action="store_true", help="draw fields in cylindrical components")
parser.add_argument("-n", "--nmap", action="store_const", default=1, help="scale of density plot colormap")

args = parser.parse_args()

coords = FieldView.Cartesian if not args.cyl else FieldView.Cylindrical

comps = {
    0: 'x' if coords == FieldView.Cartesian else 'r',
    1: 'y' if coords == FieldView.Cartesian else '\\phi',
    2: 'z',
}

vmap = np.array([-2e-2, +2e-2])

vmap_v = {
    "ions": 0.1 * vmap,
    "electrons": vmap,
}

vmap_n = np.array([0, args.nmap])

plots = []

def add(name, path, comp, dof, vmap, cmap=signed_cmap):
    diag = find_diag(f"DistributionMoment.{path}")
    if not diag is None:
        path = get_diag_path(diag)
        plane = get(diag, "region.plane")
        plots.append(gen_plot(name, path, plane, comp, dof, coords, vmap, cmap))

for s in const.sorts:
    plots.clear()

    add(f"$n_{s[0]}$", f"{s}.density.X", '', 1, vmap_n, unsigned_cmap),
    add(f"$J_{comps.get(0)}^{s[0]}$", f"{s}.current.X", 0, 3, vmap_v.get(s)),
    add(f"$J_{comps.get(1)}^{s[0]}$", f"{s}.current.X", 1, 3, vmap_v.get(s)),
    add(f"$J_{comps.get(2)}^{s[0]}$", f"{s}.current.X", 2, 3, vmap_v.get(s)),

    add(f"$n_{s[0]}$", f"{s}.density.Z", '', 1, vmap_n, unsigned_cmap),
    add(f"$J_{comps.get(0)}^{s[0]}$", f"{s}.current.Z", 0, 3, vmap_v.get(s)),
    add(f"$J_{comps.get(1)}^{s[0]}$", f"{s}.current.Z", 1, 3, vmap_v.get(s)),
    add(f"$J_{comps.get(2)}^{s[0]}$", f"{s}.current.Z", 2, 3, vmap_v.get(s)),

    const.nrows = len(plots) // 4
    const.ncols = len(plots) // const.nrows

    process_basic(f"info_{s}", time_wpe, plots)
