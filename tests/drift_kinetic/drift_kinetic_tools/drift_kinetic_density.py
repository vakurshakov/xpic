#!/usr/bin/env python3
"""Visualize particle density (DistributionMoment, moment="density") for one
or more species from a drift_kinetic run.

For every requested species the script produces a row of three panels — center
slices through the X, Y, Z cutting planes — sharing one figure so electrons
and ions can be compared at a glance. Use ``--species`` to pick subsets
(e.g. only electrons, only ions, or both); by default every species with a
density diagnostic in ``config.json`` is included.

Colormap range is fixed per species across all snapshots so the scale does
not jump from frame to frame.
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np

TOOLS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS_DIR)

from lib.plot import PlotIm, bbox, labelsize, unsigned_cmap
from lib.plot_utils import figure, subplot
from lib.constants import const, init_constants

import shutil
import matplotlib.pyplot as plt

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)


PLANES = ("X", "Y", "Z")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--species", nargs="+", default=None,
                        help="Particle sorts to plot (default: every species "
                             "with a density diagnostic in the config)")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out-subdir", default="density",
                        help="Subdirectory under <out_dir>/processed for PNGs")
    return parser.parse_args()


def plane_layout(plane: str):
    """Return ((rows, cols), (xlabel_axis, ylabel_axis), (Lx_plot, Ly_plot))."""
    if plane == "X":
        return (const.Nz, const.Ny), ("y", "z"), (const.Ly, const.Lz)
    if plane == "Y":
        return (const.Nz, const.Nx), ("x", "z"), (const.Lx, const.Lz)
    if plane == "Z":
        return (const.Ny, const.Nx), ("x", "y"), (const.Lx, const.Ly)
    raise ValueError(f"Unknown plane {plane!r}")


def iter_density_diagnostics():
    """Yield (species, explicit_out_dir or None) for every density moment."""
    seen = set()
    for diag in const.config.get("Diagnostics", []):
        if diag.get("diagnostic") != "DistributionMoment":
            continue
        if diag.get("moment") != "density":
            continue
        species = diag.get("particles")
        if species is None or species in seen:
            continue
        seen.add(species)
        yield species, diag.get("out_dir")


def candidate_dirs(species: str, explicit: str | None):
    if explicit is not None:
        return [explicit]
    return [
        os.path.join(species, "density"),
        f"{species}_density",
    ]


def list_timesteps(dir_path: str):
    entries = []
    for name in os.listdir(dir_path):
        full = os.path.join(dir_path, name)
        if os.path.isfile(full) and name.isdigit():
            entries.append((int(name), name))
    entries.sort()
    return entries


def expected_size_bytes():
    return const.Nx * const.Ny * const.Nz * 4


def locate_dir(species: str, explicit: str | None):
    expected = expected_size_bytes()
    for sub in candidate_dirs(species, explicit):
        dir_path = os.path.join(const.in_dir, sub)
        if not os.path.isdir(dir_path):
            continue
        steps = [
            (idx, name) for (idx, name) in list_timesteps(dir_path)
            if os.path.getsize(os.path.join(dir_path, name)) == expected
        ]
        if steps:
            return dir_path, steps
    return None, None


def load_frame(dir_path: str, name: str):
    path = os.path.join(dir_path, name)
    if os.path.getsize(path) != expected_size_bytes():
        return None
    raw = np.fromfile(path, dtype=np.float32,
                      count=const.Nx * const.Ny * const.Nz)
    return raw.reshape(const.Nz, const.Ny, const.Nx)


def slice_plane(frame: np.ndarray, plane: str):
    """Center cut for the requested plane. Returns 2D array."""
    if plane == "X":
        return frame[:, :, const.Nx // 2]
    if plane == "Y":
        return frame[:, const.Ny // 2, :]
    if plane == "Z":
        return frame[const.Nz // 2, :, :]
    raise ValueError(f"Unknown plane {plane!r}")


def collect_rows(species_filter):
    rows = []
    available = list(iter_density_diagnostics())
    if species_filter is not None:
        wanted = set(species_filter)
        available_names = {sp for sp, _ in available}
        missing = wanted - available_names
        for name in sorted(missing):
            print(f"[skip] no density diagnostic for species '{name}' in config")
        available = [(sp, ex) for sp, ex in available if sp in wanted]

    for species, explicit in available:
        dir_path, steps = locate_dir(species, explicit)
        if dir_path is None:
            print(f"[skip] no frames for species '{species}' "
                  f"(looked for {candidate_dirs(species, explicit)})")
            continue
        rows.append({
            "species": species,
            "dir": dir_path,
            "timesteps": steps,
        })
    return rows


def compute_vmax(rows, name_by_row):
    for row, names in zip(rows, name_by_row):
        vmax = 0.0
        for name in names:
            data = load_frame(row["dir"], name)
            if data is None:
                continue
            vmax = max(vmax, float(np.max(np.abs(data))))
        row["vmax"] = vmax if vmax > 0 else 1.0


def make_plot(fig, gs, i, j, row, plane):
    species = row["species"]
    vmax = row["vmax"]
    _, (xl, yl), (Lx_plot, Ly_plot) = plane_layout(plane)
    ax = subplot(fig, gs, j, i)
    axis_label = {
        "x": r"$x,~c/\omega_{pe}$",
        "y": r"$y,~c/\omega_{pe}$",
        "z": r"$z,~c/\omega_{pe}$",
    }
    title = rf"$n_{{\mathrm{{{species}}}}}\,(\mathrm{{plane}}\ {plane})$"
    plot = PlotIm(ax, (0.0, vmax), unsigned_cmap)
    plot.bounds = (0.0, Lx_plot, 0.0, Ly_plot)
    plot.info.set_args(
        title=title,
        xlim=(0.0, Lx_plot),
        ylim=(0.0, Ly_plot),
        xlabel=axis_label[xl],
        ylabel=axis_label[yl],
    )
    return plot


def main():
    args = parse_args()
    init_constants(args.config)

    rows = collect_rows(args.species)
    if not rows:
        print("No density diagnostics found for the requested species.")
        return

    common = set(idx for idx, _ in rows[0]["timesteps"])
    for row in rows[1:]:
        common &= set(idx for idx, _ in row["timesteps"])
    common = sorted(common)
    if not common:
        print("No timesteps common to all requested density diagnostics.")
        return

    names_per_row = []
    for row in rows:
        name_by_idx = dict(row["timesteps"])
        names_per_row.append([name_by_idx[idx] for idx in common])

    compute_vmax(rows, names_per_row)

    ncols = len(PLANES)
    nrows = len(rows)
    fig, gs = figure(ncols, nrows)

    grid = []
    for i, row in enumerate(rows):
        row_plots = [make_plot(fig, gs, i, j, row, plane)
                     for j, plane in enumerate(PLANES)]
        grid.append(row_plots)

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    for k, idx in enumerate(common):
        figname = os.path.join(out_dir, f"{idx:04d}.png")
        print(f"Processing {figname} (idx={idx})")

        for row, row_plots, names in zip(rows, grid, names_per_row):
            data = load_frame(row["dir"], names[k])
            if data is None:
                continue
            for j, plane in enumerate(PLANES):
                row_plots[j].data = slice_plane(data, plane)
                row_plots[j].draw()

        fig.suptitle(
            rf"$\omega_{{pe}}\,t = {idx * const.dts:.2f}$",
            x=0.5, y=0.995, bbox=bbox, fontsize=labelsize,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(figname, dpi=args.dpi)

        for row_plots in grid:
            for plot in row_plots:
                plot.clear()

    print(f"Frames written to {out_dir}")


if __name__ == "__main__":
    main()
