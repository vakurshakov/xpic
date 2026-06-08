#!/usr/bin/env python3
"""Visualize FieldView diagnostics (E, B, J, M) in CYLINDRICAL components.

Same layout as drift_kinetic_fields.py — one row per (field, plane) with
columns ``F_r``, ``F_phi``, ``F_z`` and ``|F|`` — but each Cartesian
vector field is projected onto cylindrical components at every cell:

    F_r   =  F_x cos(phi) + F_y sin(phi)
    F_phi = -F_x sin(phi) + F_y cos(phi)
    F_z   =  F_z

where ``phi(x, y) = arctan2(y - cy, x - cx)`` and ``(cx, cy)`` is the
cylinder axis (auto-detected from CylinderGeometry in
FieldsDamping/SetParticles, or domain centre as fallback; override with
``--center``). For plane X the in-plane x is fixed at the slice position;
similarly for plane Y, so ``phi`` only varies along one axis there. The
Z-plane is the most useful case for axially symmetric problems.
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np

TOOLS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from lib.plot import PlotIm, bbox, labelsize, signed_cmap, unsigned_cmap
from lib.plot_utils import figure, subplot
from lib.constants import const, init_constants

import shutil
import matplotlib.pyplot as plt

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)

from drift_kinetic_equilibrium import auto_center
from drift_kinetic_fields import (
    FIELD_CHOICES, PSEUDO_FIELDS,
    iter_field_diagnostics, locate_dir, candidate_dirs, list_timesteps,
    load_frame, read_b0_vector,
    plane_layout, field_tex, compute_curl_2d,
    parse_field_spec, resolve_slice_idx,
)


COMPONENT_LABELS = ["r", r"\varphi", "z"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument(
        "--fields", nargs="+", default=FIELD_CHOICES,
        help=("Fields to visualize. Plain tokens (e.g. 'E') use 2D FieldView "
              "entries; '<field>:<axis>[:<idx>]' (e.g. 'E:z:10') slices a 3D "
              "FieldView at the given cell index (default N/2). Choices: "
              + ", ".join(FIELD_CHOICES) + "."))
    parser.add_argument("--center", nargs=2, type=float, default=None,
                        metavar=("CX", "CY"),
                        help="Cylinder axis (default: auto-detect from "
                             "CylinderGeometry in FieldsDamping or "
                             "SetParticles, else domain centre)")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out-subdir", default="fields_cyl")
    return parser.parse_args()


def plane_xy_grid(plane: str, slice_idx: int | None = None):
    """For each cell of the 2D plane slice, return cell-centred (x, y) world
    coordinates with shape matching the slice (h, w). One of x, y is constant
    if the slice is along plane X or Y; both vary on plane Z. When
    ``slice_idx`` is given the constant coordinate is taken at that cell;
    otherwise the slice is centred (the historical xpic 2D FieldView default).
    """
    if plane == "X":
        # slice at x = idx*dx (or domain centre); plane shape is (Nz, Ny)
        h, w = const.Nz, const.Ny
        x_const = (slice_idx + 0.5) * const.dx if slice_idx is not None \
            else 0.5 * const.Lx
        ys = (np.arange(w) + 0.5) * const.dy
        Y = np.broadcast_to(ys[None, :], (h, w))
        X = np.full((h, w), x_const, dtype=float)
    elif plane == "Y":
        # slice at y = idx*dy (or domain centre); plane shape is (Nz, Nx)
        h, w = const.Nz, const.Nx
        y_const = (slice_idx + 0.5) * const.dy if slice_idx is not None \
            else 0.5 * const.Ly
        xs = (np.arange(w) + 0.5) * const.dx
        X = np.broadcast_to(xs[None, :], (h, w))
        Y = np.full((h, w), y_const, dtype=float)
    elif plane == "Z":
        # slice at z = 0.5 * Lz; plane shape is (Ny, Nx)
        h, w = const.Ny, const.Nx
        xs = (np.arange(w) + 0.5) * const.dx
        ys = (np.arange(h) + 0.5) * const.dy
        X, Y = np.meshgrid(xs, ys)
    else:
        raise ValueError(f"Unknown plane {plane!r}")
    return X, Y


def cyl_basis(plane: str, cx: float, cy: float, slice_idx: int | None = None):
    """cos(phi), sin(phi) per cell on the plane slice."""
    X, Y = plane_xy_grid(plane, slice_idx)
    Phi = np.arctan2(Y - cy, X - cx)
    return np.cos(Phi), np.sin(Phi)


def to_cylindrical(data_xyz: np.ndarray, cos_phi: np.ndarray,
                   sin_phi: np.ndarray) -> np.ndarray:
    """Convert (h, w, 3) Cartesian field to (h, w, 3) cylindrical (r, phi, z)
    in-place-style (returns a new array)."""
    Fx = data_xyz[..., 0]
    Fy = data_xyz[..., 1]
    Fz = data_xyz[..., 2]
    Fr = Fx * cos_phi + Fy * sin_phi
    Fphi = -Fx * sin_phi + Fy * cos_phi
    out = np.empty_like(data_xyz)
    out[..., 0] = Fr
    out[..., 1] = Fphi
    out[..., 2] = Fz
    return out


def row_frame(row, name):
    """Load a snapshot, apply per-row post-processing, then convert to
    cylindrical components in place of the Cartesian ones."""
    data = load_frame(row["dir"], name, row["plane"],
                      row.get("slice_axis"), row.get("slice_idx"))
    if data is None:
        return None
    if row.get("field") == "dB":
        data = data - row["b0"]
    elif row.get("field") == "curlM":
        # Compute the curl in Cartesian first; ``to_cylindrical`` then
        # projects the resulting vector onto (r, φ, z) per cell, which is
        # what we want for an axially-aligned configuration.
        data = compute_curl_2d(data, row["plane"])
    return to_cylindrical(data, row["cos_phi"], row["sin_phi"])


def collect_rows(field_specs, cx, cy):
    rows = []
    used_dirs = set()
    seen = set()
    b0_cached = None
    for field, slice_axis, slice_idx in field_specs:
        for explicit, plane, label in iter_field_diagnostics(field, slice_axis):
            key = (field, plane, explicit, label, slice_axis, slice_idx)
            if key in seen:
                continue
            seen.add(key)

            dir_path, steps = locate_dir(field, plane, explicit, label,
                                         slice_axis)
            if dir_path is None:
                print(f"[skip] no frames for {label} plane {plane} "
                      f"(looked for {candidate_dirs(field, plane, explicit, label, slice_axis)})")
                continue

            collision_key = (field, dir_path, slice_axis, slice_idx)
            if collision_key in used_dirs:
                print(f"[skip] {label} plane {plane}: directory {dir_path} "
                      "already used by another diagnostic — set a unique "
                      "\"out_dir\" in config to keep both slices.")
                continue
            used_dirs.add(collision_key)

            cos_phi, sin_phi = cyl_basis(plane, cx, cy, slice_idx)
            row = {
                "field": field,
                "label": label,
                "plane": plane,
                "dir": dir_path,
                "timesteps": steps,
                "is_pseudo": field in PSEUDO_FIELDS,
                "cos_phi": cos_phi,
                "sin_phi": sin_phi,
                "slice_axis": slice_axis,
                "slice_idx": slice_idx,
            }
            if field == "dB":
                if b0_cached is None:
                    b0_cached = read_b0_vector()
                row["b0"] = b0_cached
                row["b0_norm"] = float(np.linalg.norm(b0_cached))
                if row["b0_norm"] == 0.0:
                    print("[warn] |B0| = 0 in config; dB/|B0| panel will use "
                          "|B0| = 1 as a stand-in.")
            rows.append(row)
    return rows


def compute_vmax(rows, name_by_row):
    for row, names in zip(rows, name_by_row):
        vmax = 0.0
        for name in names:
            data = row_frame(row, name)
            if data is None:
                continue
            vmax = max(vmax, float(np.max(np.abs(data))))
        row["vmax"] = vmax if vmax > 0 else 1.0


def make_plot(fig, gs, i, j, row, comp_idx, layout):
    field = row["field"]
    plane = row["plane"]
    vmax = row["vmax"]
    (_, _), (xl, yl), (Lx_plot, Ly_plot) = layout
    ax = subplot(fig, gs, j, i)
    axis_label = {
        "x": r"$x,~c/\omega_{pe}$",
        "y": r"$y,~c/\omega_{pe}$",
        "z": r"$z,~c/\omega_{pe}$",
    }

    base, species_suffix = field_tex(field, row["label"], row)
    kind = "slice" if row.get("slice_axis") is not None else "plane"
    plane_suffix = rf"(\mathrm{{{kind}}}\ {plane}{species_suffix})"

    if comp_idx < 3:
        cmap = signed_cmap
        vmap = (-vmax, +vmax)
        comp = COMPONENT_LABELS[comp_idx]
        title = rf"${base}_{{{comp}}}\,{plane_suffix}$"
    elif field == "dB":
        cmap = unsigned_cmap
        b0 = row["b0_norm"] if row["b0_norm"] > 0 else 1.0
        vmap = (0.0, vmax / b0)
        title = rf"$|\delta B|/|B_0|\,{plane_suffix}$"
    else:
        cmap = unsigned_cmap
        vmap = (0.0, vmax)
        title = rf"$|{base}|\,{plane_suffix}$"

    plot = PlotIm(ax, vmap, cmap)
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
    specs = [parse_field_spec(t) for t in args.fields]
    init_constants(args.config)
    specs = [resolve_slice_idx(s) for s in specs]

    cx, cy = args.center if args.center else auto_center()
    print(f"Cylinder axis: ({cx:.2f}, {cy:.2f})")

    rows = collect_rows(specs, cx, cy)
    if not rows:
        print("No field diagnostics were found in config for:", args.fields)
        return

    common = set(idx for idx, _ in rows[0]["timesteps"])
    for row in rows[1:]:
        common &= set(idx for idx, _ in row["timesteps"])
    common = sorted(common)
    if not common:
        print("No timesteps common to all requested field diagnostics.")
        return

    names_per_row = []
    for row in rows:
        name_by_idx = dict(row["timesteps"])
        names_per_row.append([name_by_idx[idx] for idx in common])

    compute_vmax(rows, names_per_row)

    ncols = 4
    nrows = len(rows)
    fig, gs = figure(ncols, nrows)

    grid = []
    for i, row in enumerate(rows):
        layout = plane_layout(row["plane"])
        row_plots = [
            make_plot(fig, gs, i, j, row, j, layout)
            for j in range(ncols)
        ]
        grid.append(row_plots)

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    for k, idx in enumerate(common):
        figname = os.path.join(out_dir, f"{idx:04d}.png")
        print(f"Processing {figname} (idx={idx})")

        for row, row_plots, names in zip(rows, grid, names_per_row):
            data = row_frame(row, names[k])
            for c in range(3):
                row_plots[c].data = data[:, :, c]
                row_plots[c].draw()
            magnitude = np.sqrt(np.sum(data ** 2, axis=-1))
            if row["field"] == "dB":
                b0 = row["b0_norm"] if row["b0_norm"] > 0 else 1.0
                row_plots[3].data = magnitude / b0
            else:
                row_plots[3].data = magnitude
            row_plots[3].draw()

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
