#!/usr/bin/env python3
"""Visualize FieldView diagnostics (E, B, J, M) from a drift_kinetic run.

For every (field, plane) declared in ``config.json`` the script produces a
row of four panels: ``F_x``, ``F_y``, ``F_z`` components and the magnitude
``|F|``. Colormap range is fixed per row as the maximum absolute value of
the field in the plane across all available snapshots, so it does not
change from frame to frame.
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np

TOOLS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS_DIR)

from lib.plot import PlotIm, bbox, labelsize, signed_cmap, unsigned_cmap
from lib.plot_utils import figure, subplot
from lib.constants import const, init_constants

import shutil
import matplotlib.pyplot as plt

# lib.plot_utils enables TeX rendering by default; fall back if any part of
# the TeX -> PNG toolchain is missing so the script stays usable.
if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)


FIELD_CHOICES = ["E", "B", "J", "M", "rotM", "curlM", "dB"]
COMPONENT_LABELS = ["x", "y", "z"]

# Pseudo-fields read their data from a different diagnostic and apply a
# post-processing step.
#  - ``dB``    : reads ``B`` and subtracts the uniform B0 from Presets.
#  - ``curlM`` : reads ``M`` and computes ∇×M numerically on the slice via
#    central differences. The derivative along the slice normal cannot be
#    sampled and is dropped, matching the closure used by drift_kinetic_force.
PSEUDO_FIELDS = {"dB": "B", "curlM": "M"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument(
        "--fields", nargs="+", default=FIELD_CHOICES,
        help=("Fields to visualize. Each token is either a plain field name "
              "(e.g. 'E', 'curlM') taken from 2D FieldView diagnostics, or a "
              "3D-slice spec '<field>:<axis>[:<idx>]' (e.g. 'E:z:10') that "
              "reads a 3D FieldView and slices it. Axis is x/y/z; idx is the "
              "cell index, defaulting to N/2 when omitted. Field choices: "
              + ", ".join(FIELD_CHOICES) + "."))
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out-subdir", default="fields",
                        help="Subdirectory under <out_dir>/processed for PNGs")
    return parser.parse_args()


def parse_field_spec(token: str):
    """Parse one ``--fields`` token into ``(field, slice_axis, slice_idx)``.

    ``slice_axis`` is ``None`` for plain tokens (2D-FieldView mode); when set
    it is one of 'x'/'y'/'z' and the diag is treated as a 3D FieldView that
    we slice ourselves. ``slice_idx`` is ``None`` when the token only gives
    an axis — the caller fills in the default (N/2) once ``const`` is loaded.
    """
    parts = token.split(":")
    field = parts[0]
    if field not in FIELD_CHOICES:
        raise SystemExit(
            f"Unknown field {field!r} in token {token!r}; "
            f"valid fields are {FIELD_CHOICES}")
    if len(parts) == 1:
        return field, None, None
    if len(parts) > 3:
        raise SystemExit(f"Too many ':' segments in field token {token!r}")
    axis = parts[1].lower()
    if axis not in ("x", "y", "z"):
        raise SystemExit(
            f"Slice axis must be one of x/y/z; got {parts[1]!r} in {token!r}")
    idx = None
    if len(parts) == 3:
        try:
            idx = int(parts[2])
        except ValueError:
            raise SystemExit(
                f"Slice index must be an integer; got {parts[2]!r} in {token!r}")
    return field, axis, idx


def resolve_slice_idx(spec):
    """Replace ``slice_idx=None`` with the centre cell along the slice axis."""
    field, axis, idx = spec
    if axis is None or idx is not None:
        return spec
    N = {"x": const.Nx, "y": const.Ny, "z": const.Nz}[axis]
    return field, axis, N // 2


def plane_layout(plane: str):
    """Return ((rows, cols), (xlabel_axis, ylabel_axis), (Lx_plot, Ly_plot))."""
    if plane == "X":
        return (const.Nz, const.Ny), ("y", "z"), (const.Ly, const.Lz)
    if plane == "Y":
        return (const.Nz, const.Nx), ("x", "z"), (const.Lx, const.Lz)
    if plane == "Z":
        return (const.Ny, const.Nx), ("x", "y"), (const.Lx, const.Ly)
    raise ValueError(f"Unknown plane {plane!r}")


def iter_field_diagnostics(field: str, slice_axis: str | None = None):
    """Yield (explicit_out_dir or None, plane, label) for FieldView diags.

    When ``slice_axis`` is ``None`` (default) only 2D FieldView entries are
    yielded — the pre-existing behaviour. When ``slice_axis`` is set, only
    3D FieldView entries are yielded; the ``plane`` returned is the
    perpendicular plane of the slice (e.g. axis 'z' -> plane 'Z'), so the
    rest of the pipeline can treat the sliced 2D image like a plane-Z
    diagnostic. xpic treats FieldView entries without a ``region`` key as
    full 3D, so those are accepted in slice mode as well.

    ``label`` is the diagnostic's full ``field`` string (e.g. ``rotM`` or
    ``electrons/rotM``); use it for figure titles when the same field is
    written per-species so the rows can be told apart.
    """
    source = PSEUDO_FIELDS.get(field, field)
    for diag in const.config.get("Diagnostics", []):
        if diag.get("diagnostic") != "FieldView":
            continue
        diag_field = diag.get("field", "")
        if diag_field == source:
            label = source
        elif "/" in diag_field and diag_field.split("/", 1)[1] == source:
            # Species-prefixed name like "electrons/rotM"; the binary layout
            # is identical to the flat case, just under a sub-directory.
            label = diag_field
        else:
            continue
        region = diag.get("region", {})
        rtype = region.get("type")  # may be missing -> default 3D in xpic
        if slice_axis is None:
            if rtype != "2D":
                continue
            plane = region.get("plane")
            if plane not in ("X", "Y", "Z"):
                continue
            yield diag.get("out_dir"), plane, label
        else:
            if rtype == "2D":
                continue
            yield diag.get("out_dir"), slice_axis.upper(), label


def read_b0_vector():
    """Sum of all uniform B0 presets from the configuration."""
    b0 = np.zeros(3, dtype=float)
    for preset in const.config.get("Presets", []):
        if preset.get("command") != "SetMagneticField":
            continue
        if preset.get("field") != "B0":
            continue
        setter = preset.get("setter", {})
        if setter.get("name") != "SetUniformField":
            print(f"[warn] B0 setter '{setter.get('name')}' is not uniform; "
                  "dB visualization will treat its contribution as zero.")
            continue
        b0 += np.asarray(setter.get("value", [0.0, 0.0, 0.0]), dtype=float)
    return b0


def candidate_dirs(field: str, plane: str, explicit: str | None,
                   label: str | None = None, slice_axis: str | None = None):
    """Directory names to try, in order, for one (field, plane) diagnostic."""
    if explicit is not None:
        return [explicit]
    source = PSEUDO_FIELDS.get(field, field)
    # Use the per-diagnostic label (e.g. "electrons/rotM") when present so
    # the species sub-directory is searched ahead of the bare field name.
    base = label if label is not None else source
    if slice_axis is not None:
        # 3D FieldViews are typically written to a bare field directory; the
        # plane-suffixed variants only exist for 2D slices.
        return [base]
    return [
        base,
        f"{base}_plane{plane}",
        f"{base}_{plane}",
    ]


def locate_dir(field: str, plane: str, explicit: str | None,
               label: str | None = None, slice_axis: str | None = None):
    """Return (dir_path, timesteps) for the first candidate whose files match
    the expected plane size, or (None, None) if nothing fits."""
    if slice_axis is not None:
        expected_bytes = const.Nx * const.Ny * const.Nz * 3 * 4
    else:
        (h, w), _, _ = plane_layout(plane)
        expected_bytes = h * w * 3 * 4

    candidates = candidate_dirs(field, plane, explicit, label, slice_axis)
    source = PSEUDO_FIELDS.get(field, field)
    base = label if label is not None else source

    for sub in candidates:
        dir_path = os.path.join(const.in_dir, sub)

        # Only plane-specific candidates are allowed to match directories with
        # a trailing position suffix (e.g. ``E_planeY_0002``). The bare field
        # name must match exactly, otherwise it would swallow every plane.
        if not os.path.isdir(dir_path) and sub != base:
            parent, leaf = os.path.split(dir_path)
            if os.path.isdir(parent):
                prefix_matches = sorted(
                    os.path.join(parent, name)
                    for name in os.listdir(parent)
                    if name.startswith(leaf + "_") and
                    os.path.isdir(os.path.join(parent, name))
                )
                if prefix_matches:
                    dir_path = prefix_matches[-1]

        if not os.path.isdir(dir_path):
            continue

        steps = [
            (idx, name) for (idx, name) in list_timesteps(dir_path)
            if os.path.getsize(os.path.join(dir_path, name)) == expected_bytes
        ]
        if steps:
            return dir_path, steps
    return None, None


def list_timesteps(dir_path: str):
    """Return sorted list of (index, filename) pairs with numeric names."""
    entries = []
    for name in os.listdir(dir_path):
        full = os.path.join(dir_path, name)
        if os.path.isfile(full) and name.isdigit():
            entries.append((int(name), name))
    entries.sort()
    return entries


def load_frame(dir_path: str, name: str, plane: str,
               slice_axis: str | None = None, slice_idx: int | None = None):
    path = os.path.join(dir_path, name)
    if slice_axis is None:
        (h, w), _, _ = plane_layout(plane)
        expected_bytes = h * w * 3 * 4
        if os.path.getsize(path) != expected_bytes:
            return None
        raw = np.fromfile(path, dtype=np.float32, count=h * w * 3)
        return raw.reshape(h, w, 3)
    # 3D FieldView: file layout is (Nz, Ny, Nx, 3) in C order; slice at the
    # requested cell index along the chosen axis to produce the 2D image
    # whose shape matches the equivalent 2D plane.
    Nx, Ny, Nz = const.Nx, const.Ny, const.Nz
    expected_bytes = Nx * Ny * Nz * 3 * 4
    if os.path.getsize(path) != expected_bytes:
        return None
    raw = np.fromfile(path, dtype=np.float32, count=Nx * Ny * Nz * 3)
    arr = raw.reshape(Nz, Ny, Nx, 3)
    if slice_axis == "x":
        return arr[:, :, slice_idx, :]
    if slice_axis == "y":
        return arr[:, slice_idx, :, :]
    if slice_axis == "z":
        return arr[slice_idx, :, :, :]
    raise ValueError(f"Unknown slice axis {slice_axis!r}")


def compute_curl_2d(M: np.ndarray, plane: str) -> np.ndarray:
    """Numerically compute ∇×M on a 2D slice using central differences.

    ``M`` has shape ``(h, w, 3)``. The derivative along the slice's normal
    axis is unavailable from a single plane and is taken to be zero, so the
    component lying in that direction has to be reconstructed from a single
    in-plane derivative. Cell sizes come from ``const.dx/dy/dz``.
    """
    out = np.zeros_like(M)
    if plane == "Z":
        # axes (h=Ny, w=Nx); axis 0 is y, axis 1 is x. ∂/∂z is dropped.
        dMx_dy = np.gradient(M[..., 0], const.dy, axis=0)
        dMy_dx = np.gradient(M[..., 1], const.dx, axis=1)
        dMz_dy = np.gradient(M[..., 2], const.dy, axis=0)
        dMz_dx = np.gradient(M[..., 2], const.dx, axis=1)
        out[..., 0] = dMz_dy
        out[..., 1] = -dMz_dx
        out[..., 2] = dMy_dx - dMx_dy
    elif plane == "Y":
        # axes (h=Nz, w=Nx); axis 0 is z, axis 1 is x. ∂/∂y is dropped.
        dMx_dz = np.gradient(M[..., 0], const.dz, axis=0)
        dMy_dz = np.gradient(M[..., 1], const.dz, axis=0)
        dMy_dx = np.gradient(M[..., 1], const.dx, axis=1)
        dMz_dx = np.gradient(M[..., 2], const.dx, axis=1)
        out[..., 0] = -dMy_dz
        out[..., 1] = dMx_dz - dMz_dx
        out[..., 2] = dMy_dx
    elif plane == "X":
        # axes (h=Nz, w=Ny); axis 0 is z, axis 1 is y. ∂/∂x is dropped.
        dMx_dy = np.gradient(M[..., 0], const.dy, axis=1)
        dMx_dz = np.gradient(M[..., 0], const.dz, axis=0)
        dMy_dz = np.gradient(M[..., 1], const.dz, axis=0)
        dMz_dy = np.gradient(M[..., 2], const.dy, axis=1)
        out[..., 0] = dMz_dy - dMy_dz
        out[..., 1] = dMx_dz
        out[..., 2] = -dMx_dy
    else:
        raise ValueError(f"Unknown plane {plane!r}")
    return out


def row_frame(row, name):
    """Load a snapshot and apply any per-row post-processing."""
    data = load_frame(row["dir"], name, row["plane"],
                      row.get("slice_axis"), row.get("slice_idx"))
    if data is None:
        return None
    if row.get("field") == "dB":
        data = data - row["b0"]
    elif row.get("field") == "curlM":
        data = compute_curl_2d(data, row["plane"])
    return data


def collect_rows(field_specs):
    """Build per-row metadata for every (spec, diagnostic) pair.

    ``field_specs`` is the list returned by :func:`parse_field_spec` /
    :func:`resolve_slice_idx` — each element is ``(field, slice_axis,
    slice_idx)`` where ``slice_axis`` / ``slice_idx`` are ``None`` for the
    pre-existing 2D-FieldView mode.
    """
    rows = []
    # Track collisions per (field, dir, slice): two diagnostics of the same
    # field writing to the same directory step on each other's files, but B
    # and dB intentionally share the B directory, and two slices of the same
    # 3D field at different indices share its directory too — both must not
    # warn, hence the slice tuple in the key.
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

            row = {
                "field": field,
                "label": label,
                "plane": plane,
                "dir": dir_path,
                "timesteps": steps,
                "is_pseudo": field in PSEUDO_FIELDS,
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
    """Per-row absolute maximum across all shared timesteps."""
    for row, names in zip(rows, name_by_row):
        vmax = 0.0
        for name in names:
            data = row_frame(row, name)
            if data is None:
                continue
            vmax = max(vmax, float(np.max(np.abs(data))))
        row["vmax"] = vmax if vmax > 0 else 1.0


def field_tex(field: str, label: str, row: dict | None = None):
    """LaTeX expression and slice annotation for a row's field/label.

    Returns ``(base, suffix)`` where ``base`` is the math expression without
    the component subscript and ``suffix`` is the trailing annotation that
    follows in parentheses (species + slice info). For species-prefixed
    labels like ``electrons/rotM`` the species name is appended; when ``row``
    carries a 3D-slice spec the cell index and world coordinate are added
    too (e.g. ``z=10 (100 c/omega_pe)``).
    """
    if "/" in label:
        species, _ = label.split("/", 1)
        species_suffix = rf",~\mathrm{{{species}}}"
    else:
        species_suffix = ""

    if field == "dB":
        base = rf"\delta {field[1:]}"
    elif field == "rotM":
        base = r"(\nabla\times M)"
    elif field == "curlM":
        # Numerically computed from M on the slice; tag the title so it does
        # not look identical to the simulation's rotM diagnostic when both
        # are requested in the same figure.
        base = r"(\nabla\times M)"
        species_suffix += r",~\mathrm{num}"
    else:
        base = field

    if row is not None and row.get("slice_axis") is not None:
        ax = row["slice_axis"]
        idx = row["slice_idx"]
        d = {"x": const.dx, "y": const.dy, "z": const.dz}[ax]
        species_suffix += rf",~{ax}={idx}\,({idx * d:g}\,c/\omega_{{pe}})"
    return base, species_suffix


def make_plot(fig, gs, i, j, row, comp_idx, layout):
    field = row["field"]
    plane = row["plane"]
    vmax = row["vmax"]
    (_, _), (xl, yl), (Lx_plot, Ly_plot) = layout
    ax = subplot(fig, gs, j, i)
    axis_label = {
        "x": r"$x~(c/\omega_{pe})$",
        "y": r"$y~(c/\omega_{pe})$",
        "z": r"$z~(c/\omega_{pe})$",
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
    ax.set_box_aspect(1)
    return plot


def main():
    args = parse_args()
    # Token parsing is split: validate the shape up front (catches typos
    # before doing IO), then resolve default slice indices once Nx/Ny/Nz are
    # known from the config.
    specs = [parse_field_spec(t) for t in args.fields]
    init_constants(args.config)
    specs = [resolve_slice_idx(s) for s in specs]

    rows = collect_rows(specs)
    if not rows:
        print("No field diagnostics were found in config for:", args.fields)
        return

    # Shared indices (different rows can use different zero-padding widths).
    common = set(idx for idx, _ in rows[0]["timesteps"])
    for row in rows[1:]:
        common &= set(idx for idx, _ in row["timesteps"])
    common = sorted(common)
    if not common:
        print("No timesteps common to all requested field diagnostics.")
        return

    # For each row pick the actual filename for each shared index.
    names_per_row = []
    for row in rows:
        name_by_idx = dict(row["timesteps"])
        names_per_row.append([name_by_idx[idx] for idx in common])

    compute_vmax(rows, names_per_row)

    ncols = 4
    nrows = len(rows)
    fig, gs = figure(ncols, nrows, figsize=(8 * ncols, 7 * nrows))

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
