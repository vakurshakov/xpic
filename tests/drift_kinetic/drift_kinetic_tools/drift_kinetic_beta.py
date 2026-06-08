#!/usr/bin/env python3
"""Compare particle-pressure perpendicular beta with the displaced-field beta.

T_perp(t) is no longer reconstructed from M . B per cell. Instead we read
the *total* kinetic energy ``wK`` from ``temporal/dk_diagnostic.txt`` and
scale the initial temperature by its ratio:

    T_perp(t) = T_perp(0) * wK(t) / wK(0)

where ``T_perp(0)`` defaults to the ``T`` field of the first particle sort
in the config (assumed identical across sorts; override with --T-perp-0
in keV).

For every saved frame of a 2D B FieldView (plane X or Y) we still take the
central column in the perpendicular axis and average over z to get:
  <n>_z   -- n_total at central (x, y) averaged along z
  <|B|^2>_z, <|dB|>_z

Then:
  beta_perp_part(t) = 2 * <n>_z * T_perp(t) / |B0|^2
  beta_perp_disp(t) = 1 - <|B|^2>_z / |B0|^2  =  (B0^2 - <|B|^2>) / B0^2
"""

from __future__ import annotations

import os
import re
import sys
import argparse

import numpy as np

# m_e c^2 in keV — used to convert input T (in keV) to code units (m_e c^2).
M_E_C2_KEV = 511.0

TOOLS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS_DIR)

from lib.plot_utils import figure, subplot
from lib.constants import const, init_constants

import shutil
import matplotlib.pyplot as plt

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--out-subdir", default="beta",
                        help="Subdirectory under <out_dir>/processed")
    parser.add_argument("--out-name", default="beta_compare.png")
    parser.add_argument("--logy", action="store_true",
                        help="Use log scale on the beta axis")
    parser.add_argument("--T-perp-0", type=float, default=None,
                        help="Initial perpendicular temperature in keV "
                             "(default: take T from first Particles entry "
                             "in the config)")
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def load_wK_from_dk_diagnostic(path):
    """Read wK column from temporal/dk_diagnostic.txt.

    Header columns are separated by 2+ spaces (some names contain single
    spaces, e.g. 'dt * a_EJ'); we split with a regex.

    Returns dict: iteration index (Time column, int) -> wK value.
    """
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        header_line = fh.readline().rstrip("\n")
    header = re.split(r" {2,}", header_line.strip())
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if "Time" not in header or "wK" not in header:
        return None
    t_col = data[:, header.index("Time")].astype(int)
    w_col = data[:, header.index("wK")]
    return dict(zip(t_col.tolist(), w_col.tolist()))


def initial_T_keV_from_config():
    """Take T (keV) from the first Particles entry in the config."""
    parts = const.config.get("Particles", [])
    if not parts:
        return None
    T = parts[0].get("T")
    return float(T) if T is not None else None


# ---------------------------------------------------- 2D vector field (B / M)

def plane_shape_2d(plane):
    if plane == "X":
        return const.Nz, const.Ny
    if plane == "Y":
        return const.Nz, const.Nx
    raise ValueError(f"plane {plane!r} has no z-axis")


def find_field_diagnostic(field, plane=None):
    """Pick a 2D FieldView for the requested field. If `plane` is given,
    the slice plane must match; otherwise the first X/Y plane is used."""
    for diag in const.config.get("Diagnostics", []):
        if diag.get("diagnostic") != "FieldView":
            continue
        if diag.get("field") != field:
            continue
        region = diag.get("region", {})
        if region.get("type") != "2D":
            continue
        p = region.get("plane")
        if p not in ("X", "Y"):
            continue
        if plane is not None and p != plane:
            continue
        out_dir = diag.get("out_dir") or f"{field}_plane{p}"
        path = os.path.join(const.in_dir, out_dir)
        if os.path.isdir(path):
            return path, p
    return None, None


def list_timesteps(dir_path):
    entries = []
    for name in os.listdir(dir_path):
        full = os.path.join(dir_path, name)
        if os.path.isfile(full) and name.isdigit():
            entries.append((int(name), name))
    entries.sort()
    return entries


def read_b0_vector():
    b0 = np.zeros(3, dtype=float)
    for preset in const.config.get("Presets", []):
        if preset.get("command") != "SetMagneticField":
            continue
        if preset.get("field") != "B0":
            continue
        setter = preset.get("setter", {})
        if setter.get("name") == "SetUniformField":
            b0 += np.asarray(setter.get("value", [0, 0, 0]), dtype=float)
    return b0


def load_2d_vector(path, plane):
    h, w = plane_shape_2d(plane)
    expected = h * w * 3 * 4
    if os.path.getsize(path) != expected:
        return None
    raw = np.fromfile(path, dtype=np.float32, count=h * w * 3)
    return raw.reshape(h, w, 3)


# ---------------------------------------------------- 3D density

def find_density_diagnostic(species):
    for diag in const.config.get("Diagnostics", []):
        if diag.get("diagnostic") != "DistributionMoment":
            continue
        if diag.get("moment") != "density":
            continue
        if diag.get("particles") != species:
            continue
        explicit = diag.get("out_dir")
        candidates = [explicit] if explicit else [
            os.path.join(species, "density"),
            f"{species}_density",
        ]
        for sub in candidates:
            path = os.path.join(const.in_dir, sub)
            if os.path.isdir(path):
                return path
    return None


def load_density_z_column(path):
    """Density at central (x, y) → 1D z-profile."""
    expected = const.Nx * const.Ny * const.Nz * 4
    if os.path.getsize(path) != expected:
        return None
    raw = np.fromfile(path, dtype=np.float32,
                      count=const.Nx * const.Ny * const.Nz)
    frame = raw.reshape(const.Nz, const.Ny, const.Nx)
    return frame[:, const.Ny // 2, const.Nx // 2]


def density_columns(dens_dir):
    """{idx: n(z) at central (x, y)} for one species."""
    out = {}
    for idx, name in list_timesteps(dens_dir):
        col = load_density_z_column(os.path.join(dens_dir, name))
        if col is not None:
            out[idx] = col
    return out


# ---------------------------------------------------- main

def main():
    args = parse_args()
    init_constants(args.config)

    b_dir, plane = find_field_diagnostic("B")
    if b_dir is None:
        print("No 2D B FieldView found in config.")
        return

    b0 = read_b0_vector()
    b0_norm = float(np.linalg.norm(b0))
    if b0_norm == 0.0:
        print("|B0| = 0 in config; cannot normalize.")
        return
    print(f"B from {b_dir} (plane {plane}); |B0| = {b0_norm:.6g}")

    # T_perp(0) -> code units (m_e c^2). CLI overrides config.
    T0_keV = args.T_perp_0 if args.T_perp_0 is not None \
        else initial_T_keV_from_config()
    if T0_keV is None or T0_keV <= 0:
        print("[error] cannot determine T_perp(0); pass --T-perp-0 in keV")
        return
    T0_units = T0_keV / M_E_C2_KEV
    print(f"T_perp(0) = {T0_keV:.4g} keV = {T0_units:.4g} m_e c^2")

    # wK (total kinetic energy) from temporal/dk_diagnostic.txt; map by
    # iteration index (Time column).
    dk_path = os.path.join(const.in_dir, "temporal", "dk_diagnostic.txt")
    wK_by_iter = load_wK_from_dk_diagnostic(dk_path)
    if wK_by_iter is None:
        print(f"[error] cannot read wK from {dk_path}")
        return
    if 0 not in wK_by_iter or wK_by_iter[0] == 0.0:
        print("[error] wK(t=0) missing or zero; cannot scale T_perp")
        return
    wK0 = wK_by_iter[0]
    print(f"wK(t=0) = {wK0:.4g}")

    perp_w = plane_shape_2d(plane)[1]
    perp_mid = perp_w // 2

    # B summaries: <|dB|>_z / |B0|, <|B|^2>_z / |B0|^2  (for any frame).
    b_idx = []
    b_dB_norm = {}
    b_B2_norm = {}
    for idx, name in list_timesteps(b_dir):
        frame = load_2d_vector(os.path.join(b_dir, name), plane)
        if frame is None:
            continue
        column = frame[:, perp_mid, :]  # (Nz, 3)
        delta = column - b0
        d_mag = np.sqrt(np.sum(delta ** 2, axis=-1))
        b_idx.append(idx)
        b_dB_norm[idx] = float(np.mean(d_mag) / b0_norm)
        b_B2_norm[idx] = float(np.mean(np.sum(column ** 2, axis=-1))
                               / b0_norm ** 2)

    if not b_idx:
        print("No usable B frames.")
        return

    # Densities (per species, kept as z-profile at central (x, y)).
    species_n = {}
    for sort in const.config.get("Particles", []):
        name = sort.get("sort_name")
        if name is None:
            continue
        d = find_density_diagnostic(name)
        if d is None:
            print(f"[warn] no density directory for '{name}'; skipping it "
                  "in n_total.")
            continue
        species_n[name] = density_columns(d)

    # Snapshot index 'idx' corresponds to iteration count t_iter = idx * Ndts
    # (because the file name is t / diagnose_period). For each idx we look
    # up wK at that iteration in dk_diagnostic.txt.
    common = set(b_idx)
    for n_dict in species_n.values():
        common &= set(n_dict.keys())
    common = sorted(
        i for i in common
        if (i * const.Ndts) in wK_by_iter
    )

    times = np.asarray(b_idx) * const.dts
    dB_norm = np.array([b_dB_norm[i] for i in b_idx])
    B2_norm = np.array([b_B2_norm[i] for i in b_idx])
    beta_disp = 1.0 - B2_norm

    if common and species_n:
        t_part = np.asarray(common) * const.dts
        T_mean = np.empty(len(common))
        n_mean = np.empty(len(common))
        for k, idx in enumerate(common):
            n_total_z = np.zeros(const.Nz, dtype=float)
            for n_dict in species_n.values():
                n_total_z += n_dict[idx]
            n_mean[k] = float(np.mean(n_total_z))
            T_mean[k] = T0_units * wK_by_iter[idx * const.Ndts] / wK0
        beta_part = 2.0 * n_mean * T_mean / b0_norm ** 2
        print(f"  T_perp(t=0) = {T_mean[0]:.4g} m_e c^2,  "
              f"<n_total>(t=0) = {n_mean[0]:.4g},  "
              f"beta_perp_part(t=0) = {beta_part[0]:.4g}")
    else:
        t_part = None
        beta_part = None
        T_mean = None
        n_mean = None

    # Plot.
    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    fig, gs = figure(1, 1, figsize=(9, 6))
    ax = subplot(fig, gs, 0, 0)

    if beta_part is not None:
        ax.plot(t_part, beta_part,
                color="tab:blue", marker="o", markersize=3,
                label=r"$\beta_\perp^{\mathrm{part}} "
                      r"= 2\,\langle n\rangle_z\,T_\perp(t) / |B_0|^2$,"
                      r"\ $T_\perp(t) = T_\perp(0)\,w_K(t)/w_K(0)$")
    ax.plot(times, beta_disp, color="tab:red", marker="s", markersize=3,
            label=r"$\beta_\perp^{\mathrm{disp}} "
                  r"= 1 - \langle |B|^2\rangle_z / |B_0|^2$")

    ax.set_xlabel(r"$\omega_{pe}\,t$")
    ax.set_ylabel(r"$\beta_\perp$")
    ax.set_title(r"Perpendicular $\beta$: "
                 r"$T_\perp \propto w_K$ vs displaced field")
    if args.logy:
        ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()

    out_png = os.path.join(out_dir, args.out_name)
    fig.savefig(out_png, dpi=args.dpi)
    print(f"Saved {out_png}")

    # CSV.
    csv_path = os.path.join(out_dir,
                            os.path.splitext(args.out_name)[0] + ".csv")
    if beta_part is not None:
        # Match particle rows to the field-row time grid (NaN where missing).
        full_T = np.full_like(times, np.nan)
        full_n = np.full_like(times, np.nan)
        full_b = np.full_like(times, np.nan)
        for k_full, idx in enumerate(b_idx):
            if idx in common:
                j = common.index(idx)
                full_T[k_full] = T_mean[j]
                full_n[k_full] = n_mean[j]
                full_b[k_full] = beta_part[j]
        cols = [times, dB_norm, B2_norm, beta_disp, full_n, full_T, full_b]
        headers = ["t_wpe", "<|dB|>/|B0|", "<|B|^2>/|B0|^2",
                   "beta_perp_disp",
                   "<n_total>", "<T_perp>", "beta_perp_part"]
    else:
        cols = [times, dB_norm, B2_norm, beta_disp]
        headers = ["t_wpe", "<|dB|>/|B0|", "<|B|^2>/|B0|^2", "beta_perp_disp"]

    np.savetxt(csv_path, np.column_stack(cols),
               delimiter=",", header=",".join(headers), comments="")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
