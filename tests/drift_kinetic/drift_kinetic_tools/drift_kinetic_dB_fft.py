#!/usr/bin/env python3
"""Plot ln|delta B(k_z, t)| of the parallel-z Fourier mode of dB = B - B0.

For each saved timestep the 2D B slice (plane X or Y, whichever contains
the z-axis) is loaded, the uniform B0 from Presets is subtracted, and the
FFT is taken along z. The amplitude at the requested z-harmonic
(default n=1, i.e. k_z = 2*pi/L_z) is averaged in absolute value across
the perpendicular cells of the slice, and ln of that amplitude is plotted
against time.

By default both perpendicular components (B_x, B_y) are drawn on the same
axes, since for a B0 || z setup these are the unstable components whose
exponential growth this script is meant to expose.
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np

TOOLS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, TOOLS_DIR)

from lib.plot_utils import figure, subplot
from lib.constants import const, init_constants

import shutil
import matplotlib.pyplot as plt

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)


COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2, "perp": 3}
COMPONENT_LABEL = {
    "x": r"\delta B_x",
    "y": r"\delta B_y",
    "z": r"\delta B_z",
    "perp": r"\delta B_\perp",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the run")
    parser.add_argument("--components", nargs="+",
                        default=["x", "y", "z", "perp"],
                        choices=["x", "y", "z", "perp"],
                        help="dB components to plot (default: x y)")
    parser.add_argument("--mode", type=int, default=1,
                        help="z-direction harmonic to extract; "
                             "k_z = 2*pi*mode/L_z (default: 1)")
    parser.add_argument("--out-subdir", default="fft",
                        help="Subdirectory under <out_dir>/processed for the "
                             "PNG and the saved time series")
    parser.add_argument("--out-name", default="dB_fft.png")
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def plane_shape(plane: str):
    """(rows, cols) for a 2D FieldView slice. Row axis is always z for X/Y."""
    if plane == "X":
        return const.Nz, const.Ny
    if plane == "Y":
        return const.Nz, const.Nx
    if plane == "Z":
        return const.Ny, const.Nx
    raise ValueError(f"Unknown plane {plane!r}")


def find_b_diagnostic():
    """Pick the first 2D B FieldView whose slice contains the z axis."""
    for diag in const.config.get("Diagnostics", []):
        if diag.get("diagnostic") != "FieldView":
            continue
        if diag.get("field") != "B":
            continue
        region = diag.get("region", {})
        if region.get("type") != "2D":
            continue
        plane = region.get("plane")
        if plane not in ("X", "Y"):
            continue
        out_dir = diag.get("out_dir") or f"B_plane{plane}"
        dir_path = os.path.join(const.in_dir, out_dir)
        if os.path.isdir(dir_path):
            return dir_path, plane
    return None, None


def list_timesteps(dir_path: str):
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
        if setter.get("name") != "SetUniformField":
            print(f"[warn] B0 setter '{setter.get('name')}' is not uniform; "
                  "treating its contribution to B0 as zero.")
            continue
        b0 += np.asarray(setter.get("value", [0.0, 0.0, 0.0]), dtype=float)
    return b0


def load_frame(path: str, plane: str):
    h, w = plane_shape(plane)
    expected = h * w * 3 * 4
    if os.path.getsize(path) != expected:
        return None
    raw = np.fromfile(path, dtype=np.float32, count=h * w * 3)
    return raw.reshape(h, w, 3)


def mode_amplitudes(frame: np.ndarray, b0: np.ndarray, mode: int):
    """Mode amplitudes of dB at the requested z-harmonic.

    Returns a 4-vector (|dBx|, |dBy|, |dBz|, |dB_perp|) where |dB_perp| is
    sqrt(|dBx|^2 + |dBy|^2) computed per perpendicular cell before the mean,
    so it is the magnitude of the complex transverse Fourier amplitude
    (not the sum of the magnitudes of the components).
    """
    delta = frame - b0
    Nz = delta.shape[0]
    spectrum = np.fft.fft(delta, axis=0) / Nz
    line = spectrum[mode]  # (perp, 3) complex
    abs_xyz = np.abs(line)  # (perp, 3)
    abs_perp = np.sqrt(abs_xyz[:, 0] ** 2 + abs_xyz[:, 1] ** 2)  # (perp,)
    return np.array([
        abs_xyz[:, 0].mean(),
        abs_xyz[:, 1].mean(),
        abs_xyz[:, 2].mean(),
        abs_perp.mean(),
    ])


def main():
    args = parse_args()
    init_constants(args.config)

    dir_path, plane = find_b_diagnostic()
    if dir_path is None:
        print("No 2D B FieldView with a z-axis found in config.")
        return
    print(f"Reading {dir_path} (plane {plane})")

    steps = list_timesteps(dir_path)
    h, w = plane_shape(plane)
    expected = h * w * 3 * 4
    steps = [
        (idx, name) for (idx, name) in steps
        if os.path.getsize(os.path.join(dir_path, name)) == expected
    ]
    if not steps:
        print(f"No usable timestep files in {dir_path}")
        return

    Nz = const.Nz
    if args.mode <= 0 or args.mode >= Nz:
        print(f"Mode {args.mode} out of range for Nz={Nz}.")
        return

    b0 = read_b0_vector()

    times = np.empty(len(steps))
    amps = np.empty((len(steps), 4))
    for k, (idx, name) in enumerate(steps):
        frame = load_frame(os.path.join(dir_path, name), plane)
        if frame is None:
            print(f"[skip] cannot read frame {name}")
            times[k] = np.nan
            amps[k] = np.nan
            continue
        times[k] = idx * const.dts
        amps[k] = mode_amplitudes(frame, b0, args.mode)

    out_dir = os.path.join(const.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    k_z = 2.0 * np.pi * args.mode / const.Lz

    fig, gs = figure(1, 1, figsize=(8, 6))
    ax = subplot(fig, gs, 0, 0)
    for c in args.components:
        i = COMPONENT_INDEX[c]
        a = amps[:, i]
        mask = np.isfinite(a) & (a > 0)
        ax.plot(times[mask], np.log(a[mask]), marker="o", markersize=3,
                label=rf"$\ln |{COMPONENT_LABEL[c]}(k_z,\,t)|$")

    ax.set_xlabel(r"$\omega_{pe}\,t$")
    ax.set_ylabel(r"$\ln |\delta B(k_z,\,t)|$")
    ax.set_title(
        rf"$k_z = 2\pi\cdot{args.mode}/L_z "
        rf"\approx {k_z:.4g}\,\omega_{{pe}}/c$"
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    out_png = os.path.join(out_dir, args.out_name)
    fig.savefig(out_png, dpi=args.dpi)
    print(f"Saved {out_png}")

    csv_path = os.path.join(out_dir, os.path.splitext(args.out_name)[0] + ".csv")
    header = "t_wpe,|dBx|,|dBy|,|dBz|,|dB_perp|"
    np.savetxt(csv_path, np.column_stack([times, amps]),
               delimiter=",", header=header, comments="")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
