#!/usr/bin/env python3
"""Compare energy time series for drift_kinetic runs with the same ex-prefix.

Given a base run config such as
``output/drift_kinetic_energy_ei_ex1/config.json``, this script finds sibling
output directories matching ``drift_kinetic_energy_ei_ex*`` and plots three
comparisons versus omega_pe * t:

- wE(t);
- (wK_s(t) - wK_s(0)) / wK_s(0) for electrons and ions;
- (W(t) - W(0)) / W(0), where W(t) = wE(t) + wB(t) + sum_s wK_s(t).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np

if any(shutil.which(tool) is None for tool in ("latex", "dvipng")):
    plt.rc("text", usetex=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to config.json of the base run")
    parser.add_argument("--time-max", type=float, default=None,
                        help="Upper omega_pe*t limit")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out-subdir", default="energy_compare",
                        help="Subdirectory under <base_out_dir>/processed")
    return parser.parse_args()


def load_energy_table(path: Path):
    """Return (header_list, data_2d_array) for temporal/dk_diagnostic.txt."""
    with path.open("r", encoding="utf-8") as fh:
        header_line = fh.readline().rstrip("\n")
    header = re.split(r" {2,}", header_line.strip())
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(header):
        raise RuntimeError(
            f"Header has {len(header)} columns but data has {data.shape[1]}: "
            f"{path}")
    return header, data


def column(header, data, name):
    if name not in header:
        return None
    return data[:, header.index(name)]


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def run_prefix(base_name: str):
    match = re.match(r"^(.*_ex)([0-9]+)$", base_name)
    if match is None:
        raise RuntimeError(
            f"Base run name must end with '_ex<N>': {base_name}")
    return match.group(1)


def find_runs(base_config: Path):
    base_out_dir = base_config.resolve().parent
    output_root = base_out_dir.parent
    prefix = run_prefix(base_out_dir.name)
    pattern = re.compile(rf"^{re.escape(prefix)}([0-9]+)$")

    runs = []
    for candidate in output_root.iterdir():
        if not candidate.is_dir():
            continue
        match = pattern.match(candidate.name)
        if match is None:
            continue
        config_path = candidate / "config.json"
        diag_path = candidate / "temporal" / "dk_diagnostic.txt"
        if config_path.is_file() and diag_path.is_file():
            runs.append((int(match.group(1)), candidate, config_path, diag_path))

    return sorted(runs, key=lambda item: item[0])


def particle_label(config, ex_number: int):
    particles = config.get("Particles", [])
    np_by_sort = {
        str(item.get("sort_name")): item.get("Np")
        for item in particles
        if "sort_name" in item and "Np" in item
    }
    np_e = np_by_sort.get("electrons")
    np_i = np_by_sort.get("ions")

    if np_e is not None and np_i is not None:
        if np_e == np_i:
            return f"ex{ex_number}, Np={np_e:g}"
        return f"ex{ex_number}, Np_e={np_e:g}, Np_i={np_i:g}"

    if len(np_by_sort) == 1:
        only_np = next(iter(np_by_sort.values()))
        return f"ex{ex_number}, Np={only_np:g}"

    return f"ex{ex_number}"


def finite_xy(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def prepare_run(ex_number, _run_dir, config_path, diag_path, time_max):
    config = load_config(config_path)
    dt = float(config.get("Geometry", {}).get("dt", 1.0))
    header, data = load_energy_table(diag_path)

    time_iter = column(header, data, "Time")
    if time_iter is None:
        print(f"[warn] no Time column in {diag_path}; skipping")
        return None

    time = time_iter * dt
    if time_max is not None:
        mask = time <= time_max
        time = time[mask]
        data = data[mask, :]
    if time.size == 0:
        print(f"[warn] no rows with omega_pe*t <= {time_max:g} in {diag_path}")
        return None

    wE = column(header, data, "wE")
    wB = column(header, data, "wB")
    wK_columns = [
        (name[len("wK_"):], data[:, idx])
        for idx, name in enumerate(header)
        if name.startswith("wK_")
    ]

    W = None
    if wE is not None and wB is not None:
        W = wE + wB
        for _, wK in wK_columns:
            W = W + wK
    else:
        print(f"[warn] no wE/wB columns in {diag_path}; total W skipped")

    return {
        "ex_number": ex_number,
        "label": particle_label(config, ex_number),
        "time": time,
        "wE": wE,
        "wK_columns": wK_columns,
        "W": W,
    }


def setup_axis(ax, ylabel, title):
    ax.set_xlabel(r"$\omega_{pe}\,t$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)


def save_figure(fig, out_dir: Path, filename: str, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Energy comparison figure written to {out_path}")


def plot_wE(runs, out_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("tab10")
    for idx, run in enumerate(runs):
        if run["wE"] is None:
            print(f"[warn] no wE for {run['label']}; skipping")
            continue
        x, y = finite_xy(run["time"], run["wE"])
        if x.size == 0:
            print(f"[warn] no finite wE values for {run['label']}; skipping")
            continue
        ax.plot(x, y, label=run["label"], color=cmap(idx % 10), linewidth=2)
    setup_axis(ax, r"$w_E$", r"Electric-field energy")
    ax.legend(loc="best")
    save_figure(fig, out_dir, "compare_wE.png", dpi)


def plot_wK(runs, out_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("tab10")
    linestyles = {
        "electrons": "-",
        "ions": "--",
    }
    for idx, run in enumerate(runs):
        color = cmap(idx % 10)
        for sort_name, wK in run["wK_columns"]:
            if sort_name not in ("electrons", "ions"):
                continue
            wK0 = float(wK[0])
            if wK0 == 0.0:
                print(f"[warn] wK_{sort_name}(0) = 0 for {run['label']}; skipping")
                continue
            rel_wK = (wK - wK0) / wK0
            x, y = finite_xy(run["time"], rel_wK)
            if x.size == 0:
                print(f"[warn] no finite relative wK_{sort_name} values for {run['label']}; skipping")
                continue
            ax.plot(x, y, label=f"{run['label']}, {sort_name}",
                    color=color, linestyle=linestyles.get(sort_name, "-"),
                    linewidth=2)
    setup_axis(ax, "(wK - wK0) / wK0", "Relative kinetic-energy change")
    ax.legend(loc="best")
    save_figure(fig, out_dir, "compare_wK.png", dpi)


def plot_total_energy(runs, out_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("tab10")
    for idx, run in enumerate(runs):
        if run["W"] is None:
            continue
        W0 = float(run["W"][0])
        if W0 == 0.0:
            print(f"[warn] W0 = 0 for {run['label']}; skipping")
            continue
        rel_W = (run["W"] - W0) / W0
        x, y = finite_xy(run["time"], rel_W)
        if x.size == 0:
            print(f"[warn] no finite relative W values for {run['label']}; skipping")
            continue
        ax.plot(x, y, label=run["label"], color=cmap(idx % 10), linewidth=2)
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    setup_axis(ax, "(W - W0) / W0", "Relative total-energy change")
    ax.legend(loc="best")
    save_figure(fig, out_dir, "compare_total_energy.png", dpi)


def main():
    args = parse_args()
    base_config = Path(args.config)
    if not base_config.is_file():
        raise SystemExit(f"[error] config not found: {base_config}")

    raw_runs = find_runs(base_config)
    if not raw_runs:
        raise SystemExit(
            f"[error] no matching runs found for {base_config.parent.name}")

    runs = []
    for item in raw_runs:
        run = prepare_run(*item, time_max=args.time_max)
        if run is not None:
            runs.append(run)
    if not runs:
        raise SystemExit("[error] no runs with usable energy diagnostics")

    out_dir = base_config.resolve().parent / "processed" / args.out_subdir
    plot_wE(runs, out_dir, args.dpi)
    plot_wK(runs, out_dir, args.dpi)
    plot_total_energy(runs, out_dir, args.dpi)


if __name__ == "__main__":
    main()
