#!/usr/bin/env python3
"""Plot four fixed-step wall times by model from the ex1 scan_dt data.

Uses the same timing and SNES averaging as hybrid_ex1_scan_dt.py.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from hybrid_ex1_scan_dt import read_run
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[3]
CASES = (
    ("eccapfim_mi1836_dt0.2", "eccapfim", "tab:blue"),
    ("ecsimcorr_mi1836_dt0.2", "ecsimcorr", "tab:orange"),
    ("hybrid_kinetic_mi1836_dt10", "hybrid", "tab:red"),
    ("hybrid_kinetic_mi1836mm_dt10", "hybrid + mm precond", "tab:green"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path,
                        default=REPO / ".claude/figures/preconditioner_scan_dt_levels")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    configurations = []
    for suffix, label, color in CASES:
        directory = REPO / "tests/hybrid/output" / ("hybrid_ex1_" + suffix)
        run = read_run(str(directory), None)
        if not run["converged"]:
            raise RuntimeError(f"Missing completed timing: {directory}")
        configurations.append(json.loads((directory / "config.json").read_text()))
        rows.append(dict(label=label, tau=float(run["tau"]),
                         wall=float(run["wall"]),
                         snes=float(run["N_it"]) if "hybrid" in label else None,
                         source=str(directory.relative_to(REPO))))

    # Compare the same physical interval and plasma setup.
    reference = configurations[0]
    for config in configurations[1:]:
        for key in ("x", "y", "z", "t", "dx", "dy", "dz"):
            assert config["Geometry"][key] == reference["Geometry"][key]
        for species, initial in zip(config["Particles"], reference["Particles"]):
            for key in ("sort_name", "Np", "n", "q", "m", "T"):
                assert species[key] == initial[key]

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    for x, (row, (_, _, color)) in enumerate(zip(rows, CASES)):
        ax.plot(x, row["wall"], "o", color=color, ms=10, zorder=3)
        ax.annotate(f"{row['wall']:.0f} s", (x, row["wall"]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=16, color=color)
        if row["snes"] is not None:
            ax.annotate(f"{row['snes']:.1f} SNES it", (x, row["wall"]),
                        xytext=(0, -10), textcoords="offset points",
                        ha="center", va="top", fontsize=16, color=color)
    # Leave room below the fastest level for its SNES annotation.
    ax.set(xlim=(-0.45, 3.65), ylim=(-160, 2200))
    ax.set_xticks(range(len(rows)), [row["label"] for row in rows])
    ax.set_yticks(range(0, 2201, 400))
    ax.set_ylabel("wall time, s", fontsize=22)
    ax.tick_params(labelsize=18)
    ax.tick_params(axis="x", labelsize=16)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=180,
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        snes = f", {row['snes']:.1f} SNES it" if row["snes"] is not None else ""
        print(f"{row['label']}: {row['wall']:.3f} s{snes}")
    print(f"Saved {args.output}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
