#!/usr/bin/env python3
"""Compare energy conservation in the two hybrid scan_dt runs at Omega_e*tau=10."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from hybrid_ex1_energy import load_energies
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
CASES = (
    ("mi1836_dt10", "hybrid", "tab:red"),
    ("mi1836mm_dt10", "hybrid + mm precond", "tab:green"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path,
                        default=REPO / ".claude/figures/preconditioner_hybrid_energy")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    curves = []
    reference_config = None
    for suffix, label, color in CASES:
        directory = REPO / "tests/hybrid/output" / ("hybrid_ex1_hybrid_kinetic_" + suffix)
        config = json.loads((directory / "config.json").read_text())
        if reference_config is None:
            reference_config = config
        else:
            assert config["Geometry"] == reference_config["Geometry"]
            assert config["Particles"] == reference_config["Particles"]
            assert config["Presets"] == reference_config["Presets"]
        steps, _, WK0, _, _, _, dW, _, _, _ = load_energies(str(directory), config)
        time = steps * float(config["Geometry"]["dt"])
        # Accumulate the accurately recorded increments; absolute energies
        # in the diagnostics have fewer significant digits. Set W(0)-W(0)=0.
        delta = np.cumsum(dW)
        error = (delta - delta[0]) / WK0
        assert np.all(np.isfinite(error)) and WK0 > 0
        if curves:
            np.testing.assert_array_equal(time, curves[0][0])
        curves.append((time, error))
        ax.plot(time, error, color=color, lw=1.2, label=label)
        print(f"{label}: max |W(t)-W(0)|/WK(0) = {np.max(np.abs(error)):.6e}")
    ax.axhline(0, color="0.4", lw=0.7, zorder=0)
    ax.set_xlim(0, curves[0][0][-1])
    ax.margins(y=0.18)
    ax.set_xlabel(r"$\omega_{pe}t$", fontsize=22)
    ax.set_ylabel(r"$(W(t)-W(0))/W_K(0)$", fontsize=22)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(18)
    ax.tick_params(labelsize=18)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=16, framealpha=1)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=180,
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    np.savetxt(args.output.with_suffix(".csv"),
               np.column_stack((curves[0][0], curves[0][1], curves[1][1])),
               delimiter=",", header="omega_pe_t,hybrid,hybrid_mm_precond", comments="")
    print(f"Saved {args.output}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
