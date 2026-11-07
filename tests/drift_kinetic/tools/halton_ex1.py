#!/usr/bin/env python3
"""Analyze eleven 3-D frames per run: t=0 and ten DK steps with dt=10.

Density uses the quadratic charge-grid shape. E is stored on staggered Yee
faces; its norm is the sum of the component grid norms (no interpolation).
The comparison covers the complete IAW quiet loader, including its Halton /
stratified positions and antithetic velocities, versus random Maxwellians.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output" / "halton_ex1"


def read_case(path):
    with (path / "config.json").open() as stream:
        config = json.load(stream)
    geometry = config["Geometry"]
    shape = tuple(round(geometry[a] / geometry["d" + a]) for a in "zyx")
    dt = float(geometry["dt"])
    steps = round(geometry["t"] / dt)
    if steps != 10 or dt != 10:
        raise ValueError(f"{path}: expected ten steps with dt=10")

    def frames(directory, components=1):
        data = []
        for step in range(steps + 1):
            filename = path / directory / f"{step:04d}"
            values = np.fromfile(filename, dtype=np.float32).astype(float)
            expected = np.prod(shape) * components
            if values.size != expected or not np.all(np.isfinite(values)):
                raise ValueError(f"{filename}: invalid 3-D frame")
            data.append(values.reshape(shape + ((components,) if components > 1 else ())))
        return np.stack(data)

    electric = frames("E", 3)
    densities = {species: frames(f"{species}/density") for species in ("electrons", "ions")}
    for species, density in densities.items():
        if not np.allclose(density.mean(axis=(1, 2, 3)), 1, rtol=0, atol=1e-6):
            raise AssertionError(f"{path}: {species} mean density differs from 1")
    if np.any(electric[0] != 0):
        raise AssertionError(f"{path}: initial E must be zero")
    sampling = config["Sampling"]
    return dict(path=path, config=config, time=np.arange(steps + 1) * dt,
                ppc=int(sampling["markers_per_cell"]), seed=int(sampling["seed"]),
                method="halton" if sampling["algorithm"] == "halton_quiet" else "random",
                electric=electric, densities=densities,
                e_rms=np.sqrt(np.mean(np.sum(electric**2, axis=-1), axis=(1, 2, 3))),
                n_rms=np.sqrt(np.mean((densities["electrons"] - 1)**2, axis=(1, 2, 3))))


def mean_sd(values):
    values = np.asarray(values)
    return values.mean(axis=0), values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(values[0])


def write_statistics(cases, output):
    # Statistics only; full 3-D arrays remain in the native binary diagnostics.
    with (output / "statistics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "particles_per_cell", "seed", "time", "E_rms", "E_max",
                         "electron_density_rms", "electron_density_mean_abs", "electron_density_max_abs",
                         "ion_density_rms", "ion_density_mean_abs", "ion_density_max_abs"])
        for case in cases:
            for step, time in enumerate(case["time"]):
                row = [case["method"], case["ppc"], case["seed"], time, case["e_rms"][step],
                       np.sqrt(np.sum(case["electric"][step]**2, axis=-1)).max()]
                for species in ("electrons", "ions"):
                    error = np.abs(case["densities"][species][step] - 1)
                    row += [np.sqrt(np.mean(error**2)), error.mean(), error.max()]
                writer.writerow(row)


def analyze(cases, output):
    ppcs = sorted({case["ppc"] for case in cases})
    grouped = {(method, ppc): [c for c in cases if c["method"] == method and c["ppc"] == ppc]
               for method in ("halton", "random") for ppc in ppcs}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    summary = []
    n_scaling = {"halton": [], "random": []}
    n_scatter = {"halton": [], "random": []}
    marker_counts = []
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ppcs)))
    for ppc, color in zip(ppcs, colors):
        h = grouped["halton", ppc]
        r = grouped["random", ppc]
        if len(h) != 1 or len(r) != 3:
            raise ValueError(f"ppc={ppc}: expected one Halton and three random runs")
        marker_counts.append(ppc * np.prod(h[0]["electric"].shape[1:4]))
        for method, group, style in (("halton", h, "-"), ("random", r, "--")):
            time = group[0]["time"]
            for ax, key, start in ((axes[0], "e_rms", 1), (axes[1], "n_rms", 0)):
                mean, sd = mean_sd([c[key] for c in group])
                ax.plot(time[start:], mean[start:], style, color=color,
                        label=f"{method}, {ppc}/cell")
                if method == "random":
                    ax.fill_between(time[start:], np.maximum(mean[start:] - sd[start:], 1e-30),
                                    mean[start:] + sd[start:], color=color, alpha=0.15)
            # RMS over space AND the ten evolved frames, then mean/SD over
            # independent seeds. Do not treat correlated timesteps as repeats.
            mean, sd = mean_sd([np.sqrt(np.mean(c["n_rms"][1:]**2)) for c in group])
            n_scaling[method].append(float(mean))
            n_scatter[method].append(float(sd))
        he = float(np.sqrt(np.mean(h[0]["e_rms"][1:]**2)))
        re = np.array([np.sqrt(np.mean(c["e_rms"][1:]**2)) for c in r])
        if not all(h[0]["n_rms"][0] < c["n_rms"][0] for c in r) or not np.all(he < re):
            raise AssertionError(f"ppc={ppc}: Halton must reduce initial density and evolved E noise")
        rm, rs = mean_sd(re)
        gain, gain_sd = mean_sd(re / he)
        summary.append([ppc, he, float(rm), float(rs), float(gain), float(gain_sd),
                        n_scaling["halton"][-1], n_scaling["random"][-1], n_scatter["random"][-1]])
    marker_counts = np.asarray(marker_counts, dtype=float)
    for method, color in (("halton", "tab:blue"), ("random", "tab:orange")):
        label = method
        if len(ppcs) > 1:
            exponent = np.polyfit(np.log(marker_counts), np.log(n_scaling[method]), 1)[0]
            label += rf": fitted $p={exponent:.2f}$"
        axes[2].errorbar(marker_counts, n_scaling[method], yerr=n_scatter[method],
                         fmt="o-", color=color, label=label, capsize=3)
    if len(ppcs) > 1:
        reference_n = np.geomspace(marker_counts.min(), marker_counts.max(), 100)
        # Reference slopes, each normalized to its method's first sample.
        # These are visual guides, not universal convergence claims for Halton.
        for method, exponent, label, style in (
                ("random", -0.5, r"$O(N^{-1/2})$ reference", "--"),
                ("halton", -1.0, r"$O(N^{-1})$ reference", ":")):
            reference = n_scaling[method][0] * (reference_n / marker_counts[0])**exponent
            axes[2].plot(reference_n, reference, style, color="0.4", label=label)
    time_label = r"Time, $\omega_{pe}^{-1}$"
    axes[0].set(xlabel=time_label, ylabel=r"$\sqrt{\langle |E|^2\rangle_x}$", yscale="log",
                title="Electric noise: mean ± seed SD")
    axes[1].set(xlabel=time_label, ylabel=r"$\sqrt{\langle (n_e-1)^2\rangle_x}$", yscale="log",
                title="Electron density noise: mean ± seed SD")
    axes[2].set(xlabel="Total markers per species, N",
                ylabel=r"$\sqrt{\langle (n_e-1)^2\rangle_{x,t}}$",
                xscale="log", yscale="log", title="Density noise versus marker count")
    axes[2].set_xticks(marker_counts, labels=[f"{n:.0f}" for n in marker_counts])
    axes[2].tick_params(axis="x", which="minor", labelbottom=False)
    axes[2].text(0.03, 0.03, "References normalized at smallest N",
                 transform=axes[2].transAxes, fontsize=8, color="0.35")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(output / "halton_ex1.png", dpi=180)
    fig.savefig(output / "halton_ex1.pdf")
    plt.close(fig)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["particles_per_cell", "halton_E_rms", "random_E_rms_mean", "random_E_rms_sd",
                         "gain_mean", "gain_sd", "halton_density_rms", "random_density_rms_mean",
                         "random_density_rms_sd"])
        writer.writerows(summary)
    for ppc, he, rm, rs, gain, gain_sd, *_ in summary:
        print(f"ppc={ppc:3}: E RMS Halton={he:.5e}, random={rm:.5e} ± {rs:.2e}, "
              f"gain={gain:.2f} ± {gain_sd:.2f}")
    return grouped, ppcs


def plot_slices(grouped, ppcs, output):
    # Show both species for the smallest count, with the same physical slice
    # and color scale for quiet and random loading. Use one actual random
    # realization, not a seed-averaged map which would hide sampling noise.
    ppc = min(ppcs)
    h = grouped["halton", ppc][0]
    r = min(grouped["random", ppc], key=lambda c: c["seed"])
    geometry = h["config"]["Geometry"]
    y_index = h["densities"]["electrons"].shape[2] // 2
    slices = [[case["densities"][species][-1, :, y_index, :]
               for case in (h, r)] for species in ("electrons", "ions")]
    deviation = max(np.max(np.abs(values - 1)) for row in slices for values in row)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    xlo, zlo = -0.5 * geometry["dx"], -0.5 * geometry["dz"]
    for row, species in enumerate(("electrons", "ions")):
        for col, case in enumerate((h, r)):
            image = axes[row, col].imshow(slices[row][col], origin="lower", cmap="RdBu_r",
                vmin=1 - deviation, vmax=1 + deviation, interpolation="nearest",
                extent=(xlo, geometry["x"] + xlo, zlo, geometry["z"] + zlo))
            axes[row, col].set(xlabel="x", ylabel="z", title=f"{case['method']}: {species}")
    fig.colorbar(image, ax=axes, shrink=0.85, label="Number density n (reference = 1)")
    fig.suptitle(f"t={h['time'][-1]:g}, y={y_index * geometry['dy']:g}, "
                 f"{ppc} markers/cell/species; random seed={r['seed']}")
    fig.savefig(output / "density_slices.png", dpi=180)
    fig.savefig(output / "density_slices.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with (args.output / "cases.json").open() as stream:
        paths = json.load(stream)
    if not paths:
        raise ValueError("Empty scan manifest; run halton_ex1 first")
    cases = [read_case(args.output / path) for path in paths]
    write_statistics(cases, args.output)
    grouped, ppcs = analyze(cases, args.output)
    plot_slices(grouped, ppcs, args.output)


if __name__ == "__main__":
    main()
