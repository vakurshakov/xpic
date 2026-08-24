#!/usr/bin/env python3
"""Trajectory comparison of the Boris pusher and the drift-kinetic pusher.

The script loads boris_<tau>.txt and drift_kinetic_<tau>.txt traces
(columns: t, x, y, z) and draws a single projection plane (xy, xz or yz)
selected with --plane. Two curves are shown: Boris tau=0.1 (black, solid)
and drift-kinetic tau=100 (black, long dashes). For --drift exb or gradb
the start and end points of the analytic guiding-center trajectory are
marked too.
Axis limits are taken from the min/max extent of the Boris curve.
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
from lib.plot import labelsize, ticksize

import matplotlib.pyplot as plt

# Legend labels are Cyrillic; render without LaTeX (mathtext + DejaVu Sans
# still draws the Cyrillic glyphs and the math). lib.plot enables usetex on
# import, so turn it back off here.
plt.rc("text", usetex=False)

# Column index of time and of each coordinate in the *.txt trace files.
COL = {"t": 0, "x": 1, "y": 2, "z": 3}

# Legend title per drift type passed via --drift.
DRIFT_TITLES = {
    "exb": r"$\mathbf{E}\times\mathbf{B}$ - дрейф",
    "gradb": r"$\nabla B$ - дрейф",
    "trap": "Магнитная ловушка",
}

# Curves to draw: (file prefix, tau in 1/Omega_e, color, linestyle, method).
CURVES = [
    ("boris", 0.1, "black", "-", "B"),
    ("drift_kinetic", 100.0, "black", ":", "DK"),
]

# Trap comparison panels for tests/drift_kinetic/drift_kinetic_push_ex3.cpp:
# (panel label, pitch fraction, axes column, curves).
TRAP_PANELS = [
    ("(a)", 1.005, 0, [("Boris", 0.1, "black", "-", "B"),
                      ("DK", 10.0, "red", "-", "DK")]),
    ("(б)", 0.995, 1, [("Boris", 0.1, "black", "-", "B"),
                      ("DK", 10.0, "red", "-", "DK")]),
]

# Physical parameters of the ExB-drift test, mirrored from
# tests/drift_kinetic/drift_kinetic_push_ex1.cpp (E0, B0, r0, v0 and the
# charge/mass q, m from drift_kinetic_push.h).
TEST_E0 = np.array([0.0, 0.005, -0.1])
TEST_B0 = np.array([0.0, 0.0, 0.2])
TEST_R0 = np.array([0.0, 0.0, 0.0])
TEST_V0 = np.array([0.0, 0.05, 0.0])
TEST_Q = -1.0
TEST_M = +1.0

# Physical parameters of the grad-B-drift test, mirrored from
# tests/drift_kinetic/drift_kinetic_push_ex2.cpp.
GRADB_ALPHA = -0.8
GRADB_B0_MAG = 0.2
GRADB_B0_DIR = np.array([0.0, 0.0, 1.0])
GRADB_R0 = np.array([0.0, 0.0, 0.0])
GRADB_V0 = np.array([0.0, 0.05, 0.0])


def exb_theory(t):
    """Analytic guiding-center trajectory r(t) = (x, y, z) for the ExB test.

    Mirrors the reference solution of drift_kinetic_push_ex1.cpp: a uniform
    ExB drift of the guiding center plus a uniform acceleration along B due
    to the parallel electric field. The guiding center at t=0 is the initial
    particle position shifted by the gyro-radius vector.
    """
    b_len = np.linalg.norm(TEST_B0)
    b_hat = TEST_B0 / b_len
    qm = TEST_Q / TEST_M

    r_gc0 = TEST_R0 + np.cross(TEST_V0, b_hat) / (qm * b_len)
    v_drift = np.cross(TEST_E0, TEST_B0) / b_len**2
    a_parallel = TEST_Q * np.dot(TEST_E0, b_hat) / TEST_M

    t = np.asarray(t)[:, None]  # (N, 1), broadcast over the xyz axis
    r = r_gc0 + v_drift * t + b_hat * (0.5 * a_parallel * t**2)
    return r[:, 0], r[:, 1], r[:, 2]


def gradb_theory(t):
    """Analytic guiding-center trajectory r(t) for the grad-B test.

    Mirrors drift_kinetic_push_ex2.cpp: B = B0*(1 + alpha*y)*z, E = 0,
    p_parallel = 0, so only the grad-B drift remains. Since y_gc is constant,
    the drift velocity is constant and x(t) is linear.
    """
    b_hat = GRADB_B0_DIR / np.linalg.norm(GRADB_B0_DIR)
    qm = TEST_Q / TEST_M

    B0_at_r0 = GRADB_B0_MAG * (1.0 + GRADB_ALPHA * GRADB_R0[1])
    r_gc0 = GRADB_R0 + np.cross(GRADB_V0, b_hat) / (qm * B0_at_r0)
    p_perp = np.linalg.norm(GRADB_V0 - np.dot(GRADB_V0, b_hat) * b_hat)
    mu = TEST_M * p_perp**2 / (2.0 * B0_at_r0)
    v_grad_x = -(mu / (qm * TEST_M)) * GRADB_ALPHA / (
        1.0 + GRADB_ALPHA * r_gc0[1])

    t = np.asarray(t)[:, None]
    r = r_gc0 + np.array([v_grad_x, 0.0, 0.0]) * t
    return r[:, 0], r[:, 1], r[:, 2]


THEORY = {
    "exb": exb_theory,
    "gradb": gradb_theory,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=".",
                        help="Directory with the boris_*/drift_kinetic_* traces")
    parser.add_argument("--plane", choices=("xy", "xz", "yz"), default="xy",
                        help="Projection plane to draw (default: xy)")
    parser.add_argument("--drift",
                        choices=("exb", "gradb", "trap",
                                 "energy_dK", "energy_dW", "charge",
                                 "energy_charge", "energy_charge_loss",
                                 "trap_energy_dK"),
                        default="exb",
                        help="Drift type shown as the legend title "
                             "(default: exb)")
    parser.add_argument("--path", default=None,
                        help="Run name (or directory) holding the "
                             "dk_diagnostic.txt for the energy panel of "
                             "--drift trap_energy_dK, e.g. "
                             "drift_kinetic_mirror_ex1")
    parser.add_argument("--time", type=float, default=None,
                        help="Upper limit on the time axis (for --drift energy_dK)")
    parser.add_argument("--energy", type=float, default=None,
                        help="Symmetric +-limit on the energy y axis "
                             "(for --drift energy_dK/energy_dW/"
                             "energy_charge/energy_charge_loss)")
    parser.add_argument("--inset", action="store_true",
                        help="Show the settled-tail inset in the energy "
                             "panel of --drift energy_charge/"
                             "energy_charge_loss")
    parser.add_argument("--out", default=None,
                        help="Output image path "
                             "(default: <dir>/traces_<plane>.png)")
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def padded_limits(values, margin=0.05):
    """Min/max of `values` widened by `margin` of the data range on each side."""
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = margin * (hi - lo) or margin
    return lo - pad, hi + pad


def mark_endpoints(ax, xs, ys, color, size=7, zorder=5):
    ax.plot(xs[0], ys[0], "o", color="white", markeredgecolor=color,
            markeredgewidth=1.5, markersize=size, zorder=zorder)
    ax.plot(xs[-1], ys[-1], "o", color=color, markersize=size, zorder=zorder)


def load_trace(directory, prefix, tau):
    """Load <prefix>_<tau>.txt, falling back to the unsuffixed <prefix>.txt."""
    candidates = [f"{prefix}_{tau:g}.txt", f"{prefix}.txt"]
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return np.loadtxt(path, skiprows=1), name
    raise FileNotFoundError(
        f"none of {candidates} found in {directory}")


def load_named_trace(directory, name):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found in {directory}")
    return np.loadtxt(path, skiprows=1), name


def plot_trap(args):
    a, b = args.plane[0], args.plane[1]
    ca, cb = COL[a], COL[b]

    fig, gs = figure(2, 1, figsize=(14, 8))
    axes = [subplot(fig, gs, 0, 0), subplot(fig, gs, 1, 0)]

    for panel_label, pitch_frac, panel_col, curves in TRAP_PANELS:
        ax = axes[panel_col]
        boris_xs, boris_ys = [], []

        for prefix, omega_dt, color, style, method in curves:
            filename = f"{prefix}_omega_dt_{omega_dt:.4f}_pf_{pitch_frac:.3f}.txt"
            data, name = load_named_trace(args.dir, filename)
            xs, ys = data[:, ca], data[:, cb]

            zorder = 4 if prefix == "DK" else 2
            width = 3.0 if prefix == "DK" else 1.2
            ax.plot(xs, ys, color=color, linestyle=style, linewidth=width,
                    zorder=zorder,
                    label=rf"$\tau_{{{method}}}\Omega_e = {omega_dt:g}$".replace(".", ","))
            mark_endpoints(ax, xs, ys, color, zorder=zorder)
            print(f"  {method}, omega_dt={omega_dt:g}, pf={pitch_frac:.3f}: {name}")

            if prefix == "Boris":
                boris_xs.append(xs)
                boris_ys.append(ys)

        ax.set_xlim(*padded_limits(np.concatenate(boris_xs)))
        ax.set_ylim(*padded_limits(np.concatenate(boris_ys)))
        ax.set_xlabel(rf"${a}\ (c/\omega_{{pe}})$", fontsize=labelsize)
        ax.set_ylabel(rf"${b}\ (c/\omega_{{pe}})$", fontsize=labelsize)
        ax.tick_params(labelsize=ticksize)
        ax.text(0.03, 0.95, panel_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=labelsize,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, boxstyle="round,pad=0.25"))
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=1, fontsize=labelsize)
        ax.grid(True)

    fig.tight_layout()
    out = args.out or os.path.join(args.dir, f"traces_trap_{args.plane}.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}")


def _load_dt(directory):
    """Time step dt from config.json, searched in `directory` and its parent."""
    import json
    for d in (directory, os.path.dirname(os.path.abspath(directory))):
        path = os.path.join(d, "config.json")
        if os.path.exists(path):
            with open(path) as f:
                return float(json.load(f)["Geometry"]["dt"])
    raise FileNotFoundError(
        f"config.json not found in {directory} or its parent")


def _resolve_run_dir(path):
    """Locate the temporal directory holding dk_diagnostic.txt for `path`.

    `path` may be a run name under ../output, a run directory, or the
    temporal directory itself.
    """
    base = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "output"))
    candidates = [
        path,
        os.path.join(path, "temporal"),
        os.path.join(base, path),
        os.path.join(base, path, "temporal"),
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "dk_diagnostic.txt")):
            return d
    raise FileNotFoundError(
        f"dk_diagnostic.txt not found for --path {path!r} "
        f"(looked in: {', '.join(candidates)})")


def _load_dk_diagnostic(directory):
    path = os.path.join(directory, "dk_diagnostic.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"dk_diagnostic.txt not found in {directory}")
    with open(path) as f:
        header = f.readline().split()
    # "dt * a_EJ" / "wEB + wK" are written with spaces — collapse stray tokens.
    cleaned = []
    for tok in header:
        if tok in ("*", "+", "-") and cleaned:
            cleaned[-1] = cleaned[-1] + tok
        elif cleaned and cleaned[-1].endswith(("*", "+", "-")):
            cleaned[-1] = cleaned[-1] + tok
        else:
            cleaned.append(tok)
    idx = {name: i for i, name in enumerate(cleaned)}
    data = np.loadtxt(path, skiprows=1)
    return data, idx


def _load_energy_conservation(directory):
    """Load field-energy changes and losses in the B-B0 convention."""
    path = os.path.join(directory, "energy_conservation.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"energy_conservation.txt not found in {directory}")
    with open(path) as f:
        header = f.readline().split()
    idx = {name: i for i, name in enumerate(header)}
    required = ("Time", "dE", "dB", "Damped(E+B)")
    missing = [name for name in required if name not in idx]
    if missing:
        raise ValueError(
            f"{path} has no required column(s): {', '.join(missing)}")
    data = np.loadtxt(path, skiprows=1)
    return data, idx


def _plot_energy_curve(args, t, y, ylabel, out_suffix,
                       color="darkgreen", symmetric=True):
    if args.time is not None:
        mask = t <= args.time
        t = t[mask]
        y = y[mask]

    fig, gs = figure(1, 1, figsize=(14, 6))
    ax = subplot(fig, gs, 0, 0)

    ax.plot(t, y, color=color, linestyle="-", linewidth=3.0, zorder=2)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.5,
               zorder=3, label="_nolegend_")

    ax.set_xlim(0.0, float(args.time) if args.time is not None else float(t.max()))
    if symmetric:
        if args.energy is not None:
            m = float(args.energy)
        else:
            m = float(np.max(np.abs(y))) or 1.0
        ax.set_ylim(-m * 1.05, m * 1.05)
    else:
        hi = float(np.max(y)) or 1.0
        ax.set_ylim(0.0, hi * 1.05)
    ax.set_xlabel(r"$t,\ \omega_{pe}^{-1}$", fontsize=labelsize + 4)
    ax.set_ylabel(ylabel, fontsize=labelsize + 4)
    ax.tick_params(labelsize=ticksize + 4)
    ax.yaxis.get_offset_text().set_fontsize(ticksize + 4)
    ax.xaxis.get_offset_text().set_fontsize(ticksize + 4)
    ax.grid(True)

    fig.tight_layout()
    out = args.out or os.path.join(args.dir, f"traces_{out_suffix}.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"Saved {out}")


def plot_energy_dK(args):
    data, idx = _load_dk_diagnostic(args.dir)
    t = data[:, idx["Time"]]
    dK = data[:, idx["dK"]]
    wK0 = data[0, idx["wK"]]
    cum = np.cumsum(dK) / wK0
    _plot_energy_curve(args, t, cum, r"$\Delta W_K(t)/W_K(0)$", "energy_dK")


def plot_energy_dW(args):
    data, idx = _load_dk_diagnostic(args.dir)
    t = data[:, idx["Time"]]
    W = data[:, idx["wEB+wK"]]
    W0 = W[0]
    y = (W - W0) / W0
    _plot_energy_curve(args, t, y, r"$\Delta W(t)/W(0)$", "energy_dW")


def plot_charge(args):
    data, idx = _load_dk_diagnostic(args.dir)
    t = data[:, idx["Time"]]
    y = data[:, idx["N2dQ_tot"]]
    ylabel = r"$\mathcal{R}_\rho(t)$"
    _plot_energy_curve(args, t, y, ylabel, "charge",
                       color="saddlebrown", symmetric=False)


def plot_energy_charge(args, include_damping_loss=False):
    """Energy drift (top) and charge residual (bottom) stacked vertically,
    sharing a single time axis, in the drift_kinetic_exb_ex1.py style.

    When ``include_damping_loss`` is true, combine the drift-kinetic dK with
    the field-energy changes and damping loss from energy_conservation.txt.
    The latter consistently measures magnetic perturbation energy (B-B0)^2/2
    instead of the full B^2/2 used by the DK diagnostic.
    """
    data, idx = _load_dk_diagnostic(args.dir)

    # LaTeX look without a TeX install: Computer Modern mathtext on a serif
    # face, matching drift_kinetic_curv_ex3_convergence.py.
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"

    # Slightly larger fonts than the base ex1 style for this two-panel figure.
    label_fs = EX_LABELSIZE + 4
    tick_fs = EX_TICKSIZE + 4

    # Time is the step index, i.e. t/tau in dimensionless units (no dt scaling).
    t = data[:, idx["Time"]]
    # Cumulative sum of the per-step cancellation-free increments (dE, dB from
    # the WAXPY-based dW = 1/2 (X+X_prev).(X-X_prev), dK from Kahan-summed
    # kinetic energy) instead of (W - W0)/W0 from absolute energy sums: the
    # latter subtracts two large near-equal numbers and loses precision as
    # W0 dominates the difference over a long run.
    W0 = data[0, idx["wEB+wK"]]
    if include_damping_loss:
        loss_data, loss_idx = _load_energy_conservation(args.dir)
        loss_t = loss_data[:, loss_idx["Time"]]
        if loss_t.shape != t.shape or not np.allclose(loss_t, t):
            raise ValueError(
                "Time columns in dk_diagnostic.txt and "
                "energy_conservation.txt do not match")
        energy_increment = (
            data[:, idx["dK"]]
            + loss_data[:, loss_idx["dE"]]
            + loss_data[:, loss_idx["dB"]]
            + loss_data[:, loss_idx["Damped(E+B)"]]
        )
    else:
        energy_increment = data[:, idx["dE+dB+dK"]]
    dW = np.cumsum(energy_increment) / W0
    charge = data[:, idx["N2dQ_tot"]]

    if args.time is not None:
        mask = t <= args.time
        t, dW, charge = t[mask], dW[mask], charge[mask]

    fig, gs = figure(1, 2, figsize=(14, 10))
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    ax_top.plot(t, dW, color="#2ca02c", linestyle="-", linewidth=2.0, zorder=2)
    ax_top.axhline(0.0, color=EX_COLOR_TH, linestyle="--", linewidth=1.0,
                   zorder=3)
    if args.energy is not None:
        m = float(args.energy)
    else:
        m = float(np.max(np.abs(dW))) or 1.0
    ax_top.set_ylim(-m * 1.05, m * 1.05)
    ax_top.set_ylabel(r"$\delta_W(t)$", fontsize=label_fs)

    # Zoom the settled tail inside the energy panel.  Scaling both axes to
    # the last 30% of the trace makes its small residual range visible after
    # a much larger initial transient.
    if args.inset and t.size >= 2:
        tail_start = t[0] + 0.7 * (t[-1] - t[0])
        tail_mask = t >= tail_start
        tail_t = t[tail_mask]
        tail_dW = dW[tail_mask]

        ax_zoom = ax_top.inset_axes([0.38, 0.08, 0.28, 0.36])
        ax_zoom.plot(tail_t, tail_dW, color="#2ca02c",
                     linestyle="-", linewidth=1.5)
        ax_zoom.set_xlim(float(tail_t[0]), float(tail_t[-1]))

        tail_lo = float(np.min(tail_dW))
        tail_hi = float(np.max(tail_dW))
        tail_span = tail_hi - tail_lo
        tail_pad = (0.12 * tail_span if tail_span else
                    max(0.05 * abs(tail_lo), np.finfo(float).eps))
        ax_zoom.set_ylim(tail_lo - tail_pad, tail_hi + tail_pad)
        ax_zoom.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_zoom.minorticks_on()
        ax_zoom.tick_params(axis="both", which="both", direction="in",
                            top=True, right=True,
                            labelsize=max(tick_fs - 5, 8))
        ax_zoom.yaxis.get_offset_text().set_fontsize(max(tick_fs - 5, 8))
        ax_zoom.grid(True, alpha=0.25)
        ax_top.indicate_inset_zoom(
            ax_zoom, edgecolor="0.25", linewidth=0.8, alpha=0.75)

    ax_bot.plot(t, charge, color="#ff7f0e", linestyle="-", linewidth=2.0,
                zorder=2)
    ax_bot.axhline(0.0, color=EX_COLOR_TH, linestyle="--", linewidth=1.0,
                   zorder=3)
    hi = float(np.max(charge)) or 1.0
    ax_bot.set_ylim(0.0, hi * 1.05)
    ax_bot.set_ylabel(r"$\mathcal{R}_\rho(t)$", fontsize=label_fs)
    ax_bot.set_xlabel(r"$t/\tau$", fontsize=label_fs)

    x_hi = float(args.time) if args.time is not None else float(t.max())
    ax_top.set_xlim(0.0, x_hi)

    for ax, panel in ((ax_top, "(a)"), (ax_bot, "(b)")):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        _ex_style_axes(ax, panel)
        ax.tick_params(axis="both", which="both", labelsize=tick_fs)
        ax.yaxis.get_offset_text().set_fontsize(tick_fs)
    # Shared x axis: only the bottom panel keeps the time tick labels.
    ax_top.tick_params(axis="x", which="both", labelbottom=False)

    # Keep extra padding around the large math labels; the default tight
    # layout can clip the left edge of delta_W(t) for some font backends.
    fig.tight_layout(pad=2.0)
    suffix = "energy_charge_loss" if include_damping_loss else "energy_charge"
    out = args.out or os.path.join(args.dir, f"traces_{suffix}.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved {out}")


# --- trap_energy_dK: three square panels in the style of drift_kinetic_exb_ex1.py
# Sizes mirror the example exactly.
EX_COLOR_KIN = "black"
EX_COLOR_DK = "red"
EX_COLOR_TH = "black"
EX_LABELSIZE = 17
EX_TICKSIZE = 13
EX_LEGENDSIZE = 12
EX_PANELSIZE = 17


def _ex_mark_ends(ax, xs, ys, color):
    ax.plot(xs[0], ys[0], "o", mfc="white", mec=color, mew=1.5, ms=7, zorder=5)
    ax.plot(xs[-1], ys[-1], "o", color=color, ms=7, zorder=5)


def _ex_style_axes(ax, panel):
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in",
        top=True, bottom=True, left=True, right=True, labelsize=EX_TICKSIZE)
    ax.tick_params(axis="x", which="both", labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", which="both", labelleft=True, labelright=False)
    # Panel tag in the top-right corner, as in the example figures.
    ax.text(0.94, 0.98, panel, transform=ax.transAxes,
        ha="right", va="top", fontsize=EX_PANELSIZE,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6,
            boxstyle="round,pad=0.2"))


def plot_trap_energy_dK(args):
    """Three square panels in the drift_kinetic_exb_ex1.py style:
    (a), (b) magnetic-trap trajectories (two pitch fractions, Boris=kinetic in
    red, DK=drift-kinetic in blue) and (c) the energy_dK time series from the
    run given by --path, formatted like the energy panel of
    drift_kinetic_energy.py (Delta W_K / W_K(0) vs omega_pe t)."""
    if args.path is None:
        raise SystemExit(
            "--drift trap_energy_dK requires --path <run> for the energy panel")

    # LaTeX look without a TeX install: Computer Modern mathtext on a serif
    # face, matching drift_kinetic_curv_ex3_convergence.py.
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "serif"

    a, b = args.plane[0], args.plane[1]
    ca, cb = COL[a], COL[b]

    fig, gs = figure(ncols=3, nrows=1, figsize=(14, 4.5))

    # (a), (b): magnetic-trap trajectories for the two pitch fractions, same
    # data as --drift trap but recoloured (Boris -> kinetic/red,
    # DK -> drift-kinetic/blue) with the example's legend wording.
    panels = ["(a)", "(b)"]
    traj_axes = []
    legend_handles = legend_labels = None
    for col, (panel, (_, pitch_frac, _, curves)) in enumerate(
            zip(panels, TRAP_PANELS)):
        ax = subplot(fig, gs, col, 0)
        for prefix, omega_dt, _color, _style, method in curves:
            filename = f"{prefix}_omega_dt_{omega_dt:.4f}_pf_{pitch_frac:.3f}.txt"
            data, name = load_named_trace(args.dir, filename)
            xs, ys = data[:, ca], data[:, cb]

            # Draw drift-kinetic above kinetic.
            if method == "B":
                color, width, zorder = EX_COLOR_KIN, 1.0, 2
                label = rf"kinetic: $\tau = {omega_dt:g}/\Omega_e$"
            else:
                color, width, zorder = EX_COLOR_DK, 2.0, 4
                label = rf"drift-kinetic: $\tau = {omega_dt:g}/\Omega_e$"
            ax.plot(xs, ys, color=color, linewidth=width, zorder=zorder,
                    label=label)
            _ex_mark_ends(ax, xs, ys, color)
            print(f"  {panel} pf={pitch_frac:.3f} {method}: {name}")

        ax.set_xlabel(rf"${a},\ c/\omega_{{pe}}$", fontsize=EX_LABELSIZE)
        ax.set_ylabel(rf"${b},\ c/\omega_{{pe}}$", fontsize=EX_LABELSIZE)
        ax.set_box_aspect(1)
        _ex_style_axes(ax, panel)
        traj_axes.append(ax)
        # One shared legend for (a)+(b); grab the handles from the first panel.
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # (c): energy_dK from --path, in the style of the energy panel of
    # drift_kinetic_energy.py (axhline at 0, omega_pe t on x).
    energy_dir = _resolve_run_dir(args.path)
    data, idx = _load_dk_diagnostic(energy_dir)
    # Time is the step index, i.e. t/tau in dimensionless units (no dt scaling).
    t = data[:, idx["Time"]]
    dK = data[:, idx["dK"]]
    wK0 = data[0, idx["wK"]]
    cum = np.cumsum(dK) / wK0

    ax = subplot(fig, gs, 2, 0)
    ax.plot(t, cum, color="#2ca02c", lw=2.0)
    ax.axhline(0.0, color=EX_COLOR_TH, ls="--", lw=1.0)
    ax.set_xlim(0.0, float(t.max()))
    m = float(np.max(np.abs(cum))) or 1.0
    ax.set_ylim(-m * 1.05, m * 1.05)
    ax.set_xlabel(r"$t/\tau$", fontsize=EX_LABELSIZE)
    ax.set_ylabel(r"$\Delta W_K(t)/W_K(0)$", fontsize=EX_LABELSIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(EX_TICKSIZE)
    ax.set_box_aspect(1)
    _ex_style_axes(ax, "(c)")
    print(f"  (c) energy_dK from {energy_dir}: "
          f"final dW_K/W_K(0) = {cum[-1]:.3e}")

    fig.tight_layout(pad=1.2, w_pad=2.5)

    # Shared legend for (a)+(b), centred between the two panels and just a touch
    # above them.
    pos_a = traj_axes[0].get_position()
    pos_b = traj_axes[1].get_position()
    mid_x = 0.25 * (pos_a.x0 + pos_a.x1 + pos_b.x0 + pos_b.x1)
    top_y = max(pos_a.y1, pos_b.y1)
    fig.legend(legend_handles, legend_labels, loc="lower center",
               bbox_to_anchor=(mid_x, top_y + 0.01), ncol=1,
               fontsize=EX_LEGENDSIZE)

    out = args.out or os.path.join(args.dir, "traces_trap_energy_dK.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    args = parse_args()

    if args.drift == "trap":
        plot_trap(args)
        return

    if args.drift == "energy_dK":
        plot_energy_dK(args)
        return

    if args.drift == "energy_dW":
        plot_energy_dW(args)
        return

    if args.drift == "charge":
        plot_charge(args)
        return

    if args.drift == "energy_charge":
        plot_energy_charge(args)
        return

    if args.drift == "energy_charge_loss":
        plot_energy_charge(args, include_damping_loss=True)
        return

    if args.drift == "trap_energy_dK":
        plot_trap_energy_dK(args)
        return

    a, b = args.plane[0], args.plane[1]
    ca, cb = COL[a], COL[b]

    fig, gs = figure(1, 1, figsize=(14, 6))
    ax = subplot(fig, gs, 0, 0)

    boris_xs, boris_ys = [], []
    t_max = 0.0
    for prefix, tau, color, style, method in CURVES:
        data, name = load_trace(args.dir, prefix, tau)
        xs, ys = data[:, ca], data[:, cb]

        # Layering, bottom to top: theory < DK < Boris. The DK line is the
        # thickest; keeping it at least as wide as the theory line lets its
        # dashes fully cover it (no pale anti-aliased rim peeking through).
        zorder = 4 if prefix == "boris" else 2
        width = 3.0 if prefix == "drift_kinetic" else 2.0
        ax.plot(xs, ys, color=color, linestyle=style, linewidth=width,
                zorder=zorder,
                label=rf"$\tau_{{{method}}}\Omega_e = {tau:g}$".replace(".", ","))
        mark_endpoints(ax, xs, ys, color, zorder=zorder)
        print(f"  {method}, tau={tau:g}: {name}")

        t_max = max(t_max, float(data[:, COL["t"]].max()))
        if prefix == "boris":
            boris_xs.append(xs)
            boris_ys.append(ys)

    # Analytic guiding-center trajectory: a solid line between its start and
    # end points, drawn below the other curves.
    r = dict(zip("xyz", THEORY[args.drift](np.array([0.0, t_max]))))
    ax.plot(r[a], r[b], color="red", linestyle="-", linewidth=2.0,
            zorder=1, label="Теоретический дрейф")
    mark_endpoints(ax, r[a], r[b], "red", size=14, zorder=1)

    # Axis limits from the Boris curve's extent, with a small margin.
    ax.set_xlim(*padded_limits(np.concatenate(boris_xs)))
    ax.set_ylim(*padded_limits(np.concatenate(boris_ys)))

    ax.set_xlabel(rf"${a}\ (c/\omega_{{pe}})$", fontsize=labelsize)
    ax.set_ylabel(rf"${b}\ (c/\omega_{{pe}})$", fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=labelsize, title=DRIFT_TITLES[args.drift],
              title_fontsize=ticksize)
    ax.grid(True)

    fig.tight_layout()
    out = args.out or os.path.join(args.dir, f"traces_{args.plane}.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
