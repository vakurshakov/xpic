#!/usr/bin/env python3
"""Насколько выводы зависят от выбора критерия и наблюдаемой."""
import math, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
TH = pickle.load(open("theory_drift_kinetic_ringdown_ex10.pkl", "rb"))
H = pickle.load(open("harm.pkl", "rb"))
T = TH["T"]; tt = TH["t"]; nth_c = TH["nhat"]["ions"]
gam = -TH["omega0"].imag; wr = TH["omega0"].real
RUNS = ["drift_kinetic_ringdown_ex11", "drift_kinetic_ringdown_ex12",
        "drift_kinetic_ringdown_ex13", "drift_kinetic_ringdown_ex10"]
COL = dict(zip(RUNS, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))
NM = {r: f"Np={H[r]['Np']}" for r in RUNS}

def fit_window(t, c, lo, hi):
    """(omega, Gamma) из наклона ln(комплексной гармоники) на [lo,hi]."""
    w = (t >= lo) & (t <= hi)
    if w.sum() < 8: return np.nan, np.nan
    lg = np.log(c[w]); lg = lg.real + 1j * np.unwrap(np.imag(lg))
    p = np.polyfit(t[w], lg, 1)[0]
    return -p.imag, -p.real

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.4))

print("Ошибка ПОДОГНАННОГО Gamma на окне [0, t_end], в % "
      "(эталон — та же подгонка точной теории на том же окне)")
hdr = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
print(f"{'t_end/T':>8s}" + "".join(f"{x:8.1f}" for x in hdr))
for r in RUNS:
    d = H[r]; t = d["t"]
    row = f"{NM[r]:>8s}"
    xs, ys, ys_w = [], [], []
    for x in np.arange(0.25, 1.35, 0.025):
        if t[-1] < x * T: break
        _, g = fit_window(t, d["H"][:, 1], 0.0, x * T)
        wo, _ = fit_window(t, d["H"][:, 1], 0.0, x * T)
        _, g0 = fit_window(tt, nth_c, 0.0, x * T)
        w0, _ = fit_window(tt, nth_c, 0.0, x * T)
        xs.append(x); ys.append(100 * (g / g0 - 1)); ys_w.append(100 * (wo / w0 - 1))
    ax[0].plot(xs, ys, color=COL[r], lw=2, label=NM[r])
    ax[1].plot(xs, ys_w, color=COL[r], lw=2, label=NM[r])
    for x in hdr:
        v = np.interp(x, xs, ys) if xs and x <= xs[-1] else np.nan
        row += f"{v:8.1f}" if not np.isnan(v) else f"{'-':>8s}"
    print(row)

# амплитуда огибающей — то, на что смотрят глазами
for r in RUNS:
    d = H[r]; t = d["t"]
    th = np.interp(t, tt, np.abs(nth_c))
    ax[2].plot(t / T, 100 * (np.abs(d["H"][:, 1]) / th - 1), color=COL[r],
               lw=2, label=NM[r])

for a, ti, yl in ((ax[0], "ошибка подогнанного $\\Gamma$ на окне $[0,t_{end}]$", "%"),
                  (ax[1], "ошибка подогнанной $\\omega_r$ на окне $[0,t_{end}]$", "%"),
                  (ax[2], "ошибка огибающей $|\\delta n_1|$ (мгновенная)", "%")):
    a.set_title(ti); a.set_ylabel(yl); a.grid(alpha=.3); a.legend(fontsize=8)
    a.axhline(0, color="k", lw=1)
    for lvl, c in ((2, "0.55"), (5, "0.35"), (10, "0.2")):
        a.axhline(lvl, color=c, ls=":", lw=1.2); a.axhline(-lvl, color=c, ls=":", lw=1.2)
ax[0].set_xlabel("$t_{end}/T$"); ax[1].set_xlabel("$t_{end}/T$")
ax[2].set_xlabel("$t/T$")
ax[0].set_ylim(-25, 40); ax[1].set_ylim(-8, 8); ax[2].set_ylim(-35, 15)
ax[2].set_xlim(0, 1.3)
fig.tight_layout(); fig.savefig("fig18_criteria.png", dpi=130)

print("\nt_break/T при РАЗНЫХ критериях (первое устойчивое превышение)")
print(f"{'Np':>6s} {'|Gam| 2%':>9s} {'5%':>7s} {'10%':>7s} | "
      f"{'огиб 2%':>8s} {'5%':>7s} {'10%':>7s} | {'компл 5%':>9s} | {'конец':>7s}")
for r in RUNS:
    d = H[r]; t = d["t"]
    th_c = np.interp(t, tt, nth_c.real) + 1j * np.interp(t, tt, nth_c.imag)
    env = np.abs(100 * (np.abs(d["H"][:, 1]) / np.abs(th_c) - 1))
    cpx = 100 * np.abs(d["H"][:, 1] - th_c) / np.abs(th_c)
    xs, gerr = [], []
    for x in np.arange(0.25, 1.35, 0.02):
        if t[-1] < x * T: break
        _, g = fit_window(t, d["H"][:, 1], 0, x * T)
        _, g0 = fit_window(tt, nth_c, 0, x * T)
        xs.append(x); gerr.append(abs(100 * (g / g0 - 1)))
    xs = np.array(xs); gerr = np.array(gerr)
    def first(arr, grid, lvl):
        m = arr > lvl
        return grid[np.argmax(m)] if m.any() else np.nan
    row = f"{H[r]['Np']:6d}"
    for lvl in (2, 5, 10): row += f"{first(gerr, xs, lvl):9.2f}"
    row += " |"
    for lvl in (2, 5, 10): row += f"{first(env, t/T, lvl):8.2f}"
    row += f" |{first(cpx, t/T, 5):9.2f} |{t[-1]/T:8.2f}"
    print(row)
