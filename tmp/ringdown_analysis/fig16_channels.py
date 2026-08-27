#!/usr/bin/env python3
"""Почему нелинейный пол не мешает каналу m=1."""
import math, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
TH = pickle.load(open("theory_drift_kinetic_ringdown_ex10.pkl", "rb"))
H = pickle.load(open("harm.pkl", "rb"))
T = TH["T"]; tt = TH["t"]; nth_c = TH["nhat"]["ions"]; gam = -TH["omega0"].imag
RUNS = ["drift_kinetic_ringdown_ex11", "drift_kinetic_ringdown_ex12",
        "drift_kinetic_ringdown_ex13", "drift_kinetic_ringdown_ex10"]
COL = dict(zip(RUNS, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))
S = {}
for r in RUNS:
    d = H[r]; t = d["t"]
    th = np.interp(t, tt, nth_c.real) + 1j * np.interp(t, tt, nth_c.imag)
    S[r] = dict(t=t, N=d["Np"] * 288, Np=d["Np"], thabs=np.abs(th),
                res=np.abs(d["H"][:, 1] - th), n2=np.abs(d["H"][:, 2]),
                nz=np.sqrt(np.mean(np.abs(d["H"][:, 3:16])**2, axis=1)))

def med(r, key, x, half=0.05):
    s = S[r]
    if s["t"][-1] < x * T: return None
    w = np.abs(s["t"] - x * T) < half * T
    return np.median(s[key][w])

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.4))
xs = np.arange(0.15, 0.80, 0.05)
curves = {"res": [], "nz": [], "n2": []}
for x in xs:
    ok = [r for r in RUNS if med(r, "res", x) is not None
          and med(r, "res", x) < 0.25 * med(r, "thabs", x)]
    for key in curves:
        if len(ok) < 3:
            curves[key].append(np.nan); continue
        lgN = [math.log(S[r]["N"]) for r in ok]
        lgy = [math.log(med(r, key, x)) for r in ok]
        curves[key].append(-np.polyfit(lgN, lgy, 1)[0])
ax[0].plot(xs, curves["res"], "o-", color="tab:red", lw=2,
           label=r"$m=1$ — там, где сигнал")
ax[0].plot(xs, curves["nz"], "s-", color="tab:blue", lw=2, label=r"$m\geq3$")
ax[0].plot(xs, curves["n2"], "^-", color="tab:green", lw=2, label=r"$m=2$")
ax[0].axhline(0.5, color="k", ls="--", lw=1.4, label=r"чистый шум маркеров $N^{-1/2}$")
ax[0].axhline(0.0, color="k", ls=":", lw=1.4, label="чистая нелинейность (нет зависимости от $N$)")
ax[0].set_ylim(-0.5, 1.6); ax[0].set_ylabel("показатель $p$ в $\\propto N^{-p}$")
ax[0].set_title("(а) канал $m=1$ НИКОГДА не выходит на $p=0$:\n"
                "нелинейный пол сидит в $m=2$, не в $m=1$")

for r in RUNS:
    s = S[r]
    ax[1].semilogy(s["t"] / T, s["res"], color=COL[r], lw=1.5,
                   label=f"Np={s['Np']}")
ax[1].semilogy(tt / T, 0.05 * np.abs(nth_c), "k:", lw=1.8, label="5% от теории")
a1 = np.interp(tt, tt, np.abs(nth_c))
n2r = np.interp(tt, S["drift_kinetic_ringdown_ex10"]["t"],
                S["drift_kinetic_ringdown_ex10"]["n2"])
ax[1].semilogy(tt / T, n2r * a1, "k--", lw=1.8,
               label=r"оценка кубической утечки $|n_2||n_1|$")
ax[1].set_ylim(1e-5, 1e-2); ax[1].set_xlim(0, 1.2)
ax[1].set_ylabel(r"$|\delta n_1^{model}-\delta n_1^{theory}|$")
ax[1].set_title("(б) невязка $m=1$ — это шум маркеров;\n"
                "нелинейная утечка в 20 раз ниже")

for L, mk in zip((1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3), "os^Dv"):
    ts, ns = [], []
    for r in RUNS:
        s = S[r]; m = s["res"] > L
        if m.any():
            ts.append(s["t"][np.argmax(m)] / T); ns.append(s["Np"])
    ax[2].semilogx(ns, ts, mk + "-", base=2, label=f"уровень {L:.0e}")
ax[2].set_xlabel("$N_p$"); ax[2].set_ylabel("$t$, когда невязка достигает уровня, $/T$")
ax[2].set_title("(в) отдача растёт с $N_p$: наклон 0.06 T -> 0.18 T\n"
                "за удвоение (асимптота $\\ln2/2\\gamma$ = 0.46 T)")
for a in ax[:2]:
    a.set_xlabel("$t/T$")
for a in ax:
    a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig("fig16_channels.png", dpi=130)

k = TH["k"]
print("порог захвата электронов (нелинейность, которая ДЕЙСТВИТЕЛЬНО бьёт по m=1):")
for nm, E in (("ex10, dn=0.03", 5.637536721003667e-05),
              ("ex16, dn=0.01", 1.8791789070012225e-05)):
    wb = math.sqrt(E * k)
    print(f"  {nm}:  w_b/w = {wb/TH['omega0'].real:.3f}, "
          f"w_b/Gamma = {wb/gam:.2f}, tau_b = {2*math.pi/wb/T:.2f} T")
