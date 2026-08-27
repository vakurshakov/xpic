#!/usr/bin/env python3
"""Одна картинка: почему число маркеров влияет, но так неровно."""
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

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# ---- левая панель: реальные данные ----
a = ax[0]
sig = 0.05 * np.abs(nth_c)
a.semilogy(tt / T, sig, "k-", lw=3, label="ПОРОГ: 5% от сигнала\n(сигнал падает медленно, как $e^{-\\gamma t}$)")
for r in RUNS:
    d = H[r]; t = d["t"]
    th = np.interp(t, tt, nth_c.real) + 1j * np.interp(t, tt, nth_c.imag)
    res = np.abs(d["H"][:, 1] - th)
    a.semilogy(t / T, res, color=COL[r], lw=2, label=f"ШУМ при Np={d['Np']}")
    m = res > np.interp(t, tt, sig)
    if m.any():
        i = np.argmax(m)
        a.plot(t[i] / T, res[i], "o", color=COL[r], ms=13, mec="k", mew=1.5, zorder=5)
        a.annotate(f"{t[i]/T:.2f}T", (t[i] / T, res[i]), textcoords="offset points",
                   xytext=(6, -16), fontsize=10, color=COL[r], weight="bold")
a.set_xlim(0, 1.2); a.set_ylim(2e-5, 4e-3)
a.set_xlabel("$t/T$"); a.set_ylabel("амплитуда (лог. шкала)")
a.set_title("Излом = пересечение шума с порогом.\n"
            "Кружки — где именно ломается каждый прогон")
a.legend(fontsize=9, loc="lower right")
a.grid(alpha=.3, which="both")

# ---- правая панель: та же геометрия, схематично ----
b = ax[1]
x = np.linspace(0, 1.2, 400)
thr = np.log(1.5e-3) - gam * T * x            # порог: медленный спуск
b.plot(x, thr, "k-", lw=3, label="порог (медленно вниз)")
# шумовая кривая: крутой рост -> полка
def noise(x, shift):
    return np.log(2.2e-3) - shift - 3.2 * np.exp(-4.5 * x)*3.2
for shift, c, lab in ((0.0, "tab:blue", "мало маркеров"),
                      (3 * 0.347, "tab:red", "в 8 раз больше маркеров")):
    y = noise(x, shift)
    b.plot(x, y, color=c, lw=2.5, label=lab)
    d = y - thr
    i = np.argmax(d > 0)
    b.plot(x[i], y[i], "o", color=c, ms=13, mec="k", mew=1.5, zorder=5)
b.annotate("", xy=(0.0, np.log(2.2e-3) - 3*0.347 - 0.05),
           xytext=(0.0, np.log(2.2e-3) - 0.05),
           arrowprops=dict(arrowstyle="<->", lw=2, color="0.3"))
b.text(0.02, np.log(2.2e-3) - 1.7,
       "8x маркеров опускает\nшум всего в $\\sqrt{8}$ = 2.8 раза\n"
       "(это МАЛО по вертикали)", fontsize=10, color="0.25")
b.annotate("на КРУТОМ участке\nсдвиг по времени крошечный",
           xy=(0.16, noise(np.array([0.16]), 0)[0]), xytext=(0.30, -8.2),
           fontsize=10, color="tab:blue",
           arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5))
b.annotate("на ПОЛОГОМ участке\nтот же сдвиг даёт много времени",
           xy=(0.85, noise(np.array([0.85]), 3*0.347)[0]), xytext=(0.42, -6.35),
           fontsize=10, color="tab:red",
           arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.5))
b.set_xlim(0, 1.2); b.set_ylim(-9.2, -5.8)
b.set_yticks([]); b.set_xlabel("$t/T$"); b.set_ylabel("амплитуда (лог. шкала)")
b.set_title("Почему отдача такая неровная:\nвсё решает НАКЛОН шумовой кривой в точке встречи")
b.legend(fontsize=9, loc="lower right"); b.grid(alpha=.3, axis="x")

fig.tight_layout(); fig.savefig("fig17_simple.png", dpi=130)

print("сдвиг излома за удвоение маркеров = 0.35 / наклон_шума :")
for rate, où in ((15.0, "ранний, крутой участок"), (4.5, "при Np=8192"),
                 (1.0, "почти полка"), (0.001, "полка")):
    print(f"  наклон {rate:5.1f} gamma  ({où:22s}) -> "
          f"{math.log(2)/2/((rate+1)*gam)/T:.2f} T")
