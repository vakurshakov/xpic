#!/usr/bin/env python3
"""Чем именно ограничена невязка в канале m=1 (там, где живёт сигнал)?"""
import math, pickle
import numpy as np
TH = pickle.load(open("theory_drift_kinetic_ringdown_ex10.pkl", "rb"))
H = pickle.load(open("harm.pkl", "rb"))
T = TH["T"]; tt = TH["t"]; nth_c = TH["nhat"]["ions"]; gam = -TH["omega0"].imag
RUNS = ["drift_kinetic_ringdown_ex11", "drift_kinetic_ringdown_ex12",
        "drift_kinetic_ringdown_ex13", "drift_kinetic_ringdown_ex10"]
NAME = {r: f"Np={H[r]['Np']}" for r in RUNS}

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

print("1. Показатель p в  y ~ N^-p, только там, где ВСЕ включённые прогоны")
print("   ещё линейны (res < 0.25 сигнала).  Идеальный маркерный шум: p = 0.5")
print(f"{'t/T':>5s} {'p(m=1)':>8s} {'p(m>=3)':>9s} {'p(m=2)':>8s} {'вкл':>4s}  "
      f"|res| m=1: 1024 ... 8192")
for x in np.arange(0.15, 0.80, 0.05):
    ok = [r for r in RUNS if med(r, "res", x) is not None
          and med(r, "res", x) < 0.25 * med(r, "thabs", x)]
    if len(ok) < 3:
        print(f"{x:5.2f}   (линейны только {len(ok)} прогона)"); continue
    row = []
    for key in ("res", "nz", "n2"):
        lgN = [math.log(S[r]["N"]) for r in ok]
        lgy = [math.log(med(r, key, x)) for r in ok]
        row.append(-np.polyfit(lgN, lgy, 1)[0])
    vals = "  ".join(f"{med(r,'res',x):.2e}" for r in ok)
    print(f"{x:5.2f} {row[0]:8.2f} {row[1]:9.2f} {row[2]:8.2f} {len(ok):4d}  {vals}")

print("\n2. Горизонтальный сдвиг кривой |res| канала m=1")
print(f"{'уровень':>9s}" + "".join(f"{NAME[r]:>11s}" for r in RUNS)
      + "   T за удвоение Np")
for L in (1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3):
    ts = []
    for r in RUNS:
        s = S[r]; m = s["res"] > L
        ts.append(s["t"][np.argmax(m)] / T if m.any() else np.nan)
    good = [(math.log2(S[r]["Np"]), v) for r, v in zip(RUNS, ts) if not np.isnan(v)]
    sl = np.polyfit([g[0] for g in good], [g[1] for g in good], 1)[0] \
        if len(good) >= 3 else np.nan
    print(f"{L:9.1e}" + "".join(f"{v:11.3f}" for v in ts) + f"{sl:14.3f}")

print("\n3. Локальная скорость роста |res| канала m=1, в единицах gamma")
print(f"{'t/T':>5s}" + "".join(f"{NAME[r]:>11s}" for r in RUNS))
for x in np.arange(0.2, 1.15, 0.1):
    line = f"{x:5.2f}"
    for r in RUNS:
        s = S[r]
        if s["t"][-1] < (x + 0.12) * T: line += f"{'-':>11s}"; continue
        w = np.abs(s["t"] - x * T) < 0.12 * T
        line += f"{np.polyfit(s['t'][w], np.log(s['res'][w]), 1)[0]/gam:11.1f}"
    print(line)

print("\n4. Сдвиг излома = (ln2/2)/(Gamma_noise + gamma)")
for r in RUNS:
    s = S[r]
    m = s["res"] / s["thabs"] > 0.05
    if not m.any(): continue
    tb = s["t"][np.argmax(m)]
    w = np.abs(s["t"] - tb) < 0.12 * T
    rate = np.polyfit(s["t"][w], np.log(s["res"][w]), 1)[0]
    print(f"{NAME[r]:10s} t_break={tb/T:5.2f}T  Gamma_шума={rate/gam:5.1f} gamma"
          f"  -> {math.log(2)/2/(rate+gam)/T:5.2f} T за удвоение")
print(f"{'предел':10s} при Gamma_шума -> 0 (шум на полке): "
      f"{math.log(2)/2/gam/T:.2f} T за удвоение")

print("\n5. Где нелинейность вообще может попасть в m=1?")
print("   квадратичная связь m=1 x m=1 -> m=0 и m=2, в m=1 не даёт ничего.")
print("   в m=1 попадает только кубика m=2 x m=1, порядок ~ |n1|^3 / |n1| :")
for x in (0.3, 0.5, 0.7, 0.9):
    a1 = np.interp(x * T, tt, np.abs(nth_c))
    n2 = med("drift_kinetic_ringdown_ex10", "n2", x)
    if n2 is None: continue
    print(f"   t={x:.1f}T: |n1|={a1:.2e}  |n2|={n2:.2e}  оценка "
          f"|n2|*|n1| = {n2*a1:.2e}   маркерный шум m=1 при 8192 = "
          f"{med('drift_kinetic_ringdown_ex10','res',x):.2e}")
