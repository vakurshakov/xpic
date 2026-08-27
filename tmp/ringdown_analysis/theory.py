#!/usr/bin/env python3
"""Точное линейное решение Власова-Пуассона от реализованной 5-D ФР."""
import pickle, sys
import numpy as np
import rd_data as R
sys.path.insert(0, R.TESTS)
import ion_sound as IS

run = "drift_kinetic_ringdown_ex10"
ctx = IS.prepare_theory(run)
G = R.geom(run)
t, nhat, ehat = IS.solve_vlasov_poisson(
    ctx["species"], ctx["cn_hat"], ctx["u_hat"], ctx["k"],
    G["Nt"] * G["dt"], n_record=1200, omega0=ctx["omega0"], E0=ctx["E0"],
    initial_distribution=ctx["initial_distribution"])
pickle.dump(dict(t=t + ctx["t0"], T=ctx["T_wave"], omega0=ctx["omega0"],
                 k=ctx["k"], ehat=ehat, nhat=dict(nhat), E0=ctx["E0"],
                 vT={s.name: s.vT for s in ctx["species"]},
                 q={s.name: s.q for s in ctx["species"]},
                 m={s.name: s.m for s in ctx["species"]}),
            open("theory_drift_kinetic_ringdown_ex10.pkl", "wb"))
print("omega0", ctx["omega0"], "T", ctx["T_wave"])

harm = {}
for r in R.RUNS:
    tt, Hh, Gg = R.harmonics(r, "ions/density")
    te, He, _ = R.harmonics(r, "electrons/density")
    harm[r] = dict(t=tt, H=Hh, He=He, Np=Gg["Np"], Nz=Gg["Nz"],
                   Nx=Gg["Nx"], Ny=Gg["Ny"])
pickle.dump(harm, open("harm.pkl", "wb"))
print("harm.pkl:", {r: len(harm[r]["t"]) for r in harm})
