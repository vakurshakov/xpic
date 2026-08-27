#!/usr/bin/env python3
"""Загрузчик скана ringdown_ex10..ex14 по числу маркеров."""
import json, math, os, sys
import numpy as np

TESTS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", "tests", "drift_kinetic"))
OUT = os.path.join(TESTS, "output")
sys.path.insert(0, TESTS)

RUNS = ["drift_kinetic_ringdown_ex11", "drift_kinetic_ringdown_ex12",
        "drift_kinetic_ringdown_ex13", "drift_kinetic_ringdown_ex10",
        "drift_kinetic_ringdown_ex14"]


def cfg(run):
    with open(os.path.join(OUT, run, "config.json")) as f:
        return json.load(f)


def geom(run):
    c = cfg(run); g = c["Geometry"]
    return dict(Nx=round(g["x"] / g["dx"]), Ny=round(g["y"] / g["dy"]),
                Nz=round(g["z"] / g["dz"]), dz=g["dz"], Lz=g["z"], dt=g["dt"],
                dts=g["diagnose_period"], Nt=round(g["t"] / g["dt"]),
                Np=c["Particles"][0]["Np"], config=c)


def frames(run, sub, ncomp=1):
    G = geom(run)
    n = G["Nx"] * G["Ny"] * G["Nz"] * ncomp
    path = os.path.join(OUT, run, sub)
    idx, arrs = [], []
    for nm in sorted(x for x in os.listdir(path) if x.isdigit()):
        p = os.path.join(path, nm)
        if os.path.getsize(p) != n * 4:
            continue
        a = np.fromfile(p, dtype=np.float32, count=n)
        arrs.append(a.reshape(G["Nz"], G["Ny"], G["Nx"], ncomp) if ncomp > 1
                    else a.reshape(G["Nz"], G["Ny"], G["Nx"]))
        idx.append(int(nm))
    return np.array(idx), np.array(arrs), G


def harmonics(run, sub, n0=1.0, mmax=15):
    idx, arrs, G = frames(run, sub)
    prof = arrs.mean(axis=(2, 3)) / n0 - 1.0
    z = (np.arange(G["Nz"]) + 0.5) * G["dz"]
    k0 = 2.0 * math.pi / G["Lz"]
    Hh = np.empty((prof.shape[0], mmax + 1), dtype=complex)
    for m in range(mmax + 1):
        Hh[:, m] = 2.0 * np.mean(prof * np.exp(-1j * m * k0 * z), axis=1)
    return idx * G["dts"], Hh, G


def temporal(run, name):
    p = os.path.join(OUT, run, "temporal", name)
    with open(p) as f:
        head = f.readline().split()
    return head, np.loadtxt(p, skiprows=1)
