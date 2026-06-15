#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from runner import pool

for Np, tol in pool:
    script = f"{os.path.dirname(__file__)}/collect/center.py"
    directory = f"eccapfim_dx3.0_dt1.5_Np{Np:d}_Ny4_tol{tol:.0e}"
    os.system(f"mpiexec -np 8 {script} {directory}")
