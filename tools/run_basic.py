#!/usr/bin/env python3

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from lib.common import *

from runner import *

scripts = (
    # "plot/fields.py",
    # "plot/particles.py",
    "plot/forces.py",
    # "collect/center.py",
)

# dir = select_subdirectory()

def run(script, dir=""):
    os.system(f"mpiexec -np 8 {os.path.dirname(__file__)}/{script} {dir}")

for sim, Np, tol in pool:
    for script in scripts:
        run(script, f"{sim}_dx{dx:.1f}_dt{dt:.1f}_Np{Np:d}_mi{mi_me:d}_tol{tol:.0e}")

run("plot/comp_center.py")