#!/usr/bin/env python3

import os

plots = (
    "fields",
    "particles",
    "currents",
)

for plot in plots:
    os.system(f"mpiexec -np 4 {os.path.dirname(__file__)}/plots/{plot}.py")