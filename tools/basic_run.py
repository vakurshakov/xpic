#!/usr/bin/env python3

import os
from lib.common import *

plots = (
    "fields",
    "particles",
    "forces",
)

for plot in plots:
    os.system(f"mpiexec -np 4 {os.path.dirname(__file__)}/plot/{plot}.py {select_subdirectory().split("_")[0]}")
