import os
import sys
import traceback
import numpy as np
from scipy.integrate import cumulative_trapezoid

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from lib.plot import *
from lib.mpi_utils import *
from lib.data_format import *
from lib.data_consistency import *

def agg(to_agg: np.ndarray, data: np.ndarray):
    return data + (to_agg if np.any(to_agg) else np.zeros_like(data))

def sliding_average(linear: np.ndarray, w:int = 5):
    return np.convolve(linear, np.ones(w) / w, mode='valid')

def time_average(func: Callable[[int], np.ndarray], t0: int, dt: int):
    result = np.zeros_like(func(t0))
    for t in range(t0 - dt, t0):
        result += func(t)
    return result / dt

def makedirs(dirname):
    if not os.path.exists(dirname) and rank == 0:
        os.makedirs(dirname, exist_ok=True)

def dump(prefix, name, units, t, data):
    makedirs(prefix)
    with open(f"{prefix}/{name}_{t}.txt", "w") as f:
        f.write(f"Grid {name} [{units}]\n")
        for i, d in enumerate(data):
            f.write(f"{i} {d}\n")

def select_subdirectory():
    dir = None
    if rank == 0:
        res_path = "results"

        with os.scandir(res_path) as entries:
            dirs = sorted([e.name for e in entries if e.is_dir()])

        while True:
            print(f"List of {res_path} subdirectories: ")

            for i, dir in enumerate(dirs):
                print(f"{i}) {dir}")

            try:
                print(f"Select {res_path} subdirectory: ")
                i = int(input())
                dir = dirs[i]
                break
            except:
                print(traceback.print_exc(), end="\n\n")

    dir = comm.bcast(dir, root=0)
    return dir