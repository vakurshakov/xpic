import os
import sys
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

def format_time(t: int, Nt: int):
    return str(t).zfill(len(str(Nt)))

def makedirs(dirname):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if not os.path.exists(dirname) and rank == 0:
        os.makedirs(dirname, exist_ok=True)
    comm.barrier()

def dump(prefix, name, units, t, data):
    makedirs(prefix)
    with open(f"{prefix}/{name}_{t}.txt", "w") as f:
        f.write(f"Grid {name} [{units}]\n")
        for i, d in enumerate(data):
            f.write(f"{i} {d}\n")
