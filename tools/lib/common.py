import os
import sys
import traceback
import numpy as np
from scipy.integrate import cumulative_trapezoid

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from lib.constants import *

from lib.plot import *
from lib.mpi_utils import *
from lib.data_format import *
from lib.data_consistency import *

def read(file_t, c, cd):
    data_shape = [const.Ny, const.Nx, cd]
    with open(file_t, "rb") as file:
        raw = np.fromfile(file, dtype=np.float32, count=np.prod(data_shape), offset=0)
        raw = np.reshape(raw, data_shape)
    return raw[:,:,c]

def read_E(c): return lambda t: read(f"{const.in_dir}/E_ZAvg/{str(t // const.Ndts).zfill(4)}", c, 3)
def read_B(c): return lambda t: read(f"{const.in_dir}/B_ZAvg/{str(t // const.Ndts).zfill(4)}", c, 3)
def read_n(s): return lambda t: read(f"{const.in_dir}/{s}/rho_ZAvg/{str(t // const.Ndts).zfill(4)}", 0, 1) * (1 if s == "ions" else -1)
def read_J(s, c): return lambda t: read(f"{const.in_dir}/{s}/J_ZAvg/{str(t // const.Ndts).zfill(4)}", c, 3)
def read_P(s, c): return lambda t: read(f"{const.in_dir}/{s}/momentum_Z2/{str(t // const.Ndts).zfill(4)}", c, 6)

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
            except Exception:
                print(traceback.print_exc(), end="\n\n")

    dir = comm.bcast(dir, root=0)
    return dir