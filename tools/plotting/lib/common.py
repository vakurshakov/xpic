import os
import sys
import numpy as np
from scipy.integrate import cumulative_trapezoid

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from lib.plot import *
from lib.mpi_utils import *
from lib.data_format import *

makedirs(f"{output_path}/video")

def agg(to_agg: np.ndarray, data: np.ndarray):
    return data + (to_agg if np.any(to_agg) else np.zeros_like(data))

def sliding_average(linear: np.ndarray, w:int = 5):
    return np.convolve(linear, np.ones(w) / w, mode='valid')

def time_average(func: Callable[[int], np.ndarray], t0: int, dt: int):
    result = np.zeros_like(func(t0))
    for t in range(t0 - dt, t0):
        result += func(t)
    return result / dt
