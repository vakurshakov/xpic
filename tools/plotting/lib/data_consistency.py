import os
import sys
import numpy as np

from lib.data_format import *

# Timestep data consistency utils

def is_correct_timestep(t: int, view: FieldView) -> bool:
    test_filepath = view.path(t)
    file_byte_size = 4 * np.prod(view.region.size)
    return \
        os.path.isfile(test_filepath) and \
        os.path.getsize(test_filepath) == file_byte_size

def check_consistency(tmin: int, tmax: int, path: Callable[[int], str], size: np.ndarray):
    view = FieldView()
    view.path = path
    view.region.size = size

    def format_range(t1, t2):
        return f"({t1}, {t2}) [dts]"

    for t in range(tmin, tmax):
        if not is_correct_timestep(t, view):
            print(f"Data is inconsistent. Valid data range is: {format_range(tmin, t)}.")
            return
    print(f"Data range {format_range(tmin, tmax)} is consistent.")
    return

def timestep_should_be_processed(t: int, filename: str, view: FieldView, skip_processed = True):
    msg = f"{'/'.join(filename.split('/')[-2:])} {t} [dts]"
    if not is_correct_timestep(t, view):
        print(msg, "Data is incorrect, skipping")
        return False
    if skip_processed and os.path.exists(filename):
        print(msg, "Timestep was already processed, skipping.")
        return False
    print("Processing", msg)
    return True

def find_correct_timestep(t: int, t_range: np.ndarray, view: FieldView) -> int:
    for t_c in range(t_range[0], t + 1)[::-1]:
        if not is_correct_timestep(t_c, view):
            print(f"Warning! Timestep {t_c} is incorrect, first previous correct step will be used.")
            continue
        print(f"{t_c:4d} [dts]")
        return t_c
    return t_c