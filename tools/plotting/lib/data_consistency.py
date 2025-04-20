import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from configuration import *

# Timestep data consistency utils

def get_test_file(t: int) -> str | None:
    t_str = str(t).zfill(len(str(Nt)))

    for _, dirs, _ in os.walk(prefix):
        for d in dirs:
            if "E_PlaneZ" in d or "B_PlaneZ" in d:
                return (f"{prefix}/{d}/{t_str}", 4 * (3 * Nx * Ny))
            # elif d in sorts:
    return None

def is_correct_timestep(t: int) -> bool:
    test_filepath, file_byte_size = get_test_file(t)
    return \
        os.path.isfile(test_filepath) and \
        os.path.getsize(test_filepath) == file_byte_size

def check_consistency(tmin, tmax):
    def format_range(t1, t2):
        return f"({t1}, {t2}) [dts]"

    for t in range(tmin, tmax):
        if not is_correct_timestep(t):
            print(f"Data is inconsistent. Valid data range is: {format_range(tmin, t)}.")
            return
    print(f"Data range {format_range(tmin, tmax)} is consistent.")
    return

def timestep_should_be_processed(t: int, filename: str, skip_processed = True):
    msg = f"{'/'.join(filename.split('/')[-2:])} {t} [dts]"
    if not is_correct_timestep(t):
        print(msg, "Data is incorrect, skipping")
        return False
    if skip_processed and os.path.exists(filename):
        print(msg, "Timestep was already processed, skipping.")
        return False
    print("Processing", msg)
    return True

def find_correct_timestep(t: int, t_range: np.ndarray) -> int:
    for t_c in range(t_range[0], t + 1)[::-1]:
        if not is_correct_timestep(t_c):
            print(f"Warning! Timestep {t_c} is incorrect, first previous correct step will be used.")
            continue
        print(f"{t_c:4d} [dts]")
        return t_c
    return t_c