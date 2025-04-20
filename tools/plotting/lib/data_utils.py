import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from lib.xy_rphi import *
from lib.file_utils import *
from lib.data_consistency import *
from configuration import *

COS, SIN, RMAP = init_COS_SIN_RMAP((0, data_shape['Z'][0], 0, data_shape['Z'][1]))

def get_prefix(t, restarts, prefixes):
    i = 0
    for restart in restarts:
        if (t > restart):
            i += 1
    return prefixes[i]

def get_formatted_time(t: int):
    return str(t).zfill(len(str(Nt)))

def get_diag_path(diag: dict | None, prefix: str = None):
    if not diag:
        return None

    def get_suffix_2D(diag):
        plane = get(diag, "region.plane")
        pos = get(diag, "region.position")
        fill = -1

        if plane == "X": pos = round(pos / dx); fill = Nx
        elif plane == "Y": pos = round(pos / dy); fill = Ny
        elif plane == "Z": pos = round(pos / dz); fill = Nz

        s = ""
        s += f"_Plane{plane}"
        s += f"_{str(pos).zfill(len(str(fill)))}"
        return s

    suffix = ""

    if get(diag, "field"):
        suffix += get(diag, "field")
    elif get(diag, "particles") and get(diag, "moment"):
        suffix += f"{get(diag, "particles")}/{get(diag, "moment")}"

    if get(diag, "region.type") == "2D":
        suffix += get_suffix_2D(diag)

    p = get_prefix(t, restarts, prefixes) if prefix == None else prefix
    return f"{p}/{suffix}/"
