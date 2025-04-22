#!/usr/bin/env python3

import os
import json

from plotting.lib.constants import *

config_dir = os.path.join(os.path.dirname(__file__), "../")
config_path = os.path.join(config_dir, "config.json")

with open(config_path, "r") as file:
    config = json.load(file)

const = Constants()
const.init_from_config(config)
const.init_dependent()
const.input_path = os.path.join(config_dir, config["OutputDirectory"])
const.output_path = os.path.join(config_dir, config["OutputDirectory"], "processed")

# Setting up the rest of particles information
mi_me = None   # me
T_i   = None   # mec^2
T_e   = None   # mec^2
mec2  = 511.0  # KeV

if "Particles" in config:
  for sort in config["Particles"]:
    name = sort["sort_name"]

    if name == "ions":
      mi_me = sort["m"]
      T_i = sort["T"] / mec2

    elif name == "electrons":
      T_e = sort["T"] / mec2

# Setting up information from commands (presets)
def get(d: dict, path: str, default = None):
    result = default

    # Can only read top-level arrays
    def get_from_array(p: str):
        id, val = p.split(":")

        arrays = [
            "Particles",
            "Presets",
            "StepPresets",
            "Diagnostics",
        ]

        for arr in arrays:
            if not arr in config:
                continue

            for item in config[arr]:
                if id in item and val == item[id]:
                    return item

        return default

    for p in path.split("."):
        if ":" in p:
            result = get_from_array(p)
        elif d != None and p in d:
            result = d.get(p)
        else:
            return default
        d = result

    return result

# [mecwpe/e] Reference value of the magnetic field 
const.B0 = get(config, "command:SetMagneticField.setter.reference", 0)

# [1/wpe] Particles injection rate 
const.tau = get(config, "command:InjectParticles.tau", 0)

# Some utilities that would be used in plotting
def find_diag(name: str):
    names = name.split(".")

    for diag in get(config, "Diagnostics"):
        if get(diag, "diagnostic") != names[0]:
            continue

        if get(diag, "field") == names[1]:
            if get(diag, "region.type") == "3D":
                return diag
            elif get(diag, "region.type") == "2D" and get(diag, "region.plane") == names[2]:
                return diag

        if get(diag, "moment") == names[1] and get(diag, "particles") == names[2]:
            return diag
    return None

input_paths = [
    const.input_path
]

prefix = input_paths[0]

restarts = [  # in dts units
]

def get_prefix(t, restarts, prefixes):
    i = 0
    for restart in restarts:
        if (t > restart):
            i += 1
    return prefixes[i]

def get_diag_path(diag: dict | None, prefix: str = None):
    if not diag:
        return None

    def get_suffix_2D(diag):
        plane = get(diag, "region.plane")
        pos = get(diag, "region.position")
        fill = -1

        if plane == "X": pos = round(pos / const.dx); fill = const.Nx
        elif plane == "Y": pos = round(pos / const.dy); fill = const.Ny
        elif plane == "Z": pos = round(pos / const.dz); fill = const.Nz

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

    p = get_prefix(t, restarts, input_paths) if prefix == None else prefix
    return f"{p}/{suffix}/"
