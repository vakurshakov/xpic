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
const.input_path = os.path.join(config["OutputDirectory"])
const.output_path = os.path.join(config["OutputDirectory"], "processed")

# Setting up the rest of particles information
mi_me = None   # me
T_i   = None   # mec^2
T_e   = None   # mec^2
mec2  = 511.0  # KeV

if "Particles" in config:
  for sort in config["Particles"]:
    if sort.get("sort_name") is None:
        continue

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
        arr, val = p.split(":")

        if not arr in config:
            return default

        for item in config[arr]:
            if val in item.values():
                return item

        return default

    for p in path.split("."):
        if ":" in p:
            result = get_from_array(p)
        elif d != default and p in d:
            result = d.get(p)
        else:
            return default
        d = result

    return result

# [mecwpe/e] Reference value of the magnetic field
const.B0 = get(config, "Presets:SetMagneticField.setter.reference", 0)

# [1/wpe] Particles injection rate
const.tau = get(config, "StepPresets:InjectParticles.tau", 0)

input_paths = [
    const.input_path
]

prefix = input_paths[0]

restarts = [  # in dts units
]

# Some utilities that would be used in plotting
def find_diag(name: str):
    names = name.split(".")

    def _test_region(diag, pos=None):
        type = get(diag, "region.type")
        return \
            type is None or \
            type == "3D" or \
            (type == "2D" and get(diag, "region.plane") == names[pos])

    for diag in get(config, "Diagnostics"):
        if get(diag, "diagnostic") != names[0]:
            continue

        if get(diag, "field") == names[1] and \
            _test_region(diag, 2):
            return diag

        if get(diag, "particles") == names[1] and \
            get(diag, "moment") == names[2] and \
            _test_region(diag, 3):
            return diag
    return None

def get_prefix(t, restarts, prefixes):
    i = 0
    for restart in restarts:
        if (t > restart):
            i += 1
    return prefixes[i]

def get_diag_path(diag: dict):
    def get_suffix_2D(diag):
        plane = get(diag, "region.plane")
        pos = get(diag, "region.position")

        if pos is None:
            if plane == 'X':
                pos = const.Nx // 2
            if plane == 'Y':
                pos = const.Ny // 2
            if plane == 'Z':
                pos = const.Nz // 2

        return f"_plane{plane}_{pos:04d}"

    suffix = ""

    if get(diag, "field"):
        suffix += get(diag, "field")
    elif get(diag, "particles") and get(diag, "moment"):
        suffix += f"{get(diag, "particles")}/{get(diag, "moment")}"

    if get(diag, "region.type") == "2D":
        suffix += get_suffix_2D(diag)

    return suffix
