#!/usr/bin/env python3

import os
import json

xpic_dir = os.path.join(os.path.dirname(__file__), "../")

config_path = os.path.join(xpic_dir, "config.json")
config = None

with open(config_path, "r") as file:
    config = json.load(file)
    file.close()


# Common information in any case
params_path = os.path.join(xpic_dir, config["OutputDirectory"])

prefixes =  [
    os.path.join(xpic_dir, config["OutputDirectory"])
]

restarts = [  # in dts units
]

prefix = prefixes[0]


# Setting up geometry information
geometry = config["Geometry"]
dx = geometry["dx"]  # c / wpe
dy = geometry["dy"]  # c / wpe
dz = geometry["dz"]  # c / wpe
dt = geometry["dt"]  # 1 / wpe
Nx = round(geometry["x"] / dx)  # cells
Ny = round(geometry["y"] / dy)  # cells
Nz = round(geometry["z"] / dz)  # cells

time = geometry["t"]     # 1 / wpe, (number of files) * dts
Nt   = round(time / dt)  # cells

dts  = geometry["diagnose_period"]  # 1 / wpe


# Setting up particles information
sorts = []
mi_me = None   # me
T_i   = None   # mec^2
T_e   = None   # mec^2
mec2  = 511.0  # KeV

if "Particles" in config:
  for sort in config["Particles"]:
    name = sort["sort_name"]
    sorts.append(name)

    if name == "ions":
      mi_me = sort["m"]
      T_i = sort["T"] / mec2

    elif name == "electrons":
      T_e = sort["T"] / mec2


# Setting up information from commands (presets)
def get(d: dict, path: str):
    result = None

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

        return None

    for p in path.split("."):
        if ":" in p:
            result = get_from_array(p)
        elif p in d:
            result = d.get(p)
        else:
            return None
        d = result

    return result

# Buffer cells used to offset diagnostic [cells]
buff = 0

# Reference value of the magnetic field [mecwpe/e]
B0 = get(config, "command:SetMagneticField.setter.reference")

# Particles injection rate [1 / wpe]
tau = get(config, "command:InjectParticles.tau")


# Some utilities that would be used in `xplot`
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
