#!/usr/bin/env python3

import os
import json

config_dir = os.path.join(os.path.dirname(__file__), "../")
config_path = config_dir + "config.json"
config = None

with open(config_path, "r") as file:
    config = json.load(file)

# Common information in any case
output_path = os.path.join(config_dir, config["OutputDirectory"], "processed")

prefixes = [
    os.path.join(config_dir, config["OutputDirectory"])
]

prefix = prefixes[0]

restarts = [  # in dts units
]

# Setting up geometry information
geometry = config["Geometry"]
dx = geometry["dx"]  # c/wpe
dy = geometry["dy"]  # c/wpe
dz = geometry["dz"]  # c/wpe
dt = geometry["dt"]  # 1/wpe
Nx = round(geometry["x"] / dx)  # cells
Ny = round(geometry["y"] / dy)  # cells
Nz = round(geometry["z"] / dz)  # cells

time = geometry["t"]     # 1/wpe, (number of files) * dts
Nt   = round(time / dt)  # cells

dts  = geometry["diagnose_period"]  # 1/wpe

# [{Plane, (cells, cells)}] Common shapes of the data stored in files 
data_shape = {
    'X': (Ny, Nz),
    'Y': (Nx, Nz),
    'Z': (Nx, Ny),
}

# [c/wpe] Boundaries to all possible graphs in format (bx, ex, by, ey)
boundaries = {
    'X': (-0.5 * Ny * dy, +0.5 * Ny * dy, -0.5 * Nz * dz, +0.5 * Nz * dz),
    'Y': (-0.5 * Nx * dx, +0.5 * Nx * dx, -0.5 * Nz * dz, +0.5 * Nz * dz),
    'Z': (-0.5 * Nx * dx, +0.5 * Nx * dx, -0.5 * Ny * dy, +0.5 * Ny * dy),
}

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
B0 = get(config, "command:SetMagneticField.setter.reference", 0)

# [1/wpe] Particles injection rate 
tau = get(config, "command:InjectParticles.tau", 0)

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
