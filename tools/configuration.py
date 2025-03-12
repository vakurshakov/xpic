#!/usr/bin/env python3

import json

config_path = "../config.json"
config = None

with open(config_path, "r") as file:
    config = json.load(file)
    file.close()

# Common information in any case
params_path = config["OutputDirectory"]

prefixes =  [
  config["OutputDirectory"]
]

restart_timesteps = [  # in dts units
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
  for p in path.split("."):
    if p in d:
      result = d.get(p)
      d = result
    else:
      return None
  return result

# Buffer cells used to offset diagnostic [cells]
buff = None

# Reference value of the magnetic field [mecwpe/e]
B0 = get(config, "Presets.SetMagneticField.setter.reference")

# Particles injection rate [1 / wpe]
tau = get(config, "StepPresets.InjectParticles.tau")
