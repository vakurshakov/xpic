#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--name", default="config.json", help="Name of the output configuration file")
parser.add_argument("-s", "--simulation", required=True, help="Internal simulation algorithm")
parser.add_argument("-o", "--output", default="output", help="Path to the output directory")

parser.add_argument("-dx", type=float, required=True, help="Grid spacing along x-axis")
parser.add_argument("-dy", type=float, required=False, help="Grid spacing along y-axis, equal to dx if not passed")
parser.add_argument("-dz", type=float, required=False, help="Grid spacing along z-axis, equal to dx if not passed")

parser.add_argument("-dt", type=float, help="Timestep size")
parser.add_argument("-CFL", type=float, help="Courant–Friedrichs–Lewy condition fraction (i.e. dt~c*v_the/dx); used to setup the time step")

args = parser.parse_args()

config_path = os.path.join(os.path.dirname(__file__), f"../{args.name}")

T_e = 0.1
v_the = np.sqrt(T_e / 511)

args.CFL = 0.1

Nx = 25
Ny = Nx
Nz = Nx
Nt = 100

dx = args.dx
dy = args.dy if args.dy else dx
dz = args.dz if args.dz else dx
dt = args.CFL * v_the if args.CFL else args.dt

config = {
    "Simulation": args.simulation,
    "OutputDirectory": args.output,
    "Geometry": {
        "x": f"{Nx:d} [dx]",
        "y": f"{Ny:d} [dy]",
        "z": f"{Nz:d} [dz]",
        "t": f"{Nt:d} [dt]",
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "dt": dt,
        "diagnose_period": 10 * dt,
        "da_boundary_x": "DM_BOUNDARY_PERIODIC",
        "da_boundary_y": "DM_BOUNDARY_PERIODIC",
        "da_boundary_z": "DM_BOUNDARY_PERIODIC"
    },
    "Particles": [
        {
            "sort_name": "electrons",
            "Np": 100,
            "n": 1.0,
            "q": -1.0,
            "m": 1.0,
            "T": 0.1
        }
    ],
    "Presets": [
        {
            "command": "SetParticles",
            "particles": "electrons",
            "coordinate": { "name": "CoordinateInBox" },
            "momentum": { "name": "MaxwellianMomentum", "tov": True }
        }
    ],
    "Diagnostics": [
        {
            "diagnostic": "FieldView",
            "field": "E"
        },
        {
            "diagnostic": "FieldView",
            "field": "B"
        },
        {
            "diagnostic": "DistributionMoment",
            "particles": "electrons",
            "moment": "density"
        }
    ]
}


with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
