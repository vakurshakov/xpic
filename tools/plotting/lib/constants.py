#!/usr/bin/env python3

import os, sys

from dataclasses import dataclass, field
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from lib.xy_rphi import *

@dataclass
class Constants:
    input_path: str = None
    output_path: str = None

    # figure parameters
    nrows, ncols = (None,) * 2

    plot_init: int = None
    plot_time: int = None
    plot_offset: int = None

    # geometry
    dx, dy, dz, dt = (None,) * 4 # c/wpe, 1/wpe
    Nx, Ny, Nz, Nt = (None,) * 4 # cells

    diagnose_period: float = None # 1/wpe

    B0: float = 0
    sorts: list[str] = field(default_factory=list)

    # dependent variables
    data_shape: tuple = None # [{Plane, (cells, cells)}] Common shapes of the data stored in files
    boundaries: tuple = None # [c/wpe] Boundaries to all possible graphs in format (bx, ex, by, ey)
    cos, sin, rmap = (None,) * 3

    def init_from_config(self, config: dict[str, Any]):
        geometry: dict[str, Any] = config.get("Geometry")
        self.dx = geometry.get("dx")
        self.dy = geometry.get("dy")
        self.dz = geometry.get("dz")
        self.dt = geometry.get("dt")
        self.Nx = round(geometry.get("x") / self.dx)
        self.Ny = round(geometry.get("y") / self.dy)
        self.Nz = round(geometry.get("z") / self.dz)
        self.Nt = round(geometry.get("t") / self.dt)

        self.diagnose_period  = geometry["diagnose_period"]  # 1/wpe

        self.plot_init = 0
        self.plot_time = self.Nt
        self.plot_offset = int(self.diagnose_period / self.dt)

        if "Particles" in config:
            for sort in config["Particles"]:
                if not sort.get("sort_name") is None:
                    self.sorts.append(sort["sort_name"])

    def bounds_symmetric(G1: float, G2: float) -> tuple[float]:
        return (-0.5 * G1, +0.5 * G1, -0.5 * G2, +0.5 * G2)

    def init_dependent(self, bounds_callback = bounds_symmetric):
        self.data_shape = {
            'X': (self.Ny, self.Nz),
            'Y': (self.Nx, self.Nz),
            'Z': (self.Nx, self.Ny),
        }

        self.boundaries = {
            'X': bounds_callback(self.Ny * self.dy, self.Nz * self.dz),
            'Y': bounds_callback(self.Nx * self.dx, self.Nz * self.dz),
            'Z': bounds_callback(self.Nx * self.dx, self.Ny * self.dy),
        }

        self.cos, self.sin, self.rmap = init_COS_SIN_RMAP(
          (0, self.data_shape['Z'][0], 0, self.data_shape['Z'][1]))
