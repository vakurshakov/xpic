from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from lib.xy_rphi import *

class FieldView():
    # start, size -- Description of the region in the form (x, y, z)
    @dataclass
    class Region():
        dof: int
        start: np.ndarray[int]
        size: np.ndarray[int]

    Cartesian = Literal[0]
    Cylindrical = Literal[1]

    def __init__(self):
        self.path: Callable[[int], str] = None
        self.region: FieldView.Region = None
        self.coords: FieldView.Cartesian | FieldView.Cylindrical = None
        self.plane: Literal['X', 'Y', 'Z'] = None
        self.plane_position: int = None
        self.comp: int = None
        self.cos: np.ndarray = None
        self.sin: np.ndarray = None

    def init_cos_sin(self, cos: np.ndarray, sin: np.ndarray):
        self.cos = cos
        self.sin = sin

    def read(self, t: int) -> np.ndarray[np.float32]:
        # Should follow C-style ordering: (z, y, x)
        ds = list(self.region.size[:-1])
        ds.reverse()
        ds.append(self.region.size[-1])
        count = np.prod(ds)

        with open(self.path(t), "rb") as file:
            raw = np.fromfile(
                file,
                dtype=np.float32,
                count=count,
                offset=0,
            )

        raw = np.reshape(raw,ds)

        # In case we have read 3D data, we take only one slice of it to draw
        if len(raw.shape) == 4:
            plane_to_axis = { 'X': 2, 'Y': 1, 'Z': 0 }
            raw = np.take(raw, self.plane_position, axis=plane_to_axis[self.plane])

        return raw

    def parse(self, t: int) -> np.ndarray[np.float32]:
        if self.region.dof == 1:
            return self.parse_plane(t)
        elif self.region.dof == 3 and self.coords == FieldView.Cylindrical:
            return self.parse_cyl(t)
        return self.read(t)[:, :, self.comp]

    def parse_plane(self, t: int):
        data = self.read(t)
        if (len(data.shape) == 2):
            return data
        elif (len(data.shape) == 3):
            if (self.plane == "X"):
                return data.squeeze(0)
            elif (self.plane == "Y"):
                return data.squeeze(1)
            elif (self.plane == "Z"):
                return data.squeeze(2)
        return data

    def parse_cyl(self, t: int):
        if self.plane == "Z" and self.comp in (0, 1):
            data = self.read(t)
            fx = data[:, :, 0]
            fy = data[:, :, 1]
            return vx_vy_to_vr_va(fx, fy, self.cos, self.sin)[self.comp]
        # We _do_ need to remap components to get 'r', 'phi'
        elif (self.plane == "X" and self.comp == 0):
            data = self.read(t)[:, :, self.comp]
            data[:, (data.shape[1] // 2):] *= -1
            return data
        elif (self.plane == "X" and self.comp == 1):
            data = self.read(t)[:, :, self.comp]
            data[:, :(data.shape[1] // 2)] *= -1
            return data
        elif (self.plane == "Y" and self.comp == 0):
            data = self.read(t)[:, :, self.comp]
            data[:, (data.shape[1] // 2):] *= -1
            return data
        elif (self.plane == "Y" and self.comp == 1):
            data = self.read(t)[:, :, self.comp]
            data[:, :(data.shape[1] // 2)] *= -1
            return data
        return self.read(t)[:, :, self.comp]
