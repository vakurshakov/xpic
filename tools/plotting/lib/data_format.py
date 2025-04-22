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

    Comps = {
        'x': 0,
        'y': 1,
        'z': 2,

        'r': 0,
        'phi': 1,
        # 'z': 2,
    }

    Cartesian = Literal[0]
    Cylinder = Literal[1]

    def __init__(self):
        self.path: Callable[[int], str] = None
        self.region: FieldView.Region = None
        self.coords: FieldView.Cartesian | FieldView.Cylinder = None
        self.plane: Literal['X', 'Y', 'Z'] = None
        self.plane_position: int = None
        self.comp: Literal['x', 'y', 'z', 'r', 'phi'] = None
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
            return self.read(t).squeeze(2)
        elif self.region.dof == 3 and self.coords == FieldView.Cylinder:
            return self.parse_cyl(t)
        return self.read(t)[:, :, FieldView.Comps.get(self.comp)]

    def parse_cyl(self, t: int):
        # Do we need to remap 'r' and 'phi' components here?
        if self.plane != 'Z' or self.comp == 'z':
            return self.read(t)[:, :, FieldView.Comps.get(self.comp)]
        else:
            fx = self.read(t)[:, :, FieldView.Comps.get('x')]
            fy = self.read(t)[:, :, FieldView.Comps.get('y')]
            return vx_vy_to_vr_va(fx, fy, self.cos, self.sin)[FieldView.Comps.get(self.comp)]
