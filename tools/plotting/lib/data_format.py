import numpy as np

from typing import Literal
from collections.abc import Callable

from lib.data_utils import *

class FieldView():
    class Region():
        # start, size -- Description of the region in the form (x, y, z)
        def __init__(self, dof: int, start: np.ndarray[int], size: np.ndarray[int]):
            self.dof = dof
            self.start = start
            self.size = size

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
        self.comp: Literal['x', 'y', 'z', 'r', 'phi'] = None

    def read(self, t: int) -> np.ndarray[np.float32]:
        ds = list(self.region.size)
        count = np.prod(ds)

        # Should follow C-style ordering: (z, y, x)
        ds[:-1].reverse()

        with open(self.path(t), "rb") as file:
            raw = np.fromfile(
                file,
                dtype=np.float32,
                count=count,
                offset=0,
            )
        return raw.reshape(ds)

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
            return vx_vy_to_vr_va(fx, fy, COS, SIN)[FieldView.Comps.get(self.comp)]
