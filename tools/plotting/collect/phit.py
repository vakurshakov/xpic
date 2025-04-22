#!/usr/bin/env python3

from collect import *

r0    = 1 # int(1 / dx)
rmax  = 3 # int(2 / dx) 
rstep = 1

rmap = []
for r, map in enumerate(const.rmap):
    if (r0 <= r and r < rmax) and (r % rstep == 0):
        rmap.append((r * const.dx, map))

arrays: CollectionArrays = {
    "b": [[], gen_view("B_PlaneZ_05", 'z', 3)],
    "er": [[], gen_view("E_PlaneZ_05", 'r', 3)],
    "ea": [[], gen_view("E_PlaneZ_05", 'phi', 3)],

    # They will be converted to cylinder component in `parse()`
    # @todo This can be replaced with er-like variant if we use dof-storages in `DistributionMoment`
    "jri": [[], gen_view("ions/Vx_PlaneZ_05")],
    "jai": [[], gen_view("ions/Vy_PlaneZ_05")],
    "jre": [[], gen_view("electrons/Vx_PlaneZ_05")],
    "jae": [[], gen_view("electrons/Vy_PlaneZ_05")],
}

def parse(t, map):
    b, er, ea, jxi, jyi, jxe, jye = read(t, arrays, \
      ["b", "er", "ea", "jri", "jai", "jre", "jae"])
    
    jri, jai = vx_vy_to_vr_va(jxi, jyi, const.cos, const.sin)
    jre, jae = vx_vy_to_vr_va(jxe, jye, const.cos, const.sin)
    return b[map], er[map], ea[map], jri[map], jai[map], jre[map], jae[map]

def output(name, r):
    return f"{res_dir}/{name}_phit_r={r:.2f}"

for r, map in rmap:
    print(len(map[0]))

    process_collection(
        arrays,
        lambda t: parse(t, map),
        lambda name: output(name, r),
        shape=((tmax - tmin), len(map[0])))

    # Clearing the arrays, after collection
    for arr, _ in arrays.values():
        arr.clear()