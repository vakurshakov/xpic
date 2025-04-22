#!/usr/bin/env python3

from collect import *

arrays: CollectionArrays = {
    "b": [[], gen_view("B_PlaneZ_05", 'z', 3)],
    "ni": [[], gen_view("ions/Density_PlaneZ_05")],
    "ne": [[], gen_view("electrons/Density_PlaneZ_05")],

    "pri": [[], gen_view("ions/mVrVr_PlaneZ_05")],
    "pai": [[], gen_view("ions/mVphiVphi_PlaneZ_05")],
    "pre": [[], gen_view("electrons/mVrVr_PlaneZ_05")],
    "pae": [[], gen_view("electrons/mVphiVphi_PlaneZ_05")],

    "pd": [[], None],
}

def parse(t):
    b = center_avg(read(t, arrays, "b"))
    ni = center_avg(read(t, arrays, "ni"))
    ne = center_avg(read(t, arrays, "ne"))

    pri = phi_avg(read(t, arrays, "pri"))
    pai = phi_avg(read(t, arrays, "pai"))
    pre = phi_avg(read(t, arrays, "pre"))
    pae = phi_avg(read(t, arrays, "pae"))

    pr = pri + pre
    pa = pai + pae

    rs = np.arange(0, const.data_shape["Z"][0] // 2) * const.dx
    pd = cumulative_trapezoid((pr - pa) / (rs + 0.1), dx=const.dx, initial=0)
    pd = -(pd - pd[-1])
    return b, ni, ne, pri[0], pai[0], pre[0], pae[0], pd[0]

def output(name):
    return f"{res_dir}/{name}_t"

process_collection(arrays, parse, output)
