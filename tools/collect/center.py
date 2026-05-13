#!/usr/bin/env python3

from collect import *

config = select_subdirectory() # f"{sys.argv[1]}_dx3.0_dt1.5_Np1000" # ecsimcorr
init_constants(f"results/{config}/{config}.json")
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])
# const.pts = 0
# const.pte = int(3.23 * const.tau / const.dt)
const.pto = const.Ndts

arrays = {
    "ex": [],
    "ey": [],
    "ez": [],
    "bx": [],
    "by": [],
    "bz": [],
    "ni": [],
    "ne": [],
    "pxi": [],
    "pxe": [],
}

def parse(t):
    e = read_E(None)(t)
    ex = center_avg(e[:,:,0,0])
    ey = center_avg(e[:,:,0,1])
    ez = center_avg(e[:,:,0,2])
    b = read_B(None)(t)
    bx = center_avg(b[:,:,0,0])
    by = center_avg(b[:,:,0,1])
    bz = center_avg(b[:,:,0,2])
    ni = center_avg(read_n("ions")(t))
    ne = center_avg(read_n("electrons")(t))
    pxi = center_avg(read_P("ions", 0)(t))
    pxe = center_avg(read_P("electrons", 0)(t))
    return ex, ey, ez, bx, by, bz, ni, ne, pxi, pxe

def output(name):
    return f"{const.out_dir}/collections/{name}_t"

process_collection(arrays, parse, output)
