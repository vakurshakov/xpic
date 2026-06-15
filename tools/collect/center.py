#!/usr/bin/env python3

from collect import *

config = select_subdirectory() if len(sys.argv) == 1 else sys.argv[1]

init_constants(f"results/{config}/{config}.json")
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])
const.pts = 0
const.pte = int(6 * const.tau / const.dt)
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

def avg(d, w=3): 
    c1 = d.shape[1] // 2
    return d.mean(axis=0)[c1-w:c1+w].mean()

def parse(t):
    de = read_E(None)(t)
    db = read_B(None)(t)
    dni = read_n("ions")(t)
    dne = read_n("electrons")(t)
    dpxi = read_P("ions", 0)(t)
    dpxe = read_P("electrons", 0)(t)
    ex = avg(de[:,:,0,0])
    ey = avg(de[:,:,0,1])
    ez = avg(de[:,:,0,2])
    bx = avg(db[:,:,0,0])
    by = avg(db[:,:,0,1])
    bz = avg(db[:,:,0,2])
    ni = avg(dni)
    ne = avg(dne)
    pxi = avg(dpxi)
    pxe = avg(dpxe)
    return ex, ey, ez, bx, by, bz, ni, ne, pxi, pxe

def output(name):
    return f"{const.out_dir}/collections/{name}_t"

process_collection(arrays, parse, output)
