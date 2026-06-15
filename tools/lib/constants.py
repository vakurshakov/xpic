#!/usr/bin/env python3

import os, json
from typing import Final
from dataclasses import dataclass

@dataclass
class Constants:
    # directories
    config_path = None
    config = None
    in_dir = None
    in_dirs = None
    in_restarts = None
    out_dir = None

    # figure
    nrows = None
    ncols = None
    pts = None
    pte = None
    pto = None

    # geometry
    dx, dy, dz, dt = (None,) * 4 # c/wpe [3], 1/wpe
    Lx, Ly, Lz, Lt = (None,) * 4 # c/wpe [3], 1/wpe 
    Nx, Ny, Nz, Nt = (None,) * 4 # cells
    dts = None # 1/wpe
    Ndts = None # units

    # additional
    sorts = None
    B0 = None
    mi_me = None
    T_i = None # KeV
    T_e = None # KeV
    tau = None # 1/wpe

# common
mec2: Final = 511.0 # KeV

const = Constants()

def read_scalar(s):
    if type(s) != str: return s
    if s.endswith(' [dx]'): return float(s.split(' ')[0]) * const.dx
    if s.endswith(' [dy]'): return float(s.split(' ')[0]) * const.dy
    if s.endswith(' [dz]'): return float(s.split(' ')[0]) * const.dz
    if s.endswith(' [dt]'): return float(s.split(' ')[0]) * const.dt
    if s.endswith(' [c/w_pe]') or s.endswith(' [1/w_pe]'): return float(s.split(' ')[0])

def init_constants(config_path):
    global const

    with open(config_path, 'r') as file:
        config = json.load(file)

    const.config_path = config_path
    const.config = config

    geometry = config.get('Geometry')
    const.dx = geometry.get('dx')
    const.dy = geometry.get('dy')
    const.dz = geometry.get('dz')
    const.dt = geometry.get('dt')

    const.Lx = read_scalar(geometry.get('x'))
    const.Ly = read_scalar(geometry.get('y'))
    const.Lz = read_scalar(geometry.get('z'))
    const.Lt = read_scalar(geometry.get('t'))

    const.Nx = round(const.Lx / const.dx)
    const.Ny = round(const.Ly / const.dy)
    const.Nz = round(const.Lz / const.dz)
    const.Nt = round(const.Lt / const.dt)

    const.dts = read_scalar(geometry.get('diagnose_period'))
    const.Ndts = round(const.dts / const.dt)

    const.pts = 0
    const.pte = const.Nt

    if 'Particles' in config:
        const.sorts = []
        for sort in config['Particles']:
            const.sorts.append(sort.get('sort_name'))

    const.in_dir = os.path.join(config['OutputDirectory'])
    const.out_dir = os.path.join(config['OutputDirectory'], 'processed')
    return const