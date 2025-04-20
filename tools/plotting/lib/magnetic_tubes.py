import numpy as np

from scipy.integrate import cumulative_trapezoid

# Magnetic tubes utilities

def select_magnetic_tube(bz, xl):
    xs = bz.shape[0]
    zs = bz.shape[1]
    xc = xs // 2
    zc = zs // 2

    # 2 * np.pi can be removed since comparison is relative
    b_f0 = np.sum(bz[zc,xc:xl] * np.arange(0, xl - xc))

    xmap = np.zeros(zs, dtype=np.int32)
    for z in np.arange(0, zs):
        b_fz = 0
        for x in np.arange(xc, xs):
            b_fz += bz[z, x] * (x - xc)
            xmap[z] = x

            if (b_fz >= b_f0):
                break
    return xmap

def curvature_coefficients(bz_normalized, xl, xs1, xs2):
    cos_map = np.ones_like(bz_normalized[:,xl])
    for z, x in enumerate(0.5 * (xs1 + xs2)):
        cos_map[z] = bz_normalized[z,int(x)]
    return cos_map

def average_over_magnetic_tube(data, zs, xs1, xs2, xc, dx):
    f_l = np.zeros(data.shape[1])
    for z in zs:
        rs = (np.arange(xs1[z], xs2[z] + 1) - xc) * dx
        area = cumulative_trapezoid(2 * np.pi * rs, rs, initial=0)[-1]
        f_l[z] += cumulative_trapezoid(data[z, xs1[z]:(xs2[z] + 1)] * (2 * np.pi * rs), rs, initial=0)[-1] / area
        f_l[z] += cumulative_trapezoid(data[z, (2 * xc - (xs2[z] + 1)):(2 * xc - xs1[z])] * (2 * np.pi * rs), rs, initial=0)[-1] / area
    return f_l / 2

def align(fr, fz, br, bz, b):
    dot = (fr * br + fz * bz)
    return np.divide(dot, b, where=(b > 1e-3), out=np.zeros_like(b))
