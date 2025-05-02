#!/usr/bin/python

"""
    Run of this script calculates, shows and stores the Kotelnikov's
    diamagnetic bubble equilibrium with mono-energetic kernel. The
    `xpic` command, is working with _maxwellian_ kernel, so this is
    for demonstrational purposes.
"""

from header import *
from utils_mono import *

def solve_ivp(j_func):
    """
    Solves the system of ODEs equivalent to magnetic field
    equation, where `y0` - chi(r), `y1` - chi'(r) / r.
    """

    def _rhs(r, y):
        f = np.zeros_like(y)

        # y0' = chi'/r
        f[0] = y[1] * r

        # y1' = (chi'/r)' = chi''/r - chi'/r^2 = - J_{\psi}
        f[1] = -j_func(r, y[0])
        return f

    return spi.solve_ivp(_rhs, r0_rmax, y0, "RK45", t_eval=r_values, atol=atol, rtol=rtol)

def calc(d, n_func, j_func, prr_func, paa_func):
    solution = solve_ivp(j_func)
    print(solution)

    if not solution.success:
        sys.exit(1)

    r = solution.t
    chi = solution.y[0]
    b = solution.y[1]

    n = n_func(r, chi)
    j = j_func(r, chi)
    prr = prr_func(r, chi)
    paa = paa_func(r, chi)
    dprr = find_dprr(r, prr, paa)
    draw_all(d, r, chi, b, n, j, prr, paa, dprr)
    store_all(d, chi, b, n, j, prr, paa)

if __name__ == "__main__":
    calc("mono", mono_n, mono_j, mono_prr, mono_paa)