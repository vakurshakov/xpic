"""
    Functions to evaluate modified Kotelnikov's equilibrium with   \\
    maxwellian diamagnetic bubble kernel, namely                   \\
    `maxw_n()` - ion density,                                      \\
    `maxw_j()` - ion current,                                      \\
    `maxw_prr()` - rr component of ion momentum flux tensor        \\
    `maxw_paa()` - \psi \psi component of ion momentum flux tensor 
"""

from header import *
from utils_mono import *

# integral of mono-energetic function over maxwellian velocity distribution
def _integrated_mono(r, chi, mono_func, power):
    def _maxwell(x, power):
        return np.pow(x, power) * np.exp(- x * x / 2)

    def _integrand(x):
        return mono_func(r, np.divide(chi, x, where=(x > v0_tolerance))) * _maxwell(x, power)

    return spi.quad(_integrand, 0, np.inf, epsabs=atol, epsrel=rtol)

# returns evaluated function for maxwellian distribution function
def _evaluate_maxw(r, chi, mono_func, power):
    func = np.zeros_like(r)

    def _impl(_r, _c):
        f, err = _integrated_mono(_r, _c, mono_func, power)
        print(f"r: {_r:<5.2f}  f: {f: 7.5e}  err: {err: 7.5e}")
        return f

    if func.ndim == 0:
        func = _impl(r, chi)
    else:
        for i, (_r, _c) in enumerate(zip(r, chi)):
            func[i] = _impl(_r, _c)

    return func

def maxw_n(r, chi): return _evaluate_maxw(r, chi, mono_n, 1)
def maxw_j(r, chi): return _evaluate_maxw(r, chi, mono_j, 2)
def maxw_prr(r, chi): return _evaluate_maxw(r, chi, mono_prr, 3)
def maxw_paa(r, chi): return _evaluate_maxw(r, chi, mono_paa, 3)
