"""
    Functions to evaluate modified Kotelnikov's equilibrium with   \\
    mono-energetic diamagnetic bubble kernel, namely               \\
    `mono_n()` - ion density,                                      \\
    `mono_j()` - ion current,                                      \\
    `mono_prr()` - rr component of ion momentum flux tensor        \\
    `mono_paa()` - \psi \psi component of ion momentum flux tensor 
"""

from header import *

def _mono_impl(r, chi, f):
  d = np.zeros_like(r)

  def _impl(_r, _c):
    if _r < a:
        s1 = +1
        s2 = -1
    else:
        s1 = +max(min((a - _c) / _r, 1), -1)
        s2 = -max(min((a + _c) / _r, 1), -1)
    return f(s1, s2)

  if d.ndim == 0:
    d = _impl(r, chi)
  else: 
    for i, (_r, _c) in enumerate(zip(r, chi)):
      d[i] = _impl(_r, _c)
  return d

def mono_n(r, chi):
  return _mono_impl(r, chi, lambda s1, s2: \
    (np.arcsin(s1) - np.arcsin(s2)) / np.pi)

def mono_j(r, chi):
  return _mono_impl(r, chi, lambda s1, s2: \
    (-np.sqrt(1.0 - s1 * s1) + np.sqrt(1.0 - s2 * s2)) / np.pi)

def _p_impl(r, chi, sign):
  return 0.5 * (mono_n(r, chi) + _mono_impl(r, chi, lambda s1, s2: \
    sign / np.pi * (s1 * np.sqrt(1 - np.square(s1)) - s2 * np.sqrt(1 - np.square(s2)))))

def mono_prr(r, chi): return _p_impl(r, chi, +1)
def mono_paa(r, chi): return _p_impl(r, chi, -1)
