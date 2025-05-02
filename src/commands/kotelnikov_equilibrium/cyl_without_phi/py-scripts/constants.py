"""
    Parameters and numerical constants for Kotelnikov's equilibrium,
    for reference see https://doi.org//10.1088/1361-6587/ab8a63.
"""

from pathlib import Path
import numpy as np

# Diamagnetic bubble parameters

a = 1.5
"[c/wpi] - radius of the diamagnetic bubble."

B_in = 1e-2
"[B_v = sqrt(4 pi n0 T_i)] - magnetic field at the center of the bubble."

out_dir = Path(__file__).parent / f"../cache/a{a:.1f}_B{B_in:.2f}/"
"Directory where quantities needed for the equilibrium will be stored."

out_dir.mkdir(parents=True, exist_ok=True)

# Numerical parameters

r0_rmax = (a, 25)
"[c/wpi] - region where we want to find a solution."

dr = 0.02
"[c/wpi] - width of numerical mesh where solutions are defined."

r_values = np.linspace(r0_rmax[0], r0_rmax[1], int((r0_rmax[1] - r0_rmax[0]) / dr) + 1)
"[c/wpi] - array of evenly spaced points where the solution will be defined."

y0 = (0, B_in)
"[(B_v * (c/wpi)^2, B_v)] - initial values for ODE solution vector (chi(r), chi'(r) / r)."

# tolerances in case of dividing by zero
r0_tolerance = 1e-3
v0_tolerance = 1e-3

# ODE solver tolerances
rtol = 1e-8
atol = 1e-8
