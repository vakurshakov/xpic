#!/usr/bin/python

"""
    Run of this script calculates, shows and stores the Kotelnikov's
    diamagnetic bubble equilibrium with maxwellian kernel. By default,
    `xpic` command will look for caches from this script to initialize
    the simulation.
"""

from header import *
from calc_mono import *
from utils_maxw import *

if __name__ == "__main__":
    calc("maxw", maxw_n, maxw_j, maxw_prr, maxw_paa)
