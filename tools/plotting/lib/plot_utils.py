# By default we are not using interactive backend
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from lib.plot_colormaps import *

plt.rc('text', usetex=True)
plt.rc('axes.formatter', use_mathtext=True)

def subplot(fig, gs, x=0, y=0):
    return fig.add_subplot(gs[x + y * gs.ncols])

bbox = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.25')

def figure(ncols: int, nrows: int, width_ratios: list[int] = None, height_ratios: list[int] = None, figsize=None):
    if figsize == None:
        figsize = (8 * ncols * 1.2, 8 * nrows * 1.2)

    fig = plt.figure(figsize=figsize)

    if width_ratios == None:
        width_ratios = [1] * ncols
    if height_ratios == None:
        height_ratios = [1] * nrows

    gs = plt.GridSpec(
        figure=fig,
        ncols=ncols,
        nrows=nrows,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )
    return fig, gs

def find_exp(number):
    return int(np.floor(np.log10(np.abs(number))))
