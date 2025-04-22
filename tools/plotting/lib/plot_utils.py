# By default we are not using interactive backend
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from lib.plot_colormaps import *

plt.rc('text', usetex=True)
plt.rc('axes.formatter', use_mathtext=True)

# Font sizes
titlesize = 36
labelsize = 34
ticksize  = 30

# Utilities to set font sizes externally
def set_titlesize(new_titlesize):
    global titlesize
    titlesize = new_titlesize

def set_labelsize(new_labelsize):
    global labelsize
    labelsize = new_labelsize

def set_ticksize(new_ticksize):
    global ticksize
    ticksize = new_ticksize

def subplot(fig, gs, x=0, y=0):
    return fig.add_subplot(gs[x + y * gs.ncols])

bbox = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.25')

def annotate_x(axis, annotation, x=0.5, y=1, size=titlesize, ha="center", bbox=bbox):
    axis.annotate(
        annotation,
        xy=(x, y),
        xytext=(0, 1),
        xycoords="axes fraction",
        textcoords="offset points",
        ha=ha,
        va="baseline",
        size=size,
        bbox=bbox
    )

def annotate_y(axis, annotation):
    axis.annotate(
        annotation,
        xy=(0, 0.5),
        xytext=(-axis.yaxis.labelpad - 1, 0),
        xycoords=axis.yaxis.label,
        textcoords="offset points",
        ha="right",
        va="center",
        rotation=90,
        size=titlesize,
    )

def figure(ncols: int, nrows: int, width_ratios: list[int] = None, height_ratios: list[int] = None):
    fig = plt.figure(figsize=(8 * ncols * 1.2, 8 * nrows * 1.2))

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
