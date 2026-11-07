#!/usr/bin/env python3

from plot import *

config = select_subdirectory() # sys.argv[1] # f"{sys.argv[1]}_dx3.0_dt1.5_Np1000" # ecsimcorr
init_constants(f"results/{config}/{config}.json")
const.B0 = float(const.config["Presets"][0]["setter"]["value"][2])
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])
const.pto = 2 * const.Ndts

fig, gs = figure(4, 3)

def plot(i, j, title, vmap, cmap = signed_cmap):
    plot = PlotIm(subplot(fig, gs, i, j), vmap, cmap)
    bx = -const.Lx / 2
    ex = +const.Lx / 2
    by = 0
    ey = const.Ly
    plot.bounds = (bx, ex, by, ey)
    plot.info.set_args(
        title=title,
        xlim=(bx, ex),
        ylim=(by, ey),
        xticks=np.linspace(bx, ex, 5),
        yticks=np.linspace(by, ey, 5),
        xlabel='$x,~c/\\omega_{pe}$',
        ylabel='$y,~c/\\omega_{pe}$',
    )
    return plot

vmap = np.array([-0.01, +0.01])

jx = plot(0, 0, "$J_x$", vmap)
jy = plot(0, 1, "$J_y$", vmap)
jz = plot(0, 2, "$J_z$", vmap)

rbx = plot(1, 0, "$(\\nabla \\times B)_x$", vmap)
rby = plot(1, 1, "$(\\nabla \\times B)_y$", vmap)
rbz = plot(1, 2, "$(\\nabla \\times B)_z$", vmap)

dtex = plot(2, 0, "$\\partial E_x / \\partial t = - J_x + (\\nabla \\times B)_x$", vmap)
dtey = plot(2, 1, "$\\partial E_y / \\partial t = - J_y + (\\nabla \\times B)_y$", vmap)
dtez = plot(2, 2, "$\\partial E_z / \\partial t = - J_z + (\\nabla \\times B)_z$", vmap)

bx = plot(3, 0, "$B_x$", vmap)
by = plot(3, 1, "$B_y$", vmap)
bz = plot(3, 2, "$B_z$", (0, 1.0 * const.B0), unsigned_cmap)

plots = (jx, jy, jz, rbx, rby, rbz, dtex, dtey, dtez, bx, by, bz)

def callback(t):
    jx.data = read_J("ions", 0)(t) + read_J("electrons", 0)(t)
    jy.data = read_J("ions", 1)(t) + read_J("electrons", 1)(t)
    jz.data = read_J("ions", 2)(t) + read_J("electrons", 2)(t)

    bx.data = read_B(0)(t)
    by.data = read_B(1)(t)
    bz.data = read_B(2)(t)

    rbx.data = +np.gradient(bz.data, const.dy, axis=0)
    rby.data = -np.gradient(bz.data, const.dx, axis=1)
    rbz.data = +np.gradient(by.data, const.dx, axis=1) \
                -np.gradient(bx.data, const.dy, axis=0)

    dtex.data = -jx.data + rbx.data
    dtey.data = -jy.data + rby.data
    dtez.data = -jz.data + rbz.data

    for f in plots: 
        f.draw()
        f.info.draw()

process_plots(fig, "check_currents", time_tau, plots, callback)
