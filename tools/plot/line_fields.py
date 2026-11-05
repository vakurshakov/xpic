#!/usr/bin/env python3

from plot import *

config = select_subdirectory() # sys.argv[1]
init_constants(f"results/{config}/{config}.json")
const.B0 = float(const.config["Presets"][0]["setter"]["value"][2])
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])
const.min_x = read_scalar(const.config["StepPresets"][0]["geometry"]["min"][0]) - const.Lx / 2
const.max_x = read_scalar(const.config["StepPresets"][0]["geometry"]["max"][0]) - const.Lx / 2
const.min_y = read_scalar(const.config["StepPresets"][0]["geometry"]["min"][1]) 
const.max_y = read_scalar(const.config["StepPresets"][0]["geometry"]["max"][1])
const.pto = 2 * const.Ndts

vmap = np.array([-0.02, +0.02])
fig, gs = figure(3, 1)

bx = -const.Lx / 2
ex = +const.Lx / 2
xs = np.linspace(bx, ex, const.Nx)

def plot(j, title):
    plot = PlotLinear(subplot(fig, gs, j, 0), vmap)
    plot.info.set_args(
        title=title,
        xlim=(bx, ex),
        xticks=np.linspace(bx, ex, 5),
        xlabel='$x,~c/\\omega_{pe}$',
    )
    return plot

e = plot(0, "$E$")
b = plot(1, "$B$")
bz = plot(2, "$B_z$")

def callback(t):
    xmin = [const.min_x, const.min_x]
    xmax = [const.max_x, const.max_x]

    # dn = read_n("ions")(t).mean(0)

    dx = read_E(0)(t).mean(0)
    dy = read_E(1)(t).mean(0)
    dz = read_E(2)(t).mean(0)
    e.axis.plot(xs, dx, label="$E_x$")
    e.axis.plot(xs, dy, label="$E_y$")
    e.axis.plot(xs, dz, label="$E_z$")
    e.axis.plot(xmin, vmap, ls="--", c="black")
    e.axis.plot(xmax, vmap, ls="--", c="black")
    e.axis.legend(fontsize=0.8*FontSize.label)
    e.info.draw()

    dx = read_B(0)(t).mean(0)
    dy = read_B(1)(t).mean(0)
    dz = read_B(2)(t).mean(0)
    b.axis.plot(xs, dx, label="$B_x$")
    b.axis.plot(xs, dy, label="$B_y$")
    b.axis.plot(xmin, vmap, ls="--", c="black")
    b.axis.plot(xmax, vmap, ls="--", c="black")
    b.axis.legend(fontsize=0.8*FontSize.label)
    b.info.draw()

    bz.axis.plot(xs, dz)
    bz.axis.plot(xmin, (0, const.B0), ls="--", c="black")
    bz.axis.plot(xmax, (0, const.B0), ls="--", c="black")
    bz.info.draw()

process_plots(fig, "line_fields", time_tau, [e, b, bz], callback)
