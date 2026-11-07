#!/usr/bin/env python3

from plot import *

config = select_subdirectory() if len(sys.argv) == 1 else sys.argv[1]

init_constants(f"results/{config}/{config}.json")
const.B0 = float(const.config["Presets"][0]["setter"]["value"][2])
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])
const.min_x = read_scalar(const.config["StepPresets"][0]["geometry"]["min"][0]) - const.Lx / 2
const.max_x = read_scalar(const.config["StepPresets"][0]["geometry"]["max"][0]) - const.Lx / 2
const.min_y = read_scalar(const.config["StepPresets"][0]["geometry"]["min"][1])
const.max_y = read_scalar(const.config["StepPresets"][0]["geometry"]["max"][1])
const.T_i = round(read_scalar(const.config["Particles"][0]["T"]) / mec2, 2)
const.T_e = round(read_scalar(const.config["Particles"][1]["T"]) / mec2, 3)
const.pto = const.Ndts

vmap_n = (0, 3)

vmap_j = {
    "ions": (-2e-3, +2e-3),
    "electrons": (-2e-2, +2e-2),
}

vmap_p = {
    "ions": (0, const.T_i),
    "electrons": (0, const.T_e)
}

fig, gs = figure(3, 3)

FontSize.title = 30
FontSize.label = 28
FontSize.ticks = 24

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


def add(i, j, title, vmap, cmap, read):
    plots.append((plot(i, j, title, vmap, cmap), read))

for s in const.sorts:
    fig.clear()
    plots = []

    add(0, 0, f"$n_{s[0]}$", vmap_n, unsigned_cmap, read_n(s)),
    add(0, 1, f"$J_x^{s[0]}$", vmap_j[s], signed_cmap, read_J(s, 0)),
    add(1, 1, f"$J_y^{s[0]}$", vmap_j[s], signed_cmap, read_J(s, 1)),
    add(2, 1, f"$J_z^{s[0]}$", vmap_j[s], signed_cmap, read_J(s, 2)),
    add(0, 2, f"$\\Pi_{{xx}}^{s[0]}$", vmap_p[s], unsigned_cmap, read_P(s, 0)),
    add(1, 2, f"$\\Pi_{{yy}}^{s[0]}$", vmap_p[s], unsigned_cmap, read_P(s, 3)),
    add(2, 2, f"$\\Pi_{{zz}}^{s[0]}$", vmap_p[s], unsigned_cmap, read_P(s, 5)),
    # add(0, 3, f"$\\Pi_{{xy}}^{s[0]}$", (-0.1 * vmap_p[s][1], +0.1 * vmap_p[s][1]), signed_cmap, read_P(s, 1)),
    # add(1, 3, f"$\\Pi_{{xz}}^{s[0]}$", (-0.1 * vmap_p[s][1], +0.1 * vmap_p[s][1]), signed_cmap, read_P(s, 2)),
    # add(2, 3, f"$\\Pi_{{yz}}^{s[0]}$", (-0.1 * vmap_p[s][1], +0.1 * vmap_p[s][1]), signed_cmap, read_P(s, 4)),
    plots = np.asarray(plots)

    def callback(t):
        for plot, read in plots:
            plot.data = read(t)
            plot.draw()
            plot.axis.plot([const.min_x, const.min_x], [const.min_y, const.max_y], ls="--", c="black")
            plot.axis.plot([const.max_x, const.max_x], [const.min_y, const.max_y], ls="--", c="black")

    process_plots(fig, f"info_{s}", time_tau, plots[:,0], callback)
