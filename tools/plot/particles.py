#!/usr/bin/env python3

from plot import *

config = f"{sys.argv[1]}_dx3.0_dt1.5_Np1000"
init_constants(f"results/{config}/{config}.json")
const.B0 = float(const.config["Presets"][0]["setter"]["value"][2])
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])
const.min_x = read_scalar(const.config["StepPresets"][0]["geometry"]["min"][0]) - const.Lx / 2
const.max_x = read_scalar(const.config["StepPresets"][0]["geometry"]["max"][0]) - const.Lx / 2
const.min_y = read_scalar(const.config["StepPresets"][0]["geometry"]["min"][1])
const.max_y = read_scalar(const.config["StepPresets"][0]["geometry"]["max"][1])
const.T_i = round(read_scalar(const.config["Particles"][0]["T"]) / mec2, 2)
const.T_e = round(read_scalar(const.config["Particles"][1]["T"]) / mec2, 3)
const.pto = 2 * const.Ndts

def read(file_t, c, cd):
    data_shape = [const.Ny, const.Nx, cd]
    with open(file_t, "rb") as file:
        raw = np.fromfile(file, dtype=np.float32, count=np.prod(data_shape), offset=0)
        raw = np.reshape(raw, data_shape)
    return raw[:,:,c]

def read_J(c): return lambda t: read(f"{const.in_dir}/{s}/J_ZAvg/{str(t // const.Ndts).zfill(4)}", c, 3)
def read_P(c): return lambda t: read(f"{const.in_dir}/{s}/momentum_Z2/{str(t // const.Ndts).zfill(4)}", c, 6)
def read_rho(): return lambda t: read(f"{const.in_dir}/{s}/rho_ZAvg/{str(t // const.Ndts).zfill(4)}", 0, 1) * (1 if s == "ions" else -1)

vmap_n = (0, 3)

vmap_s = {
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

    add(0, 0, f"$n_{s[0]}$", vmap_n, unsigned_cmap, read_rho()),
    add(0, 1, f"$J_x^{s[0]}$", vmap_s[s], signed_cmap, read_J(0)),
    add(1, 1, f"$J_y^{s[0]}$", vmap_s[s], signed_cmap, read_J(1)),
    add(2, 1, f"$J_z^{s[0]}$", vmap_s[s], signed_cmap, read_J(2)),
    add(0, 2, f"$\\Pi_{{xx}}^{s[0]}$", vmap_p[s], unsigned_cmap, read_P(0)),
    add(1, 2, f"$\\Pi_{{yy}}^{s[0]}$", vmap_p[s], unsigned_cmap, read_P(3)),
    add(2, 2, f"$\\Pi_{{zz}}^{s[0]}$", vmap_p[s], unsigned_cmap, read_P(5)),
    plots = np.asarray(plots)

    def callback(t):
        for plot, read in plots:
            plot.data = read(t)
            plot.draw()
            plot.axis.plot([const.min_x, const.min_x], [const.min_y, const.max_y], ls="--", c="black")
            plot.axis.plot([const.max_x, const.max_x], [const.min_y, const.max_y], ls="--", c="black")

    process_plots(fig, f"info_{s}", time_tau, plots[:,0], callback)
