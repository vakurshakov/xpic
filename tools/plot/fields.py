#!/usr/bin/env python3

# This is the most basic example of the plot

from plot import *

# 0) Config path is declared explicitly, input and output directories are read from it

config = f"{sys.argv[1]}_dx3.0_dt1.5_Np1000" # ecsimcorr
init_constants(f"results/{config}/{config}.json")
const.B0 = float(const.config["Presets"][0]["setter"]["value"][2])
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])
const.min_x = read_scalar(const.config["StepPresets"][0]["geometry"]["min"][0]) - const.Lx / 2
const.max_x = read_scalar(const.config["StepPresets"][0]["geometry"]["max"][0]) - const.Lx / 2
const.min_y = read_scalar(const.config["StepPresets"][0]["geometry"]["min"][1])
const.max_y = read_scalar(const.config["StepPresets"][0]["geometry"]["max"][1])
const.pto = 2 * const.Ndts

# 1a) Data shape, it is not declared in files themselves
# 1b) Where to read data and how to convert it into flat array for `plt.imshow`

def read(file_t, c):
    data_shape = [const.Ny, const.Nx, 3]
    with open(file_t, "rb") as file:
        raw = np.fromfile(file, dtype=np.float32, count=np.prod(data_shape), offset=0)
        raw = np.reshape(raw, data_shape)
    return raw[:,:,c]

def read_E(c): return lambda t: read(f"{const.in_dir}/E_ZAvg/{str(t // const.Ndts).zfill(4)}", c)
def read_B(c): return lambda t: read(f"{const.in_dir}/B_ZAvg/{str(t // const.Ndts).zfill(4)}", c)

# 2a) Location of plot (`i`, `j` indices)
# 2b) Colormap boundaries
# 2c) Output is `plt.Axis` to process it further, maybe
# 3) Choosen timestep-sequence should be declarated

vmap = np.array([-0.02, +0.02])
fig, gs = figure(3, 2)

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

plots = np.asarray((
    (plot(0, 0, "$E_x(x, y)$", vmap), read_E(0)),
    (plot(1, 0, "$E_y(x, y)$", vmap), read_E(1)),
    (plot(2, 0, "$E_z(x, y)$", vmap), read_E(2)),

    (plot(0, 1, "$B_x(x, y)$", vmap), read_B(0)),
    (plot(1, 1, "$B_y(x, y)$", vmap), read_B(1)),
    (plot(2, 1, "$B_z(x, y)$", (0, 1.0 * const.B0), unsigned_cmap), read_B(2)),
))

def callback(t):
    for plot, read in plots:
        plot.data = read(t)
        plot.draw()
        plot.axis.plot([const.min_x, const.min_x], [const.min_y, const.max_y], ls="--", c="black")
        plot.axis.plot([const.max_x, const.max_x], [const.min_y, const.max_y], ls="--", c="black")

process_plots(fig, "fields", time_tau, plots[:,0], callback)
