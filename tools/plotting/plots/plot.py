import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from lib.common import *
from configuration import *

# @todo move this into a `lib.common`, or even into `lib.plot``

# `PlotIm` generation utility
def gen_plot(title: str, path: str, plane: str, comp: str, dof: int, vmap: tuple[float], \
        cmap: plt.Colormap = signed_cmap, buff: int = 0, const = const):
    view = FieldView()
    view.path = lambda t: f"{const.input_path}/{path}/{format_time(t, const.Nt)}"
    view.region = FieldView.Region(dof, (0, 0, 0), (*const.data_shape[plane], dof))
    view.coords = FieldView.Cartesian if not comp in ['r', 'phi'] else FieldView.Cylinder
    if view.coords == FieldView.Cylinder: view.init_cos_sin(const.cos, const.sin)
    view.plane = plane
    view.comp = comp

    plot = PlotIm(view, vmap, cmap)

    axis_args = {
        'X': [ "$(y, z)$", 'y', 'z' ],
        'Y': [ "$(x, z)$", 'x', 'z' ],
        'Z': [ "$(x, y)$", 'x', 'y' ]
    }

    plot.bounds = const.boundaries[plane]
    bx = plot.bounds[0] + buff * const.dx
    ex = plot.bounds[1] - buff * const.dx
    by = plot.bounds[2] + (buff * const.dy if plane == 'Z' else 0)
    ey = plot.bounds[3] - (buff * const.dy if plane == 'Z' else 0)

    plot.info.set_args(
        title=title + axis_args[plane][0],
        xlim=(bx, ex),
        ylim=(by, ey),
        xticks=np.linspace(bx, ex, 5),
        yticks=np.linspace(by, ey, 5),
        xlabel=f"${axis_args[plane][1]},""~c/\\omega_{pe}$",
        ylabel=f"${axis_args[plane][2]},""~c/\\omega_{pe}$",
    )
    return plot

# `PlotLinear` generation utility
def gen_linear(title: str, plane: str, vmap: tuple[float], buff: int = 0, const = const, **kwargs):
    plot = PlotLinear(vmap)

    bx = 0
    ex = const.boundaries[plane][1] - buff * const.dx

    plot.info.set_args(
        title=title,
        xlim=(bx, ex),
        xticks=np.linspace(bx, ex, 5),
        xlabel="$r,~c/\\omega_{pe}$",
    )

    plot.plot_info = {**plot.plot_info, **kwargs}
    return plot

# This is the outline of all time-dependent plotting process
def process_plots(out: str, time: Callable[[int], str], plots: tuple[PlotIm | PlotLinear], callback: Callable[[int], None], const = const):
    ntot = len(plots)
    nrows = int(np.sqrt(ntot))
    ncols = ntot // nrows

    if not const.ncols is None and not const.nrows is None:
        nrows = const.nrows
        ncols = const.ncols

    fig, gs = figure(ncols, nrows)

    for i, plot in enumerate(plots):
        plot.set_axis(subplot(fig, gs, i % ncols, i // ncols))

    res_dir = f"{const.output_path}/{out}"
    makedirs(res_dir)

    offset = int(const.diagnose_period / const.dt)
    t_range = mpi_consecutive_t_range(0, const.Nt, offset)

    for t in t_range:
        filename = f"{res_dir}/{format_time(t // offset, const.Nt)}.png"
        if not timestep_should_be_processed(t, filename, plots[0].view, False):
            return

        callback(t)

        fig.suptitle(time(t), x=0.50, y=0.99, bbox=bbox, fontsize=labelsize)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        fig.savefig(filename)

        for plot in plots:
            plot.clear()

# The most basic plots are of `FieldView` diagnostics, it is reading data and drawing it
def process_basic(out: str, time: Callable[[int], str], plots: tuple[PlotIm], const = const):
    def callback(t):
        for plot in plots:
            plot.data = plot.view.parse(t)
            plot.draw()

    process_plots(out, time, plots, callback, const) 