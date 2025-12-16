from lib.plot_utils import *
from lib.data_format import FieldView

from typing import Any
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# Main classes for plotting

class PlotAxisInfo:
    def __init__(self, axis):
        self.axis: plt.Axes = axis
        self.args: dict[str, Any] = {}

    def set_args(self, **kwargs):
        supported_names = [
            "title",
            "xlim", "ylim",
            "xticks", "yticks",
            "xlabel", "ylabel",
            "xticklabels", "yticklabels",
        ]

        for name, arg in kwargs.items():
            if not name in supported_names:
                raise NotImplementedError("No parameter named " + name)
            self.args[name] = arg

    def draw(self):
        ax = self.axis
        ax.tick_params(labelsize=ticksize, pad=8)

        for name, arg in self.args.items():
            if name == "title":
                ax.set_title(arg, fontsize=titlesize, pad=0, y=1.05)
            elif name == "xlabel":
                ax.set_xlabel(arg, fontsize=labelsize, labelpad=12)
            elif name == "ylabel":
                ax.set_ylabel(arg, fontsize=labelsize, labelpad=10)
            elif name == "xticklabels":
                ax.set_xticklabels(arg, fontsize=labelsize)
            elif name == "yticklabels":
                ax.set_yticklabels(arg, fontsize=labelsize)
            else:
                # These parameters are passed as is, e.g. xlim, xticks
                ax.set(**{name: arg})

class PlotIm:
    def __init__(
        self,
        view: FieldView,
        vmap: tuple[float] = (None, None),
        cmap: str = "plasma",
        bounds: tuple[float] = None,
        axis: plt.Axes = None):

        self.view: FieldView = view
        self.data: np.ndarray[np.float32] = None
        self.bounds: tuple[float] = bounds

        self.axis: plt.Axes = axis
        self.info = PlotAxisInfo(axis)

        self.im: plt.AxesImage = None
        self.cbar: plt.Colorbar = None

        self.vmin: float = vmap[0]
        self.vmax: float = vmap[1]
        self.cmap: plt.Colormap = cmap

    def set_axis(self, axis: plt.Axes):
        self.axis = axis
        self.info.axis = axis

    def draw(self, **kwargs):
        self.axis.set_aspect(1.0)

        if self.bounds == None:
            self.bounds = (0, self.data.shape[1], 0, self.data.shape[0])

        self.im = self.axis.imshow(
            self.data,
            cmap=self.cmap,
            interpolation="gaussian",
            origin="lower",
            aspect="auto",
            extent=(
                self.bounds[0],
                self.bounds[1],
                self.bounds[2],
                self.bounds[3],
            ),
            vmin=self.vmin,
            vmax=self.vmax,
        )

        self.draw_cbar(**kwargs)
        self.info.draw()

    def draw_cbar(self, **kwargs):
        args = {
            "add": True,
            "pad": 0.2,
            "ticks_num": 5,
            "exponential": False,
            "orientation": "vertical",
        }

        for iname in kwargs:
            for name in args:
                if iname == name:
                    args[name] = kwargs[iname]

        if not args["add"] or not self.cbar is None or \
            self.vmin is None or self.vmax is None:
            return

        if (np.abs(self.vmin) > 0 and np.abs(self.vmin) < 1e-4) or \
            (np.abs(self.vmax) > 0 and np.abs(self.vmax) < 1e-4):
            args["exponential"] = True

        fig = self.axis.figure
        divider = make_axes_locatable(self.axis)

        side = "right" if (args["orientation"] == "vertical") else "bottom"
        cax = divider.append_axes(side, size="5%", pad=args["pad"])
        cax.tick_params(labelsize=ticksize, pad=10)

        self.cbar = fig.colorbar(
            self.im, orientation=args["orientation"], cax=cax,
            ticks=np.linspace(self.vmin, self.vmax, args["ticks_num"]),
        )

        if not args["exponential"]:
            return

        self.cbar.formatter.set_powerlimits((0, 0))

        yax = self.cbar.ax.yaxis
        yax.OFFSETTEXTPAD = 8
        yax.get_offset_text().set_size(0.82 * ticksize)

    def clear(self):
        self.axis.cla()

class PlotLinear:
    def __init__(
        self,
        vmap: tuple[float] = None,
        axis: plt.Axes = None):

        self.data: np.ndarray[np.float32] = None

        self.axis: plt.Axes = axis
        self.plot: plt.Line2D = None

        self.info = PlotAxisInfo(axis)

        if vmap != None:
            self.info.args["ylim"] = vmap
            self.info.args["yticks"] = np.linspace(vmap[0], vmap[1], 5)

        self.plot_info: dict[str, Any] = {}

    def set_axis(self, axis: plt.Axes):
        self.axis = axis
        self.info.axis = axis

    def draw(self, x):
        self.plot = self.axis.plot(x, self.data, **self.plot_info)
        self.draw_info()

    def draw_info(self):
        self.info.draw()

        if "label" in self.plot_info:
            self.axis.legend()

        yax = self.axis.yaxis
        yax.OFFSETTEXTPAD = 8
        yax.get_offset_text().set_size(0.82 * ticksize)

    def clear(self):
        self.axis.cla()
