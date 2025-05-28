#!/usr/bin/env python3

import argparse

from plot import *

view = FieldView()
view.path = lambda t: f"{const.input_path}/electrons/velocity/{format_time(t, const.Nt)}"
view.region = FieldView.Region(1, (0, 0), (200, 200))
view.coords = FieldView.Cartesian
view.comp = 0

plot = PlotIm(view, (0, 1), unsigned_cmap)

bx = -1
ex = +1
by = -1
ey = +1
plot.bounds = (bx, ex, by, ey)

plot.info.set_args(
    title="$f(v_x, v_y)$",
    xlim=(bx, ex),
    ylim=(by, ey),
    xticks=np.linspace(bx, ex, 5),
    yticks=np.linspace(by, ey, 5),
    xlabel="$v_x,~c$",
    ylabel="$v_y,~c$",
)

process_basic("distribution", time_wpe, [plot])
