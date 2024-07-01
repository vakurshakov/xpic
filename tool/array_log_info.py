#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "axes.formatter.use_mathtext": True})

from configuration import config

def draw_evolution(data, name):
    plt.plot(np.arange(len(data)), data)
    plt.title(name)
    plt.show()

def draw_parametric(x_data, y_data, name):
    plt.plot(x_data, y_data)
    plt.title(name)
    plt.show()

if __name__ == "__main__":
    filename = f"../{config['Out_dir']}/particle_iterations.bin"
    dtype = np.dtype([
        ("i",  np.float32),
        ("dt", np.float32),
        ("mu", np.float32),
        ("x",  np.float32),
        ("y",  np.float32),
        ("z",  np.float32),
        ("px", np.float32),
        ("py", np.float32),
        ("pz", np.float32),
    ])
    count = 2811
    data = np.fromfile(filename, dtype=dtype, count=count)

    draw_evolution(data["x"], "$v_{\|}^0 = 0.3 v_{crit}$")
    # draw_parametric(data["x"], data["z"], "$v_{\|}^0 = 0.3 v_{crit}$")
