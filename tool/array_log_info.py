#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "axes.formatter.use_mathtext": True})

# from configuration import config

def plot_evolution(t, data):
    plt.plot(t, data)

def plot_parametric(x_data, y_data, **kwargs):
    if "emphasis" in kwargs:
        plt.scatter(x_data, y_data, **kwargs["emphasis"])
        kwargs.pop("emphasis") # To avoid collision with `plt.plot()` named arguments

    plt.plot(x_data, y_data, **kwargs)

def read_binary(filename, count=-1):
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
    return np.fromfile(filename, dtype=dtype, count=count)

def read_text(filename):
    with open(filename) as f:
        names = f.readline()
        dimensions = f.readline()
        # print(names, dimensions)  # `dimensions` are _only_ to double check

        arrays = [[] for n in names.split()]

        for l in f.readlines():
            for i, d in enumerate(l.split()):
                arrays[i].append(float(d))

        return dict(zip(names.split(), arrays))


def get_args(c, m):
    return {"c": c, "lw": 1, "emphasis": {"c": c, "marker": m, "s": 10, "alpha": 0.8}}

r_args = get_args("red", "x")
b_args = get_args("blue", "s")
g_args = get_args("green", "+")
p_args = get_args("purple", "$O$")

tmax = 150

dt = 0.5
ts = np.arange(0, tmax) * dt

prefix = "./tests/chin_output/chin_polarization"

l = "a"
if l == "a":
    data = read_text(f"{prefix}_EB1A.txt")
    plt.scatter(data["y"][:tmax], data["z"][:tmax], **g_args["emphasis"])
    data = read_text(f"{prefix}_EB1B.txt")
    plt.scatter(data["y"][:tmax], data["z"][:tmax], **r_args["emphasis"])
    data = read_text(f"{prefix}_EBLF.txt")
    plt.scatter(data["y"][:tmax], data["z"][:tmax], **p_args["emphasis"])
    data = read_text(f"{prefix}_EB2B.txt")

    b_args.pop("emphasis")
    plot_parametric(data["y"][:tmax], data["z"][:tmax], **b_args)
    plt.xlim((-0.1, 0.1))
    plt.ylim((-5, 60))

elif l == "b":
    data = read_text(f"{prefix}_EB1A.txt")
    plt.plot(ts, data["z"][:tmax] - (0.01 * ts * ts), lw=1, c="g") #, **g_args["emphasis"])
    data = read_text(f"{prefix}_EB1B.txt")
    plt.plot(ts, data["z"][:tmax] - (-0.01 * ts + 0.01 * ts * ts), lw=1, c="r") #, **r_args["emphasis"])
    data = read_text(f"{prefix}_EBLF.txt")
    plt.plot(ts, data["z"][:tmax] - (-0.01 * ts + 0.01 * ts * ts), lw=1, c="purple") #, **p_args["emphasis"])
    data = read_text(f"{prefix}_EB2B.txt")
    plt.plot(ts, data["z"][:tmax] - (-0.005 * ts + 0.01 * ts * ts), lw=1, c="b") #, **b_args)

    rg = 0.001
    plt.plot([ts[0], ts[-1]], [-rg, -rg], lw=1, c="black", alpha=0.8)
    plt.plot([ts[0], ts[-1]], [+rg, +rg], lw=1, c="black", alpha=0.8)
    Rg = 0.052467
    plt.plot([ts[0], ts[-1]], [-Rg, -Rg], lw=1, c="black", alpha=0.8)
    plt.plot([ts[0], ts[-1]], [+Rg, +Rg], lw=1, c="black", alpha=0.8)
    plt.xlim((0, 50))

plt.title("Gyro orbits")
plt.grid(c="grey", alpha=0.6)
plt.show()