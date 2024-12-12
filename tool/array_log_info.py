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
    return {"c": c, "lw": 1, "emphasis": {"c": c, "marker": m, "s": 5, "alpha": 0.8}}

r_args = get_args("red", "o")
b_args = get_args("blue", "s")
g_args = get_args("green", "d")
p_args = get_args("purple", "<")

tmax = 5

# dt = 0.16
# ts = np.arange(0, tmax) * dt
# vd = -0.001266
# zd = ts * vd

prefix = "./tests/chin_output/chin_gyration"

data = read_text(f"{prefix}_M2A.txt")
plot_parametric(data["x"][:tmax], data["y"][:tmax], **r_args)
data = read_text(f"{prefix}_M2B.txt")
plot_parametric(data["x"][:tmax], data["y"][:tmax], **p_args)
data = read_text(f"{prefix}_B2B.txt")
plot_parametric(data["x"][:tmax], data["y"][:tmax], **b_args)
data = read_text(f"{prefix}_C2A.txt")
plot_parametric(data["x"][:tmax], data["y"][:tmax], **g_args)

phi = np.linspace(0, 2 * np.pi)
rg = 0.5
plot_parametric(rg * np.cos(phi), rg * np.sin(phi), c="black", lw=1, alpha=0.8)


plt.title("Gyro orbits")
plt.grid(c="grey", alpha=0.6)
plt.xlim((-0.65, 0.65))
plt.ylim((-0.65, 0.65))
plt.show()