#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "axes.formatter.use_mathtext": True})

# from configuration import config

def plot_evolution(t, data):
    plt.plot(t, data)

def plot_parametric(x_data, y_data, **kwargs):
    if ["emphasis" in kwargs]:
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

tmax = 1000
dt = 0.16

vd = -0.001266

ts = np.arange(0, tmax) * dt
zd = ts * vd

data = read_text("./tests/chin_curvilinear_b_B1A.txt")
plt.plot(ts, data["z"][:tmax] - zd, alpha=0.8, c="green") #, **g_args["emphasis"]) # data["y"][:tmax],
data = read_text("./tests/chin_curvilinear_b_B1B.txt")
plt.plot(ts, data["z"][:tmax] - zd, alpha=0.8, c="red") #, **r_args["emphasis"]) # data["y"][:tmax],
data = read_text("./tests/chin_curvilinear_b_BLF.txt")
plt.plot(ts, data["z"][:tmax] - zd, alpha=0.8, c="purple") #, **p_args["emphasis"]) # data["y"][:tmax],


plt.title("Gyro orbits")
plt.grid(c="grey", alpha=0.6)
# plt.xlim((10-15, 10+15))
# plt.ylim((-1, +0.5))
plt.xlim((0, 10))
plt.ylim((-0.015, +0.020))
plt.show()