#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "axes.formatter.use_mathtext": True})

# from configuration import config

def plot_evolution(data):
    plt.plot(np.arange(len(data)), data)

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


m_args = {"c": "red", "lw": 1, "emphasis": {"c": "red", "marker": "o", "s": 5}}

data = read_text("./tests/chin_gyration_M1A.txt")
plot_parametric(data["x"], data["y"], **m_args)

data = read_text("./tests/chin_gyration_M1B.txt")
plot_parametric(data["x"], data["y"], **m_args)


b_args = {"c": "blue", "lw": 1, "emphasis": {"c": "blue", "marker": "s", "s": 5}}

data = read_text("./tests/chin_gyration_B1A.txt")
plot_parametric(data["x"], data["y"], **b_args)

data = read_text("./tests/chin_gyration_B1B.txt")
plot_parametric(data["x"], data["y"], **b_args)


plt.title("Gyro orbits")
plt.grid(c="grey", alpha=0.6)
plt.xlim((-1.25, +1.25))
plt.ylim((-1.25, +1.25))
plt.show()