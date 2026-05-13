#!/usr/bin/env python3

from plot import *

config = f"{sys.argv[1]}_dx3.0_dt1.5_Np1000" # ecsimcorr
init_constants(f"results/{config}/{config}.json")
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])
const.min_x = read_scalar(const.config["StepPresets"][0]["geometry"]["min"][0]) - const.Lx / 2
const.max_x = read_scalar(const.config["StepPresets"][0]["geometry"]["max"][0]) - const.Lx / 2
const.min_y = read_scalar(const.config["StepPresets"][0]["geometry"]["min"][1]) 
const.max_y = read_scalar(const.config["StepPresets"][0]["geometry"]["max"][1])
const.pto = 2 * const.Ndts

def read(file_t, c, cd):
    data_shape = [const.Ny, const.Nx, cd]
    with open(file_t, "rb") as file:
        raw = np.fromfile(file, dtype=np.float32, count=np.prod(data_shape), offset=0)
        raw = np.reshape(raw, data_shape)
    return raw[:,:,c]

def read_E(c): return lambda t: read(f"{const.in_dir}/E_ZAvg/{str(t // const.Ndts).zfill(4)}", c, 3)
def read_B(c): return lambda t: read(f"{const.in_dir}/B_ZAvg/{str(t // const.Ndts).zfill(4)}", c, 3)
def read_J(s, c): return lambda t: read(f"{const.in_dir}/{s}/J_ZAvg/{str(t // const.Ndts).zfill(4)}", c, 3)
def read_P(s, c): return lambda t: read(f"{const.in_dir}/{s}/momentum_Z2/{str(t // const.Ndts).zfill(4)}", c, 6)
def read_n(s):    return lambda t: read(f"{const.in_dir}/{s}/rho_ZAvg/{str(t // const.Ndts).zfill(4)}", 0, 1) * (1 if s == "ions" else -1)

vmap = 0.002 * np.array([-1, +1])

fig, gs = figure(1, 2)#, figsize=())

def plot(j, title, vmap):
    plot = PlotLinear(subplot(fig, gs, 0, j), vmap)
    bx = -const.Lx / 2
    ex = +const.Lx / 2
    plot.info.set_args(
        title=title,
        xlim=(bx, ex),
        ylim=vmap,
        xticks=np.linspace(bx, ex, 5),
        yticks=np.linspace(vmap[0], vmap[1], 5),
        xlabel='$x,~c/\\omega_{pe}$',
    )
    return plot

fi = plot(0, "\\rm Forces on ions", vmap)
fe = plot(1, "\\rm Forces on electrons", vmap)

def avg1(d): return sliding_average(d, 3)
def avg2(d): return sliding_average(d, 8)

d_c = const.Ny // 2
d_w = 5
def parse(r, t): return r(t).mean(axis=0) # [d_c-d_w:d_c+d_w,:]

def callback(t):
    ex = parse(read_E(0), t)
    bz = parse(read_B(2), t)

    ni = parse(read_n("ions"), t)
    jyi = parse(read_J("ions", 1), t)
    pxi = parse(read_P("ions", 0), t)

    ne = parse(read_n("electrons"), t)
    jye = parse(read_J("electrons", 1), t)
    pxe = parse(read_P("electrons", 0), t)

    dx = const.dx
    xs = (np.arange(-const.Nx // 2, const.Nx // 2) + 0.5) * dx
    xs1 = avg1(xs)
    xs2 = avg2(xs)

    def impl(f, s, n, jy, px):
        ax = f.axis
        
        dfp = np.interp(xs, xs1, np.gradient(avg1(px), dx))
        dfe = np.interp(xs, xs2, avg2(n * ex))
        dfb = jy * bz
        ax.plot(xs, dfp, label=f"$\\partial_x \\Pi_{{xx}}$", lw=2)
        ax.plot(xs, dfe, label=f"$n E_r$", lw=2)
        ax.plot(xs, dfb, label=f"$J_{{\\phi}} B_z$", lw=2)

        ax.plot(xs, -dfp + dfe * (1 if s == "i" else -1) + dfb, label=f"$m n \\partial_t v$", lw=2)


        f.draw_info()
        f.axis.grid()
        f.axis.legend(fontsize=0.9*FontSize.label, loc="lower right")

    impl(fi, 'i', ni, jyi, pxi)
    impl(fe, 'e', ne, jye, pxe)

process_plots(fig, "forces", time_tau, (fi, fe), callback)
