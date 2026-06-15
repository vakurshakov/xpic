#!/usr/bin/env python3

from plot import *

config = select_subdirectory() if len(sys.argv) == 1 else sys.argv[1]

init_constants(f"results/{config}/{config}.json")
const.tau = read_scalar(const.config["StepPresets"][3]["tau"])
const.mi_me = read_scalar(const.config["Particles"][0]["m"])
const.pts = 0
# const.pte = 20 * const.Ndts
const.pto = const.Ndts

vmap = 0.002 * np.array([-1, +1])

fig, gs = figure(3, 1, figsize=(24, 8))#, figsize=())

def plot(j, title, vmap):
    plot = PlotLinear(subplot(fig, gs, j, 0), vmap)
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
fp = plot(2, "\\rm Total pressures", (0, 0))
fp.info.set_args(
    ylim=(0, 0.025),
    yticks=np.linspace(0, 0.025, 6),
)

def avg1(d): return sliding_average(d, 3)
def avg2(d): return sliding_average(d, 8)

d_c = const.Ny // 2
d_w = 5
def parse(r, t): return r(t).mean(axis=0)

ntw = 6
tw = ntw * const.pto
dfts = np.ndarray((ntw, const.Nx))

def callback(t):
    ex = parse(read_E(0), t)
    bz = parse(read_B(2), t)

    ni = parse(read_n("ions"), t) * (+1)
    jxi = parse(read_J("ions", 0), t)
    jyi = parse(read_J("ions", 1), t)
    pxi = parse(read_P("ions", 0), t)

    ne = parse(read_n("electrons"), t) * (-1)
    jxe = parse(read_J("electrons", 0), t)
    jye = parse(read_J("electrons", 1), t)
    pxe = parse(read_P("electrons", 0), t)

    dx = const.dx
    xs = (np.arange(-const.Nx // 2, const.Nx // 2) + 0.5) * dx
    xs1 = avg1(xs)
    xs2 = avg2(xs)

    def impl(f, n, jy, px):
        ax = f.axis
        dfp = np.interp(xs, xs1, np.gradient(avg1(px), dx))
        dfe = np.interp(xs, xs2, avg2(n * ex))
        dfb = jy * bz
        ax.plot(xs, dfp, label=f"$\\partial_x \\Pi_{{xx}}$", lw=2)
        ax.plot(xs, dfe, label=f"$e n E_x$", lw=2)
        ax.plot(xs, dfb, label=f"$J_y B_z$", lw=2)
        ax.plot(xs, -dfp + dfe + dfb, label=f"$m \\partial_t (n v_x)$", lw=2)

    impl(fi, ni, jyi, pxi)
    impl(fe, ne, jye, pxe)

    def symint(d):
        r = np.zeros_like(xs)
        c = r.shape[0] // 2
        r[:c] = np.cumsum(d[:c])
        r[:c] -= r[c-1]
        r[c:] = -np.cumsum(d[c:][::-1])[::-1]
        r[c:] -= r[c]
        r -= (r[0] + r[-1]) / 2
        return r

    ax = fp.axis
    dfp = pxi + pxe
    dfb = symint((jyi + jye) * bz)
    dfti = -symint(jxi) * const.mi_me
    dfte = +symint(jxe)
    ax.plot(xs, dfp, label=f"$\\Pi_{{xx}}$", lw=2)
    ax.plot(xs, dfb, label=f"$\\int J_y B_z \\, dx$", lw=2)
    # ax.plot(xs, (dfp - dfb), label=f"$\\Pi_{{xx}} - \\int J_y B_z \\, dx$", lw=2)

    '''
    i = (t % tw) // const.pto
    dfts[i] = dfti + dfte
    diff = np.ndarray((ntw-1, const.Nx))
    
    for j in range(ntw-1):
        diff[j] = dfts[(i-j)%ntw] - dfts[(i-j-1)%ntw]
    
    ax.plot(xs, dfti, label=f"$-\\int (n v_x)_i \\, dx$", lw=2, color="C3")
    
    ax.plot(xs, diff.mean(axis=0), label=f"$-\\int m \\partial_t (n v_x) \\, dx$", lw=2, color="C4")
    ax.fill_between(xs, np.min(diff, axis=0), np.max(diff, axis=0), alpha=0.2, color="C4")
    '''

    for f in [fi, fe, fp]: 
        f.draw_info()
        f.axis.grid()
        f.axis.legend(fontsize=0.9*FontSize.label, loc="upper right")

process_plots(fig, "forces", time_tau, (fi, fe, fp), callback)
