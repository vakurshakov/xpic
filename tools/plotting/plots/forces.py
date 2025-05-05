#!/usr/bin/env python3

from plot import *

const.ncols=1
const.nrows=2

vmap = 0.002 * np.array([-1, +1])

def gen_view(path, plane, comp, dof, coords=FieldView.Cartesian):
    view = FieldView()
    view.path = lambda t: f"{const.input_path}/{path}/{format_time(t, const.Nt)}"
    view.region = FieldView.Region(dof, (0, 0, 0), (*const.data_shape[plane], dof))

    view.coords = coords
    if view.coords == FieldView.Cylindrical:
      view.init_cos_sin(const.cos, const.sin)

    view.plane = plane
    view.comp = comp
    return view

ni = gen_view("ions/density_planeZ_0001", 'Z', 0, 1)
jai = gen_view("ions/current_planeZ_0001", 'Z', 1, 3, FieldView.Cylindrical)
pri = gen_view("ions/momentum_flux_diag_cyl_planeZ_0001", 'Z', 0, 3)
pai = gen_view("ions/momentum_flux_diag_cyl_planeZ_0001", 'Z', 1, 3)

ne = gen_view("electrons/density_planeZ_0001", 'Z', 0, 1)
jae = gen_view("electrons/current_planeZ_0001", 'Z', 1, 3, FieldView.Cylindrical)
pre = gen_view("electrons/momentum_flux_diag_cyl_planeZ_0001", 'Z', 0, 3)
pae = gen_view("electrons/momentum_flux_diag_cyl_planeZ_0001", 'Z', 1, 3)

er = gen_view("E_planeZ_0001", 'Z', 0, 3, FieldView.Cylindrical)
bz = gen_view("B_planeZ_0001", 'Z', 2, 3, FieldView.Cylindrical)

fi = gen_linear("\\rm Forces on ions", 'Z', vmap)
fe = gen_linear("\\rm Forces on electrons", 'Z', vmap)


def phi_avg(data):
    return phi_averaged(data, const.rmap)

def callback(t):
    er_d = phi_avg(er.parse(t))
    bz_d = phi_avg(bz.parse(t))

    def avg1(d): return sliding_average(d, 3)
    def avg2(d): return sliding_average(d, 7)

    dr = const.dx
    rs = (np.arange(0, const.data_shape["Z"][0] // 2) + 0.5) * dr
    rs1 = avg1(rs)
    rs2 = avg2(rs)

    def _impl(s, n, ja, pr, pa, f):
        ni_d = phi_avg(n.parse(t))
        jai_d = phi_avg(ja.parse(t))
        pri_d = phi_avg(pr.parse(t))
        pai_d = phi_avg(pa.parse(t))

        ax = f.axis
        ax.plot(rs1, np.gradient(avg1(pri_d), dr), label="$\\partial_r \\Pi^{" + s + "}_{rr}$", linewidth=2)
        ax.plot(rs1, avg1(np.divide(pri_d - pai_d, rs)), label="$\\Delta \\Pi^{" + s + "}_{rr}$", linewidth=2)
        ax.plot(rs2, avg2(ni_d * er_d), label="$n_{" + s + "} E_r$", linewidth=2)
        ax.plot(rs, jai_d * bz_d, label="$J^{" + s + "}_{\\phi} B_z$", linewidth=2)
        ax.plot(rs2, avg2(-(np.gradient(pri_d, dr) + np.divide(pri_d - pai_d, rs)) + ni_d * er_d + jai_d * bz_d), label="\\rm res. force", linewidth=2, linestyle="--", color="red")

        f.draw_info()
        f.axis.grid()
        f.axis.legend(fontsize=ticksize * 0.8, loc="lower right")
    
    _impl('i', ni, jai, pri, pai, fi)
    _impl('e', ne, jae, pre, pae, fe)

process_plots("forces", lambda t: f"$t = {t * const.dt:.3f}$", (fi, fe), callback)