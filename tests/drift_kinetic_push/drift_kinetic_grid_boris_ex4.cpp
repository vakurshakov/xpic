#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: magnetic mirror with double-Gaussian mirrors.\n"
  "Guiding-center push (analytic/grid) should match the Boris pusher.\n";

using namespace drift_kinetic_test_utils;
using namespace gaussian_magnetic_mirror;

void get_analytical_fields(const Vector3R&, const Vector3R& rn, Vector3R& E_p,
  Vector3R& B_p, Vector3R& gradB_p)
{
  get_fields(rn, rn, E_p, B_p, gradB_p);
  E_p = {0.0, 0.0, 0.0};
}

void get_grid_fields(const Vector3R&, const Vector3R& rn, Vector3R& E_p,
  Vector3R& B_p, Vector3R& gradB_p)
{
  get_grid(rn, rn, E_p, B_p, gradB_p);
  E_p = {0.0, 0.0, 0.0};
} 

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dx = 0.00125;
  dt = omega_dt / B_min;
  geom_nx = (PetscInt )(2 * L / dx);

  overwrite_config(2 * Rc, 2 * Rc, 2 * L, 10*dt, dx, dx, dx, dt, dt);

  FieldContext context;

  PetscCall(context.initialize([&](PetscInt i, PetscInt j, PetscInt k, Vector3R& E_g, Vector3R& B_g, Vector3R& gradB_g) {
    Vector3R r(i * dx, j * dy, k * dz);
    get_grid_fields(r, r, E_g, B_g, gradB_g);
  }));

  DriftKineticEsirkepov esirkepov(
    context.E_arr, context.B_arr, nullptr, nullptr);

  esirkepov.set_dBidrj(context.dBdx_arr, context.dBdy_arr, context.dBdz_arr);

  constexpr PetscReal v_par = 0.01 / M_SQRT2;
  constexpr PetscReal v_perp = 0.01 / M_SQRT2;
  constexpr Vector3R r0(Rc + 0.02, Rc, L);
  constexpr Vector3R v0(v_perp, 0.0, v_par);
  Point point_init(r0, v0);

  auto f_anal = [&](const Vector3R& r0, const Vector3R& rn, Vector3R& E_p,
      Vector3R& B_p, Vector3R& gradB_p){
        E_p = {};
        B_p = {};
        gradB_p = {};
        Vector3R Es_p, Bs_p, gradBs_p;
        Vector3R E_dummy, B_dummy, gradB_dummy;

        Vector3R pos = (rn - r0);

        auto coords = cell_traversal_new(rn, r0);
        PetscInt Nsegments = (PetscInt)coords.size() - 1;

        pos[X] = pos[X] != 0 ? pos[X] : Nsegments;
        pos[Y] = pos[Y] != 0 ? pos[Y] : Nsegments;
        pos[Z] = pos[Z] != 0 ? pos[Z] : Nsegments;

        get_analytical_fields(r0, rn, E_dummy, B_p, gradB_dummy);

      for (PetscInt s = 1; s < (PetscInt)coords.size(); ++s) {
        auto&& rs0 = coords[s - 1];
        auto&& rsn = coords[s - 0];

        get_analytical_fields(rs0, rsn, Es_p, B_dummy, gradBs_p);

        Vector3R drs{
          rsn[X] != rs0[X] ? rsn[X] - rs0[X] : 1.0,
          rsn[Y] != rs0[Y] ? rsn[Y] - rs0[Y] : 1.0,
          rsn[Z] != rs0[Z] ? rsn[Z] - rs0[Z] : 1.0,
        };

        E_p += Es_p.elementwise_product(drs);
        gradB_p += gradBs_p.elementwise_product(drs);
      }
      E_p = E_p.elementwise_division(pos);
      gradB_p = gradB_p.elementwise_division(pos);
      };

    auto f_grid = [&](const Vector3R& r0, const Vector3R& rn, //
      Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p) {
      E_p = {};
      B_p = {};
      gradB_p = {};
      Vector3R Es_p, Bs_p, gradBs_p;
      Vector3R E_dummy, B_dummy, gradB_dummy;

      Vector3R pos = (rn - r0);

      auto coords = cell_traversal_new(rn, r0);
      PetscInt Nsegments = (PetscInt)coords.size() - 1;

      pos[X] = pos[X] != 0 ? pos[X] / dx : Nsegments;
      pos[Y] = pos[Y] != 0 ? pos[Y] / dy : Nsegments;
      pos[Z] = pos[Z] != 0 ? pos[Z] / dz : Nsegments;

      esirkepov.interpolate(E_dummy, B_p, gradB_dummy, rn, r0);

      for (PetscInt s = 1; s < (PetscInt)coords.size(); ++s) {
        auto&& rs0 = coords[s - 1];
        auto&& rsn = coords[s - 0];

        esirkepov.interpolate(Es_p, B_dummy, gradBs_p, rsn, rs0);

        E_p += Es_p;
        gradB_p += gradBs_p;
      }
      E_p = E_p.elementwise_division(pos);
      gradB_p = gradB_p.elementwise_division(pos);
    };

  Vector3R B0{
    -0.5 * (r0.x() - Rc) * get_dBz_dz(r0.z() - L),
    -0.5 * (r0.y() - Rc) * get_dBz_dz(r0.z() - L),
    get_Bz_corr(r0),
  };

  Vector3R B0_anal = B0;
  PointByField point_analytical(point_init, B0_anal, m, q / m);
  Vector3R B0_grid = B0;
  PointByField point_grid(point_init, B0_grid, m, q / m);
  Point point_boris(point_init);

  ComparisonStats stats;
  stats.ref_mu = point_grid.mu();
  stats.ref_energy = get_kinetic_energy(point_grid);

  LOG("B0_anal: ({:.6e} {:.6e} {:.6e}) |B0_anal|={:.6e}", REP3_A(B0_anal), B0_anal.length());
  LOG("B0_grid: ({:.6e} {:.6e} {:.6e}) |B0_grid|={:.6e}", REP3_A(B0_grid), B0_grid.length());
  LOG("mu_anal={:.6e} mu_grid={:.6e}", point_analytical.mu(), point_grid.mu());

  DriftKineticPush push_analytical;
  push_analytical.set_qm(q / m);
  push_analytical.set_mp(m);
  push_analytical.set_fields_callback(f_anal);

  DriftKineticPush push_grid;
  push_grid.set_qm(q / m);
  push_grid.set_mp(m);
  push_grid.set_fields_callback(f_grid);

  BorisPush push_boris;
  push_boris.set_qm(q / m);

  TraceTriplet trace(__FILE__, std::format("omega_dt_{:.4f}", omega_dt),
    1., point_analytical, point_grid, point_boris);

  Vector3R E_analytical, B_analytical, gradB_analytical;
  Vector3R E_grid, B_grid, gradB_grid;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_analytical_old = point_analytical;
    const PointByField point_grid_old = point_grid;

    PetscCall(trace.diagnose(t));

    push_analytical.process(dt, point_analytical, point_analytical_old);
    push_grid.process(dt, point_grid, point_grid_old);
    boris_step(push_boris, point_boris, get_analytical_fields);

    f_anal(point_analytical_old.r, point_analytical.r,
      E_analytical, B_analytical, gradB_analytical);

    f_grid(point_grid_old.r, point_grid.r, E_grid, B_grid, gradB_grid);

    if (t < 20) {
      Vector3R b_analytical = B_analytical.normalized();
      Vector3R b_grid = B_grid.normalized();

      PetscReal bgradB_analytical = b_analytical.dot(gradB_analytical);
      PetscReal bgradB_grid = b_grid.dot(gradB_grid);

      PetscReal Vh_analytical = 0.5 * (point_analytical_old.p_parallel + point_analytical.p_parallel);
      PetscReal Vh_grid = 0.5 * (point_grid_old.p_parallel + point_grid.p_parallel);

      LOG("t={} b·gradB: anal={:.6e} grid={:.6e} b·Vp(=Vh): anal={:.6e} grid={:.6e}",
        t, bgradB_analytical, bgradB_grid, Vh_analytical, Vh_grid);
    }

    if (t < 20) {
      Vector3R b_analytical = B_analytical.normalized();
      Vector3R b_grid = B_grid.normalized();
      PetscReal bgradB_analytical = b_analytical.dot(gradB_analytical);
      PetscReal bgradB_grid = b_grid.dot(gradB_grid);
      LOG("t={} b·gradB: anal={:.6e} grid={:.6e}", t, bgradB_analytical, bgradB_grid);
    }

    update_comparison_stats(stats,  //
      point_analytical, point_grid, point_boris,  //
      B_analytical, gradB_analytical, B_grid, gradB_grid);
  }

  PetscCall(print_statistics(stats, point_analytical, point_grid, point_boris));

  PetscCall(context.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
