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

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dx = 0.1;
  dt = omega_dt / B_min;
  geom_nx = (PetscInt)(2 * L / dx);

  World::set_geometry(geom_nx, geom_nx, geom_nx, 2000, dx, dx, dx, dt, dt);

  FieldContext context;

  PetscCall(context.initialize([&](PetscInt i, PetscInt j, PetscInt k, Vector3R& E_g, Vector3R& B_g, Vector3R& gradB_g) {
    Vector3R r(i * dx, j * dy, k * dz);
    get_analytical_fields(r, r, E_g, B_g, gradB_g);
  }));

  DriftKineticEsirkepov esirkepov(
    context.E_arr, context.B_arr, nullptr, context.gradB_arr);

  constexpr PetscReal v_par = 0.1 / M_SQRT2;
  constexpr PetscReal v_perp = 0.1 / M_SQRT2;
  constexpr Vector3R r0(Rc + 0.1, Rc, L);
  constexpr Vector3R v0(v_perp, 0.0, v_par);
  Point point_init(r0, v0);

  Vector3R B0{
    -0.5 * (r0.x() - Rc) * get_dBz_dz(r0.z() - L),
    -0.5 * (r0.y() - Rc) * get_dBz_dz(r0.z() - L),
    get_Bz_corr(r0),
  };

  PointByField point_analytical(point_init, B0, m, q / m);
  PointByField point_grid(point_init, B0, m, q / m);
  Point point_boris(point_init);

  ComparisonStats stats;
  stats.ref_mu = point_grid.mu();
  stats.ref_energy = get_kinetic_energy(point_grid);

  DriftKineticPush push_analytical;
  push_analytical.set_qm(q / m);
  push_analytical.set_mp(m);
  push_analytical.set_fields_callback(get_analytical_fields);

  DriftKineticPush push_grid;
  push_grid.set_qm(q / m);
  push_grid.set_mp(m);
  push_grid.set_fields_callback(
    [&](const Vector3R& r0_local, const Vector3R& rn_local, Vector3R& E_p,
      Vector3R& B_p, Vector3R& gradB_p) {
      esirkepov.interpolate(E_p, B_p, gradB_p, rn_local, r0_local);
    });

  BorisPush push_boris;
  push_boris.set_qm(q / m);

  TraceTriplet trace(__FILE__, std::format("omega_dt_{:.4f}", omega_dt),
    geom_nt / 1000, point_analytical, point_grid, point_boris);

  Vector3R E_analytical, B_analytical, gradB_analytical;
  Vector3R E_grid, B_grid, gradB_grid;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_analytical_old = point_analytical;
    const PointByField point_grid_old = point_grid;

    PetscCall(trace.diagnose(t));

    push_analytical.process(dt, point_analytical, point_analytical_old);
    push_grid.process(dt, point_grid, point_grid_old);
    boris_step(push_boris, point_boris, get_analytical_fields);

    get_analytical_fields(point_analytical_old.r, point_analytical.r,
      E_analytical, B_analytical, gradB_analytical);

    esirkepov.interpolate(
      E_grid, B_grid, gradB_grid, point_grid.r, point_grid_old.r);

    update_comparison_stats(stats,  //
      point_analytical, point_grid, point_boris,  //
      B_analytical, gradB_analytical, B_grid, gradB_grid);
  }

  PetscCall(print_statistics(stats, point_analytical, point_grid, point_boris));

  PetscCall(context.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
