#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic Esirkepov vs analytical: comparison of fields and trajectories\n"
  "calculated by drift_kinetic_push on analytical fields vs on grid filled with\n"
  "analytical fields. Controls boundary violations and particle displacement < cell size.\n";

constexpr Vector3R E0(0.0, 0.0, 0.0);
constexpr Vector3R B0(0.0, 0.0, 2.0);
constexpr Vector3R gradB0(1.0, 0.0, 0.0);

void get_analytical_fields(const Vector3R&, const Vector3R& rn, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = E0;
  B_p = B0 + (rn-Vector3R(2.0,2.0,2.0)).elementwise_product(gradB0);
  gradB_p = gradB0;
}

using namespace drift_kinetic_test_utils;

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / B0.length();
  geom_nt = 100;
  diagnose_period = geom_nt / 1;
  geom_nx = 20;
  geom_ny = 20;
  geom_nz = 20;
  dx = 1.;
  dy = 1.;
  dz = 1.;


  World::set_geometry(
    geom_nx, geom_ny, geom_nz, geom_nt,
    dx, dy, dz, dt,
    diagnose_period);

  World world;
  PetscCall(world.initialize());

  Vec E_vec, B_vec, gradB_vec;
  PetscCall(DMCreateGlobalVector(world.da, &E_vec));
  PetscCall(DMCreateGlobalVector(world.da, &B_vec));
  PetscCall(DMCreateGlobalVector(world.da, &gradB_vec));

  PetscCall(initialize_grid_fields(
    world.da,
    E_vec,
    B_vec,
    gradB_vec,
    [&](PetscInt i, PetscInt j, PetscInt k, Vector3R& E_cell, Vector3R& B_cell, Vector3R& gradB_cell) {
      Vector3R rn((i - 2) * dx, (j - 2) * dy, (k - 2) * dz);
      E_cell = E0;
      B_cell = B0 + rn.elementwise_product(gradB0);
      gradB_cell = gradB0;
    }));

  Vector3R*** E_arr, ***B_arr, ***gradB_arr;
  PetscCall(DMDAVecGetArrayRead(world.da, E_vec, &E_arr));
  PetscCall(DMDAVecGetArrayRead(world.da, B_vec, &B_arr));
  PetscCall(DMDAVecGetArrayRead(world.da, gradB_vec, &gradB_arr));

  std::unique_ptr<DriftKineticEsirkepov> esirkepov =
    std::make_unique<DriftKineticEsirkepov>(E_arr, B_arr, nullptr, gradB_arr);



  constexpr Vector3R r0(2.0, 2.0, 2.0);
  constexpr Vector3R v0(1.0, 0.0, 1.0);
  Point point_init(r0, v0);
  PointByField point_analytical(point_init, B0, 1.0);
  PointByField point_grid(point_init, B0, 1.0);

  DriftKineticPush push_analytical;
  push_analytical.set_qm(1.0);
  push_analytical.set_mp(1.0);
  push_analytical.set_fields_callback(get_analytical_fields);

  DriftKineticPush push_grid;
  push_grid.set_qm(q/m);
  push_grid.set_mp(m);
  push_grid.set_fields_callback([&](const Vector3R& r0, const Vector3R& rn, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p) {
    esirkepov->interpolate(E_p, B_p, gradB_p, rn, r0);
  });

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace_analytical(__FILE__, id + "_analytical", point_analytical, geom_nt / 10);
  PointByFieldTrace trace_grid(__FILE__, id + "_grid", point_grid, geom_nt / 10);

  Vector3R domain_min(0.0, 0.0, 0.0);
  Vector3R domain_max(geom_nx*dx, geom_ny*dy, geom_nz*dz);
  PetscReal min_cell_size = std::min({2 * dx, 2 * dy, 2 * dz});

  FieldComparisonStats stats;
  constexpr PetscReal field_tolerance = 1e-10;
  constexpr PetscReal position_tolerance = 1e-6;

  Vector3R start_r = point_analytical.r;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_analytical_old = point_analytical;
    const PointByField point_grid_old = point_grid;

    PetscCall(trace_analytical.diagnose(t));
    PetscCall(trace_grid.diagnose(t));

    push_analytical.process(dt, point_analytical, point_analytical_old);
    push_grid.process(dt, point_grid, point_grid_old);

    PetscCall(update_grid_comparison_step(
      point_analytical,
      point_analytical_old,
      point_grid,
      point_grid_old,
      push_analytical.get_iteration_number(),
      push_grid.get_iteration_number(),
      min_cell_size,
      domain_min,
      domain_max,
      position_tolerance,
      field_tolerance,
      stats,
      [&](const PointByField& old_point, const PointByField& new_point,
          Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p) {
        get_analytical_fields(old_point.r, new_point.r, E_p, B_p, gradB_p);
        return PETSC_SUCCESS;
      },
      [&](const PointByField& old_point, const PointByField& new_point,
          Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p) {
        esirkepov->interpolate(E_p, B_p, gradB_p, new_point.r, old_point.r);
        return PETSC_SUCCESS;
      }));

    if (t % (geom_nt / 10) == 0) {
      PetscCall(log_progress(id.c_str(), t, geom_nt, stats.max_position_error));
    }
  }

  PetscReal m = 1.0;
  PetscReal q = 1.0;
  PetscReal B = B0.length();

  Vector3R V_gradB;
  V_gradB += Vector3R(0.0, 0.0, v0.z());
  V_gradB += POW2(v0.z()) / POW3(B) * B0.cross(gradB0);
  V_gradB += point_analytical.mu_p / (m * q * POW2(B)) * B0.cross(gradB0);

  PetscReal T = dt * (PetscReal)(geom_nt + 1);
  Vector3R r_theory = start_r + V_gradB * T;


  PetscCheck(equal_tol(point_analytical.r, r_theory, 1e-8), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Analytical guiding center must drift with grad-B drift. Result: (%.8e %.8e %.8e), theory: (%.8e %.8e %.8e)", REP3_A(point_analytical.r), REP3_A(r_theory));

  PetscCheck(equal_tol(point_grid.r, r_theory, 1e-8), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Grid guiding center must drift with grad-B drift. Result: (%.8e %.8e %.8e), theory: (%.8e %.8e %.8e)", REP3_A(point_grid.r), REP3_A(r_theory));

  PetscCall(DMDAVecRestoreArrayRead(world.da, E_vec, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(world.da, B_vec, &B_arr));
  PetscCall(DMDAVecRestoreArrayRead(world.da, gradB_vec, &gradB_arr));

  PetscCall(VecDestroy(&E_vec));
  PetscCall(VecDestroy(&B_vec));
  PetscCall(VecDestroy(&gradB_vec));

  stats.simulation_time = dt * geom_nt;
  stats.total_steps = geom_nt;
  stats.final_position_analytical = point_analytical.r;
  stats.final_position_grid = point_grid.r;

  PetscCall(print_field_statistics(stats));

  PetscCall(world.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
