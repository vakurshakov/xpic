#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: magnetic mirror. Particle should be reflected at mirror\n"
  "points, center stays between plugs.\n";

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 20.0;

PetscReal get_B(PetscReal z) {
  return B_min + (B_max - B_min) * ((z-L) * (z-L)) / (L * L);
}

PetscReal get_gradB(PetscReal z) {
  return 2.0 * (B_max - B_min) * (z-L) / (L * L);
};

void get_analytical_fields(
  const Vector3R&, const Vector3R& rn, Vector3R&, Vector3R& B_p, Vector3R& gradB_p)
{
  B_p = Vector3R{0, 0, get_B(rn.z())};
  gradB_p = Vector3R{0, 0, get_gradB(rn.z())};
}

using namespace drift_kinetic_test_utils;


int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  PetscReal d = 0.1;
  dt = omega_dt / B_min;
  geom_nt = 150'000;
  diagnose_period = geom_nt / 4;
  geom_nx = 5;
  geom_ny = 5;
  geom_nz = 4*L/d;
  dx = d;
  dy = d;
  dz = d;

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
    [&](PetscInt, PetscInt, PetscInt k, Vector3R& E_cell, Vector3R& B_cell, Vector3R& gradB_cell) {
      const PetscReal z_bz = k * dz;
      E_cell = {0.0, 0.0, 0.0};
      B_cell = {0.0, 0.0, get_B(z_bz)};
      gradB_cell = {0.0, 0.0, get_gradB(z_bz)};
    }));

  Vector3R*** E_arr, ***B_arr, ***gradB_arr;
  PetscCall(DMDAVecGetArrayRead(world.da, E_vec, &E_arr));
  PetscCall(DMDAVecGetArrayRead(world.da, B_vec, &B_arr));
  PetscCall(DMDAVecGetArrayRead(world.da, gradB_vec, &gradB_arr));

  std::unique_ptr<DriftKineticEsirkepov> esirkepov =
    std::make_unique<DriftKineticEsirkepov>(E_arr, B_arr, nullptr, gradB_arr);


  constexpr Vector3R r0(2.0, 2.0, L);
  constexpr PetscReal v_perp = 1.0;
  constexpr PetscReal v_par = 1.0;
  constexpr Vector3R v0(v_perp, 0.0, v_par);
  Point point_init(r0+correction::rho(v0, {0.0, 0.0, B_min}, q/m), v0);
  PointByField point_analytical(point_init, {0.0, 0.0, B_min}, m, q/m);
  PointByField point_grid(point_init, {0.0, 0.0, B_min}, m, q/m);

  DriftComparisonStats stats;

  DriftKineticPush push_analytical;
  push_analytical.set_qm(q/m);
  push_analytical.set_mp(m);
  push_analytical.set_fields_callback(get_analytical_fields);

  DriftKineticPush push_grid;
  push_grid.set_qm(q/m);
  push_grid.set_mp(m);
  push_grid.set_fields_callback([&](const Vector3R& r0, const Vector3R& rn, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p) {
    esirkepov->interpolate(E_p, B_p, gradB_p, rn, r0);
  });

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace_analytical(__FILE__, id + "_analytical", point_analytical, geom_nt / 1000);
  PointByFieldTrace trace_grid(__FILE__, id + "_grid", point_grid, geom_nt / 1000);

  PetscReal z_max = L;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_analytical_old = point_analytical;
    const PointByField point_grid_old = point_grid;

    PetscCall(trace_analytical.diagnose(t));
    PetscCall(trace_grid.diagnose(t));

    push_analytical.process(dt, point_analytical, point_analytical_old);
    push_grid.process(dt, point_grid, point_grid_old);

    PetscCheck(std::abs(point_analytical.r.z()-L) <= z_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle must not escape magnetic mirror! z = %.6e, allowed max = %.6e", point_analytical.r.z(), z_max);

    PetscCheck(std::abs(point_grid.r.z()-L) <= z_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Grid particle must not escape magnetic mirror! z = %.6e, allowed max = %.6e", point_grid.r.z(), z_max);

    const PetscReal position_error = (point_analytical.r - point_grid.r).length();
    stats.max_position_error = std::max(stats.max_position_error, position_error);

    Vector3R E_analytical;
    Vector3R B_analytical;
    Vector3R gradB_analytical;
    Vector3R E_grid;
    Vector3R B_grid;
    Vector3R gradB_grid;

    get_analytical_fields(point_analytical_old.r, point_analytical.r,
      E_analytical, B_analytical, gradB_analytical);
    esirkepov->interpolate(E_grid, B_grid, gradB_grid, point_grid.r, point_grid_old.r);

    const PetscReal error_B = (B_analytical - B_grid).length();
    const PetscReal error_gradB = (gradB_analytical - gradB_grid).length();

    stats.max_field_error_B = std::max(stats.max_field_error_B, error_B);
    stats.max_field_error_gradB = std::max(stats.max_field_error_gradB, error_gradB);
  }

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

  PetscCall(print_drift_statistics(stats));

  PetscCall(world.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
