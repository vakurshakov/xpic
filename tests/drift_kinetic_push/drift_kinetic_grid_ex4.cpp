#include "drift_kinetic_push.h"

#include <algorithm>

static constexpr char help[] =
  "Test: magnetic mirror. Particle should be reflected at mirror\n"
  "points, center stays between plugs.\n";

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 20.0;
constexpr PetscReal q = 1.0;
constexpr PetscReal m = 1.0;

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

struct TestStatistics {
  PetscInt total_steps = 0;
  PetscReal simulation_time = 0.0;

  PetscReal max_position_error = 0.0;
  PetscReal max_field_error_B = 0.0;
  PetscReal max_field_error_gradB = 0.0;

  Vector3R final_position_analytical;
  Vector3R final_position_grid;
};

PetscErrorCode print_statistics(const TestStatistics& stats)
{
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n=== TEST STATISTICS ===\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Simulation time: %.6e\n", stats.simulation_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Total steps: %d\n", stats.total_steps));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nField comparison errors:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max B field error:     %.8e\n", stats.max_field_error_B));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max gradB field error: %.8e\n", stats.max_field_error_gradB));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nTrajectory comparison:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max position error:    %.8e\n", stats.max_position_error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Final pos analytical:  (%.6e %.6e %.6e)\n",
           REP3_A(stats.final_position_analytical)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Final pos grid:        (%.6e %.6e %.6e)\n",
           REP3_A(stats.final_position_grid)));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode initialize_grid_fields(DM da, const Vector3R& dr, Vec E_vec, Vec B_vec, Vec gradB_vec)
{
  PetscFunctionBeginUser;

  Vector3R*** E_arr, ***B_arr, ***gradB_arr;

  PetscCall(DMDAVecGetArrayWrite(da, E_vec, &E_arr));
  PetscCall(DMDAVecGetArrayWrite(da, B_vec, &B_arr));
  PetscCall(DMDAVecGetArrayWrite(da, gradB_vec, &gradB_arr));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  for (PetscInt k = start[Z]; k < start[Z] + size[Z]; k++) {
    for (PetscInt j = start[Y]; j < start[Y] + size[Y]; j++) {
      for (PetscInt i = start[X]; i < start[X] + size[X]; i++) {
        const PetscReal z_bz = (k) * dr.z();

        E_arr[k][j][i] = Vector3R{0.0, 0.0, 0.0};

        B_arr[k][j][i].x() = 0.0;
        B_arr[k][j][i].y() = 0.0;
        B_arr[k][j][i].z() = get_B(z_bz);

        gradB_arr[k][j][i].x() = 0.0;
        gradB_arr[k][j][i].y() = 0.0;
        gradB_arr[k][j][i].z() = get_gradB(z_bz);
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, E_vec, &E_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, B_vec, &B_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, gradB_vec, &gradB_arr));

  PetscFunctionReturn(PETSC_SUCCESS);
}


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

  PetscCall(initialize_grid_fields(world.da, {dx,dy,dz}, E_vec, B_vec, gradB_vec));

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
  Point point_init(r0, v0);
  PointByField point_analytical(point_init, {0.0, 0.0, B_min}, m);
  PointByField point_grid(point_init, {0.0, 0.0, B_min}, m);

  TestStatistics stats;

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


  // Maximum value of z-coordinates of the particle
  PetscReal z_max = L;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_analytical_old = point_analytical;
    const PointByField point_grid_old = point_grid;

    PetscCall(trace_analytical.diagnose(t));
    PetscCall(trace_grid.diagnose(t));

    push_analytical.process(dt, point_analytical, point_analytical_old);
    push_grid.process(dt, point_grid, point_grid_old);

    /// @todo Check that `mu_p` is conserved in a mirror

    PetscCheck(std::abs(point_analytical.r.z()-L) <= z_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle must not escape magnetic mirror! z = %.6e, allowed max = %.6e", point_analytical.r.z(), z_max);

    PetscCheck(std::abs(point_grid.r.z()-L) <= z_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Grid particle must not escape magnetic mirror! z = %.6e, allowed max = %.6e", point_grid.r.z(), z_max);

    const PetscReal position_error = (point_analytical.r - point_grid.r).length();
    stats.max_position_error = std::max(stats.max_position_error, position_error);

    Vector3R E_analytical, B_analytical, gradB_analytical;
    Vector3R E_grid, B_grid, gradB_grid;

    get_analytical_fields(point_analytical_old.r, point_analytical.r, E_analytical, B_analytical, gradB_analytical);
    esirkepov->interpolate(E_grid, B_grid, gradB_grid, point_grid.r, point_grid_old.r);

    const PetscReal error_B = (B_analytical - B_grid).length();
    const PetscReal error_gradB = (gradB_analytical - gradB_grid).length();

    stats.max_field_error_B = std::max(stats.max_field_error_B, error_B);
    stats.max_field_error_gradB = std::max(stats.max_field_error_gradB, error_gradB);
  }

  stats.simulation_time = dt * geom_nt;
  stats.total_steps = geom_nt;
  stats.final_position_analytical = point_analytical.r;
  stats.final_position_grid = point_grid.r;

  PetscCall(DMDAVecRestoreArrayRead(world.da, E_vec, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(world.da, B_vec, &B_arr));
  PetscCall(DMDAVecRestoreArrayRead(world.da, gradB_vec, &gradB_arr));

  PetscCall(VecDestroy(&E_vec));
  PetscCall(VecDestroy(&B_vec));
  PetscCall(VecDestroy(&gradB_vec));

  PetscCall(print_statistics(stats));

  PetscCall(world.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
