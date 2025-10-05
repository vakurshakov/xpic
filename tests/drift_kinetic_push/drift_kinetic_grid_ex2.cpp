#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic Esirkepov vs analytical: comparison of fields and trajectories\n"
  "calculated by drift_kinetic_push on analytical fields vs on grid filled with\n"
  "analytical fields. Controls boundary violations and particle displacement < cell size.\n";

constexpr Vector3R E0(0.0, 1.0, -1.0);
constexpr Vector3R B0(0.0, 0.0, 1.0);
constexpr Vector3R gradB0(0.0, 0.0, 0.0);

struct TestStatistics {
  PetscReal max_field_error_E = 0.0;
  PetscReal max_field_error_B = 0.0;
  PetscReal max_field_error_gradB = 0.0;
  PetscReal max_position_error = 0.0;
  PetscReal max_displacement_per_step = 0.0;
  PetscInt total_iterations_analytical = 0;
  PetscInt total_iterations_grid = 0;
  PetscInt boundary_violations = 0;
  PetscInt cell_size_violations = 0;
  Vector3R final_position_analytical;
  Vector3R final_position_grid;
  PetscReal simulation_time = 0.0;
  PetscInt total_steps = 0;
};

void get_analytical_fields(const Vector3R&, const Vector3R&, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = E0;
  B_p = B0;
  gradB_p = gradB0;
}

PetscErrorCode initialize_grid_fields(DM da, Vec E_vec, Vec B_vec, Vec gradB_vec)
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
        E_arr[k][j][i] = E0;
        B_arr[k][j][i] = B0;
        gradB_arr[k][j][i] = gradB0;
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, E_vec, &E_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, B_vec, &B_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, gradB_vec, &gradB_arr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_boundaries(const PointByField& point, const Vector3R& domain_min,
                               const Vector3R& domain_max, TestStatistics& stats)
{
  PetscFunctionBeginUser;

  bool violation = false;
  if (point.r.x() < domain_min.x() || point.r.x() > domain_max.x() ||
      point.r.y() < domain_min.y() || point.r.y() > domain_max.y() ||
      point.r.z() < domain_min.z() || point.r.z() > domain_max.z()) {
    violation = true;
    stats.boundary_violations++;
  }

  PetscCheck(!violation, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Particle exceeded domain boundaries at position (%.6e, %.6e, %.6e)",
    REP3_A(point.r));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_displacement(const PointByField& point_new, const PointByField& point_old,
                                 PetscReal min_cell_size, TestStatistics& stats)
{
  PetscFunctionBeginUser;

  PetscReal displacement = (point_new.r - point_old.r).length();
  stats.max_displacement_per_step = std::max(stats.max_displacement_per_step, displacement);

  if (displacement >= min_cell_size) {
    stats.cell_size_violations++;
    PetscCheck(false, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle displacement (%.6e) exceeds cell size (%.6e)", displacement, min_cell_size);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode compare_fields_and_update_stats(const Vector3R& E_analytical, const Vector3R& B_analytical,
                                              const Vector3R& gradB_analytical, const Vector3R& E_grid,
                                              const Vector3R& B_grid, const Vector3R& gradB_grid,
                                              TestStatistics& stats, PetscReal tolerance)
{
  PetscFunctionBeginUser;

  PetscReal error_E = (E_analytical - E_grid).length();
  PetscReal error_B = (B_analytical - B_grid).length();
  PetscReal error_gradB = (gradB_analytical - gradB_grid).length();

  stats.max_field_error_E = std::max(stats.max_field_error_E, error_E);
  stats.max_field_error_B = std::max(stats.max_field_error_B, error_B);
  stats.max_field_error_gradB = std::max(stats.max_field_error_gradB, error_gradB);

  PetscCheck(equal_tol(E_analytical, E_grid, tolerance), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Electric field mismatch. Analytical: (%.8e %.8e %.8e), Grid: (%.8e %.8e %.8e), Error: %.8e",
    REP3_A(E_analytical), REP3_A(E_grid), error_E);

  PetscCheck(equal_tol(B_analytical, B_grid, tolerance), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Magnetic field mismatch. Analytical: (%.8e %.8e %.8e), Grid: (%.8e %.8e %.8e), Error: %.8e",
    REP3_A(B_analytical), REP3_A(B_grid), error_B);

  PetscCheck(equal_tol(gradB_analytical, gradB_grid, tolerance), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Gradient B field mismatch. Analytical: (%.8e %.8e %.8e), Grid: (%.8e %.8e %.8e), Error: %.8e",
    REP3_A(gradB_analytical), REP3_A(gradB_grid), error_gradB);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode print_statistics(const TestStatistics& stats)
{
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n=== TEST STATISTICS ===\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Simulation time: %.6e\n", stats.simulation_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Total steps: %d\n", stats.total_steps));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nField comparison errors:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max E field error:     %.8e\n", stats.max_field_error_E));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max B field error:     %.8e\n", stats.max_field_error_B));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max gradB field error: %.8e\n", stats.max_field_error_gradB));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nTrajectory comparison:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max position error:    %.8e\n", stats.max_position_error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Final pos analytical:  (%.6e %.6e %.6e)\n",
           REP3_A(stats.final_position_analytical)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Final pos grid:        (%.6e %.6e %.6e)\n",
           REP3_A(stats.final_position_grid)));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nIteration statistics:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Total iterations (analytical): %d\n", stats.total_iterations_analytical));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Total iterations (grid):       %d\n", stats.total_iterations_grid));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Avg iterations (analytical):   %.2f\n",
           (PetscReal)stats.total_iterations_analytical / stats.total_steps));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Avg iterations (grid):         %.2f\n",
           (PetscReal)stats.total_iterations_grid / stats.total_steps));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nConstraint violations:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Boundary violations:    %d\n", stats.boundary_violations));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Cell size violations:   %d\n", stats.cell_size_violations));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max displacement/step:  %.8e\n", stats.max_displacement_per_step));

  PetscFunctionReturn(PETSC_SUCCESS);
}

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
  geom_ny = 5;
  geom_nz = 100;
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

  PetscCall(initialize_grid_fields(world.da, E_vec, B_vec, gradB_vec));

  Vector3R*** E_arr, ***B_arr, ***gradB_arr;
  PetscCall(DMDAVecGetArrayRead(world.da, E_vec, &E_arr));
  PetscCall(DMDAVecGetArrayRead(world.da, B_vec, &B_arr));
  PetscCall(DMDAVecGetArrayRead(world.da, gradB_vec, &gradB_arr));

  std::unique_ptr<DriftKineticEsirkepov> esirkepov =
    std::make_unique<DriftKineticEsirkepov>(E_arr, B_arr, nullptr, gradB_arr);



  constexpr Vector3R r0(2.0, 2.0, 2.0);
  constexpr Vector3R v0(0.0, 1.0, 0.0);
  Point point_init(r0, v0);
  PointByField point_analytical(point_init, B0, 1.0);
  PointByField point_grid(point_init, B0, 1.0);

  DriftKineticPush push_analytical;
  push_analytical.set_qm(-1.0);
  push_analytical.set_mp(1.0);
  push_analytical.set_fields_callback(get_analytical_fields);

  DriftKineticPush push_grid;
  push_grid.set_qm(-1.0);
  push_grid.set_mp(1.0);
  push_grid.set_fields_callback([&](const Vector3R& r0, const Vector3R& rn, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p) {
    esirkepov->interpolate(E_p, B_p, gradB_p, rn, r0);
  });

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace_analytical(__FILE__, id + "_analytical", point_analytical, geom_nt / 100);
  PointByFieldTrace trace_grid(__FILE__, id + "_grid", point_grid, geom_nt / 100);

  Vector3R domain_min(0.0, 0.0, 0.0);
  Vector3R domain_max(geom_nx*dx, geom_ny*dy, geom_nz*dz);
  PetscReal min_cell_size = std::min({dx, dy, dz});

  TestStatistics stats;
  constexpr PetscReal field_tolerance = 1e-10;
  constexpr PetscReal position_tolerance = 1e-6;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Starting simulation with %d steps, dt = %.6e\n", geom_nt, dt));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Domain: [%.1f, %.1f] x [%.1f, %.1f] x [%.1f, %.1f]\n",
           domain_min.x(), domain_max.x(), domain_min.y(), domain_max.y(), domain_min.z(), domain_max.z()));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Min cell size: %.6e\n", min_cell_size));


  Vector3R start_r = point_analytical.r;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_analytical_old = point_analytical;
    const PointByField point_grid_old = point_grid;

    PetscCall(trace_analytical.diagnose(t));
    PetscCall(trace_grid.diagnose(t));
    push_analytical.process(dt, point_analytical, point_analytical_old);
    push_grid.process(dt, point_grid, point_grid_old);


    stats.total_iterations_analytical += push_analytical.get_iteration_number();
    stats.total_iterations_grid += push_grid.get_iteration_number();


    PetscCall(check_boundaries(point_analytical, domain_min, domain_max, stats));
    PetscCall(check_boundaries(point_grid, domain_min, domain_max, stats));

    PetscCall(check_displacement(point_analytical, point_analytical_old, min_cell_size, stats));
    PetscCall(check_displacement(point_grid, point_grid_old, min_cell_size, stats));

    Vector3R E_analytical, B_analytical, gradB_analytical;
    Vector3R E_grid, B_grid, gradB_grid;

    get_analytical_fields(point_analytical.r, point_analytical.r, E_analytical, B_analytical, gradB_analytical);
    esirkepov->interpolate(E_grid, B_grid, gradB_grid, point_grid.r, point_grid_old.r);

    PetscCall(compare_fields_and_update_stats(E_analytical, B_analytical, gradB_analytical,
                                            E_grid, B_grid, gradB_grid, stats, field_tolerance));

    PetscReal position_error = (point_analytical.r - point_grid.r).length();
    stats.max_position_error = std::max(stats.max_position_error, position_error);

    PetscCheck(equal_tol(point_analytical.r, point_grid.r, position_tolerance), PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Position mismatch at step %d. Analytical: (%.8e %.8e %.8e), Grid: (%.8e %.8e %.8e), Error: %.8e",
      t, REP3_A(point_analytical.r), REP3_A(point_grid.r), position_error);

    if (t % (geom_nt / 10) == 0) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step %d/%d, max position error: %.8e\n",
               t, geom_nt, stats.max_position_error));
    }
  }

  stats.simulation_time = dt * geom_nt;
  stats.total_steps = geom_nt;
  stats.final_position_analytical = point_analytical.r;
  stats.final_position_grid = point_grid.r;

  PetscCall(print_statistics(stats));

  PetscReal T = dt * (PetscReal)(geom_nt+1);
  PetscReal q = push_analytical.get_qm();
  PetscReal E_par = E0.z();
  PetscReal p_par_theory = q * E_par * T;

  PetscCheck(equal_tol(point_analytical.p_parallel, p_par_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Analytical p_parallel should match theory. Result: %.6e, theory: %.6e",
    point_analytical.p_parallel, p_par_theory);

  PetscCheck(equal_tol(point_grid.p_parallel, p_par_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Grid p_parallel should match theory. Result: %.6e, theory: %.6e",
    point_grid.p_parallel, p_par_theory);

  PetscReal z_theory = 0.5 * q * E_par * T * T;

  PetscCheck(equal_tol(point_analytical.r.z(), z_theory + r0.z(), 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Analytical z should match theory. Result: %.6e, theory: %.6e",
    point_analytical.r.z(), z_theory);

  PetscCheck(equal_tol(point_grid.r.z(), z_theory + r0.z(), 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Grid z should match theory. Result: %.6e, theory: %.6e",
    point_grid.r.z(), z_theory);

  Vector3R V_drift = E0.cross(B0) / (B0.length() * B0.length());
  Vector3R r_theory = start_r + V_drift * T + Vector3R{0, 0, z_theory};

  PetscCheck(equal_tol(point_analytical.r, r_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Analytical guiding center must move with ExB drift. Result: (%.6e %.6e %.6e), theory: (%.6e %.6e %.6e)", REP3_A(point_analytical.r), REP3_A(r_theory));

  PetscCheck(equal_tol(point_grid.r, r_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Grid guiding center must move with ExB drift. Result: (%.6e %.6e %.6e), theory: (%.6e %.6e %.6e)", REP3_A(point_grid.r), REP3_A(r_theory));

  PetscCall(DMDAVecRestoreArrayRead(world.da, E_vec, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(world.da, B_vec, &B_arr));
  PetscCall(DMDAVecRestoreArrayRead(world.da, gradB_vec, &gradB_arr));

  PetscCall(VecDestroy(&E_vec));
  PetscCall(VecDestroy(&B_vec));
  PetscCall(VecDestroy(&gradB_vec));

  PetscCall(world.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
