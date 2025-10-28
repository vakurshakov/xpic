#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic Esirkepov vs analytical: comparison of fields and "
  "trajectories\n"
  "calculated by drift_kinetic_push on analytical fields vs on grid filled "
  "with\n"
  "analytical fields. Controls boundary violations and particle displacement < "
  "cell size.\n";

constexpr Vector3R E0(0.0, 1.0, -1.0);
constexpr Vector3R B0(0.0, 0.0, 1.0);
constexpr Vector3R gradB0(0.0, 0.0, 0.0);

void get_analytical_fields(const Vector3R&, const Vector3R&, Vector3R& E_p,
  Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = E0;
  B_p = B0;
  gradB_p = gradB0;
}

using namespace drift_kinetic_test_utils;

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  drift_kinetic_test_utils::grid::FieldWorldContext context;
  PetscCall(context.init([&]() {
    World::set_geometry(20, 5, 100, 100, 1., 1., 1., omega_dt / B0.length(),
      omega_dt / B0.length());
  }));

  PetscCall(initialize_grid_fields(context.dm(), context.E_vec(), context.B_vec(), context.gradB_vec(),
    [&](PetscInt, PetscInt, PetscInt, Vector3R& E_cell, Vector3R& B_cell, Vector3R& gradB_cell) {
            E_cell = E0;
            B_cell = B0;
            gradB_cell = gradB0;
    }));

  drift_kinetic_test_utils::grid::FieldArrayTripletRead arrays(
    context.dm(), context.E_vec(), context.B_vec(), context.gradB_vec());

  DriftKineticEsirkepov esirkepov(
    arrays.E(), arrays.B(), nullptr, arrays.gradB());

  constexpr Vector3R r0(2.0, 2.0, 2.0);
  constexpr Vector3R v0(0.0, 0.1, 0.0);
  Point point_init(r0 + correction::rho(v0, B0, -q / m), v0);
  PointByField point_analytical(point_init, B0, 1.0, -q / m);
  PointByField point_grid(point_init, B0, 1.0, -q / m);
  Point point_boris(point_init);

  DriftKineticPush push_analytical;
  push_analytical.set_qm(-q / m);
  push_analytical.set_mp(m);
  push_analytical.set_fields_callback(get_analytical_fields);

  DriftKineticPush push_grid;
  push_grid.set_qm(-q / m);
  push_grid.set_mp(m);
  push_grid.set_fields_callback(
    [&](const Vector3R& r0, const Vector3R& rn, Vector3R& E_p, Vector3R& B_p,
      Vector3R& gradB_p) {
      esirkepov.interpolate(E_p, B_p, gradB_p, rn, r0);
    });

  DriftComparisonStats drift_stats;
  BorisComparisonStats boris_stats;
  boris_stats.mu_reference = point_grid.mu();
  boris_stats.energy_reference = compute_kinetic_energy(point_grid);

  SortParameters sort_parameters{"boris_sort", 1, 1.0, q, m};
  interfaces::Particles particles(context.world(), sort_parameters);

  BorisPush push_boris;
  push_boris.set_qm(-q / m);

  auto id = std::format("omega_dt_{:.4f}", omega_dt);
  PointByFieldTrace trace_analytical(
    __FILE__, id + "_analytical", point_analytical, geom_nt / 100);
  PointByFieldTrace trace_grid(__FILE__, id + "_grid", point_grid, geom_nt / 100);
  PointTrace trace_boris(__FILE__, id + "_boris", point_boris, geom_nt / 100);

  Vector3R start_r = point_analytical.r;
  Vector3R E_analytical;
  Vector3R B_analytical;
  Vector3R gradB_analytical;
  Vector3R E_grid;
  Vector3R B_grid;
  Vector3R gradB_grid;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_analytical_old = point_analytical;
    const PointByField point_grid_old = point_grid;

    PetscCall(trace_analytical.diagnose(t));
    PetscCall(trace_grid.diagnose(t));
    PetscCall(trace_boris.diagnose(t));
    push_analytical.process(dt, point_analytical, point_analytical_old);
    push_grid.process(dt, point_grid, point_grid_old);
    boris_step(push_boris, point_boris, particles, [&](const Vector3R&) {
      return B0;
    });

    const PetscReal position_error = (point_analytical.r - point_grid.r).length();
    drift_stats.max_position_error =
      std::max(drift_stats.max_position_error, position_error);

    get_analytical_fields(point_analytical_old.r, point_analytical.r,
      E_analytical, B_analytical, gradB_analytical);
    esirkepov.interpolate(
      E_grid, B_grid, gradB_grid, point_grid.r, point_grid_old.r);

    update_boris_comparison_step(
      point_grid, point_boris, B_analytical, gradB_analytical, B_grid, gradB_grid,
      [&](const Vector3R&) {
        return B0;
      },
      drift_stats, boris_stats, 0.0);
  }

  PetscReal T = dt * (PetscReal)(geom_nt + 1);
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

  PetscCall(arrays.restore());
  PetscCall(particles.finalize());

  PetscCall(finalize_and_print_statistics(drift_stats, point_analytical, point_grid, dt, geom_nt, &boris_stats, &point_boris));

  PetscCall(context.destroy());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
