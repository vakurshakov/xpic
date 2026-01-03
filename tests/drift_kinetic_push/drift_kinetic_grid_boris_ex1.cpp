#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic Esirkepov vs analytical: comparison of fields and \n"
  "trajectories calculated by kinetic_push on analytical fields\n"
  "vs on grid filled with analytical fields. Controls boundary violations\n"
  "and particle displacement < cell size.\n";

constexpr Vector3R E0(0, 1, -1);
constexpr Vector3R B0(0, 0, 1);

void get_analytical_fields(const Vector3R&, const Vector3R&, //
  Vector3R& E_p, Vector3R& B_p, Vector3R&)
{
  E_p = E0;
  B_p = B0;
}

using namespace drift_kinetic_test_utils;

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / B0.length();
  overwrite_config(20., 5., 100., 10., 1.0, 1.0, 1.0, dt, dt);

  FieldContext context;

  PetscCall(context.initialize([&](PetscInt, PetscInt, PetscInt, Vector3R& E_g, Vector3R& B_g, Vector3R&) {
      E_g = E0;
      B_g = B0;
  }));

  DriftKineticEsirkepov esirkepov(
    context.E_arr, context.B_arr, nullptr, nullptr);

  esirkepov.set_dBidrj(context.dBdx_arr, context.dBdy_arr, context.dBdz_arr);

  constexpr Vector3R r0(2, 2, 2);
  constexpr Vector3R v0(0, 0.1, 0);
  Point point_init(r0, v0);

  PointByField point_analytical(point_init, B0, 1, q / m);
  PointByField point_grid(point_init, B0, 1, q / m);
  Point point_boris(point_init);

  DriftKineticPush push_analytical;
  push_analytical.set_qm(q / m);
  push_analytical.set_mp(m);
  push_analytical.set_fields_callback(get_analytical_fields);

  DriftKineticPush push_grid;
  push_grid.set_qm(q / m);
  push_grid.set_mp(m);
  push_grid.set_fields_callback(
    [&](const Vector3R& r0, const Vector3R& rn, //
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
    });

  BorisPush push_boris;
  push_boris.set_qm(q / m);

  ComparisonStats stats;
  stats.ref_mu = point_grid.mu();
  stats.ref_energy = get_kinetic_energy(point_grid);

  TraceTriplet trace(__FILE__, std::format("omega_dt_{:.4f}", omega_dt),
    geom_nt / 100, point_analytical, point_grid, point_boris);

  Vector3R start_r = point_analytical.r;

  Vector3R E_analytical, B_analytical, gradB_analytical;
  Vector3R E_grid, B_grid, gradB_grid;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_analytical_old = point_analytical;
    const PointByField point_grid_old = point_grid;
    std::cout << point_grid.x() << " " << point_grid.y() << " " << point_grid.z() << std::endl;
    std::cout << point_analytical.x() << " " << point_analytical.y() << " " << point_analytical.z() << std::endl;
    PetscCall(trace.diagnose(t));

    push_analytical.process(dt, point_analytical, point_analytical_old);
    push_grid.process(dt, point_grid, point_grid_old);
    boris_step(push_boris, point_boris, get_analytical_fields);

    get_analytical_fields(point_analytical_old.r, point_analytical.r,
      E_analytical, B_analytical, gradB_analytical);

    esirkepov.interpolate(
      E_grid, B_grid, gradB_grid, point_grid.r, point_grid_old.r);

    update_comparison_stats(stats, //
      point_analytical, point_grid, point_boris, //
      B_analytical, gradB_analytical, B_grid, gradB_grid);
  }

  PetscReal T = dt * (PetscReal)(geom_nt + 1);
  PetscReal q = push_analytical.get_qm();
  PetscReal E_par = E0.z();
  PetscReal p_par_theory = q * E_par * T;

  PetscCheck(equal_tol(point_analytical.p_parallel, p_par_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Analytical p_parallel should match theory. Result: %.6e, theory: %.6e", point_analytical.p_parallel, p_par_theory);

  PetscCheck(equal_tol(point_grid.p_parallel, p_par_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Grid p_parallel should match theory. Result: %.6e, theory: %.6e", point_grid.p_parallel, p_par_theory);

  PetscReal z_theory = 0.5 * q * E_par * T * T;

  PetscCheck(equal_tol(point_analytical.z(), z_theory + r0.z(), 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Analytical z should match theory. Result: %.6e, theory: %.6e", point_analytical.z(), z_theory);

  PetscCheck(equal_tol(point_grid.z(), z_theory + r0.z(), 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Grid z should match theory. Result: %.6e, theory: %.6e", point_grid.z(), z_theory);

  Vector3R V_drift = E0.cross(B0) / (B0.length() * B0.length());
  Vector3R r_theory = start_r + V_drift * T + Vector3R{0, 0, z_theory};

  PetscCheck(equal_tol(point_analytical.r, r_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Analytical guiding center must move with ExB drift. Result: (%.6e %.6e %.6e), theory: (%.6e %.6e %.6e)", REP3_A(point_analytical.r), REP3_A(r_theory));

  PetscCheck(equal_tol(point_grid.r, r_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Grid guiding center must move with ExB drift. Result: (%.6e %.6e %.6e), theory: (%.6e %.6e %.6e)", REP3_A(point_grid.r), REP3_A(r_theory));

  PetscCall(print_statistics(stats, point_analytical, point_grid, point_boris));

  PetscCall(context.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
