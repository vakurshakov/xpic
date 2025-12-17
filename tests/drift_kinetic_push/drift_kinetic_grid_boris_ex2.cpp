#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic Esirkepov vs analytical: comparison of fields and\n"
  "trajectories calculated by drift_kinetic_push on analytical fields vs\n"
  "on grid filled with analytical fields. Controls boundary violations and\n"
  "particle displacement < cell size.\n";

using namespace drift_kinetic_test_utils;

constexpr Vector3R r0(2.0, 2.0, 2.0);
constexpr Vector3R v0(0.1, 0.0, 0.1);

constexpr Vector3R E0(0, 0, 0);
constexpr Vector3R B0(0, 0, 2);
constexpr Vector3R gradB0(1, 0, 0);

void get_analytical_fields(const Vector3R&, const Vector3R& rn, //
  Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = E0;
  B_p = B0 + (rn - r0).dot(gradB0) * B0.normalized();
  gradB_p = gradB0;
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / B0.length();

  overwrite_config(20., 20., 20., 100., 1.0, 1.0, 1.0, dt, dt);

  FieldContext context;

  PetscCall(context.initialize([&](PetscInt i, PetscInt j, PetscInt k, Vector3R& E_g, Vector3R& B_g, Vector3R& gradB_g) {
    Vector3R r((i + 0.5) * dx, (j + 0.5) * dy, k * dz);
    get_analytical_fields(r, r, E_g, B_g, gradB_g);
  }));

  DriftKineticEsirkepov esirkepov(
    context.E_arr, context.B_arr, nullptr, nullptr);

  esirkepov.set_dBidrj(context.dBdx_arr, context.dBdy_arr, context.dBdz_arr);

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
    geom_nt / 10, point_analytical, point_grid, point_boris);

  Vector3R start_r = point_analytical.r;
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

    update_comparison_stats(stats, //
      point_analytical, point_grid, point_boris, //
      B_analytical, gradB_analytical, B_grid, gradB_grid);
  }

  PetscReal B = B0.length();

  Vector3R V_gradB;
  V_gradB += Vector3R(0, 0, v0.z());
  V_gradB += m * POW2(v0.z()) / (q * POW3(B)) * B0.cross(gradB0);
  V_gradB += point_analytical.mu_p / (q * POW2(B)) * B0.cross(gradB0);

  PetscReal T = dt * (PetscReal)(geom_nt + 1);
  Vector3R r_theory = start_r + V_gradB * T;

  PetscCheck(equal_tol(point_analytical.r, r_theory, 1e-8), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Analytical guiding center must drift with grad-B drift. Result: (%.8e %.8e %.8e), theory: (%.8e %.8e %.8e)", REP3_A(point_analytical.r), REP3_A(r_theory));

  PetscCheck(equal_tol(point_grid.r, r_theory, 1e-8), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Grid guiding center must drift with grad-B drift. Result: (%.8e %.8e %.8e), theory: (%.8e %.8e %.8e)", REP3_A(point_grid.r), REP3_A(r_theory));

  PetscCall(print_statistics(stats, point_analytical, point_grid, point_boris));

  PetscCall(context.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
