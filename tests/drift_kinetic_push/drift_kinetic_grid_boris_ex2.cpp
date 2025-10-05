#include "drift_kinetic_push.h"

#include "src/algorithms/boris_push.h"
#include "src/interfaces/particles.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <memory>

static constexpr char help[] =
  "Test: magnetic mirror with double-Gaussian mirrors.\n"
  "Guiding-center push (analytic/grid) should match the Boris pusher.\n";

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 5.0;          // half-length of the trap and grid shift
constexpr PetscReal D = 1.0;          // mirror width squared
constexpr PetscReal q = 1.0;
constexpr PetscReal m = 1.0;

constexpr PetscReal axis_x = L;
constexpr PetscReal axis_y = L;
constexpr PetscReal axis_z = L;

inline PetscReal gauss_profile(PetscReal z, PetscReal z0)
{
  return std::exp(-POW2(z - z0) / D);
}

PetscReal get_Bz(PetscReal z)
{
  return B_min + (B_max - B_min) * (gauss_profile(z, -L) + gauss_profile(z, +L));
}

PetscReal get_dBz_dz(PetscReal z)
{
  const PetscReal coeff = (B_max - B_min);
  return coeff * ((-2.0 * (z + L) / D * gauss_profile(z, -L)) +
                  (-2.0 * (z - L) / D * gauss_profile(z, +L)));
}

PetscReal get_d2Bz_dz2(PetscReal z)
{
  const PetscReal coeff = (B_max - B_min);
  const PetscReal t1 = (z + L);
  const PetscReal t2 = (z - L);
  return coeff * ((-2.0 / D + 4.0 * POW2(t1 / D)) * gauss_profile(z, -L) +
                  (-2.0 / D + 4.0 * POW2(t2 / D)) * gauss_profile(z, +L));
}

PetscReal get_d3Bz_dz3(PetscReal z)
{
  const PetscReal coeff = (B_max - B_min);
  const PetscReal t1 = (z + L);
  const PetscReal t2 = (z - L);
  return coeff * ((12.0 * t1 / (D * D) - 8.0 * POW3(t1 / D)) * gauss_profile(z, -L) +
                  (12.0 * t2 / (D * D) - 8.0 * POW3(t2 / D)) * gauss_profile(z, +L));
}

Vector3R to_magnetic_axis(const Vector3R& r)
{
  return {r.x() - axis_x, r.y() - axis_y, r.z() - axis_z};
}

Vector3R get_B_vector(const Vector3R& r)
{
  const Vector3R local = to_magnetic_axis(r);
  const PetscReal x = local.x();
  const PetscReal y = local.y();
  const PetscReal z = local.z();
  const PetscReal r2 = x * x + y * y;

  const PetscReal dBz_dz = get_dBz_dz(z);
  const PetscReal d2Bz_dz2 = get_d2Bz_dz2(z);
  const PetscReal Bz = get_Bz(z) - 0.25 * r2 * d2Bz_dz2;

  return {
    -0.5 * x * dBz_dz,
    -0.5 * y * dBz_dz,
     Bz,
  };
}

Vector3R get_gradB_vector(const Vector3R& r)
{
  const Vector3R local = to_magnetic_axis(r);
  const PetscReal x = local.x();
  const PetscReal y = local.y();
  const PetscReal z = local.z();
  const PetscReal r2 = x * x + y * y;
  const PetscReal radial = std::sqrt(r2);

  const PetscReal d2Bz_dz2 = get_d2Bz_dz2(z);
  const PetscReal d3Bz_dz3 = get_d3Bz_dz3(z);

  const PetscReal dB_dr = -0.5 * radial * d2Bz_dz2;
  const PetscReal dB_dz = get_dBz_dz(z) - 0.25 * r2 * d3Bz_dz3;

  if (radial > 1e-12) {
    const PetscReal inv_r = 1.0 / radial;
    return {x * inv_r * dB_dr, y * inv_r * dB_dr, dB_dz};
  }

  return {0.0, 0.0, dB_dz};
}

void get_analytical_fields(
  const Vector3R&, const Vector3R& rn, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = {0.0, 0.0, 0.0};
  B_p = get_B_vector(rn);
  gradB_p = get_gradB_vector(rn);
}

struct DriftComparisonStats {
  PetscInt total_steps = 0;
  PetscReal simulation_time = 0.0;

  PetscReal max_position_error = 0.0;
  PetscReal max_field_error_B = 0.0;
  PetscReal max_field_error_gradB = 0.0;

  Vector3R final_position_analytical;
  Vector3R final_position_grid;
};

struct BorisComparisonStats {
  PetscReal mu_reference = 0.0;
  PetscReal energy_reference = 0.0;

  PetscReal max_z_error = 0.0;
  PetscReal max_parallel_error = 0.0;
  PetscReal max_mu_error = 0.0;
  PetscReal max_energy_error = 0.0;

  Vector3R final_position_boris;
};

PetscErrorCode print_statistics(
  const DriftComparisonStats& drift_stats, const BorisComparisonStats& boris_stats)
{
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n=== DRIFT VS GRID STATISTICS ===\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Simulation time: %.6e\n", drift_stats.simulation_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Total steps: %d\n", drift_stats.total_steps));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nField comparison errors:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max B field error:     %.8e\n", drift_stats.max_field_error_B));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max gradB field error: %.8e\n", drift_stats.max_field_error_gradB));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nTrajectory comparison:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max position error:    %.8e\n", drift_stats.max_position_error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Final pos analytical:  (%.6e %.6e %.6e)\n",
           REP3_A(drift_stats.final_position_analytical)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Final pos grid:        (%.6e %.6e %.6e)\n",
           REP3_A(drift_stats.final_position_grid)));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n=== DRIFT VS BORIS STATISTICS ===\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Reference mu:      %.8e\n", boris_stats.mu_reference));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Reference energy:  %.8e\n", boris_stats.energy_reference));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Max z error:       %.8e\n", boris_stats.max_z_error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Max p_parallel err %.8e\n", boris_stats.max_parallel_error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Max mu error:      %.8e\n", boris_stats.max_mu_error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Max energy error:  %.8e\n", boris_stats.max_energy_error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Final pos boris:   (%.6e %.6e %.6e)\n",
           REP3_A(boris_stats.final_position_boris)));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode initialize_grid_fields(
  DM da, const Vector3R& dr, Vec E_vec, Vec B_vec, Vec gradB_vec)
{
  PetscFunctionBeginUser;

  Vector3R*** E_arr;
  Vector3R*** B_arr;
  Vector3R*** gradB_arr;

  PetscCall(DMDAVecGetArrayWrite(da, E_vec, &E_arr));
  PetscCall(DMDAVecGetArrayWrite(da, B_vec, &B_arr));
  PetscCall(DMDAVecGetArrayWrite(da, gradB_vec, &gradB_arr));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  for (PetscInt k = start[Z]; k < start[Z] + size[Z]; ++k) {
    for (PetscInt j = start[Y]; j < start[Y] + size[Y]; ++j) {
      for (PetscInt i = start[X]; i < start[X] + size[X]; ++i) {
        const PetscReal x_i = i * dr.x();
        const PetscReal y_j = j * dr.y();
        const PetscReal z_k = k * dr.z();

        E_arr[k][j][i] = {0.0, 0.0, 0.0};

        const Vector3R pos_bx{x_i, (j + 0.5) * dr.y(), (k + 0.5) * dr.z()};
        const Vector3R pos_by{(i + 0.5) * dr.x(), y_j, (k + 0.5) * dr.z()};
        const Vector3R pos_bz{(i + 0.5) * dr.x(), (j + 0.5) * dr.y(), z_k};

        const Vector3R Bx_vec = get_B_vector(pos_bx);
        const Vector3R By_vec = get_B_vector(pos_by);
        const Vector3R Bz_vec = get_B_vector(pos_bz);
        B_arr[k][j][i].x() = Bx_vec.x();
        B_arr[k][j][i].y() = By_vec.y();
        B_arr[k][j][i].z() = Bz_vec.z();

        const Vector3R gradBx_vec = get_gradB_vector(pos_bx);
        const Vector3R gradBy_vec = get_gradB_vector(pos_by);
        const Vector3R gradBz_vec = get_gradB_vector(pos_bz);
        gradB_arr[k][j][i].x() = gradBx_vec.x();
        gradB_arr[k][j][i].y() = gradBy_vec.y();
        gradB_arr[k][j][i].z() = gradBz_vec.z();
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, E_vec, &E_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, B_vec, &B_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, gradB_vec, &gradB_arr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal compute_kinetic_energy(const PointByField& point)
{
  const PetscReal p_perp = point.p_perp_ref();
  const PetscReal p_parallel = point.p_par();
  return 0.5 * (p_perp * p_perp + p_parallel * p_parallel);
}

PetscReal compute_kinetic_energy(const Point& point)
{
  return 0.5 * point.p.squared();
}

PetscReal compute_mu(const Point& point)
{
  const Vector3R B = get_B_vector(point.r);
  const PetscReal p_perp = point.p.transverse_to(B).length();
  return 0.5 * m * p_perp * p_perp / B.length();
}

PetscReal compute_parallel_velocity(const Point& point)
{
  const Vector3R B = get_B_vector(point.r);
  const Vector3R b = B.normalized();
  return point.p.dot(b);
}

void boris_step(BorisPush& push, Point& point, interfaces::Particles& particles)
{
  const Vector3R E_p = {0.0, 0.0, 0.0};
  const Vector3R B_p = get_B_vector(point.r);
  push.set_fields(E_p, B_p);
  push.process(dt, point, particles);
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  const PetscReal d = 0.05;
  dt = omega_dt / get_Bz(L);
  geom_nt = 20'000;
  diagnose_period = geom_nt / 10;
  geom_nx = static_cast<PetscInt>(2 * L / d);
  geom_ny = static_cast<PetscInt>(2 * L / d);
  geom_nz = static_cast<PetscInt>(2 * L / d);
  dx = d;
  dy = d;
  dz = d;

  World::set_geometry(
    geom_nx, geom_ny, geom_nz, geom_nt,
    dx, dy, dz, dt,
    diagnose_period);

  World world;
  PetscCall(world.initialize());

  Vec E_vec;
  Vec B_vec;
  Vec gradB_vec;
  PetscCall(DMCreateGlobalVector(world.da, &E_vec));
  PetscCall(DMCreateGlobalVector(world.da, &B_vec));
  PetscCall(DMCreateGlobalVector(world.da, &gradB_vec));

  PetscCall(initialize_grid_fields(world.da, {dx, dy, dz}, E_vec, B_vec, gradB_vec));

  Vector3R*** E_arr;
  Vector3R*** B_arr;
  Vector3R*** gradB_arr;
  PetscCall(DMDAVecGetArrayRead(world.da, E_vec, &E_arr));
  PetscCall(DMDAVecGetArrayRead(world.da, B_vec, &B_arr));
  PetscCall(DMDAVecGetArrayRead(world.da, gradB_vec, &gradB_arr));

  auto esirkepov = std::make_unique<DriftKineticEsirkepov>(E_arr, B_arr, nullptr, gradB_arr);

  constexpr PetscReal v_perp = 0.1/std::sqrt(2.);
  constexpr PetscReal v_par = 0.1/std::sqrt(2.);
  const Vector3R r0(axis_x + 0.1, axis_y, axis_z);
  const Vector3R v0(v_perp, 0.0, v_par);

  const PetscReal B_mag = get_B_vector(r0).length();
  const PetscReal rho = v_perp / B_mag;
  Point point_init(r0 - Vector3R{0.0, rho, 0.0}, v0);
  PointByField point_analytical(point_init, get_B_vector(r0), m);
  PointByField point_grid(point_init, get_B_vector(r0), m);

  Point point_boris(r0, v0);

  DriftComparisonStats drift_stats;
  BorisComparisonStats boris_stats;
  boris_stats.mu_reference = point_grid.mu();
  boris_stats.energy_reference = compute_kinetic_energy(point_grid);

  SortParameters sort_parameters{"boris_sort", 1, 1.0, q, m};
  interfaces::Particles particles(world, sort_parameters);

  DriftKineticPush push_analytical;
  push_analytical.set_qm(q / m);
  push_analytical.set_mp(m);
  push_analytical.set_fields_callback(get_analytical_fields);

  DriftKineticPush push_grid;
  push_grid.set_qm(q / m);
  push_grid.set_mp(m);
  push_grid.set_fields_callback([
    &](const Vector3R& r0_local, const Vector3R& rn_local, Vector3R& E_p,
        Vector3R& B_p, Vector3R& gradB_p) {
    esirkepov->interpolate(E_p, B_p, gradB_p, rn_local, r0_local);
  });

  BorisPush push_boris;
  push_boris.set_qm(q / m);

  const auto id = std::format("omega_dt_{:.4f}", omega_dt);
  PointByFieldTrace trace_analytical(__FILE__, id + "_analytical", point_analytical, geom_nt / 1000);
  PointByFieldTrace trace_grid(__FILE__, id + "_grid", point_grid, geom_nt / 1000);
  PointTrace trace_boris(__FILE__, id + "_boris", point_boris, geom_nt / 1000);

  //const PetscReal z_limit = 2*L + 1e-2;
  //const PetscReal r_limit = 2*L + 1e-2;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_analytical_old = point_analytical;
    const PointByField point_grid_old = point_grid;

    PetscCall(trace_analytical.diagnose(t));
    PetscCall(trace_grid.diagnose(t));
    PetscCall(trace_boris.diagnose(t));

    push_analytical.process(dt, point_analytical, point_analytical_old);
    push_grid.process(dt, point_grid, point_grid_old);
    boris_step(push_boris, point_boris, particles);

    //PetscCheck(std::abs(point_analytical.r.z() - axis_z) <= z_limit, PETSC_COMM_WORLD, PETSC_ERR_USER,
    //  "Particle escaped magnetic mirror (analytic). z = %.6e, allowed = %.6e",
    //  point_analytical.r.z() - axis_z, z_limit);
//
    //PetscCheck(std::abs(point_grid.r.z() - axis_z) <= z_limit, PETSC_COMM_WORLD, PETSC_ERR_USER,
    //  "Particle escaped magnetic mirror (grid). z = %.6e, allowed = %.6e",
    //  point_grid.r.z() - axis_z, z_limit);
//
    //PetscCheck(std::abs(point_boris.r.z() - axis_z) <= z_limit, PETSC_COMM_WORLD, PETSC_ERR_USER,
    //  "Particle escaped magnetic mirror (Boris). z = %.6e, allowed = %.6e",
    //  point_boris.r.z() - axis_z, z_limit);

    //PetscCheck((point_boris.r - Vector3R{axis_x, axis_y, axis_z}).length() <= r_limit,
    //  PETSC_COMM_WORLD, PETSC_ERR_USER,
    //  "Particle escaped radial well. r = %.6e, allowed = %.6e",
    //  (point_boris.r - Vector3R{axis_x, axis_y, axis_z}).length(), r_limit);

    const PetscReal position_error = (point_analytical.r - point_grid.r).length();
    drift_stats.max_position_error = std::max(drift_stats.max_position_error, position_error);

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
    drift_stats.max_field_error_B = std::max(drift_stats.max_field_error_B, error_B);
    drift_stats.max_field_error_gradB = std::max(drift_stats.max_field_error_gradB, error_gradB);

    const PetscReal z_error = std::abs((point_grid.r.z() - axis_z) - (point_boris.r.z() - axis_z));
    boris_stats.max_z_error = std::max(boris_stats.max_z_error, z_error);

    const PetscReal parallel_velocity_boris = compute_parallel_velocity(point_boris);
    const PetscReal parallel_error = std::abs(point_grid.p_par() - parallel_velocity_boris);
    boris_stats.max_parallel_error = std::max(boris_stats.max_parallel_error, parallel_error);

    const PetscReal mu_boris = compute_mu(point_boris);
    const PetscReal mu_error = std::abs(point_grid.mu() - mu_boris);
    boris_stats.max_mu_error = std::max(boris_stats.max_mu_error, mu_error);

    const PetscReal energy_drift = compute_kinetic_energy(point_grid);
    const PetscReal energy_boris = compute_kinetic_energy(point_boris);
    const PetscReal energy_error = std::abs(energy_drift - energy_boris);
    boris_stats.max_energy_error = std::max(boris_stats.max_energy_error, energy_error);
  }

  drift_stats.simulation_time = dt * geom_nt;
  drift_stats.total_steps = geom_nt;
  drift_stats.final_position_analytical = point_analytical.r;
  drift_stats.final_position_grid = point_grid.r;
  boris_stats.final_position_boris = point_boris.r;

  PetscCall(DMDAVecRestoreArrayRead(world.da, E_vec, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(world.da, B_vec, &B_arr));
  PetscCall(DMDAVecRestoreArrayRead(world.da, gradB_vec, &gradB_arr));

  PetscCall(VecDestroy(&E_vec));
  PetscCall(VecDestroy(&B_vec));
  PetscCall(VecDestroy(&gradB_vec));

  PetscCall(particles.finalize());
  PetscCall(print_statistics(drift_stats, boris_stats));

  PetscCall(world.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
