#include "src/algorithms/drift_kinetic_push.h"
#include "src/algorithms/drift_kinetic_implicit.h" 
#include "src/algorithms/boris_push.h"
#include "src/interfaces/particles.h"
#include "src/utils/vector3.h"
#include "tests/common.h"
#include "src/utils/world.h"
#include "src/utils/utils.h"

constexpr PetscReal q = 1.0;
constexpr PetscReal m = 1.0;

PetscErrorCode get_omega_dt(PetscReal& omega_dt)
{
  PetscFunctionBeginUser;
  PetscBool flg;
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-omega_dt", &omega_dt, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify the timestep (Omega * dt) with '-omega_dt'");
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace correction {
  Vector3R rho(const Vector3R& vp, const Vector3R& Bp, PetscReal qm) {
    return vp.cross(Bp.normalized())/(qm*Bp.length());
  }
}

namespace quadratic_magnetic_mirror {

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 10.0;   // half-length of the mirror
constexpr PetscReal Rc = 20.0;  // width of the radial well

PetscReal get_Bz(PetscReal z)
{
  return B_min + (B_max - B_min) * (z * z) / (L * L);
}

PetscReal get_B(PetscReal r, PetscReal z)
{
  return get_Bz(z) * (1.0 + 0.5 * (r * r) / (Rc * Rc));
}

PetscReal get_dBz_dz(PetscReal z)
{
  return 2 * (B_max - B_min) * z / (L * L);
}

void get_mirror_fields(const Vector3R& pos, Vector3R& B_p, Vector3R& gradB_p)
{
  PetscReal x = pos.x();
  PetscReal y = pos.y();
  PetscReal z = pos.z();
  PetscReal r = std::sqrt(x * x + y * y);

  PetscReal Bz = get_Bz(z);
  PetscReal B = get_B(r, z);

  B_p = Vector3R{0.0, 0.0, B};

  PetscReal dBz_dz = get_dBz_dz(z);
  PetscReal dB_dz = dBz_dz * (1.0 + 0.5 * (r * r) / (Rc * Rc));
  PetscReal dB_dr = Bz * r / (Rc * Rc);

  gradB_p = (r > 1e-10) //
    ? Vector3R{x / r * dB_dr, y / r * dB_dr, dB_dz}
    : Vector3R{0.0, 0.0, dB_dz};
}

}  // namespace quadratic_magnetic_mirror

namespace gaussian_magnetic_mirror {

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 5.0;      // Half the length of the trap
constexpr PetscReal D = 1.0;      // Mirror width
constexpr PetscReal S = POW2(D);  // Mirror width squared

inline PetscReal exp(PetscReal z, PetscReal z0)
{
  return std::exp(-POW2(z - z0) / S);
}

// Center field profile on the axis (double gauss)
PetscReal get_Bz(PetscReal z)
{
  return B_min + (B_max - B_min) * (exp(z, -L) + exp(z, +L));
}

PetscReal get_dBz_dz(PetscReal z)
{
  return (B_max - B_min) * //
    ((-2.0 * (z + L) / S * exp(z, -L)) + //
      (-2.0 * (z - L) / S * exp(z, +L)));
}

PetscReal get_d2Bz_dz2(PetscReal z)
{
  PetscReal t1 = (z + L);
  PetscReal t2 = (z - L);
  return (B_max - B_min) * //
    ((-2.0 / S + 4.0 * POW2(t1 / S)) * exp(z, -L) + //
      (-2.0 / S + 4.0 * POW2(t2 / S)) * exp(z, +L));
}

PetscReal get_d3Bz_dz3(PetscReal z)
{
  PetscReal t1 = (z + L);
  PetscReal t2 = (z - L);
  return (B_max - B_min) * //
    ((12.0 * t1 / (S * S) - 8.0 * POW3(t1 / S)) * exp(z, -L) + //
      (12.0 * t2 / (S * S) - 8.0 * POW3(t2 / S)) * exp(z, +L));
}

// Bz field off axis
PetscReal get_Bz_corr(const Vector3R& r)
{
  return get_Bz(r.z()) - 0.25 * (POW2(r.x()) + POW2(r.y())) * get_d2Bz_dz2(r.z());
}

void get_fields(const Vector3R&, const Vector3R& pos, Vector3R&, Vector3R& B_p, Vector3R& gradB_p)
{
  PetscReal x = pos.x();
  PetscReal y = pos.y();
  PetscReal z = pos.z();
  PetscReal r2 = x * x + y * y;
  PetscReal r = std::sqrt(r2);

  // 1) Calculation of the axial field and its derivatives
  PetscReal Bz = get_Bz(z);
  PetscReal dBz_dz = get_dBz_dz(z);
  PetscReal d2Bz_dz2 = get_d2Bz_dz2(z);
  PetscReal d3Bz_dz3 = get_d3Bz_dz3(z);

  // 2) Computation of the magnetic field vector B_p in the paraxial
  // approximation: transverse components, follow from div(B) = 0,
  // longitudinal component with second-order correction by radius.
  B_p = Vector3R{
    -0.5 * x * dBz_dz,
    -0.5 * y * dBz_dz,
    Bz - 0.25 * r2 * d2Bz_dz2,
  };

  // 3) Calculation of the field modulus gradient |B|
  PetscReal dB_dr = -0.5 * r * d2Bz_dz2;
  PetscReal dB_dz = dBz_dz - 0.25 * r2 * d3Bz_dz3;

  gradB_p = (r > 1e-12) //
    ? Vector3R{(x / r) * dB_dr, (y / r) * dB_dr, dB_dz}
    : Vector3R{0, 0, dB_dz};
}

}  // namespace gaussian_magnetic_mirror

namespace drift_kinetic_test_utils {

template <typename Func>
auto make_translated_field_getter(Func&& func, Vector3R offset)
{
  using FuncT = std::decay_t<Func>;
  return [func = FuncT(std::forward<Func>(func)), offset](const Vector3R& pos, auto&&... rest)
    -> decltype(auto)
  {
    return func(pos - offset, std::forward<decltype(rest)>(rest)...);
  };
}

struct FieldComparisonStats {
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

inline PetscErrorCode print_drift_statistics(
  const DriftComparisonStats& stats,
  const char* header = "\n=== TEST STATISTICS ===\n")
{
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s", header));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Simulation time: %.6e\n", stats.simulation_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Total steps: %d\n", stats.total_steps));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nField comparison errors:\n"));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD, "  Max B field error:     %.8e\n", stats.max_field_error_B));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD, "  Max gradB field error: %.8e\n", stats.max_field_error_gradB));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nTrajectory comparison:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max position error:    %.8e\n", stats.max_position_error));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "  Final pos analytical:  (%.6e %.6e %.6e)\n",
    REP3_A(stats.final_position_analytical)));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "  Final pos grid:        (%.6e %.6e %.6e)\n",
    REP3_A(stats.final_position_grid)));

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode print_statistics(
  const DriftComparisonStats& drift_stats,
  const BorisComparisonStats& boris_stats)
{
  PetscFunctionBeginUser;

  PetscCall(print_drift_statistics(drift_stats, "\n=== DRIFT VS GRID STATISTICS ===\n"));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n=== DRIFT VS BORIS STATISTICS ===\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Reference mu:      %.8e\n", boris_stats.mu_reference));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Reference energy:  %.8e\n", boris_stats.energy_reference));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Max z error:       %.8e\n", boris_stats.max_z_error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Max p_parallel err %.8e\n", boris_stats.max_parallel_error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Max mu error:      %.8e\n", boris_stats.max_mu_error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Max energy error:  %.8e\n", boris_stats.max_energy_error));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "Final pos boris:   (%.6e %.6e %.6e)\n",
    REP3_A(boris_stats.final_position_boris)));

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscReal compute_kinetic_energy(const PointByField& point)
{
  const PetscReal p_perp = point.p_perp_ref();
  const PetscReal p_parallel = point.p_par();
  return 0.5 * (p_perp * p_perp + p_parallel * p_parallel);
}

inline PetscReal compute_kinetic_energy(const Point& point)
{
  return 0.5 * point.p.squared();
}

template <typename GetB>
inline PetscReal compute_mu(const Point& point, const GetB& get_B)
{
  const Vector3R B = get_B(point.r);
  const PetscReal p_perp = point.p.transverse_to(B).length();
  return 0.5 * m * p_perp * p_perp / B.length();
}

template <typename GetB>
inline PetscReal compute_parallel_velocity(const Point& point, const GetB& get_B)
{
  const Vector3R B = get_B(point.r);
  const Vector3R b = B.normalized();
  return point.p.dot(b);
}

template <typename BorisPushLike, typename PointType, typename ParticlesLike, typename GetB>
inline void boris_step(
  BorisPushLike& push,
  PointType& point,
  ParticlesLike& particles,
  const GetB& get_B)
{
  const Vector3R E_p = {0.0, 0.0, 0.0};
  const Vector3R B_p = get_B(point.r);
  push.set_fields(E_p, B_p);
  push.process(dt, point, particles);
}

template <typename GetB, typename GetGradB>
inline PetscErrorCode initialize_staggered_grid_fields(
  DM da,
  const Vector3R& dr,
  Vec E_vec,
  Vec B_vec,
  Vec gradB_vec,
  const GetB& get_B,
  const GetGradB& get_gradB)
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

        const Vector3R Bx_vec = get_B(pos_bx);
        const Vector3R By_vec = get_B(pos_by);
        const Vector3R Bz_vec = get_B(pos_bz);
        B_arr[k][j][i].x() = Bx_vec.x();
        B_arr[k][j][i].y() = By_vec.y();
        B_arr[k][j][i].z() = Bz_vec.z();

        const Vector3R gradBx_vec = get_gradB(pos_bx);
        const Vector3R gradBy_vec = get_gradB(pos_by);
        const Vector3R gradBz_vec = get_gradB(pos_bz);

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

template <typename FillFunc>
inline PetscErrorCode initialize_grid_fields(
  DM da,
  Vec E_vec,
  Vec B_vec,
  Vec gradB_vec,
  FillFunc&& fill_cell)
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
        fill_cell(i, j, k, E_arr[k][j][i], B_arr[k][j][i], gradB_arr[k][j][i]);
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, E_vec, &E_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, B_vec, &B_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, gradB_vec, &gradB_arr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode check_boundaries(
  const PointByField& point,
  const Vector3R& domain_min,
  const Vector3R& domain_max,
  FieldComparisonStats& stats)
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

inline PetscErrorCode check_displacement(
  const PointByField& point_new,
  const PointByField& point_old,
  PetscReal min_cell_size,
  FieldComparisonStats& stats)
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

inline PetscErrorCode compare_fields_and_update_stats(
  const Vector3R& E_analytical,
  const Vector3R& B_analytical,
  const Vector3R& gradB_analytical,
  const Vector3R& E_grid,
  const Vector3R& B_grid,
  const Vector3R& gradB_grid,
  FieldComparisonStats& stats,
  PetscReal tolerance)
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

inline PetscErrorCode print_field_statistics(
  const FieldComparisonStats& stats,
  const char* header = "\n=== TEST STATISTICS ===\n")
{
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s", header));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Simulation time: %.6e\n", stats.simulation_time));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Total steps: %d\n", stats.total_steps));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nField comparison errors:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max E field error:     %.8e\n", stats.max_field_error_E));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max B field error:     %.8e\n", stats.max_field_error_B));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max gradB field error: %.8e\n", stats.max_field_error_gradB));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nTrajectory comparison:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Max position error:    %.8e\n", stats.max_position_error));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "  Final pos analytical:  (%.6e %.6e %.6e)\n",
    REP3_A(stats.final_position_analytical)));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "  Final pos grid:        (%.6e %.6e %.6e)\n",
    REP3_A(stats.final_position_grid)));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nIteration statistics:\n"));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "  Total iterations (analytical): %d\n",
    stats.total_iterations_analytical));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "  Total iterations (grid):       %d\n",
    stats.total_iterations_grid));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "  Avg iterations (analytical):   %.2f\n",
    (PetscReal)stats.total_iterations_analytical / stats.total_steps));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "  Avg iterations (grid):         %.2f\n",
    (PetscReal)stats.total_iterations_grid / stats.total_steps));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nConstraint violations:\n"));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "  Boundary violations:    %d\n",
    stats.boundary_violations));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "  Cell size violations:   %d\n",
    stats.cell_size_violations));
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "  Max displacement/step:  %.8e\n",
    stats.max_displacement_per_step));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename AnalyticalFieldsFunc, typename GridFieldsFunc>
inline PetscErrorCode update_grid_comparison_step(
  const PointByField& point_analytical,
  const PointByField& point_analytical_old,
  const PointByField& point_grid,
  const PointByField& point_grid_old,
  PetscInt iterations_analytical,
  PetscInt iterations_grid,
  PetscReal min_cell_size,
  const Vector3R& domain_min,
  const Vector3R& domain_max,
  PetscReal position_tolerance,
  PetscReal field_tolerance,
  FieldComparisonStats& stats,
  const AnalyticalFieldsFunc& analytical_fields_func,
  const GridFieldsFunc& grid_fields_func)
{
  PetscFunctionBeginUser;

  stats.total_iterations_analytical += iterations_analytical;
  stats.total_iterations_grid += iterations_grid;

  PetscCall(check_boundaries(point_analytical, domain_min, domain_max, stats));
  PetscCall(check_boundaries(point_grid, domain_min, domain_max, stats));

  PetscCall(check_displacement(point_analytical, point_analytical_old, min_cell_size, stats));
  PetscCall(check_displacement(point_grid, point_grid_old, min_cell_size, stats));

  Vector3R E_analytical;
  Vector3R B_analytical;
  Vector3R gradB_analytical;
  Vector3R E_grid;
  Vector3R B_grid;
  Vector3R gradB_grid;

  PetscCall(analytical_fields_func(
    point_analytical_old, point_analytical, E_analytical, B_analytical, gradB_analytical));
  PetscCall(grid_fields_func(point_grid_old, point_grid, E_grid, B_grid, gradB_grid));

  PetscReal position_error = (point_analytical.r - point_grid.r).length();
  stats.max_position_error = std::max(stats.max_position_error, position_error);

  PetscCheck(equal_tol(point_analytical.r, point_grid.r, position_tolerance), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Position mismatch. Analytical: (%.8e %.8e %.8e), Grid: (%.8e %.8e %.8e), Error: %.8e",
    REP3_A(point_analytical.r), REP3_A(point_grid.r), position_error);

  PetscCall(compare_fields_and_update_stats(
    E_analytical, B_analytical, gradB_analytical,
    E_grid, B_grid, gradB_grid,
    stats, field_tolerance));

  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename GetBFunc>
inline void update_boris_comparison_step(
  const PointByField& point_grid,
  const Point& point_boris,
  const Vector3R& B_analytical,
  const Vector3R& gradB_analytical,
  const Vector3R& B_grid,
  const Vector3R& gradB_grid,
  const GetBFunc& get_B_vector,
  DriftComparisonStats& drift_stats,
  BorisComparisonStats& boris_stats,
  PetscReal reference_z = 0.0)
{
  const PetscReal error_B = (B_analytical - B_grid).length();
  const PetscReal error_gradB = (gradB_analytical - gradB_grid).length();
  drift_stats.max_field_error_B = std::max(drift_stats.max_field_error_B, error_B);
  drift_stats.max_field_error_gradB = std::max(drift_stats.max_field_error_gradB, error_gradB);

  const PetscReal z_error =
    std::abs((point_grid.r.z() - reference_z) - (point_boris.r.z() - reference_z));
  boris_stats.max_z_error = std::max(boris_stats.max_z_error, z_error);

  const PetscReal parallel_velocity_boris = compute_parallel_velocity(point_boris, get_B_vector);
  const PetscReal parallel_error = std::abs(point_grid.p_par() - parallel_velocity_boris);
  boris_stats.max_parallel_error = std::max(boris_stats.max_parallel_error, parallel_error);

  const PetscReal mu_boris = compute_mu(point_boris, get_B_vector);
  const PetscReal mu_error = std::abs(point_grid.mu() - mu_boris);
  boris_stats.max_mu_error = std::max(boris_stats.max_mu_error, mu_error);

  const PetscReal energy_drift = compute_kinetic_energy(point_grid);
  const PetscReal energy_boris = compute_kinetic_energy(point_boris);
  const PetscReal energy_error = std::abs(energy_drift - energy_boris);
  boris_stats.max_energy_error = std::max(boris_stats.max_energy_error, energy_error);
}

inline PetscErrorCode log_progress(
  const char* label,
  PetscInt step,
  PetscInt total_steps,
  PetscReal metric_value,
  const char* metric_name = "max position error")
{
  PetscFunctionBeginUser;

  const char* tag = (label && label[0] != '\0') ? label : "progress";
  PetscCall(PetscPrintf(
    PETSC_COMM_WORLD,
    "[%s] Step %d/%d, %s: %.8e\n",
    tag,
    step,
    total_steps,
    metric_name,
    metric_value));

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline void finalize_drift_stats(
  DriftComparisonStats& stats,
  PetscReal dt_local,
  PetscInt total_steps,
  const PointByField& point_analytical,
  const PointByField& point_grid)
{
  stats.simulation_time = dt_local * total_steps;
  stats.total_steps = total_steps;
  stats.final_position_analytical = point_analytical.r;
  stats.final_position_grid = point_grid.r;
}

inline void finalize_boris_stats(
  BorisComparisonStats& stats,
  const Point& point_boris)
{
  stats.final_position_boris = point_boris.r;
}

inline void finalize_field_stats(
  FieldComparisonStats& stats,
  PetscReal dt_local,
  PetscInt total_steps,
  const PointByField& point_analytical,
  const PointByField& point_grid)
{
  stats.simulation_time = dt_local * total_steps;
  stats.total_steps = total_steps;
  stats.final_position_analytical = point_analytical.r;
  stats.final_position_grid = point_grid.r;
}

inline PetscErrorCode finalize_and_print_statistics(
  DriftComparisonStats& drift_stats,
  const PointByField& point_analytical,
  const PointByField& point_grid,
  PetscReal dt_local,
  PetscInt total_steps,
  BorisComparisonStats* boris_stats = nullptr,
  const Point* point_boris = nullptr,
  FieldComparisonStats* field_stats = nullptr,
  const char* header = "\n=== TEST STATISTICS ===\n")
{
  PetscFunctionBeginUser;

  finalize_drift_stats(drift_stats, dt_local, total_steps, point_analytical, point_grid);
  if (field_stats) {
    finalize_field_stats(*field_stats, dt_local, total_steps, point_analytical, point_grid);
  }

  if (boris_stats && point_boris) {
    finalize_boris_stats(*boris_stats, *point_boris);
    PetscCall(print_statistics(drift_stats, *boris_stats));
  } else {
    PetscCall(print_drift_statistics(drift_stats, header));
  }

  if (field_stats) {
    PetscCall(print_field_statistics(*field_stats));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace drift_kinetic_test_utils
