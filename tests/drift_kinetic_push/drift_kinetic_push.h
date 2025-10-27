#include <array>
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

  if (0) {
    LOG("%s", header);
    LOG("Simulation time: %.6e", stats.simulation_time);
    LOG("Total steps: %d", stats.total_steps);

    LOG("\nField comparison errors:");
    LOG("  Max B field error:     %.8e", stats.max_field_error_B);
    LOG("  Max gradB field error: %.8e", stats.max_field_error_gradB);

    LOG("\nTrajectory comparison:");
    LOG("  Max position error:    %.8e", stats.max_position_error);
    LOG(
      "  Final pos analytical:  (%.6e %.6e %.6e)",
      REP3_A(stats.final_position_analytical));
    LOG(
      "  Final pos grid:        (%.6e %.6e %.6e)",
      REP3_A(stats.final_position_grid));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode print_statistics(
  const DriftComparisonStats& drift_stats,
  const BorisComparisonStats& boris_stats)
{
  PetscFunctionBeginUser;

  PetscCall(print_drift_statistics(drift_stats, "\n=== DRIFT VS GRID STATISTICS ===\n"));

  if (0) {
    LOG("\n=== DRIFT VS BORIS STATISTICS ===");
    LOG("Reference mu:      %.8e", boris_stats.mu_reference);
    LOG("Reference energy:  %.8e", boris_stats.energy_reference);
    LOG("Max z error:       %.8e", boris_stats.max_z_error);
    LOG("Max p_parallel err %.8e", boris_stats.max_parallel_error);
    LOG("Max mu error:      %.8e", boris_stats.max_mu_error);
    LOG("Max energy error:  %.8e", boris_stats.max_energy_error);
    LOG(
      "Final pos boris:   (%.6e %.6e %.6e)",
      REP3_A(boris_stats.final_position_boris));
  }

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

inline std::array<Vector3R, 3> yee_E_positions(
  PetscReal x_i,
  PetscReal y_j,
  PetscReal z_k,
  const Vector3R& dr)
{
  return {
    Vector3R{x_i + 0.5 * dr.x(), y_j, z_k},
    Vector3R{x_i, y_j + 0.5 * dr.y(), z_k},
    Vector3R{x_i, y_j, z_k + 0.5 * dr.z()},
  };
}

inline std::array<Vector3R, 3> yee_B_positions(
  PetscReal x_i,
  PetscReal y_j,
  PetscReal z_k,
  const Vector3R& dr)
{
  return {
    Vector3R{x_i, y_j + 0.5 * dr.y(), z_k + 0.5 * dr.z()},
    Vector3R{x_i + 0.5 * dr.x(), y_j, z_k + 0.5 * dr.z()},
    Vector3R{x_i + 0.5 * dr.x(), y_j + 0.5 * dr.y(), z_k},
  };
}

namespace grid {

class FieldArrayReadHandle {
public:
  FieldArrayReadHandle(DM dm, Vec vec)
    : dm_(dm)
    , vec_(vec)
  {
    PetscErrorCode ierr = DMDAVecGetArrayRead(dm_, vec_, &array_);
    if (PetscUnlikely(ierr != 0)) {
      PetscCallAbort(PETSC_COMM_WORLD, ierr);
    }
  }

  ~FieldArrayReadHandle()
  {
    PetscErrorCode ierr = restore();
    if (PetscUnlikely(ierr != 0)) {
      PetscCallAbort(PETSC_COMM_WORLD, ierr);
    }
  }

  FieldArrayReadHandle(const FieldArrayReadHandle&) = delete;
  FieldArrayReadHandle& operator=(const FieldArrayReadHandle&) = delete;

  Vector3R*** get() const { return array_; }

  PetscErrorCode restore()
  {
    if (array_ != nullptr) {
      PetscErrorCode ierr = DMDAVecRestoreArrayRead(dm_, vec_, &array_);
      array_ = nullptr;
      return ierr;
    }
    return PETSC_SUCCESS;
  }

private:
  DM dm_;
  Vec vec_;
  Vector3R*** array_ = nullptr;
};

class FieldArrayWriteHandle {
public:
  FieldArrayWriteHandle(DM dm, Vec vec)
    : dm_(dm)
    , vec_(vec)
  {
    PetscErrorCode ierr = DMDAVecGetArrayWrite(dm_, vec_, &array_);
    if (PetscUnlikely(ierr != 0)) {
      PetscCallAbort(PETSC_COMM_WORLD, ierr);
    }
  }

  ~FieldArrayWriteHandle()
  {
    PetscErrorCode ierr = restore();
    if (PetscUnlikely(ierr != 0)) {
      PetscCallAbort(PETSC_COMM_WORLD, ierr);
    }
  }

  FieldArrayWriteHandle(const FieldArrayWriteHandle&) = delete;
  FieldArrayWriteHandle& operator=(const FieldArrayWriteHandle&) = delete;

  Vector3R*** get() const { return array_; }

  PetscErrorCode restore()
  {
    if (array_ != nullptr) {
      PetscErrorCode ierr = DMDAVecRestoreArrayWrite(dm_, vec_, &array_);
      array_ = nullptr;
      return ierr;
    }
    return PETSC_SUCCESS;
  }

private:
  DM dm_;
  Vec vec_;
  Vector3R*** array_ = nullptr;
};

class FieldArrayTripletRead {
public:
  FieldArrayTripletRead(DM dm, Vec E_vec, Vec B_vec, Vec gradB_vec): E_(dm, E_vec), B_(dm, B_vec), gradB_(dm, gradB_vec){}

  Vector3R*** E() const { return E_.get(); }
  Vector3R*** B() const { return B_.get(); }
  Vector3R*** gradB() const { return gradB_.get(); }

  PetscErrorCode restore()
  {
    PetscFunctionBeginUser;
    PetscCall(E_.restore());
    PetscCall(B_.restore());
    PetscCall(gradB_.restore());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  FieldArrayReadHandle E_;
  FieldArrayReadHandle B_;
  FieldArrayReadHandle gradB_;
};

class FieldArrayTripletWrite {
public:
  FieldArrayTripletWrite(DM dm, Vec E_vec, Vec B_vec, Vec gradB_vec): E_(dm, E_vec), B_(dm, B_vec), gradB_(dm, gradB_vec) {}

  Vector3R*** E() const { return E_.get(); }
  Vector3R*** B() const { return B_.get(); }
  Vector3R*** gradB() const { return gradB_.get(); }

  PetscErrorCode restore()
  {
    PetscFunctionBeginUser;
    PetscCall(E_.restore());
    PetscCall(B_.restore());
    PetscCall(gradB_.restore());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  FieldArrayWriteHandle E_;
  FieldArrayWriteHandle B_;
  FieldArrayWriteHandle gradB_;
};

class FieldWorldContext {
public:
  FieldWorldContext() = default;

  FieldWorldContext(const FieldWorldContext&) = delete;
  FieldWorldContext& operator=(const FieldWorldContext&) = delete;

  ~FieldWorldContext()
  {
    if (initialized_) {
      PetscErrorCode ierr = destroy();
      if (PetscUnlikely(ierr != 0)) {
        PetscCallAbort(PETSC_COMM_WORLD, ierr);
      }
    }
  }

  template <typename GeometrySetter>
  PetscErrorCode init(GeometrySetter&& geometry_setter)
  {
    PetscFunctionBeginUser;
    PetscCheck(!initialized_, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
      "FieldWorldContext is already initialized");

    geometry_setter();

    PetscCall(world_.initialize());
    PetscCall(DMCreateGlobalVector(world_.da, &E_vec_));
    PetscCall(DMCreateGlobalVector(world_.da, &B_vec_));
    PetscCall(DMCreateGlobalVector(world_.da, &gradB_vec_));

    initialized_ = true;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode destroy()
  {
    PetscFunctionBeginUser;
    if (!initialized_) {
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscCall(VecDestroy(&E_vec_));
    PetscCall(VecDestroy(&B_vec_));
    PetscCall(VecDestroy(&gradB_vec_));
    PetscCall(world_.finalize());

    initialized_ = false;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  DM dm() const { return world_.da; }
  Vec E_vec() const { return E_vec_; }
  Vec B_vec() const { return B_vec_; }
  Vec gradB_vec() const { return gradB_vec_; }
  World& world() { return world_; }

private:
  World world_;
  Vec E_vec_ = nullptr;
  Vec B_vec_ = nullptr;
  Vec gradB_vec_ = nullptr;
  bool initialized_ = false;
};

template <typename FillFunc>
inline PetscErrorCode initialize_grid_fields(
  DM da,
  Vec E_vec,
  Vec B_vec,
  Vec gradB_vec,
  FillFunc&& fill_cell)
{
  PetscFunctionBeginUser;

  FieldArrayTripletWrite arrays(da, E_vec, B_vec, gradB_vec);
  Vector3R*** E_arr = arrays.E();
  Vector3R*** B_arr = arrays.B();
  Vector3R*** gradB_arr = arrays.gradB();

  Vector3I start;
  Vector3I size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  for (PetscInt k = start[Z]; k < start[Z] + size[Z]; ++k) {
    for (PetscInt j = start[Y]; j < start[Y] + size[Y]; ++j) {
      for (PetscInt i = start[X]; i < start[X] + size[X]; ++i) {
        fill_cell(i, j, k, E_arr[k][j][i], B_arr[k][j][i], gradB_arr[k][j][i]);
      }
    }
  }

  PetscCall(arrays.restore());
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace grid

using grid::initialize_grid_fields;

template <typename GetE, typename GetB, typename GetGradB>
inline PetscErrorCode initialize_staggered_grid_fields(
  DM da,
  const Vector3R& dr,
  Vec E_vec,
  Vec B_vec,
  Vec gradB_vec,
  const GetE& get_E,
  const GetB& get_B,
  const GetGradB& get_gradB)
{
  PetscFunctionBeginUser;

  grid::FieldArrayTripletWrite arrays(da, E_vec, B_vec, gradB_vec);
  Vector3R*** E_arr = arrays.E();
  Vector3R*** B_arr = arrays.B();
  Vector3R*** gradB_arr = arrays.gradB();

  auto assign_components = [](Vector3R& cell, const auto& getter, const auto& positions) {
    cell.x() = getter(positions[0]).x();
    cell.y() = getter(positions[1]).y();
    cell.z() = getter(positions[2]).z();
  };

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  for (PetscInt k = start[Z]; k < start[Z] + size[Z]; ++k) {
    for (PetscInt j = start[Y]; j < start[Y] + size[Y]; ++j) {
      for (PetscInt i = start[X]; i < start[X] + size[X]; ++i) {
        const PetscReal x_i = i * dr.x();
        const PetscReal y_j = j * dr.y();
        const PetscReal z_k = k * dr.z();

        const auto e_positions = yee_E_positions(x_i, y_j, z_k, dr);
        assign_components(E_arr[k][j][i], get_E, e_positions);

        const auto b_positions = yee_B_positions(x_i, y_j, z_k, dr);
        assign_components(B_arr[k][j][i], get_B, b_positions);
        assign_components(gradB_arr[k][j][i], get_gradB, b_positions);
      }
    }
  }

  PetscCall(arrays.restore());
  PetscFunctionReturn(PETSC_SUCCESS);
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
  auto zero_E = [](const Vector3R&) { return Vector3R{0.0, 0.0, 0.0}; };
  return initialize_staggered_grid_fields(da, dr, E_vec, B_vec, gradB_vec, zero_E, get_B, get_gradB);
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

inline PetscErrorCode finalize_and_print_statistics(
  DriftComparisonStats& drift_stats,
  const PointByField& point_analytical,
  const PointByField& point_grid,
  PetscReal dt_local,
  PetscInt total_steps,
  BorisComparisonStats* boris_stats = nullptr,
  const Point* point_boris = nullptr,
  const char* header = "\n=== TEST STATISTICS ===\n")
{
  PetscFunctionBeginUser;

  finalize_drift_stats(drift_stats, dt_local, total_steps, point_analytical, point_grid);

  if (boris_stats && point_boris) {
    finalize_boris_stats(*boris_stats, *point_boris);
    PetscCall(print_statistics(drift_stats, *boris_stats));
  } else {
    PetscCall(print_drift_statistics(drift_stats, header));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace drift_kinetic_test_utils
