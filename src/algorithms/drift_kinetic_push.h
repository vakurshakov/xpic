#ifndef SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H
#define SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H

#include "src/interfaces/point.h"

/// @brief Fully implicit drift-kinetic guiding-center pusher.
/// Advances guiding-center position and parallel velocity.
class DriftKineticPush {
public:
  DriftKineticPush() = default;
  DriftKineticPush(PetscReal qm, PetscReal mp);

  /// @brief Sets iteration tolerances and limit.
  void set_tolerances(PetscReal atol, PetscReal rtol, PetscInt maxit);

  /// @brief Sets the charge-to-mass ratio.
  void set_qm(PetscReal qm);
  /// @brief Sets the particle mass.
  void set_mp(PetscReal mp);
  PetscReal get_mp() const;
  PetscReal get_qm() const;

  PetscInt get_iteration_number() const;
  bool has_converged() const;

  /// @brief Returns the last position residual.
  PetscReal get_FRk() const;

  /// @brief Returns the last parallel velocity residual.
  PetscReal get_FVk() const;

  /// @brief Callback for interpolating particle fields.
  using SetFields = std::function<PetscErrorCode(const Vector3R&,
    const Vector3R&, Vector3R&, PetscReal&, Vector3R&, Vector3R&, Vector3R&)>;

  void set_fields_callback(SetFields&& callback);

  /// @warning `pn` and `p0` must be distinct objects.
  /// Interpolation errors are returned; nonlinear convergence is reported
  /// separately by `has_converged()`.
  PetscErrorCode process(PetscReal dt, PointByField& pn, const PointByField& p0);

private:
  PetscErrorCode process_picard(
    PetscReal dt, PointByField& pn, const PointByField& p0);

  /// @brief Evaluates drift velocity and acceleration.
  PetscErrorCode evaluate_rhs(const PointByField& pn, const PointByField& p0);

  PetscErrorCode update_fields(const PointByField& pn, const PointByField& p0);
  void update_effective_field();
  void update_V_dgc();
  void update_V_par(const PointByField& pn, const PointByField& p0);
  void update_a_par();
  void update_r(PetscReal dt, PointByField& pn, const PointByField& p0);
  void update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0);

  /// @brief Checks convergence using both residuals.
  bool check_convergence(
    PetscReal dt, const PointByField& pn, const PointByField& p0);

  // Particle parameters
  PetscReal qm = 0;
  PetscReal mp = 0;

  SetFields set_fields;

  // Nonlinear iteration state and residuals
  PetscInt it = 0;
  PetscInt maxit = 100;
  // `atol` is a velocity, the position residual uses `dt * atol`. Both
  // residuals are also scaled by `rtol` with the step: the displacement and
  // |V_par| + |dt * a_par|, since the rounding error of the parallel velocity
  // is relative to it and not to its small increment.
  PetscReal atol = 1e-16;
  PetscReal rtol = 1e-13;
  PetscReal FRk, FVk;
  bool converged = false;

  // Right-hand side that built the current `pn`, see `check_convergence()`
  Vector3R V_dgc_prev;
  PetscReal a_par_prev;
  bool has_prev = false;  // False before the first update of `pn`.

  // Fields interpolated to the particle (set by `set_fields`), at t=n+1/2
  Vector3R E;      // Midpoint electric field.
  Vector3R b;      // Unit magnetic direction.
  Vector3R gradB;  // Magnetic magnitude gradient.
  Vector3R rotB;   // Magnetic field curl.
  PetscReal lenB;  // Magnetic field magnitude.

  // Quantities derived from the interpolated fields
  Vector3R B_eff;      // Effective magnetic field.
  PetscReal lenB_eff;  // Effective field parallel projection.
  Vector3R b_eff;      // Effective field scaled by its parallel projection.
  Vector3R F_eff;      // Effective force.

  // Integrated guiding-center motion
  Vector3R V_dgc;   // Guiding-center drift velocity.
  PetscReal V_par;  // Midpoint parallel velocity.
  PetscReal a_par;  // Midpoint parallel acceleration.
};

#endif  // SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H
