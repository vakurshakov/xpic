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
  using SetFields = std::function<void(const Vector3R&, const Vector3R&,
    Vector3R&, PetscReal&, Vector3R&, Vector3R&, Vector3R&)>;

  void set_fields_callback(SetFields&& callback);

  /// @warning `pn` and `p0` must be distinct objects.
  void process(PetscReal dt, PointByField& pn, const PointByField& p0);

private:
  void process_picard(PetscReal dt, PointByField& pn, const PointByField& p0);

  /// @brief Performs one Picard update.
  void step(const PetscReal dt, PointByField& pn, const PointByField& p0);

  /// @brief Evaluates drift velocity and acceleration.
  void evaluate_rhs(const PointByField& pn, const PointByField& p0);

  void update_fields(const PointByField& pn, const PointByField& p0);
  void update_V_dgc(const PointByField& pn, const PointByField& p0);
  void update_V_par(const PointByField& pn, const PointByField& p0);
  void update_a_par(const PointByField& pn, const PointByField& p0);
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
  PetscReal atol = 1e-12;
  PetscReal rtol = 1e-12;
  PetscReal FRk, FVk;
  bool converged = false;

  // Fields interpolated to the particle (set by `set_fields`), at t=n+1/2
  Vector3R E;      // Midpoint electric field.
  Vector3R b;      // Unit magnetic direction.
  Vector3R gradB;  // Magnetic magnitude gradient.
  Vector3R rotB;   // Magnetic field curl.
  PetscReal lenB;  // Magnetic field magnitude.

  // Quantities derived from the interpolated fields
  Vector3R rotb;       // Unit magnetic direction curl.
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
