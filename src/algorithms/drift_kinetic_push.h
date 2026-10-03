#ifndef SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H
#define SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H

#include "src/interfaces/particles.h"
#include "src/interfaces/point.h"
#include "src/utils/utils.h"

/// @file drift_kinetic_push.h
/// @brief Implicit drift-kinetic particle pusher.

/// @brief Advances guiding-center position and parallel velocity.
class DriftKineticPush {
public:
  /// @brief Creates a pusher with unset particle parameters.
  DriftKineticPush() = default;
  /// @brief Sets the charge-to-mass ratio and particle mass.
  DriftKineticPush(PetscReal qm, PetscReal mp);

  /// @brief Sets iteration tolerances and limit.
  void set_tolerances(PetscReal atol, PetscReal rtol, PetscInt maxit);

  /// @brief Sets the charge-to-mass ratio.
  void set_qm(PetscReal qm);
  /// @brief Sets the particle mass.
  void set_mp(PetscReal mp);
  /// @brief Returns the particle mass.
  PetscReal get_mp() const;
  /// @brief Returns the charge-to-mass ratio.
  PetscReal get_qm() const;

  /// @brief Returns the last iteration count.
  PetscInt get_iteration_number() const;

  /// @brief Reports convergence of the last move.
  bool has_converged() const;

  /// @brief Returns the last position residual.
  PetscReal get_FRk() const;

  /// @brief Returns the last parallel velocity residual.
  PetscReal get_FVhk() const;

  /// @brief Callback for interpolating particle fields.
  using SetFields = std::function<void(
    const Vector3R&, const Vector3R&, Vector3R&, PetscReal&, Vector3R&, Vector3R&, Vector3R&)>;

  /// @brief Sets the field interpolation callback.
  void set_fields_callback(SetFields&& callback);

  /// @brief Advances a particle by one timestep.
  /// @warning `pn` and `p0` must be distinct objects.
  void process(PetscReal dt, PointByField& pn, const PointByField& p0);

private:
  /// @brief Solves the particle move with Picard iteration.
  void process_picard(PetscReal dt, PointByField& pn, const PointByField& p0);

  /// @brief Performs one Picard update.
  void step(const PetscReal dt, PointByField& pn, const PointByField& p0);
  /// @brief Evaluates drift velocity and acceleration.
  void evaluate_rhs(const PointByField& pn, const PointByField& p0);

  /// @brief Checks convergence using both residuals.
  bool check_discrepancy(PetscReal dt, const PointByField& pn, const PointByField& p0);
  /// @brief Computes the position residual.
  PetscReal get_residue_r(PetscReal dt, const PointByField& pn, const PointByField& p0);
  /// @brief Computes the parallel velocity residual.
  PetscReal get_residue_v(PetscReal dt, const PointByField& pn, const PointByField& p0);

  /// @brief Updates the guiding-center drift velocity.
  void update_Vp(const PointByField& pn, const PointByField& p0);
  /// @brief Updates the midpoint parallel velocity.
  void update_Vh(const PointByField& pn, const PointByField& p0);
  /// @brief Updates the parallel acceleration.
  void update_ah(const PointByField& pn, const PointByField& p0);
  /// @brief Updates the position.
  void update_r(PetscReal dt, PointByField& pn, const PointByField& p0);
  /// @brief Updates the parallel velocity.
  void update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0);

  /// @brief Updates fields and derived quantities.
  void update_fields(const PointByField& pn, const PointByField& p0);

  // -- Particle parameters -------------------------------------------------
  PetscReal qm = 0;  ///< Charge-to-mass ratio.
  PetscReal mp = 0;  ///< Macroparticle mass.

  SetFields set_fields;

  // -- Nonlinear iteration state and residuals -----------------------------
  PetscInt it = 0;
  PetscInt maxit = 100;
  PetscReal atol = 1e-12;
  PetscReal rtol = 1e-12;
  PetscReal FRk, FVhk;
  bool converged = false;

  // -- Fields interpolated to the particle (set by `set_fields`) -----------
  Vector3R Eh;      ///< Midpoint electric field.
  Vector3R bh;      ///< Unit magnetic direction.
  Vector3R gradBh;  ///< Magnetic magnitude gradient.
  Vector3R rotBh;   ///< Magnetic field curl.
  PetscReal lenBh;  ///< Magnetic field magnitude.

  // -- Quantities derived from the interpolated fields ---------------------
  Vector3R rotbh;       ///< Unit magnetic direction curl.
  Vector3R Bh_eff;      ///< Effective magnetic field.
  PetscReal lenBh_eff;  ///< Effective field parallel projection.
  Vector3R bh_eff;      ///< Effective field scaled by its parallel projection.
  Vector3R F_eff;       ///< Effective force.

  // -- Integrated guiding-center motion ------------------------------------
  Vector3R Vp;     ///< Guiding-center drift velocity.
  PetscReal Vh;    ///< Midpoint parallel velocity.
  PetscReal ah;    ///< Midpoint parallel acceleration.
};

#endif  // SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H
