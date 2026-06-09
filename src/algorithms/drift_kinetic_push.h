#ifndef SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H
#define SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H

#include "src/interfaces/particles.h"
#include "src/interfaces/point.h"
#include "src/utils/utils.h"

/// @file drift_kinetic_push.h
/// @brief Fully implicit guiding-center pusher for drift-kinetic electrons.
///
/// @details Advances the guiding center of a single electron over one timestep
/// by solving the drift-kinetic equations of motion with a fully implicit
/// (Crank-Nicolson) scheme. All field quantities are evaluated at the midpoint
/// of the trajectory @f$t = (n+1/2)\tau@f$, which makes the resulting set of
/// equations nonlinear; they are solved here by Picard (fixed-point) iteration.
class DriftKineticPush {
public:
  DriftKineticPush() = default;
  DriftKineticPush(PetscReal qm, PetscReal mp);

  /// @brief Sets numerical tolerances controlling the nonlinear iteration.
  void set_tolerances(PetscReal atol, PetscReal rtol, PetscInt maxit);

  /// @brief Charge-to-mass ratio, assumed constant during the iteration.
  void set_qm(PetscReal qm);
  /// @brief Macroparticle mass, assumed constant during the iteration.
  void set_mp(PetscReal mp);
  PetscReal get_mp() const;
  PetscReal get_qm() const;

  /// @brief Number of nonlinear iterations used by the last `process()` call.
  PetscInt get_iteration_number() const;

  /// @brief Returns true if the last call to `process()` converged within tolerances.
  bool has_converged() const;

  using SetFields = std::function<void(
    const Vector3R&, const Vector3R&, Vector3R&, Vector3R&, Vector3R&, Vector3R&, Vector3R&)>;

  /// @brief Callback supplying the midpoint field values interpolated to the
  /// particle: @f$\mathbf{E}@f$, @f$\mathbf{B}@f$, @f$\mathbf{b}@f$,
  /// @f$\nabla B@f$ and @f$\mathrm{rot}\,\mathbf{B}@f$.
  void set_fields_callback(SetFields&& callback);

  /// @brief Nonlinear move of point `pn` by timestep shift `dt`.
  /// @warning `pn` and `p0` cannot be the same as `pn` will be updated.
  void process(PetscReal dt, PointByField& pn, const PointByField& p0);

private:
  /// @brief One Picard sweep: refresh midpoint state and update position/momentum.
  void step(const PetscReal dt, PointByField& pn, const PointByField& p0);

  /// @brief Checks nonlinear residuals and determines convergence.
  bool check_discrepancy(PetscReal dt, const PointByField& pn, const PointByField& p0);
  PetscReal get_residue_r(PetscReal dt, const PointByField& pn, const PointByField& p0);
  PetscReal get_residue_v(PetscReal dt, const PointByField& pn, const PointByField& p0);

  /// @brief Evaluates the guiding-center drift velocity `Vp` (drift-velocity equation).
  void update_Vp(const PointByField& pn, const PointByField& p0);
  /// @brief Evaluates the midpoint parallel velocity `Vh`.
  void update_Vh(const PointByField& pn, const PointByField& p0);
  /// @brief Evaluates the parallel acceleration `ah` (parallel-momentum equation).
  void update_ah(const PointByField& pn, const PointByField& p0);
  /// @brief Advances the position from the drift velocity `Vp`.
  void update_r(PetscReal dt, PointByField& pn, const PointByField& p0);
  /// @brief Advances the parallel momentum from the acceleration `ah`.
  void update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0);

  /// @brief Refreshes interpolated fields and derived quantities at the midpoint.
  void update_fields(const PointByField& pn, const PointByField& p0);

  // -- Particle parameters -------------------------------------------------
  PetscReal qm = 0;  ///< Charge-to-mass ratio.
  PetscReal mp = 0;  ///< Macroparticle mass.

  SetFields set_fields;

  // -- Nonlinear iteration state and residuals -----------------------------
  PetscInt it = 0;          ///< Current iteration index.
  PetscInt maxit = 60;      ///< Maximum number of iterations.
  PetscReal atol = 1e-12;   ///< Absolute tolerance.
  PetscReal rtol = 1e-12;   ///< Relative tolerance.
  Vector3R dRk = Vector3R{0, 0, 0};  ///< Position increment of the previous iteration.
  PetscReal dVhk = 0;       ///< Parallel-velocity increment of the previous iteration.
  PetscReal FRk, FVhk;      ///< Position and velocity residuals.
  bool converged = false;

  // -- Fields interpolated to the particle (set by `set_fields`) -----------
  Vector3R Eh;      ///< Midpoint electric field @f$\mathbf{E}^{n+1/2}@f$.
  Vector3R Bh;      ///< Midpoint magnetic field @f$\mathbf{B}^{n+1/2}@f$.
  Vector3R bh;      ///< Unit vector along @f$\mathbf{B}^{n+1/2}@f$.
  Vector3R gradBh;  ///< Gradient of the field magnitude @f$\nabla B@f$.
  Vector3R rotBh;   ///< Curl of the magnetic field @f$\mathrm{rot}\,\mathbf{B}@f$.

  // -- Quantities derived from the interpolated fields ---------------------
  PetscReal lenBh;      ///< Magnitude of the magnetic field @f$|\mathbf{B}^{n+1/2}|@f$.
  Vector3R rotbh;       ///< Curl of the unit field @f$\mathrm{rot}\,\mathbf{b}@f$.
  Vector3R Bh_eff;      ///< Effective magnetic field @f$\mathbf{B}^{*}@f$.
  PetscReal lenBh_eff;  ///< Parallel projection @f$B^{*}_\parallel@f$.
  Vector3R bh_eff;      ///< Effective field direction @f$\mathbf{B}^{*}/B^{*}_\parallel@f$.
  Vector3R F_eff;       ///< Effective force @f$q\mathbf{E}-\mu\nabla B@f$.

  // -- Integrated guiding-center motion ------------------------------------
  Vector3R Vp;     ///< Guiding-center drift velocity.
  PetscReal Vh;    ///< Midpoint parallel velocity @f$v_\parallel^{n+1/2}@f$.
  PetscReal ah;    ///< Midpoint parallel acceleration.
};

#endif  // SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H
