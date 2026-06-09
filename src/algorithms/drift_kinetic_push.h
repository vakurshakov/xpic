#ifndef SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H
#define SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H

#include "src/interfaces/particles.h"
#include "src/interfaces/point.h"
#include "src/utils/utils.h"

class DriftKineticPush {
public:
  DriftKineticPush() = default;
  DriftKineticPush(PetscReal qm, PetscReal mp);

  /// @brief Sets numerical tolerances controlling iterative process.
  void set_tolerances(PetscReal atol, PetscReal rtol, PetscInt maxit);

  /// @brief Assumed to be constant during the iteration.
  void set_qm(PetscReal qm);
  void set_mp(PetscReal mp);
  PetscReal get_mp() const;
  PetscReal get_qm() const;

  /// @brief Once `process()` is complete, this will return the number of nonlinear iterations used.
  PetscInt get_iteration_number() const;

  /// @brief Returns true if the last call to `process()` converged within tolerances
  bool has_converged() const;

  using SetFields = std::function<void(
    const Vector3R&, const Vector3R&, Vector3R&, Vector3R&, Vector3R&, Vector3R&, Vector3R&)>;

  /// @brief Callback signature for supplying field values and gradients.
  void set_fields_callback(SetFields&& callback);

  /// @brief Nonlinear move of point `pn` by timestep shift `dt`.
  /// @warning `pn` and `p0` cannot be the same as `pn` will be updated.
  void process(PetscReal dt, PointByField& pn, const PointByField& p0);

private:

  /// @brief Initializes cached fields and unit vectors before the nonlinear loop.
  void step(const PetscReal dt, PointByField& pn, const PointByField& p0);

  /// @brief Checks nonlinear residuals and determines convergence.
  bool check_discrepancy(PetscReal dt,const PointByField& pn, const PointByField& p0);
  PetscReal get_residue_r(PetscReal dt, const PointByField& pn, const PointByField& p0);
  PetscReal get_residue_v(PetscReal dt, const PointByField& pn, const PointByField& p0);

  /// @brief Evaluates drift velocity `Vp` for the current iteration.
  void update_Vp(const PointByField& pn, const PointByField& p0);
  void update_Vh(const PointByField& pn, const PointByField& p0);
  void update_ah(const PointByField& pn, const PointByField& p0);
  /// @brief Updates the spatial position using drift velocity `Vp`.
  void update_r(PetscReal dt, PointByField& pn, const PointByField& p0);
  /// @brief Advances the parallel momentum using the drift-kinetic equation.
  void update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0);

  /// @brief Refreshes fields and derived quantities at the midpoint trajectory.
  void update_fields(const PointByField& pn, const PointByField& p0);

  /// @brief Iteration counters and tolerances.
  PetscInt it = 0;
  PetscInt maxit = 30;
  PetscReal atol = 1e-10;
  PetscReal rtol = 1e-10;
  Vector3R dRk = Vector3R{0, 0, 0};
  PetscReal dVhk = 0;
  PetscReal FRk, FVhk;
  bool converged = false;

  /// @brief Particle parameters.
  PetscReal qm = 0;
  PetscReal mp = 0;

  SetFields set_fields;

  /// @brief Cached velocities and fields.
  PetscReal Vh;
  PetscReal ah;
  Vector3R Vp;

  Vector3R Eh;
  Vector3R Bh;
  Vector3R gradBh;
  Vector3R rotBh;

  /// @brief Unit vector aligned with `Bh`.
  Vector3R bh;
  Vector3R rotbh;
  Vector3R Bh_eff;
  Vector3R bh_eff;
  Vector3R F_eff;
  /// @brief Magnitude of the magnetic field.
  PetscReal lenBh;
  PetscReal lenBh_eff;
};

#endif  // SRC_ALGORITHMS_DRIFT_KINETIC_PUSH_H
