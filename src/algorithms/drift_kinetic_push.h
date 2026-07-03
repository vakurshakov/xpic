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
/// equations nonlinear. They are solved as a fixed-point problem on the Picard
/// map @f$G@f$, either by a safeguarded, adaptively damped Anderson(1)
/// acceleration (@ref DriftKineticSolver::DKANDERSON, the default) or by the
/// classical Picard iteration (@ref DriftKineticSolver::DKPICARD).

/// @brief Selects the nonlinear fixed-point solver used by `DriftKineticPush::process()`.
enum class DriftKineticSolver {
  DKANDERSON,  ///< Safeguarded, adaptively damped Anderson(1) acceleration (default).
  DKPICARD,    ///< Classical (undamped) Picard fixed-point iteration.
};

class DriftKineticPush {
public:
  DriftKineticPush() = default;
  DriftKineticPush(PetscReal qm, PetscReal mp);

  /// @brief Selects the nonlinear solver used by `process()`.
  void set_solver(DriftKineticSolver solver);

  /// @brief Sets numerical tolerances controlling the nonlinear iteration.
  void set_tolerances(PetscReal atol, PetscReal rtol, PetscInt maxit);

  /// @brief Tunes the Anderson(1) under-relaxation safeguards (base and minimum damping).
  void set_relaxation(PetscReal beta0, PetscReal beta_min);
  /// @brief Sets the clipping bounds of the Anderson(1) extrapolation coefficient.
  void set_anderson_clip(PetscReal gamma_min, PetscReal gamma_max);
  /// @brief Sets the growth factor of @f$\|f\|@f$ above which the iteration is
  /// treated as diverging and control is handed back to the outer substep logic.
  void set_divergence_factor(PetscReal div_factor);

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
    const Vector3R&, const Vector3R&, Vector3R&, PetscReal&, Vector3R&, Vector3R&, Vector3R&)>;

  /// @brief Callback supplying the midpoint field values interpolated to the
  /// particle: @f$\mathbf{E}@f$, @f$\mathbf{B}@f$, @f$\mathbf{b}@f$,
  /// @f$\nabla B@f$ and @f$\mathrm{rot}\,\mathbf{B}@f$.
  void set_fields_callback(SetFields&& callback);

  /// @brief Nonlinear move of point `pn` by timestep shift `dt`.
  /// @warning `pn` and `p0` cannot be the same as `pn` will be updated.
  void process(PetscReal dt, PointByField& pn, const PointByField& p0);

private:
  /// @brief Safeguarded, adaptively damped Anderson(1) acceleration of the
  /// Picard map. The map and its residual are
  /// @f$G(x) = (R^n + \tau\,V_p(x),\ v^n + \tau\,a_h(x))@f$ and
  /// @f$f(x) = G(x) - x@f$ with @f$x = (R^{n+1}, v_\parallel^{n+1})@f$.
  void process_anderson(PetscReal dt, PointByField& pn, const PointByField& p0);
  /// @brief Classical Picard (fixed-point) iteration over the same map.
  void process_picard(PetscReal dt, PointByField& pn, const PointByField& p0);

  /// @brief Working state of the Anderson(1) acceleration during one solve: the
  /// current iterate/image/residual @f$(x, g, f)@f$, the previous triple kept as
  /// history and the adaptive under-relaxation coefficient.
  struct AndersonState {
    Vector3R  x_r{}, g_r{}, f_r{};        ///< Current iterate, image and residual (position).
    PetscReal x_v = 0, g_v = 0, f_v = 0;  ///< Current iterate, image and residual (parallel momentum).
    Vector3R  xr_prev{}, gr_prev{}, fr_prev{};        ///< Previous iterate/image/residual (position).
    PetscReal xv_prev = 0, gv_prev = 0, fv_prev = 0;  ///< Previous iterate/image/residual (parallel momentum).
    PetscReal fnorm_prev = PETSC_MAX_REAL;  ///< ‖f‖ of the previous iteration.
    PetscReal fnorm0 = -1.0;                ///< ‖f‖ of the first iteration (divergence reference).
    PetscReal beta = 0;                     ///< Current under-relaxation coefficient.
    bool have_history = false;              ///< Whether a previous triple is available.
  };

  /// @brief Evaluates the Picard map at the current iterate `pn`: records the
  /// iterate, forms the image `g`, the residual `f = g - x` and its norm into `st`.
  void evaluate_map(PetscReal dt, const PointByField& pn, const PointByField& p0, AndersonState& st);
  /// @brief True if the residual norm is non-finite or has grown past `div_factor`.
  bool is_diverging(const AndersonState& st) const;
  /// @brief Safeguarded Anderson(1) (or damped-Picard fallback) update of `pn`.
  void anderson_update(PointByField& pn, AndersonState& st);
  /// @brief Stores the current iterate/image/residual as history for the next step.
  void shift_history(AndersonState& st);

  /// @brief One Picard sweep: refresh midpoint state and update position/momentum.
  void step(const PetscReal dt, PointByField& pn, const PointByField& p0);
  /// @brief Evaluates RHS Vp and ah for the current nonlinear state pn.
  void evaluate_rhs(const PointByField& pn, const PointByField& p0);
  /// @brief Residual norm of the last process() call (for substep control).
  PetscReal get_residual_norm() const;

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

  // -- Solver selection ----------------------------------------------------
  DriftKineticSolver solver = DriftKineticSolver::DKANDERSON;  ///< Nonlinear solver used by `process()`.

  // -- Nonlinear iteration state and residuals -----------------------------
  PetscInt it = 0;
  PetscInt maxit = 60;
  PetscReal atol = 1e-12;
  PetscReal rtol = 1e-12;
  PetscReal FRk, FVhk;
  bool converged = false;

  // -- Anderson(1) / adaptive damping --------------------------------------
  PetscReal beta0 = 0.7;       ///< Base under-relaxation coefficient.
  PetscReal beta_min = 0.05;   ///< Lower bound on the damping coefficient.
  PetscReal gamma_min = -5.0, gamma_max = 5.0;  ///< Clipping of the Anderson step.
  PetscReal div_factor = 1e4;  ///< Growth of ‖f‖ that is treated as divergence.
  PetscReal fnorm = 0;         ///< ‖f‖ of the last iteration (diagnostic).

  // -- Fields interpolated to the particle (set by `set_fields`) -----------
  Vector3R Eh;      ///< Midpoint electric field @f$\mathbf{E}^{n+1/2}@f$.
  Vector3R bh;      ///< Unit vector along @f$\mathbf{B}^{n+1/2}@f$.
  Vector3R gradBh;  ///< Gradient of the field magnitude @f$\nabla B@f$.
  Vector3R rotBh;   ///< Curl of the magnetic field @f$\mathrm{rot}\,\mathbf{B}@f$.
  PetscReal lenBh;      ///< Magnitude of the magnetic field @f$|\mathbf{B}^{n+1/2}|@f$.

  // -- Quantities derived from the interpolated fields ---------------------
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
