#include "drift_kinetic_push.h"

/// @file drift_kinetic_push.cpp
/// @brief Implementation of the fully implicit drift-kinetic guiding-center pusher.
/// @see drift_kinetic_push.h for the description of the algorithm and the mapping
/// of the members to the drift-kinetic equations of motion.

DriftKineticPush::DriftKineticPush(PetscReal qm, PetscReal mp)
  : qm(qm), mp(mp)
{
}

void DriftKineticPush::set_tolerances(
  PetscReal atol, PetscReal rtol, PetscInt maxit)
{
  this->atol = atol;
  this->rtol = rtol;
  this->maxit = maxit;
}

void DriftKineticPush::set_qm(PetscReal qm)
{
  this->qm = qm;
}

void DriftKineticPush::set_mp(PetscReal mp)
{
  this->mp = mp;
}

PetscReal DriftKineticPush::get_mp() const
{
  return this->mp;
}

PetscReal DriftKineticPush::get_qm() const
{
  return this->qm;
}

PetscInt DriftKineticPush::get_iteration_number() const
{
  return it;
}

bool DriftKineticPush::has_converged() const
{
  return converged;
}

void DriftKineticPush::set_fields_callback(SetFields&& callback)
{
  this->set_fields = std::move(callback);
}

/// @brief Solves the midpoint equations of motion as a fixed-point problem by
/// classical Picard iteration and reports non-convergence.
void DriftKineticPush::process(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  converged = false;
  FRk = FVhk = PETSC_MAX_REAL;

  process_picard(dt, pn, p0);

  if (!converged)
    LOG("WARNING: DK Push not converged: it={}, FRk={: .4e}, FVhk={: .4e}",
        it, FRk, FVhk);
}

/// @brief Classical Picard iteration over the nonlinear midpoint equations until
/// the position and parallel-momentum residuals fall below the tolerances.
void DriftKineticPush::process_picard(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  for (it = 0; it < maxit; ++it) {
    // Evaluate RHS at the current nonlinear guess pn (sets Vp, ah).
    evaluate_rhs(pn, p0);

    // Check the true nonlinear residual before taking the next step.
    if (check_discrepancy(dt, pn, p0)) { converged = true; return; }

    // Picard update of position and parallel momentum.
    update_r(dt, pn, p0);
    update_v_parallel(dt, pn, p0);
  }

  // Final check after the last Picard update.
  evaluate_rhs(pn, p0);
  converged = check_discrepancy(dt, pn, p0);
}

/// @brief Single Picard sweep: recompute the midpoint state, then update the
/// guiding-center position and parallel momentum from it.
void DriftKineticPush::step(const PetscReal dt, PointByField& pn, const PointByField& p0) {
  update_Vh(pn, p0);
  update_fields(pn, p0);
  update_Vp(pn, p0);
  update_ah(pn, p0);
  update_r(dt, pn, p0);
  update_v_parallel(dt, pn, p0);
}

void DriftKineticPush::evaluate_rhs(
  const PointByField& pn,
  const PointByField& p0)
{
  update_Vh(pn, p0);
  update_fields(pn, p0);
  update_Vp(pn, p0);
  update_ah(pn, p0);
}

/// @brief Drift-velocity equation: parallel streaming along the effective field
/// plus the @f$(q\mathbf{E}-\mu\nabla B)\times\mathbf{b}/(qB)@f$ drift.
void DriftKineticPush::update_Vp(const PointByField& pn, const PointByField& p0) {
  Vp = Vh * bh_eff + F_eff.cross(bh / (qm * mp * lenBh_eff));
}

/// @brief Midpoint parallel velocity @f$v_\parallel^{n+1/2}=(v^{n+1}+v^{n})/2@f$.
void DriftKineticPush::update_Vh(const PointByField& pn, const PointByField& p0) {
  Vh = 0.5 * (pn.p_parallel + p0.p_parallel);
}

/// @brief Parallel-momentum equation: effective force projected on the effective field.
void DriftKineticPush::update_ah(const PointByField& pn, const PointByField& p0) {
  ah = F_eff.dot(bh_eff) / mp;
}

bool DriftKineticPush::check_discrepancy(PetscReal dt, const PointByField& pn, const PointByField& p0){
  FRk = get_residue_r(dt, pn, p0);
  FVhk = get_residue_v(dt, pn, p0);

  // The residual of each implicit equation is measured against the size of the
  // update that equation produces over the step: the drift displacement
  // `dt * Vp` for the position and the parallel-velocity increment `dt * ah`.
  // This makes the test a bound on the *relative* error of the discrete
  // equations of motion, independent of the (arbitrary) coordinate origin and
  // of the absolute particle position. The relative scale is what carries the
  // tolerance; `atol` is only a floor guarding the degenerate case of a
  // vanishing update (Vp -> 0 or ah -> 0).
  const PetscReal r_scale = atol + rtol * (dt * Vp).length();
  const PetscReal v_scale = atol + rtol * std::abs(dt * ah);

  return (FRk <= r_scale) && (FVhk <= v_scale);
}

PetscReal DriftKineticPush::get_residue_r( PetscReal dt, const PointByField& pn, const PointByField& p0)
{
  return (pn.r - p0.r - dt * Vp).length();
}

PetscReal DriftKineticPush::get_residue_v(PetscReal dt, const PointByField& pn,
  const PointByField& p0)
{
  return std::abs(pn.p_parallel - p0.p_parallel - dt * ah);
}

/// @brief Implicit position update @f$\mathbf{R}^{n+1}=\mathbf{R}^{n}+\tau\mathbf{V}_p@f$.
void DriftKineticPush::update_r(PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pn.r = p0.r + dt * Vp;
}

/// @brief Implicit parallel-momentum update @f$v^{n+1}=v^{n}+\tau\,a@f$.
void DriftKineticPush::update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pn.p_parallel = p0.p_parallel + dt * ah;
}

/// @brief Refreshes the midpoint state: interpolates the fields to the particle,
/// then forms the curl of the unit field, the effective field @f$\mathbf{B}^{*}@f$
/// and the effective force @f$q\mathbf{E}-\mu\nabla B@f$ used by the motion equations.
void DriftKineticPush::update_fields(const PointByField& pn, const PointByField& p0) {
  // Interpolated fields at the midpoint of the trajectory.
  set_fields(p0.r, pn.r, Eh, lenBh, bh, gradBh, rotBh);

  // Derived quantities.
  rotbh = (bh.cross(gradBh) + rotBh) / lenBh;

  Bh_eff = lenBh * bh + (Vh / qm) * rotbh;
  lenBh_eff = bh.dot(Bh_eff);
  bh_eff = Bh_eff / lenBh_eff;

  F_eff = (qm * mp * Eh - p0.mu_p * gradBh);
}
