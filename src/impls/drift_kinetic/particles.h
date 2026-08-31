#ifndef SRC_DRIFT_KINETIC_PARTICLES_H
#define SRC_DRIFT_KINETIC_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/shape.h"

namespace drift_kinetic {

class Simulation;
class InjectParticles;
class SetPairedParticles;

/// @brief Builds a `PointByField` treating `point.r` as the guiding center
/// (no Larmor shift). Parallel momentum, perpendicular momentum and magnetic
/// moment are still computed from the locally interpolated `Bp`.
PointByField make_point_at_gc(
  const Point& point, const Vector3R& Bp, PetscReal mp);

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const SortParameters& parameters);
  PetscErrorCode initialize_point_by_field(const Arr B_arr);
  PetscErrorCode finalize() override;

  PetscErrorCode sync_dk_curr_storage();
  PetscErrorCode prepare_storage();
  PetscErrorCode form_iteration();
  PetscErrorCode restore_from_prev_storage();
  void set_energy_audit(bool value) { energy_audit_enabled_ = value; }

  /// @brief Three-level decomposition of the per-particle energy defect,
  /// filled by `form_iteration()` when the energy audit is enabled (i.e. on
  /// the final, post-SNES deposition of the step). Each level mirrors one
  /// discrete identity of the scheme; all sums carry the deposition weight
  /// `n_Np`, so `D_total` is directly comparable to the global
  /// `dK-dMB+dt*dEJ` column:
  ///  - `D_push`  = sum of dK_par + mu*dt*V.gradB - q*dt*E.V, the discrete
  ///    parallel-work identity with V = (R1-R0)/dt — nonzero means the
  ///    pusher did not solve the discrete equations (or the Vp/ah formulas
  ///    are inconsistent);
  ///  - `D_gradB` = sum of mu*(dt*V.gradB - hatb.(B^{n+1/2}(R1) -
  ///    B^{n+1/2}(R0))), the energy-consistent gradB interpolation rule with
  ///    hatb = b/|b|^2, b = (b^n + b^{n+1})/2 — nonzero (with small D_push)
  ///    points at `interpolate_gradB()`, segmentation or periodic wrapping;
  ///  - `D_total` = sum of dK_par + mu*(B^{n+1}(R1) - B^n(R0)) - W_E - W_M,
  ///    where W_M is the particle's share of M^{n+1/2}.(B^{n+1}-B^n) exactly
  ///    as `decomposition_M()` deposits it — the particle-side estimate of
  ///    the global defect. With exact identities D_total = D_push - D_gradB;
  ///    if the first two are small but this is not, look at the M deposition
  ///    weights, time interpolation of B or substepping.
  /// `max_*` are the largest per-particle magnitudes (same weights).
  struct EnergyAudit {
    PetscReal D_push = 0.0;
    PetscReal D_gradB = 0.0;
    PetscReal D_total = 0.0;
    PetscReal max_D_push = 0.0;
    PetscReal max_D_gradB = 0.0;
  };

  const EnergyAudit& energy_audit() const { return audit_; }

  PetscReal kinetic_energy_local() const;
  PetscReal get_average_iteration_number() const;
  PetscInt get_max_iteration_number() const;
  PetscInt get_push_retries() const { return push_retries_; }
  PetscInt get_push_leaf_failures() const { return push_leaf_failures_; }
  PetscReal get_max_push_leaf_residue_r() const
  {
    return max_push_leaf_residue_r_;
  }
  PetscReal get_max_push_leaf_residue_v() const
  {
    return max_push_leaf_residue_v_;
  }

  void set_fail_on_terminal_nonconvergence(bool value)
  {
    fail_on_terminal_nonconvergence_ = value;
  }
  bool fail_on_terminal_nonconvergence() const
  {
    return fail_on_terminal_nonconvergence_;
  }
  const std::vector<std::list<PointByField>>& get_dk_curr_storage() const
  {
    return dk_curr_storage;
  }

  /// @brief Owning simulation, used by diagnostics that need the current
  /// fields (e.g. interpolating |B| at the gyrocenter).
  const Simulation& simulation() const { return simulation_; }

  void set_coord_is_gc(bool v) { coord_is_gc_ = v; }
  bool coord_is_gc() const { return coord_is_gc_; }

  Vec M;
  Vec M_loc;
  Arr M_arr;

  Arr Bn_arr;
  Arr Bn1_arr;

protected:
  PetscReal n_Np(const PointByField& point) const;
  PetscReal qn_Np(const PointByField& point) const;
  PetscErrorCode update_cells_seq();
  PetscErrorCode update_cells_mpi();
  PetscErrorCode correct_coordinates(PointByField& point);
  std::vector<std::list<PointByField>> dk_curr_storage;
  std::vector<std::vector<PointByField>> dk_prev_storage;
  PetscInt size = 0;
  PetscReal avgit = 0.0;
  PetscInt maxit = 0;
  PetscInt push_retries_ = 0;
  PetscInt push_leaf_failures_ = 0;
  PetscReal max_push_leaf_residue_r_ = 0.0;
  PetscReal max_push_leaf_residue_v_ = 0.0;
  bool fail_on_terminal_nonconvergence_ = false;
  bool coord_is_gc_ = false;
  bool energy_audit_enabled_ = false;
  EnergyAudit audit_;
  Simulation& simulation_;


  friend class InjectParticles;
  friend class SetPairedParticles;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_PARTICLES_H
