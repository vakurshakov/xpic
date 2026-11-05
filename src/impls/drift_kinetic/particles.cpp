#include "particles.h"

#include "src/algorithms/drift_kinetic_push.h"
#include "src/algorithms/implicit_drift_kinetic.h"
#include "src/impls/drift_kinetic/adaptive_substepping.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/utils.h"

namespace drift_kinetic {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world, parameters), simulation_(simulation)
{
}

PetscErrorCode Particles::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(interfaces::Particles::initialize());

  PetscCall(storage.initialize());
  PetscCall(storage.register_attribute("r", 3));
  PetscCall(storage.register_attribute("p", 3));
  PetscCall(storage.register_attribute("p_parallel", 1));
  PetscCall(storage.register_attribute("mu_p", 1));
  PetscCall(storage.register_previous("r"));
  PetscCall(storage.register_previous("p"));
  PetscCall(storage.register_previous("p_parallel"));
  PetscCall(storage.register_previous("mu_p"));
  PetscCall(storage.commit_attributes());

  PetscCall(DMCreateGlobalVector(da, &M));
  PetscCall(DMCreateLocalVector(da, &M_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&M));
  PetscCall(VecDestroy(&M_loc));
  PetscCall(interfaces::Particles::finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::form_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMDAVecGetArrayWrite(da, M_loc, &M_arr));

  avgit = 0;
  maxit = 0;
  retries = 0;
  leaf_failures = 0;
  max_leaf_residue_r = 0;
  max_leaf_residue_v = 0;

  const PetscReal q = parameters.q;
  const PetscReal m = parameters.m;
  const PetscReal mpw = parameters.n / (PetscReal)parameters.Np;

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attributes(f));
  PetscCall(storage.get_previous(f));
  const DkPointArrays arrays(f);
  const DkPointArrays arrays_prev(f, "_prev");

#pragma omp parallel for reduction(+ : avgit, retries, leaf_failures) \
  reduction(max : maxit, max_leaf_residue_r, max_leaf_residue_v)
  for (PetscInt i = 0; i < f.size; i++) {
    PointByField curr = arrays.load(i);
    PointByField prev = arrays_prev.load(i);

    DriftKineticPush push(q / m, m);
    DriftKineticEsirkepov util(world, E_arr, Bn0_arr, B_arr, Bn1_arr, J_arr, M_arr);

    push.set_fields_callback(
      [&](const Vector3R& r0, const Vector3R& rn, Vector3R& E_p,
        PetscReal& lenB_p, Vector3R& b_p, Vector3R& gradB_p, Vector3R& rotB_p)
      {
        util.interpolate(E_p, lenB_p, b_p, gradB_p, rotB_p, rn, r0);
      });

    auto attempt = [&](PetscReal dtau, PointByField& pn, const PointByField& p0)
    {
      push.process(dtau, pn, p0);
      return AdaptiveSubstepAttempt{
        push.has_converged(),
        push.get_iteration_number(),
        push.get_FRk(),
        push.get_FVk(),
      };
    };

    auto accept = [&](PetscReal dtau, PointByField& pn, const PointByField& p0)
    {
      const PetscReal a0 = q * mpw * (dtau / dt);
      const PetscReal b0 = pn.mu_p * mpw * (dtau / dt);
      const Vector3R Vh_sub = (pn.r - p0.r) / dtau;
      util.decomposition(pn.r, p0.r, Vh_sub, a0, b0);
    };

    // Halving does not conserve energy: every substep is pushed and
    // deposited with the whole-step layers B^n, B^{n+1}, so the mu B terms
    // of the substeps do not telescope. The whole step is accepted instead,
    // a non-converged push is counted as a leaf failure.
    AdaptiveSubstepStats stats;
    const AdaptiveSubstepAttempt result = attempt(dt, curr, prev);
    accept(dt, curr, prev);

    stats.iteration_sum = result.iterations;
    stats.max_iterations = result.iterations;

    if (!result.converged) {
      stats.leaf_failures = 1;
      stats.max_leaf_residue_r =
        isfinite(result.residue_r) ? result.residue_r : PETSC_MAX_REAL;
      stats.max_leaf_residue_v =
        isfinite(result.residue_v) ? result.residue_v : PETSC_MAX_REAL;
    }

    avgit += (PetscReal)stats.iteration_sum / (PetscReal)size;
    maxit = std::max(maxit, stats.max_iterations);
    retries += stats.retries;
    leaf_failures += stats.leaf_failures;
    max_leaf_residue_r = std::max(max_leaf_residue_r, stats.max_leaf_residue_r);
    max_leaf_residue_v = std::max(max_leaf_residue_v, stats.max_leaf_residue_v);

    curr.p = (curr.r - prev.r) / dt;
    arrays.store(i, curr);
  }

  PetscCall(storage.restore_attributes(f));
  PetscCall(DMDAVecRestoreArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, M_loc, &M_arr));
  PetscCall(DMLocalToGlobal(da, J_loc, ADD_VALUES, J));
  PetscCall(DMLocalToGlobal(da, M_loc, ADD_VALUES, M));

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &retries, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &leaf_failures, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &max_leaf_residue_r, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &max_leaf_residue_v, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));

  if (leaf_failures > 0 && !fail_on_terminal_nonconvergence) {
    LOG("WARNING: DK Push terminal non-convergence for sort={}:", parameters.sort_name);
    LOG("  leaf_failures={}, max FRk={: .4e}, max FVk={: .4e}", leaf_failures, max_leaf_residue_r, max_leaf_residue_v);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Appends the kinetic points converted to the guiding-center form.
PetscErrorCode Particles::ingest(const Point* points, PetscInt n)
{
  PetscFunctionBeginUser;
  const PetscReal qm = parameters.q / parameters.m;
  const PetscReal mp = parameters.m;

  PetscCall(DMGlobalToLocal(simulation_.da, simulation_.B, INSERT_VALUES, simulation_.B_loc));
  PetscCall(DMDAVecGetArrayRead(simulation_.da, simulation_.B_loc, &simulation_.B_arr));

  DriftKineticEsirkepov esirkepov(world, simulation_.B_arr);

  PetscInt first;
  PetscCall(storage.grow(n, &first));

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attributes(f));
  const DkPointArrays arrays(f);

  for (PetscInt i = 0; i < n; i++) {
    const Point& point = points[i];

    Vector3R B_p{};
    PetscCall(esirkepov.interpolate_B(B_p, point.r));
    arrays.store(first + i,
      coord_is_gc ? PointByField::from_guiding_center(point, B_p, mp)
                  : PointByField::from_particle_position(point, B_p, mp, qm));
  }

  PetscCall(storage.restore_attributes(f));
  PetscCall(DMDAVecRestoreArrayRead(simulation_.da, simulation_.B_loc, &simulation_.B_arr));

  // The guiding center may lie outside of the box of the particle
  PetscCall(storage.update_cells());
  PetscCall(storage.sort());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Saves current states before the nonlinear iteration.
PetscErrorCode Particles::prepare_storage()
{
  PetscFunctionBeginUser;
  PetscCall(storage.get_size(&size));
  PetscCall(storage.save_previous());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Particles::kinetic_energy_local() const
{
  PetscReal energy = 0.0;
  const PetscReal mpw = parameters.n / static_cast<PetscReal>(parameters.Np);
  PetscCallAbort(PETSC_COMM_WORLD, DMGlobalToLocal(simulation_.da, simulation_.B, INSERT_VALUES, simulation_.B_loc));
  PetscCallAbort(PETSC_COMM_WORLD, DMDAVecGetArrayRead(simulation_.da, simulation_.B_loc, &simulation_.B_arr));

  DriftKineticEsirkepov esirkepov(world, simulation_.B_arr);

  ParticleStorage::Attributes f;
  PetscCallAbort(PETSC_COMM_WORLD, storage.get_attributes(f));
  const DkPointArrays arrays(f);

#pragma omp parallel for reduction(+ : energy)
  for (PetscInt i = 0; i < f.size; i++) {
    const PointByField point = arrays.load(i);

    Vector3R B_p{};
    esirkepov.interpolate_B(B_p, point.r);

    energy += POW2(point.p_parallel) +  //
      2.0 * point.mu_p * B_p.length() / parameters.m;
  }

  PetscCallAbort(PETSC_COMM_WORLD, storage.restore_attributes(f));
  PetscCallAbort(PETSC_COMM_WORLD, DMDAVecRestoreArrayRead(simulation_.da, simulation_.B_loc, &simulation_.B_arr));
  return 0.5 * parameters.m * mpw * energy;
}

PetscReal Particles::get_average_iteration_number() const
{
  return avgit;
}

PetscInt Particles::get_max_iteration_number() const
{
  return maxit;
}

PetscInt Particles::get_push_retries() const
{
  return retries;
}

PetscInt Particles::get_push_leaf_failures() const
{
  return leaf_failures;
}

PetscReal Particles::get_max_push_leaf_residue_r() const
{
  return max_leaf_residue_r;
}

PetscReal Particles::get_max_push_leaf_residue_v() const
{
  return max_leaf_residue_v;
}

void Particles::set_fail_on_terminal_nonconvergence(bool value)
{
  fail_on_terminal_nonconvergence = value;
}

bool Particles::get_fail_on_terminal_nonconvergence() const
{
  return fail_on_terminal_nonconvergence;
}

const Simulation& Particles::simulation() const
{
  return simulation_;
}

void Particles::set_coord_is_gc(bool v)
{
  coord_is_gc = v;
}

}  // namespace drift_kinetic
