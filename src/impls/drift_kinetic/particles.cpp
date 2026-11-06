#include "particles.h"

#include "src/algorithms/drift_kinetic_push.h"
#include "src/algorithms/implicit_drift_kinetic.h"
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

  const PetscReal q = parameters.q;
  const PetscReal m = parameters.m;
  const PetscReal mpw = parameters.mpw;
  PushStatistics stats;

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attributes(f));
  PetscCall(storage.get_previous(f));
  const DkPointArrays arrays(f);
  const DkPointArrays arrays_prev(f, "_prev");

#pragma omp parallel reduction(drift_kinetic_stats : stats)
  {
    DriftKineticPush push(q / m, m);
    DriftKineticEsirkepov util(
      world, E_arr, Bn0_arr, B_arr, Bn1_arr, J_arr, M_arr);
    push.set_fields_callback(
      [&](const Vector3R& r0, const Vector3R& rn, Vector3R& E_p,
        PetscReal& lenB_p, Vector3R& b_p, Vector3R& gradB_p, Vector3R& rotB_p)
      {
        return util.interpolate(E_p, lenB_p, b_p, gradB_p, rotB_p, rn, r0);
      });

    AdaptiveSubstepping substeps(
      dt, q * mpw, mpw, substepping ? max_substep_depth : 0, stats);
#pragma omp for schedule(static)
    for (PetscInt i = 0; i < f.size; i++) {
      PointByField current = arrays.load(i);
      const PointByField previous = arrays_prev.load(i);
      substeps.process(current, previous, push, util);
      arrays.store(i, current);
    }
  }

  PetscCall(storage.restore_attributes(f));
  PetscCall(DMDAVecRestoreArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, M_loc, &M_arr));
  PetscCall(DMLocalToGlobal(da, J_loc, ADD_VALUES, J));
  PetscCall(DMLocalToGlobal(da, M_loc, ADD_VALUES, M));
  PetscCall(push_stats.finish(stats, PETSC_COMM_WORLD,
    parameters.sort_name, fail_on_terminal_nonconvergence));
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
  return push_stats.average_iterations();
}

PetscInt Particles::get_max_iteration_number() const
{
  return push_stats.max_iterations;
}

PetscInt Particles::get_push_retries() const
{
  return push_stats.retries;
}

PetscInt Particles::get_push_leaf_failures() const
{
  return push_stats.leaf_failures;
}

PetscReal Particles::get_max_push_leaf_residue_r() const
{
  return push_stats.max_leaf_residue_r;
}

PetscReal Particles::get_max_push_leaf_residue_v() const
{
  return push_stats.max_leaf_residue_v;
}

const PushStatistics& Particles::get_push_stats() const
{
  return push_stats;
}

void Particles::set_substepping(bool enabled, PetscInt depth)
{
  substepping = enabled;
  max_substep_depth = depth;
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
