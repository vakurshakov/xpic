#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/diagnostics/particles_energy.h"
#include "src/impls/ecsimcorr/simulation.h"

namespace ecsimcorr {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : ecsim::Particles(simulation, parameters), simulation_(simulation)
{
  DM da = world.da;
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &local_currJe));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &global_currJe));

  PetscCallAbort(PETSC_COMM_WORLD, PetscClassIdRegister("ecsimcorr::Particles", &classid));
  PetscCallAbort(PETSC_COMM_WORLD, PetscLogEventRegister("first_push", classid, &events[0]));
  PetscCallAbort(PETSC_COMM_WORLD, PetscLogEventRegister("second_push", classid, &events[1]));
  PetscCallAbort(PETSC_COMM_WORLD, PetscLogEventRegister("final_update", classid, &events[2]));
}

PetscErrorCode Particles::first_push()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArray(world.da, local_currJe, &currJe));

  PetscLogEventBegin(events[0], local_currJe, 0, 0, 0);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
      const Vector3R old_r = point.r;
      BorisPush::update_r((0.5 * dt), point);

      Shape shape;
      shape.setup(old_r, point.r, shape_radius2, shape_func2);
      decompose_esirkepov_current(shape, point);
    }
  }

  PetscLogEventEnd(events[0], local_currJe, 0, 0, 0);

  PetscCall(DMDAVecRestoreArray(world.da, local_currJe, &currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::second_push()
{
  PetscFunctionBeginUser;
  pred_w = 0.0;
  PetscCall(DMDAVecGetArray(world.da, local_currJe, &currJe));

  PetscLogEventBegin(events[1], 0, 0, 0, 0);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
      const Vector3R old_r = point.r;
      const Vector3R old_v = point.p;

      Shape shape;
      shape.setup(point.r, shape_radius1, shape_func1);

      Vector3R E_p;
      Vector3R B_p;
      SimpleInterpolation interpolation(shape);
      interpolation.process({{E_p, E}}, {{B_p, B}});

      BorisPush push(charge(point) / mass(point), E_p, B_p);
      push.update_vEB(dt, point);
      push.update_r((0.5 * dt), point);

      shape.setup(old_r, point.r, shape_radius2, shape_func2);
      decompose_esirkepov_current(shape, point);

      PetscReal frac = charge(point) * density(point) / particles_number(point);

#pragma omp atomic update
      pred_w += 0.5 * (old_v + point.p).dot(E_p) * frac;
    }
  }

  PetscLogEventEnd(events[1], 0, 0, 0, 0);

  PetscCall(DMDAVecRestoreArray(world.da, local_currJe, &currJe));
  PetscCall(DMLocalToGlobal(world.da, local_currJe, ADD_VALUES, global_currJe));
  PetscCall(VecAXPY(simulation_.currJe, 1.0, global_currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::final_update()
{
  PetscFunctionBeginUser;
  PetscCall(VecDot(global_currJe, simulation_.Ec, &corr_w));

  PetscReal K0 = energy;
  PetscLogEventBegin(events[2], 0, 0, 0, 0);

  calculate_energy();
  PetscReal K = energy;

  PetscReal lambda2 = 1.0 + dt * (corr_w - pred_w) / K;
  PetscReal lambda = std::sqrt(lambda2);

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage)
    for (auto& point : cell)
      point.p *= lambda;

  PetscLogEventEnd(events[2], 0, 0, 0, 0);

  lambda_dK = (lambda2 - 1.0) * K;
  pred_dK = K - K0;
  corr_dK = lambda2 * K - K0;
  energy = lambda2 * K;

  LOG("  Velocity renormalization for \"{}\"", parameters.sort_name);
  LOG("    predicted field work [(ECSIM) * E_pred]: {:.7f}", pred_w);
  LOG("    corrected field work [(Esirkepov) * E_corr]: {:.7f}", corr_w);
  LOG("    lambda: {:.7f}, lambda^2: {:.7f}", lambda, lambda2);
  LOG("    d(energy) pred.: {:.7f}, corr.: {:.7f}, lambda: {:.7f}", pred_dK, corr_dK, lambda_dK);
  LOG("    energy prev.: {:.7f}, curr.: {:.7f}, diff: {:.7f}", K0, energy, energy - K0 /* == corr_dK */);
  PetscFunctionReturn(PETSC_SUCCESS);
}

void Particles::decompose_esirkepov_current(const Shape& shape, const Point& point)
{
  PetscFunctionBeginUser;
  const PetscReal alpha =
    charge(point) * density(point) / (particles_number(point) * (6.0 * dt));

  EsirkepovDecomposition decomposition(shape, alpha);
  PetscCallVoid(decomposition.process(currJe));
}

PetscErrorCode Particles::calculate_energy()
{
  PetscFunctionBeginUser;
  energy = 0.0;

  const PetscReal m = parameters.m;
  const PetscInt Np = parameters.Np;

#pragma omp parallel for reduction(+ : energy), \
  schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage)
    for (auto& point : cell)
      energy += ParticlesEnergy::get(point.p, m, Np);

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(ecsim::Particles::clear_sources());
  PetscCall(VecSet(local_currJe, 0.0));
  PetscCall(VecSet(global_currJe, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(ecsim::Particles::finalize());
  PetscCall(VecDestroy(&local_currJe));
  PetscCall(VecDestroy(&global_currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ecsimcorr
