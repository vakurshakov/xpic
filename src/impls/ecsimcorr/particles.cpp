#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/diagnostics/energy.h"
#include "src/impls/ecsimcorr/simulation.h"
#include "src/utils/profiler.h"

namespace ecsimcorr {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : ecsim::Particles(simulation, parameters), simulation_(simulation)
{
}

PetscErrorCode Particles::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(ecsim::Particles::initialize());

  PetscCall(DMCreateGlobalVector(da, &currJe));
  PetscCall(DMCreateLocalVector(da, &currJe_loc));

  J = currJe;
  J_loc = currJe_loc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::first_push()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArray(da, currJe_loc, &currJe_arr));

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attributes(f));
  const PointArrays arrays(f);

#pragma omp parallel for
  for (PetscInt i = 0; i < f.size; i++) {
    Point point = arrays.load(i);
    const Vector3R old_r = point.r;

    Profiler::push_phase(phase_push);
    BorisPush::update_r(0.5 * dt, point);
    Profiler::pop_phase(phase_push);

    Profiler::push_phase(phase_decompose);
    Shape shape;
    shape.setup(old_r, point.r, shape_radius2, shape_func2);
    decompose_esirkepov_current(shape, point);
    Profiler::pop_phase(phase_decompose);
    arrays.store(i, point);
  }

  PetscCall(Profiler::flush());
  PetscCall(storage.restore_attributes(f));

  PetscCall(DMDAVecRestoreArray(da, currJe_loc, &currJe_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::second_push()
{
  PetscFunctionBeginUser;
  pred_w = 0.0;
  PetscCall(DMDAVecGetArray(da, currJe_loc, &currJe_arr));

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attributes(f));
  const PointArrays arrays(f);

#pragma omp parallel for
  for (PetscInt i = 0; i < f.size; i++) {
    Point point = arrays.load(i);
    const Vector3R old_r = point.r;
    const Vector3R old_v = point.p;

    Profiler::push_phase(phase_interpolate);
    Vector3R E_p = interpolate_E_s1(E_arr, point.r);
    Vector3R B_p = interpolate_B_s1(B_arr, point.r);
    Profiler::pop_phase(phase_interpolate);

    Profiler::push_phase(phase_push);
    BorisPush push(parameters.q / parameters.m, E_p, B_p);
    push.update_vEB(dt, point);
    push.update_r(0.5 * dt, point);
    Profiler::pop_phase(phase_push);

    Profiler::push_phase(phase_decompose);
    Shape shape;
    shape.setup(old_r, point.r, shape_radius2, shape_func2);
    decompose_esirkepov_current(shape, point);
    Profiler::pop_phase(phase_decompose);
    arrays.store(i, point);

#pragma omp atomic update
    pred_w += parameters.q * parameters.mpw * 0.5 * (old_v + point.p).dot(E_p);
  }

  PetscCall(Profiler::flush());
  PetscCall(storage.restore_attributes(f));

  // Because we manually calculated `pred_w`, it is needed to reduce it between ranks
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &pred_w, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

  PetscCall(DMDAVecRestoreArray(da, currJe_loc, &currJe_arr));
  PetscCall(DMLocalToGlobal(da, currJe_loc, ADD_VALUES, currJe));
  PetscCall(VecAXPY(simulation_.currJe, 1.0, currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::final_update()
{
  PetscFunctionBeginUser;
  PetscCall(VecDot(currJe, simulation_.Ec, &corr_w));

  PetscReal K0 = energy;
  calculate_energy();
  PetscReal K = energy;

  PetscReal lambda2 = 1.0 + dt * (corr_w - pred_w) / K;
  PetscReal lambda = std::sqrt(lambda2);

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attribute(f, "p"));
  PetscReal* p = f["p"];

#pragma omp parallel for
  for (PetscInt i = 0; i < 3 * f.size; i++)
    p[i] *= lambda;

  PetscCall(storage.restore_attributes(f));

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
  EsirkepovDecomposition decomposition(
    shape, parameters.q * parameters.mpw / (6.0 * dt));
  decomposition.process(currJe_arr);
}

PetscErrorCode Particles::calculate_energy()
{
  PetscFunctionBeginUser;
  energy = 0.0;

  const PetscReal m = parameters.m;
  const PetscReal mpw = parameters.n / parameters.Np;

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attribute(f, "p"));
  const PetscReal* p = f["p"];

#pragma omp parallel for reduction(+ : energy)
  for (PetscInt i = 0; i < f.size; i++)
    energy += Energy::get_kinetic(Vector3R(p + 3 * i), m, mpw);

  PetscCall(storage.restore_attributes(f));

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(ecsim::Particles::clear_sources());
  PetscCall(VecSet(currJe, 0.0));
  PetscCall(VecSet(currJe_loc, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(ecsim::Particles::finalize());
  PetscCall(VecDestroy(&currI));
  PetscCall(VecDestroy(&currI_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ecsimcorr
