#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/impls/basic/simulation.h"
#include "src/utils/profiler.h"

namespace basic {

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
  PetscCall(storage.commit_attributes());

  PetscCall(Profiler::register_event("interpolate", &phase_interpolate));
  PetscCall(Profiler::register_event("push", &phase_push));
  PetscCall(Profiler::register_event("decompose", &phase_decompose));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::push()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(da, J_loc, &J_arr));

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attributes(f));
  const PointArrays arrays(f);

#pragma omp parallel for
  for (PetscInt i = 0; i < f.size; i++) {
    Point point = arrays.load(i);
    const Vector3R old_r = point.r;

    BorisPush push;
    push.set_qm(parameters.q / parameters.m);

    (void)Profiler::push_phase(phase_push);
    push.update_r(dt / 2.0, point);
    (void)Profiler::pop_phase(phase_push);

    (void)Profiler::push_phase(phase_interpolate);
    Shape shape;
    shape.setup(point.r, shr, sfunc);

    Vector3R E_p, B_p;
    SimpleInterpolation interpolation(shape);
    interpolation.process({{E_p, E_arr}}, {{B_p, B_arr}});
    (void)Profiler::pop_phase(phase_interpolate);

    (void)Profiler::push_phase(phase_push);
    push.set_fields(E_p, B_p);
    push.update_vEB(dt, point);
    push.update_r(dt / 2.0, point);
    (void)Profiler::pop_phase(phase_push);

    (void)Profiler::push_phase(phase_decompose);
    shape.setup(old_r, point.r, shr, sfunc);
    EsirkepovDecomposition decomposition(
      shape, parameters.q * parameters.mpw / (6.0 * dt));
    decomposition.process(J_arr);
    (void)Profiler::pop_phase(phase_decompose);

    arrays.store(i, point);
  }

  PetscCall(Profiler::flush());
  PetscCall(storage.restore_attributes(f));

  PetscCall(DMDAVecRestoreArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMLocalToGlobal(da, J_loc, ADD_VALUES, J));
  PetscCall(VecAXPY(simulation_.J, 1.0, J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace basic
