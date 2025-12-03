#include "particles.h"

#include "src/algorithms/boris_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/impls/basic/simulation.h"

namespace basic {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world, parameters), simulation_(simulation)
{
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &J));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &J_loc));
}

PetscErrorCode Particles::push()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(da, J_loc, &J_arr));

#pragma omp parallel for schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (auto& cell : storage) {
    for (auto& point : cell) {
      const Vector3R old_r = point.r;

      BorisPush push;
      push.set_qm(parameters.q / parameters.m);

      push.update_r(dt / 2.0, point);

      Shape shape;
      shape.setup(point.r, shr, sfunc);

      Vector3R E_p, B_p;
      SimpleInterpolation interpolation(shape);
      interpolation.process({{E_p, E_arr}}, {{B_p, B_arr}});

      push.set_fields(E_p, B_p);
      push.update_vEB(dt, point);
      push.update_r(dt / 2.0, point);

      shape.setup(old_r, point.r, shr, sfunc);
      EsirkepovDecomposition decomposition(shape, qn_Np(point) / (6.0 * dt));
      decomposition.process(J_arr);
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMLocalToGlobal(da, J_loc, ADD_VALUES, J));
  PetscCall(VecAXPY(simulation_.J, 1.0, J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&J_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace basic
