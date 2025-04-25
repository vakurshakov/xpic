#include "particles.h"

#include "src/algorithms/crank_nicolson_push.h"
#include "src/algorithms/esirkepov_decomposition.h"
#include "src/algorithms/simple_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/diagnostics/particles_energy.h"
#include "src/impls/eccapfim/simulation.h"

namespace eccapfim {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world, parameters),
    previous_storage(world.size.elements_product()),
    simulation_(simulation)
{
  DM da = world.da;
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &local_J));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &global_J));
}

PetscErrorCode Particles::form_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(world.da, local_J, &J));

  // #pragma omp parallel for
  for (PetscInt g = 0; g < (PetscInt)storage.size(); ++g) {
    for (PetscInt i = 0; auto& point : storage[g]) {
      const auto& point_0(previous_storage[g][i]);

      CrankNicolsonPush push(charge(point) / mass(point));

      /// @todo This `Shape` and `SimpleInterpolation`, `SimpleDecomposition` below
      /// should be replaced with charge-conserving ones, using non-symmetric particle shape
      Shape shape;

      push.set_fields_callback(
        [&](const Vector3R& r, Vector3R& E_p, Vector3R& B_p) {
          shape.setup(r, shape_radius, shape_func);

          SimpleInterpolation interpolation(shape);
          interpolation.process({{E_p, E}}, {{B_p, B}});
        });

      /// @todo This `dt` can be different from the global one, if we introduce sub-stepping
      /// @todo Get some feedback about convergence success, number of iterations during the `process()`
      push.process(dt, point, point_0);

      /// @todo For now, we have to check that particle doesn't pass the cell boundaries
      PetscAssertAbort((point.r - point_0.r).abs_max() < dx, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Particle cannot move farther than one cell at a time");

      /// @todo Separate particle advancement and current decomposition onto different stages?
      Vector3R J_p = macro_q(point) * 0.5 * (point.p + point_0.p);
      SimpleDecomposition decomposition(shape, J_p);
      decomposition.process(J);

      i++;
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(world.da, local_J, &J));
  PetscCall(DMLocalToGlobal(world.da, local_J, ADD_VALUES, global_J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(local_J, 0.0));
  PetscCall(VecSet(global_J, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::prepare_storage()
{
  PetscFunctionBeginUser;
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    auto&& curr = storage[g];
    if (curr.empty())
      continue;

    auto&& prev = previous_storage[g];
    prev = std::vector(curr.begin(), curr.end());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&local_J));
  PetscCall(VecDestroy(&global_J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace eccapfim
