#include "particles.h"

#include "src/algorithms/crank_nicolson_push.h"
#include "src/algorithms/implicit_esirkepov.h"
#include "src/algorithms/simple_decomposition.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/diagnostics/particles_energy.h"
#include "src/impls/eccapfim/cell_traversal.h"
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

PetscReal Particles::get_average_iteration_number() const
{
  return avgit;
}

PetscErrorCode Particles::form_iteration()
{
  /// @todo Create a separate shape only for ImplicitEsirkepov interpolation/decomposition
  using namespace ImplicitEsirkepov;

  std::vector<Vector3R> coords;
  PetscReal path;

  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(world.da, local_J, &J));

  // #pragma omp parallel for
  for (PetscInt g = 0; g < (PetscInt)storage.size(); ++g) {
    for (PetscInt i = 0; auto& point : storage[g]) {
      const auto& point_0(previous_storage[g][i]);

      CrankNicolsonPush push(q_m(point));

      push.set_fields_callback(  //
        [&](const Vector3R& rn, const Vector3R& r0, Vector3R& E_p, Vector3R& B_p) {
          path = (rn - r0).length();
          coords = cell_traversal(rn, r0);

          for (PetscInt s = 0; s < (PetscInt)coords.size() - 1; ++s) {
            auto&& rs0 = coords[s + 0];
            auto&& rsn = coords[s + 1];

            Vector3R Ei_p, Bi_p;
            interpolation(Ei_p, E, rsn, rs0);
            interpolation(Bi_p, B, rsn, rs0);

            PetscReal beta = path > 0 ? (rsn - rs0).length() / path : 1.0;
            E_p += Ei_p * beta;
            B_p += Bi_p * beta;
          }
        });

      push.process(dt, point, point_0);
      avgit += push.get_iteration_number() / size;

      path = (point.r - point_0.r).length();
      coords = cell_traversal(point.r, point_0.r);

      for (PetscInt s = 0; s < (PetscInt)coords.size() - 1; ++s) {
        auto&& rs0 = coords[s + 0];
        auto&& rsn = coords[s + 1];

        PetscReal alpha = (qn_Np(point) / 6.0) * (rsn - rs0).length() / path;
        decomposition(J, rsn, rs0, 0.5 * (point.p + point_0.p), alpha);
      }

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

    size += (PetscInt)curr.size();
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
