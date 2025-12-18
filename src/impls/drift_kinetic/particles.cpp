#include "particles.h"

// #include "src/algorithms/drift_kinetic_push.h"
// #include "src/algorithms/drift_kinetic_implicit.h"
#include "src/impls/drift_kinetic/simulation.h"

namespace drift_kinetic {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world, parameters),
    dk_curr_storage(world.size.elements_product()),
    dk_prev_storage(world.size.elements_product()),
    simulation_(simulation)
{
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &J));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &M));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &J_loc));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &M_loc));
}

PetscErrorCode Particles::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&M));
  PetscCall(VecDestroy(&J_loc));
  PetscCall(VecDestroy(&M_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::form_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMDAVecGetArrayWrite(da, M_loc, &M_arr));

  PetscReal q = parameters.q;
  PetscReal m = parameters.m;

#pragma omp parallel for
  for (PetscInt g = 0; g < (PetscInt)dk_curr_storage.size(); ++g) {
    PetscReal path;
    std::vector<Vector3R> coords;

    for (PetscInt i = 0; auto& curr : dk_curr_storage[g]) {
      const auto& prev(dk_prev_storage[g][i]);

      /// @todo this part should reuse the logic from:
      /// tests/drift_kinetic_push/drift_kinetic_push.h:620 implicit_test_utils::interpolation_test()
#if 0
      CrankNicolsonPush push(q / m);
      ImplicitEsirkepov util(E_arr, B_arr, J_arr);

      push.set_fields_callback(  //
        [&](const Vector3R& rn, const Vector3R& r0, Vector3R& E_p, Vector3R& B_p) {
          path = (rn - r0).length();
          coords = cell_traversal(rn, r0);

          for (PetscInt s = 1; s < (PetscInt)coords.size(); ++s) {
            auto&& rs0 = coords[s - 1];
            auto&& rsn = coords[s - 0];

            Vector3R Es_p, Bs_p;
            util.interpolate(Es_p, Bs_p, rsn, rs0);

            PetscReal beta = path > 0 ? (rsn - rs0).length() / path : 1.0;
            E_p += Es_p * beta;
            B_p += Bs_p * beta;
          }
        });

      for (PetscReal dtau = 0.0, tau = 0.0; tau < dt; tau += dtau) {
        PetscReal dtx = process_bound(curr.px(), curr.x(), xb, xe);
        PetscReal dty = process_bound(curr.py(), curr.y(), yb, ye);
        PetscReal dtz = process_bound(curr.pz(), curr.z(), zb, ze);

        dtau = std::min({dt - tau, dtx, dty, dtz});

        push.process(dtau, curr, prev);

        path = (curr.r - prev.r).length();
        coords = cell_traversal(curr.r, prev.r);

        PetscReal a0 = qn_Np(curr);
        Vector3R vh = 0.5 * (curr.p + prev.p);

        for (PetscInt s = 1; s < (PetscInt)coords.size(); ++s) {
          auto&& rs0 = coords[s - 1];
          auto&& rsn = coords[s - 0];

          PetscReal a = a0 * (rsn - rs0).length() / path;
          util.decompose(a, vh, rsn, rs0);
        }

        correct_coordinates(curr);
      }
#endif

      i++;
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, M_loc, &M_arr));
  PetscCall(DMLocalToGlobal(da, M_loc, ADD_VALUES, M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::prepare_storage()
{
  PetscFunctionBeginUser;
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    if (auto& curr = dk_curr_storage[g]; !curr.empty()) {
      auto& prev = dk_prev_storage[g];
      prev = std::vector(curr.begin(), curr.end());
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace drift_kinetic
