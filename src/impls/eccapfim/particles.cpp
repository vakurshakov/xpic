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

PetscReal Particles::get_average_number_of_traversed_cells() const
{
  return avgcell;
}


PetscErrorCode Particles::form_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(world.da, local_J, &J));

  ImplicitEsirkepov util(E, B, J);

  avgit = 0.0;
  avgcell = 0.0;

  PetscReal path;
  std::vector<Vector3R> coords;

  PetscReal xb = (world.start[X] - 0.5) * dx;
  PetscReal yb = (world.start[Y] - 0.5) * dy;
  PetscReal zb = (world.start[Z] - 0.5) * dz;

  PetscReal xe = xb + (world.size[X] + 1.0) * dx;
  PetscReal ye = yb + (world.size[Y] + 1.0) * dy;
  PetscReal ze = zb + (world.size[Z] + 1.0) * dz;

  static const PetscReal max = std::numeric_limits<double>::max();

  auto process_bound = [&](PetscReal vh, PetscReal x, PetscReal xb, PetscReal xe) {
    if (vh > 0)
      return (xe - x) / vh;
    else if (vh < 0)
      return (xb - x) / vh;
    else
      return max;
  };

// #pragma omp parallel for firstprivate(util)
  for (PetscInt g = 0; g < (PetscInt)storage.size(); ++g) {
    for (PetscInt i = 0; auto& point : storage[g]) {
      const auto& point_0(previous_storage[g][i]);

      CrankNicolsonPush push(q_m(point));

      push.set_fields_callback(  //
        [&](const Vector3R& rn, const Vector3R& r0, Vector3R& E_p, Vector3R& B_p) {
          path = (rn - r0).length();
          coords = cell_traversal(rn, r0);

          for (PetscInt s = 1; s < (PetscInt)coords.size(); ++s) {
            auto&& rs0 = coords[s - 1];
            auto&& rsn = coords[s - 0];

            Vector3R Ei_p, Bi_p;
            util.interpolate(Ei_p, Bi_p, rsn, rs0);

            PetscReal beta = path > 0 ? (rsn - rs0).length() / path : 1.0;
            E_p += Ei_p * beta;
            B_p += Bi_p * beta;
          }
        });

      for (PetscReal dtau = 0.0, dtau_sum = 0.0; dtau_sum < dt; dtau_sum += dtau) {
        PetscReal dtx = process_bound(point.px(), point.x(), xb, xe);
        PetscReal dty = process_bound(point.py(), point.y(), yb, ye);
        PetscReal dtz = process_bound(point.pz(), point.z(), zb, ze);

        dtau = std::min({dt - dtau_sum, dtx, dty, dtz});

        push.process(dtau, point, point_0);
        avgit += (PetscReal)(push.get_iteration_number() + 1) / size;

        path = (point.r - point_0.r).length();
        coords = cell_traversal(point.r, point_0.r);
        avgcell += (PetscReal)(coords.size() - 1) / size;

        PetscReal a0 = qn_Np(point);
        Vector3R vh = 0.5 * (point.p + point_0.p);

        for (PetscInt s = 1; s < (PetscInt)coords.size(); ++s) {
          auto&& rs0 = coords[s - 1];
          auto&& rsn = coords[s - 0];

          PetscReal a = a0 * (rsn - rs0).length() / path;
          util.decompose(a, vh, rsn, rs0);
        }

        correct_coordinates(point);
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
  size = 0;

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
