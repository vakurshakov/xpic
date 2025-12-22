#include "particles.h"

#include "src/algorithms/crank_nicolson_push.h"
#include "src/algorithms/implicit_esirkepov.h"
#include "src/impls/eccapfim/cell_traversal.h"
#include "src/impls/eccapfim/simulation.h"

namespace eccapfim {

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world, parameters),
    previous_storage(world.size.elements_product()),
    simulation_(simulation)
{
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &J));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &J_loc));
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
  PetscCall(DMDAVecGetArrayWrite(da, J_loc, &J_arr));

  avgit = 0.0;
  avgcell = 0.0;

  PetscReal q = parameters.q;
  PetscReal m = parameters.m;
  PetscReal mpw = parameters.n / parameters.Np;

  PetscReal xb = (world.start[X] - 0.5) * dx;
  PetscReal yb = (world.start[Y] - 0.5) * dy;
  PetscReal zb = (world.start[Z] - 0.5) * dz;

  PetscReal xe = (world.end[X] + 0.5) * dx;
  PetscReal ye = (world.end[Y] + 0.5) * dy;
  PetscReal ze = (world.end[Z] + 0.5) * dz;

  PetscReal max = std::numeric_limits<double>::max();

  auto process_bound = [&](PetscReal vh, PetscReal x, PetscReal xb, PetscReal xe) {
    if (vh > 0 && abs(xe - x) > 1e-7)
      return (xe - x) / vh;
    else if (vh < 0 && abs(xb - x) > 1e-7)
      return (xb - x) / vh;
    else
      return max;
  };

  auto bound_periodic = [&](PetscReal& s, Axis axis) {
    if (s < 0.0) {
      s = Geom[axis] - (0.0 - s);
      return true;
    }
    else if (s > Geom[axis]) {
      s = 0.0 + (s - Geom[axis]);
      return true;
    }
    return false;
  };

#pragma omp parallel for reduction(+ : avgit, avgcell)
  for (PetscInt g = 0; g < (PetscInt)storage.size(); ++g) {
    ImplicitEsirkepov util(E_arr, B_arr, J_arr);

    for (PetscInt i = 0; auto& curr : storage[g]) {
      curr = previous_storage[g][i];
      Point tmp = curr;

      PetscReal tau = 0, dtau = 0, dtx, dty, dtz;

      for (; tau < dt; tau += dtau) {
        auto& pn = curr;
        auto& p0 = tmp;

        // This is a guess, not a proper calculation of `dtau`
        Vector3R vh = 0.5 * (pn.p + p0.p);

        dtx = process_bound(vh.x(), p0.x(), xb, xe);
        dty = process_bound(vh.y(), p0.y(), yb, ye);
        dtz = process_bound(vh.z(), p0.z(), zb, ze);
        dtau = std::min({dt - tau, dtx, dty, dtz});

        const PetscReal a0 = q * mpw;
        const PetscReal alpha = 0.5 * dtau * (q / m);

        const PetscInt cn_maxit = 30;
        const PetscReal cn_atol = 1e-8;

        PetscInt it = 0, s;
        PetscReal rn = max, d, ds, bs;

        Vector3R E_p, B_p, rsn, rs0;
        std::vector<Vector3R> coords;

        auto set_fields = [&]() {
          E_p = Vector3R{};
          B_p = Vector3R{};

          d = (pn.r - p0.r).length();
          coords = cell_traversal(pn.r, p0.r);

          for (s = 1; s < (PetscInt)coords.size(); s++) {
            rs0 = coords[s - 1];
            rsn = coords[s - 0];
            ds = (rsn - rs0).length();
            bs = (d > 0 ? ds / d : 1.0);

            // No (dtau / dt) here, this is a field on substep `tau`
            Vector3R Es_p, Bs_p;
            util.interpolate(Es_p, Bs_p, rsn, rs0);
            E_p += Es_p * bs;
            B_p += Bs_p * bs;
          }
        };

        set_fields();

        for (; it < cn_maxit && rn > cn_atol; it++) {
          Vector3R a, b, w;
          a = alpha * E_p;
          b = alpha * B_p;
          w = p0.p + a;
          vh = (w + w.cross(b) + b * w.dot(b)) / (1.0 + b.squared());

          pn.r = p0.r + dtau * vh;
          pn.p = 2.0 * vh - p0.p;

          set_fields();
          rn = ((pn.p - p0.p) - (dtau * q / m) * (E_p + vh.cross(B_p))).length();
        }

        avgit += (PetscReal)it / size;
        avgcell += (PetscReal)(coords.size() - 1) / size;

        d = (pn.r - p0.r).length();
        coords = cell_traversal(pn.r, p0.r);

        for (s = 1; s < (PetscInt)coords.size(); s++) {
          rs0 = coords[s - 1];
          rsn = coords[s - 0];
          ds = (rsn - rs0).length();
          bs = (d > 0 ? ds / d : 1.0);

          util.decompose(a0 * bs * (dtau / dt), vh, rsn, rs0);
        }

        bool reset = false;
        reset |= bound_periodic(pn.r[X], X);
        reset |= bound_periodic(pn.r[Y], Y);
        reset |= bound_periodic(pn.r[Z], Z);

        if (reset)
          p0 = pn;
      }

      i++;
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMLocalToGlobal(da, J_loc, ADD_VALUES, J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(J, 0.0));
  PetscCall(VecSet(J_loc, 0.0));
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
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&J_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace eccapfim
