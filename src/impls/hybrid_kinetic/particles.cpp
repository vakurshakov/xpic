#include "particles.h"

#include "src/algorithms/implicit_esirkepov.h"
#include "src/impls/eccapfim/particles.h"

namespace hybrid_kinetic {

// Picard tolerances of the Crank-Nicolson push. The residual of the velocity
// equation is compared with the mid-step velocity `vh`, so `cn_rtol` is the
// relative accuracy of the velocity, independent of the mass, B and dt;
// `cn_atol` is the rounding error of |v| ~ c, so it is always achievable.
static constexpr PetscReal cn_atol = 1e-16;
static constexpr PetscReal cn_rtol = 1e-13;
static constexpr PetscInt cn_maxit = 100;

KineticParticles::KineticParticles(
  const World& world, const SortParameters& parameters)
  : interfaces::Particles(world, parameters),
    previous_storage(world.size.elements_product())
{
}

PetscErrorCode KineticParticles::prepare_storage()
{
  PetscFunctionBeginUser;
  size = 0;

  for (PetscInt g = 0; g < world.size.elements_product(); g++) {
    previous_storage[g].assign(storage[g].begin(), storage[g].end());
    size += (PetscInt)storage[g].size();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Restarts each particle from layer n and pushes it through the current field
// iterate. The step is split at the ghost-region boundary of the subdomain,
// substeps are chained, the particle is wrapped by the periodic boundaries.
PetscErrorCode KineticParticles::form_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(da, J_loc, &J_arr));

  avgit = 0;

  const PetscReal q = parameters.q;
  const PetscReal m = parameters.m;
  const PetscReal mpw = parameters.mpw;

  const Vector3R rb{
    (world.start[X] - 0.5) * dx,
    (world.start[Y] - 0.5) * dy,
    (world.start[Z] - 0.5) * dz,
  };

  const Vector3R re{
    (world.end[X] + 0.5) * dx,
    (world.end[Y] + 0.5) * dy,
    (world.end[Z] + 0.5) * dz,
  };

  auto process_bound = [&](PetscReal vh, PetscReal x, Axis a)
  {
    if (vh > 0 && std::abs(re[a] - x) > 1e-7)
      return (re[a] - x) / vh;
    if (vh < 0 && std::abs(rb[a] - x) > 1e-7)
      return (rb[a] - x) / vh;
    return std::numeric_limits<PetscReal>::max();
  };

  auto bound_periodic = [&](PetscReal& s, Axis a)
  {
    if (s < 0)
      s += Geom[a];
    else if (s > Geom[a])
      s -= Geom[a];
  };

#pragma omp parallel for reduction(+ : avgit)
  for (PetscInt g = 0; g < (PetscInt)storage.size(); g++) {
    ImplicitEsirkepov util(E_arr, B_arr, J_arr);

    for (PetscInt i = 0; auto& curr : storage[g]) {
      curr = previous_storage[g][i];
      Point p0 = curr;
      Point& pn = curr;

      for (PetscReal tau = 0, dtau = 0; tau < dt; tau += dtau) {
        Vector3R vh = 0.5 * (pn.p + p0.p);

        dtau = std::min({
          dt - tau,
          process_bound(vh[X], p0.r[X], X),
          process_bound(vh[Y], p0.r[Y], Y),
          process_bound(vh[Z], p0.r[Z], Z),
        });

        const PetscReal alpha = 0.5 * dtau * (q / m);

        Vector3R E_p, B_p;
        std::vector<Vector3R> coords;

        auto for_segments = [&](auto&& callback)
        {
          const PetscReal d = (pn.r - p0.r).length();
          coords = eccapfim::cell_traversal(pn.r, p0.r);

          for (PetscInt s = 1; s < (PetscInt)coords.size(); s++) {
            const PetscReal ds = (coords[s] - coords[s - 1]).length();
            callback(coords[s], coords[s - 1], (d > 0 ? ds / d : 1.0));
          }
        };

        auto set_fields = [&]
        {
          E_p = Vector3R{};
          B_p = Vector3R{};

          for_segments(
            [&](const Vector3R& rsn, const Vector3R& rs0, PetscReal bs)
            {
              Vector3R Es_p, Bs_p;
              util.interpolate(Es_p, Bs_p, rsn, rs0);
              E_p += Es_p * bs;
              B_p += Bs_p * bs;
            });
        };

        auto get_residue = [&]
        {
          return ((pn.p - p0.p) - (dtau * q / m) * (E_p + vh.cross(B_p)))
            .length();
        };

        set_fields();
        // const PetscReal r0 = get_residue();
        PetscReal rn;

        // At least one iteration, so that the deposited current `vh`
        // always matches the displacement `pn.r - p0.r`.
        PetscInt it = 0;
        do {
          const Vector3R a = alpha * E_p;
          const Vector3R b = alpha * B_p;
          const Vector3R w = p0.p + a;
          vh = (w + w.cross(b) + b * w.dot(b)) / (1.0 + b.squared());

          pn.r = p0.r + dtau * vh;
          pn.p = 2.0 * vh - p0.p;

          set_fields();
          rn = get_residue();
          it++;
          // } while (rn > cn_atol + cn_rtol * r0 && it < cn_maxit);
        } while (rn > cn_atol + cn_rtol * vh.length() && it < cn_maxit);

        avgit += (PetscReal)it / (PetscReal)size;

        for_segments(
          [&](const Vector3R& rsn, const Vector3R& rs0, PetscReal bs)
          {
            util.decompose(q * mpw * bs * (dtau / dt), vh, rsn, rs0);
          });

        for (Axis a : {X, Y, Z})
          bound_periodic(pn.r[a], a);

        p0 = pn;
      }

      i++;
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMLocalToGlobal(da, J_loc, ADD_VALUES, J));

  PetscMPIInt mpi_size;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &avgit, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  avgit /= (PetscReal)mpi_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal KineticParticles::kinetic_energy_local() const
{
  PetscReal w = 0;

#pragma omp parallel for reduction(+ : w)
  for (auto&& cell : storage) {
    for (auto&& point : cell)
      w += point.p.squared();
  }
  return 0.5 * parameters.m * parameters.mpw * w;
}

PetscReal KineticParticles::get_average_iteration_number() const
{
  return avgit;
}

}  // namespace hybrid_kinetic
