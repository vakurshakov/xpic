#include "set_paired_particles.h"

#include "src/algorithms/implicit_drift_kinetic.h"
#include "src/diagnostics/energy.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/geometries.h"
#include "src/utils/utils.h"

namespace drift_kinetic {

SetPairedParticles::SetPairedParticles(            //
  Particles& ionized,                              //
  Particles& ejected,                              //
  PetscInt number_of_particles,                    //
  const CoordinateGenerator& generate_coordinate,  //
  const MomentumGenerator& generate_momentum_i,    //
  const MomentumGenerator& generate_momentum_e)
  : ionized_(ionized),
    ejected_(ejected),
    number_of_particles_(number_of_particles),
    generate_coordinate_(generate_coordinate),
    generate_momentum_i_(generate_momentum_i),
    generate_momentum_e_(generate_momentum_e)
{
}

PetscErrorCode SetPairedParticles::add_particle(Particles& particles,
  DriftKineticEsirkepov& esirkepov, const Point& point, bool& is_added)
{
  PetscFunctionBeginUser;
  Vector3I vg{
    FLOOR_STEP(point.x(), dx) - particles.world.start[X],
    FLOOR_STEP(point.y(), dy) - particles.world.start[Y],
    FLOOR_STEP(point.z(), dz) - particles.world.start[Z],
  };

  if (!is_point_within_bounds(vg, 0, particles.world.size))
    PetscFunctionReturn(PETSC_SUCCESS);

  const PetscReal mp = particles.parameters.m;
  const PetscReal qm = particles.parameters.q / particles.parameters.m;

  Vector3R B_p{};
  PetscCall(esirkepov.interpolate_B(B_p, point.r));

  PetscInt g = particles.world.s_g(REP3_A(vg));
  if (particles.coord_is_gc())
    particles.dk_curr_storage[g].emplace_back(make_point_at_gc(point, B_p, mp));
  else
    particles.dk_curr_storage[g].emplace_back(point, B_p, mp, qm);

  // Mirror the point into the legacy `interfaces::Particles::storage` so that
  // density/current diagnostics (`DistributionMoment::collect` iterates
  // `particles.storage`) can see the loaded particles. With `coord_is_gc=true`
  // the stored `r` is the guiding center — exactly what the diagnostic should
  // bin in DK runs.
  particles.storage[g].emplace_back(point);

  is_added = true;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetPairedParticles::execute(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  if (executed_)
    PetscFunctionReturn(PETSC_SUCCESS);
  executed_ = true;

  PetscReal energy_i = 0.0;
  PetscReal energy_e = 0.0;
  PetscInt added_particles = 0;

  const PetscReal mi = ionized_.parameters.m;
  const PetscReal mpwi = ionized_.parameters.n / ionized_.parameters.Np;

  const PetscReal me = ejected_.parameters.m;
  const PetscReal mpwe = ejected_.parameters.n / ejected_.parameters.Np;

  Simulation& simulation = ionized_.simulation_;
  PetscCall(DMGlobalToLocal(
    simulation.da, simulation.B, INSERT_VALUES, simulation.B_loc));
  PetscCall(DMDAVecGetArrayRead(simulation.da, simulation.B_loc, &simulation.B_arr));

  DriftKineticEsirkepov esirkepov(simulation.B_arr);

  for (PetscInt p = 0; p < number_of_particles_; ++p) {
    const Vector3R shared_coordinate = generate_coordinate_();
    const Vector3R pi = generate_momentum_i_(shared_coordinate);
    const Vector3R pe = generate_momentum_e_(shared_coordinate);

    bool is_added = false;
    PetscCall(add_particle(ionized_, esirkepov,
      Point(shared_coordinate, pi), is_added));
    PetscCall(add_particle(ejected_, esirkepov,
      Point(shared_coordinate, pe), is_added));

    if (is_added) {
      energy_i += Energy::get_kinetic(pi, mi, mpwi);
      energy_e += Energy::get_kinetic(pe, me, mpwe);
      ++added_particles;
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(
    simulation.da, simulation.B_loc, &simulation.B_arr));

  PetscCall(log_statistics(added_particles, energy_i, energy_e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetPairedParticles::log_statistics(PetscInt added_particles,
  PetscReal energy_i, PetscReal energy_e) const
{
  PetscFunctionBeginUser;
  LOG("  Paired particles have been set into \"{}\" + \"{}\"",
    ionized_.parameters.sort_name, ejected_.parameters.sort_name);

  PetscCall(MPIUtils::log_statistics("    ", added_particles, PETSC_COMM_WORLD));

  PetscReal energy[2]{energy_i, energy_e};
  PetscCallMPI(MPI_Allreduce(
    MPI_IN_PLACE, energy, 2, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  LOG("    energy added into \"{}\": {:6.4e}",
    ionized_.parameters.sort_name, energy[0]);
  LOG("    energy added into \"{}\": {:6.4e}",
    ejected_.parameters.sort_name, energy[1]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace drift_kinetic
