#include "inject_particles.h"

#include "src/algorithms/implicit_drift_kinetic.h"
#include "src/diagnostics/energy.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/geometries.h"
#include "src/utils/utils.h"

namespace drift_kinetic {

InjectParticles::InjectParticles(                  //
  Particles& ionized,                              //
  Particles& ejected,                              //
  PetscInt injection_start,                        //
  PetscInt injection_end,                          //
  PetscInt per_step_particles_num,                 //
  const CoordinateGenerator& generate_coordinate,  //
  const MomentumGenerator& generate_momentum_i,    //
  const MomentumGenerator& generate_momentum_e)
  : ionized_(ionized),
    ejected_(ejected),
    injection_start_(injection_start),
    injection_end_(injection_end),
    per_step_particles_num_(per_step_particles_num),
    generate_coordinate_(generate_coordinate),
    generate_momentum_i_(generate_momentum_i),
    generate_momentum_e_(generate_momentum_e)
{
}

PetscErrorCode InjectParticles::add_particle(Particles& particles,
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

  is_added = true;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode InjectParticles::execute(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t < injection_start_ || t > injection_end_)
    PetscFunctionReturn(PETSC_SUCCESS);

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

  /// @note Unlike `::InjectParticles`, the loop is sequential: the points are
  /// emplaced into `std::list` cells of `dk_curr_storage` directly.
  for (PetscInt p = 0; p < per_step_particles_num_; ++p) {
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

PetscErrorCode InjectParticles::log_statistics(PetscInt added_particles,
  PetscReal energy_i, PetscReal energy_e) const
{
  PetscFunctionBeginUser;
  LOG("  Particles have been injected");

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
