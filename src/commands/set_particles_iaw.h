#ifndef SRC_COMMANDS_SET_PARTICLES_IAW_H
#define SRC_COMMANDS_SET_PARTICLES_IAW_H

#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"


/**
 * @brief Loads one sort with the density, bulk velocity and parallel
 * temperature of an ion-acoustic wave eigenmode, as pairs of particles that
 * share a coordinate and have mirrored thermal velocities.
 */
class SetParticlesIonAcousticWave : public interfaces::Command {
public:
  SetParticlesIonAcousticWave(interfaces::Particles& particles,
    PetscInt number_of_pairs, PetscReal force_electric_amplitude,
    PetscReal omega_real, PetscReal gamma, const Vector3R& wave_number,
    const Vector3R& field_phase, const Vector3R& density_amplitude,
    const Vector3R& density_phase);

  PetscErrorCode execute(PetscInt t) override;

private:
  Vector3R pair_coordinate(PetscInt pair) const;

  void local_parallel_parameters(const Vector3R& coordinate,
    PetscReal& bulk_velocity, PetscReal& parallel_scale) const;

  PetscErrorCode log_statistics();

  interfaces::Particles& particles;
  PetscInt number_of_pairs;

  Axis axis = Z;
  PetscReal n_pa;      // density perturbation amplitude (along `axis`)
  PetscReal n_pm;      // density perturbation wave number (along `axis`)
  PetscReal n_pp;      // density perturbation phase (along `axis`)
  PetscReal length;    // domain length along `axis`
  PetscReal k_par;     // wave number (parallel to `axis`)
  PetscReal v2_0;      // initial value of second moment (v^2)
  PetscComplex n_p;    // initial density perturbation
  PetscComplex v_p;    // initial velocity perturbation
  PetscComplex nv2_p;  // initial temperature perturbation

  PetscReal added_energy = 0;
  PetscInt added_particles = 0;
};

#endif  // SRC_COMMANDS_SET_PARTICLES_IAW_H
