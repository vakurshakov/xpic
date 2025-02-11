#ifndef SRC_DIAGNOSTICS_PARTICLES_ENERGY_H
#define SRC_DIAGNOSTICS_PARTICLES_ENERGY_H

#include <petscdmda.h>
#include <petscvec.h>

#include "src/pch.h"
#include "src/interfaces/diagnostic.h"
#include "src/interfaces/particles.h"
#include "src/utils/sync_binary_file.h"
#include "src/utils/vector3.h"

class ParticlesEnergy : public interfaces::Diagnostic {
public:
  using ParticlesPointersVector = std::vector<const interfaces::Particles*>;

  ParticlesEnergy(ParticlesPointersVector particles);
  ParticlesEnergy(const std::string& out_dir, ParticlesPointersVector particles);

  PetscErrorCode diagnose(timestep_t t) override;
  PetscErrorCode calculate_energies();
  std::vector<PetscReal> get_energies() const;

  static PetscReal get(const Vector3R& p, PetscReal m, PetscInt Np);

private:
  SyncBinaryFile file_;

  ParticlesPointersVector particles_;
  std::vector<Vector3R> energies_;
};

#endif  // SRC_DIAGNOSTICS_PARTICLES_ENERGY_H
