#ifndef SRC_DIAGNOSTICS_PARTICLES_ENERGY_H
#define SRC_DIAGNOSTICS_PARTICLES_ENERGY_H

#include "src/pch.h"
#include "src/interfaces/diagnostic.h"
#include "src/interfaces/particles.h"
#include "src/utils/sync_file.h"

class ParticlesEnergy : public interfaces::Diagnostic {
  friend class EnergyConservation;

public:
  ParticlesEnergy(std::vector<const interfaces::Particles*> particles);

  ParticlesEnergy( //
    const std::string& out_dir, //
    std::vector<const interfaces::Particles*> particles);

  PetscErrorCode diagnose(PetscInt t) override;
  PetscErrorCode calculate_energies();
  std::vector<PetscReal> get_energies() const;

  static PetscReal get(const Vector3R& p, PetscReal m, PetscInt Np);

private:
  SyncFile file_;

  std::vector<const interfaces::Particles*> particles_;
  std::vector<Vector3R> energies_;
};

#endif  // SRC_DIAGNOSTICS_PARTICLES_ENERGY_H
