#ifndef SRC_DIAGNOSTICS_ENERGY_CONSERVATION_H
#define SRC_DIAGNOSTICS_ENERGY_CONSERVATION_H

#include "src/interfaces/diagnostic.h"
#include "src/interfaces/simulation.h"
#include "src/diagnostics/fields_energy.h"
#include "src/diagnostics/particles_energy.h"
#include "src/utils/sync_file.h"

class EnergyConservation : public interfaces::Diagnostic {
public:
  DEFAULT_MOVABLE(EnergyConservation);

  EnergyConservation( //
    const interfaces::Simulation& simulation,
    std::shared_ptr<FieldsEnergy> fields_energy,
    std::shared_ptr<ParticlesEnergy> particles_energy);

  PetscErrorCode diagnose(PetscInt t) override;

private:
  PetscErrorCode write_header();

  SyncFile file_;

  const interfaces::Simulation& simulation;
  std::shared_ptr<FieldsEnergy> fields_energy;
  std::shared_ptr<ParticlesEnergy> particles_energy;
};

#endif  // SRC_DIAGNOSTICS_ENERGY_CONSERVATION_H
