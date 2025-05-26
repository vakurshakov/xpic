#ifndef SRC_DIAGNOSTICS_ENERGY_CONSERVATION_H
#define SRC_DIAGNOSTICS_ENERGY_CONSERVATION_H

#include "src/interfaces/simulation.h"
#include "src/diagnostics/fields_energy.h"
#include "src/diagnostics/particles_energy.h"
#include "src/diagnostics/utils/table_diagnostic.h"

class EnergyConservation : public TableDiagnostic {
public:
  EnergyConservation( //
    const interfaces::Simulation& simulation,
    std::shared_ptr<FieldsEnergy> fields_energy,
    std::shared_ptr<ParticlesEnergy> particles_energy);

protected:
  EnergyConservation(const interfaces::Simulation&);

  PetscErrorCode initialize() override;
  PetscErrorCode add_titles() override;
  PetscErrorCode add_args(PetscInt t) override;

  // Vec B0;
  PetscReal dF = 0.0;
  PetscReal dK = 0.0;

  const interfaces::Simulation& simulation;
  std::shared_ptr<FieldsEnergy> fields_energy;
  std::shared_ptr<ParticlesEnergy> particles_energy;
};

#endif  // SRC_DIAGNOSTICS_ENERGY_CONSERVATION_H
