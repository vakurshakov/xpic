#ifndef SRC_DIAGNOSTICS_ENERGY_CONSERVATION_H
#define SRC_DIAGNOSTICS_ENERGY_CONSERVATION_H

#include "src/interfaces/simulation.h"
#include "src/diagnostics/energy.h"
#include "src/diagnostics/utils/table_diagnostic.h"

class EnergyConservation : public TableDiagnostic {
public:
  EnergyConservation( //
    const interfaces::Simulation& simulation, std::shared_ptr<Energy> energy);

protected:
  EnergyConservation(const interfaces::Simulation&);

  PetscErrorCode diagnose(PetscInt t) override;
  PetscErrorCode initialize() override;
  PetscErrorCode add_columns(PetscInt t) override;

  PetscReal E0, B0, dE, dB, dF, dK;
  std::vector<PetscReal> K0, K;

  const interfaces::Simulation& simulation;
  std::shared_ptr<Energy> energy;
};

#endif  // SRC_DIAGNOSTICS_ENERGY_CONSERVATION_H
