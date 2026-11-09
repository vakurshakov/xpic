#ifndef SRC_DIAGNOSTICS_CHARGE_CONSERVATION_H
#define SRC_DIAGNOSTICS_CHARGE_CONSERVATION_H

#include "src/interfaces/simulation.h"
#include "src/diagnostics/table_diagnostic.h"

class ChargeConservation : public TableDiagnostic {
public:
  ChargeConservation(const interfaces::Simulation& simulation);

  PetscErrorCode initialize() override;

private:
  PetscErrorCode add_columns(PetscInt t) override;

  const interfaces::Simulation& simulation;

  PetscInt event_diagnose;
};

#endif  // SRC_DIAGNOSTICS_CHARGE_CONSERVATION_H
