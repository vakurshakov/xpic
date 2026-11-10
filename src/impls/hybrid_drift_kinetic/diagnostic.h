#ifndef SRC_IMPLS_HYBRID_DRIFT_KINETIC_DIAGNOSTIC_H
#define SRC_IMPLS_HYBRID_DRIFT_KINETIC_DIAGNOSTIC_H

#include "src/diagnostics/table_diagnostic.h"

namespace hybrid_drift_kinetic {

class Simulation;

/// @brief Energy balance of fields, kinetic and drift-kinetic sorts.
class EnergyConservation : public TableDiagnostic {
public:
  EnergyConservation(const Simulation& simulation);
  PetscErrorCode initialize() override;
  PetscErrorCode add_columns(PetscInt t) override;

private:
  PetscErrorCode calculate_kinetic_energies(std::vector<PetscReal>& K) const;

  const Simulation& simulation;

  PetscReal w_E0 = 0;         // Electric field energy at the previous step.
  PetscReal w_B0 = 0;         // Magnetic field energy at the previous step.
  std::vector<PetscReal> K0;  // Kinetic energies at the previous step.
};

}  // namespace hybrid_drift_kinetic

#endif  // SRC_IMPLS_HYBRID_DRIFT_KINETIC_DIAGNOSTIC_H
