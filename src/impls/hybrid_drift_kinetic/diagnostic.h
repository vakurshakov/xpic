#ifndef SRC_IMPLS_HYBRID_DRIFT_KINETIC_DIAGNOSTIC_H
#define SRC_IMPLS_HYBRID_DRIFT_KINETIC_DIAGNOSTIC_H

#include "src/impls/drift_kinetic/diagnostic.h"

namespace hybrid_drift_kinetic {

class Simulation;

class EnergyConservation : public drift_kinetic::EnergyConservation {
public:
  EnergyConservation(const Simulation& simulation);

protected:
  PetscErrorCode fill_energy_cons(PetscInt t) override;

private:
  const Simulation& simulation;
};

class ConvergenceHistory : public drift_kinetic::ConvergenceHistory {
public:
  ConvergenceHistory(const Simulation& simulation);
  PetscErrorCode add_columns(PetscInt t) override;

private:
  const Simulation& hybrid_simulation;
};

}  // namespace hybrid_drift_kinetic

#endif  // SRC_IMPLS_HYBRID_DRIFT_KINETIC_DIAGNOSTIC_H
