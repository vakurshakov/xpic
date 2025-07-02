#ifndef SRC_ECSIMCORR_ENERGY_CONSERVATION_H
#define SRC_ECSIMCORR_ENERGY_CONSERVATION_H

#include "src/diagnostics/energy_conservation.h"
#include "src/impls/ecsimcorr/simulation.h"

namespace ecsimcorr {

class EnergyConservation : public ::EnergyConservation {
public:
  EnergyConservation(const ecsimcorr::Simulation& simulation);

private:
  PetscErrorCode add_columns(PetscInt t) override;

  Vec B, B0;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_ENERGY_CONSERVATION_H
