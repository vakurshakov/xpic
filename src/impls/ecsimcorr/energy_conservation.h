#ifndef SRC_ECSIMCORR_ENERGY_CONSERVATION_H
#define SRC_ECSIMCORR_ENERGY_CONSERVATION_H

#include "src/diagnostics/energy_conservation.h"
#include "src/impls/ecsimcorr/simulation.h"

namespace ecsimcorr {

class EnergyConservation : public ::EnergyConservation {
public:
  DEFAULT_MOVABLE(EnergyConservation);

  EnergyConservation(const ecsimcorr::Simulation& simulation);

  EnergyConservation( //
    const interfaces::Simulation& simulation,
    std::shared_ptr<FieldsEnergy> fields_energy,
    std::shared_ptr<ParticlesEnergy> particles_energy);

private:
  PetscErrorCode add_titles() override;
  PetscErrorCode add_args(PetscInt t) override;

  Vec B, B0;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_ENERGY_CONSERVATION_H
