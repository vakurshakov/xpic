#ifndef SRC_ECSIMCORR_ENERGY_H
#define SRC_ECSIMCORR_ENERGY_H

#include "src/interfaces/diagnostic.h"
#include "src/diagnostics/fields_energy.h"
#include "src/diagnostics/particles_energy.h"
#include "src/impls/ecsimcorr/simulation.h"
#include "src/utils/sync_file.h"

namespace ecsimcorr {

class EnergyDiagnostic : public interfaces::Diagnostic {
public:
  DEFAULT_MOVABLE(EnergyDiagnostic);

  EnergyDiagnostic(const Simulation& simulation);

  PetscErrorCode diagnose(timestep_t t) override;

private:
  PetscErrorCode write_header();

  const Simulation& simulation;

  SyncFile file_;

  Vec E;
  Vec B;
  Vec B0;

  std::unique_ptr<FieldsEnergy> fields_energy;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_ENERGY_H
