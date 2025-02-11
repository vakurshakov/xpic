#ifndef SRC_ECSIMCORR_CHARGE_CONSERVATION_H
#define SRC_ECSIMCORR_CHARGE_CONSERVATION_H

#include "src/interfaces/diagnostic.h"
#include "src/diagnostics/distribution_moment.h"
#include "src/impls/ecsimcorr/simulation.h"
#include "src/utils/sync_file.h"

namespace ecsimcorr {

class ChargeConservation : public interfaces::Diagnostic {
public:
  DEFAULT_MOVABLE(ChargeConservation);

  ChargeConservation(const Simulation& simulation);

  PetscErrorCode diagnose(PetscInt t) override;

private:
  PetscErrorCode write_header();

  const Simulation& simulation;

  SyncFile file_;

  std::vector<std::unique_ptr<DistributionMoment>> charge_densities;
  Mat divE;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_CHARGE_CONSERVATION_H
