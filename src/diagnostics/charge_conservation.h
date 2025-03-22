#ifndef SRC_DIAGNOSTICS_CHARGE_CONSERVATION_H
#define SRC_DIAGNOSTICS_CHARGE_CONSERVATION_H

#include "src/diagnostics/distribution_moment.h"
#include "src/diagnostics/utils/table_diagnostic.h"

class ChargeConservation : public TableDiagnostic {
public:
  DEFAULT_MOVABLE(ChargeConservation);

  ChargeConservation(DM da, std::vector<Vec> current_densities,
    std::vector<const interfaces::Particles*> particles);

private:
  PetscErrorCode initialize() override;
  PetscErrorCode add_titles() override;
  PetscErrorCode add_args() override;

  Mat divE;
  std::vector<Vec> current_densities;
  std::vector<std::unique_ptr<DistributionMoment>> charge_densities;
};

#endif  // SRC_DIAGNOSTICS_CHARGE_CONSERVATION_H
