#ifndef SRC_DIAGNOSTICS_CHARGE_CONSERVATION_H
#define SRC_DIAGNOSTICS_CHARGE_CONSERVATION_H

#include "src/interfaces/particles.h"
#include "src/diagnostics/distribution_moment.h"
#include "src/diagnostics/utils/table_diagnostic.h"

class ParticlesChargeDensity final : public DistributionMoment {
public:
  ParticlesChargeDensity(const interfaces::Particles& particles);

private:
  struct Shape;
  PetscErrorCode collect() override;
  PetscErrorCode set_data_views(const FieldView::Region& region) override;
};

class ChargeConservation : public TableDiagnostic {
public:
  ChargeConservation(DM da, std::vector<Vec> current_densities,
    std::vector<const interfaces::Particles*> particles);

private:
  PetscErrorCode initialize() override;
  PetscErrorCode add_columns(PetscInt t) override;

  Mat divE;
  std::vector<Vec> current_densities;
  std::vector<std::unique_ptr<DistributionMoment>> charge_densities;
};

#endif  // SRC_DIAGNOSTICS_CHARGE_CONSERVATION_H
