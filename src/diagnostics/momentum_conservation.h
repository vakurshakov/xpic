#ifndef SRC_DIAGNOSTICS_MOMENTUM_CONSERVATION_H
#define SRC_DIAGNOSTICS_MOMENTUM_CONSERVATION_H

#include "src/interfaces/particles.h"
#include "src/diagnostics/distribution_moment.h"
#include "src/diagnostics/utils/table_diagnostic.h"

class MomentumConservation : public TableDiagnostic {
public:
  MomentumConservation(
    DM da, Vec E, std::vector<const interfaces::Particles*> particles);

private:
  PetscErrorCode calculate();
  PetscErrorCode initialize() override;
  PetscErrorCode add_titles() override;
  PetscErrorCode add_args(PetscInt t) override;

  DM da;
  Vec E;
  std::vector<const interfaces::Particles*> particles;
  std::vector<Vector3R> P0, P1, QE;
};

#endif  // SRC_DIAGNOSTICS_MOMENTUM_CONSERVATION_H
