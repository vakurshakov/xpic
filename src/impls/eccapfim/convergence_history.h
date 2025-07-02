#ifndef SRC_DIAGNOSTICS_CONVERGENCE_HISTORY_H
#define SRC_DIAGNOSTICS_CONVERGENCE_HISTORY_H

#include "src/diagnostics/utils/table_diagnostic.h"
#include "src/impls/eccapfim/simulation.h"

namespace eccapfim {

class ConvergenceHistory : public TableDiagnostic {
public:
  ConvergenceHistory(const Simulation& simulation);

private:
  PetscErrorCode add_titles() override;
  PetscErrorCode add_args(PetscInt t) override;

  const Simulation& simulation;
};

}

#endif  // SRC_DIAGNOSTICS_MOMENTUM_CONSERVATION_H
