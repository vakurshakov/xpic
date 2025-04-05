#ifndef SRC_DIAGNOSTICS_BUILDERS_DIAGNOSTIC_BUILDER_H
#define SRC_DIAGNOSTICS_BUILDERS_DIAGNOSTIC_BUILDER_H

#include "src/pch.h"
#include "src/interfaces/builder.h"
#include "src/interfaces/diagnostic.h"
#include "src/interfaces/simulation.h"
#include "src/utils/configuration.h"

class DiagnosticBuilder : public interfaces::Builder {
public:
  DEFAULT_MOVABLE(DiagnosticBuilder);

  DiagnosticBuilder(
    interfaces::Simulation& simulation, std::vector<Diagnostic_up>& result);

protected:
  using Diagnostics_vector = std::vector<Diagnostic_up>;
  Diagnostics_vector& diagnostics_;
};


PetscErrorCode build_diagnostics(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& result);

#endif  // SRC_DIAGNOSTICS_BUILDERS_DIAGNOSTIC_BUILDER_H
