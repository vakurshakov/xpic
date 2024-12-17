#ifndef SRC_BASIC_BUILDERS_DIAGNOSTIC_BUILDER_H
#define SRC_BASIC_BUILDERS_DIAGNOSTIC_BUILDER_H

#include "src/pch.h"
#include "src/interfaces/builder.h"
#include "src/interfaces/diagnostic.h"
#include "src/impls/basic/simulation.h"
#include "src/utils/configuration.h"


class DiagnosticBuilder : public interfaces::Builder {
public:
  DEFAULT_MOVABLE(DiagnosticBuilder);

  DiagnosticBuilder(const interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& result);

protected:
  const basic::Particles& get_sort(const std::string& name) const;

  using Diagnostics_vector = std::vector<Diagnostic_up>;
  Diagnostics_vector& diagnostics_;
};

PetscErrorCode build_diagnostics(
  const interfaces::Simulation& simulation, std::vector<Diagnostic_up>& result);


#endif  // SRC_BASIC_BUILDERS_DIAGNOSTIC_BUILDER_H
