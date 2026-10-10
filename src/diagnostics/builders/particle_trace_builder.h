#ifndef SRC_DIAGNOSTICS_BUILDERS_PARTICLE_TRACE_BUILDER_H
#define SRC_DIAGNOSTICS_BUILDERS_PARTICLE_TRACE_BUILDER_H

#include "src/diagnostics/builders/diagnostic_builder.h"

class ParticleTraceBuilder : public DiagnosticBuilder {
public:
  ParticleTraceBuilder(interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& diagnostics);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the ParticleTrace diagnostic description:\n"
      "{\n"
      "  \"diagnostic\": \"ParticleTrace\", -- Name of the diagnostic, constant.\n"
      "  \"particles\": \"name\", -- Sort name, the first stored particle is traced.\n"
      "  \"skip\": 1, -- Optional, trace is written each `skip`-th timestep.\n"
      "}";
    return help;
  }
};

#endif  // SRC_DIAGNOSTICS_BUILDERS_PARTICLE_TRACE_BUILDER_H
