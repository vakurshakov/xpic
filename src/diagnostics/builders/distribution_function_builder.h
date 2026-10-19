#ifndef SRC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_FUNCTION_BUILDER_H
#define SRC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_FUNCTION_BUILDER_H

#include "src/diagnostics/builders/distribution_moment_builder.h"

class DistributionFunctionBuilder : public DistributionMomentBuilder {
public:
  DistributionFunctionBuilder(interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& diagnostics);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the DistributionFunction diagnostics description:\n"
      "{\n"
      "  \"diagnostic\": \"DistributionFunction\" -- Name of the diagnostic.\n"
      "  \"particles\": \"electrons\", -- Name from `Particles` settings.\n"
      "  \"projection\": \"vx_vy\", -- Name of the projector, available\n"
      "                           values are: vx_vy, vz_vxy, vr_vphi.\n"
      "  \"geometry\": {}, -- Where particles should be placed to collect v.\n"
      "  \"vmin\": [vx, vy], -- Minimum velocity of a diagnosed region.\n"
      "  \"vmax\": [vx, vy], -- Maximum velocity of a diagnosed region.\n"
      "  \"dv\": [dvx, dvy] -- Velocity spacing in a diagnostic region.\n"
      "}";
    return help;
  }

private:
  PetscErrorCode build_dk_distribution_function(const Configuration::json_t& info);
};

#endif  // SRC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_FUNCTION_BUILDER_H
