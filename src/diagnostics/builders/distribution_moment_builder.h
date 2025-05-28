#ifndef SRC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H
#define SRC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H

#include "src/diagnostics/builders/field_view_builder.h"

class DistributionMomentBuilder : public FieldViewBuilder {
public:
  DistributionMomentBuilder(interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& diagnostics);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the DistributionMoment diagnostics description:\n"
      "{\n"
      "  \"diagnostic\": \"DistributionMoment\" -- Name of the diagnostic.\n"
      "  \"particles\": \"electrons\", -- Name from `Particles` settings.\n"
      "  \"moment\": \"Density\", -- Name of the moment getter, available\n"
      "                           getters are: density, current,\n"
      "                           momentum_flux, momentum_flux_cyl,\n"
      "                           momentum_flux_diag, momentum_flux_diag_cyl.\n"
      "  \"start\": [ox, oy, oz], -- Starting point of a diagnosed region, in\n"
      "                           global coordinates of c/w_pe units.\n"
      "                           Optional, zeros will be used if empty.\n"
      "  \"size\": [sx, sy, sz] -- Sizes of a diagnosed region along each\n"
      "                         direction, in global coordinates of\n"
      "                         c/w_pe units. Optional, \"Geometry\"\n"
      "                         settings will be used if empty.\n"
      "}";
    return help;
  }
};

#endif  // SRC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H
