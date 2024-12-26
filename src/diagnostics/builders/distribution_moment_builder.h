#ifndef SRC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H
#define SRC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H

#include "src/diagnostics/builders/diagnostic_builder.h"
#include "src/diagnostics/distribution_moment.h"

class DistributionMomentBuilder : public DiagnosticBuilder {
public:
  DistributionMomentBuilder(const interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& diagnostics);

  PetscErrorCode build(const Configuration::json_t& info) override;

private:
  std::string_view usage_message() const override
  {
    std::string_view help =
      "\n Structure of the DistributionMoment diagnostics description:\n"
      "{\n"
      "  \"diagnostic\": \"DistributionMoment\" -- Name of the diagnostic.\n"
      "  \"particles\": \"electrons\", -- Name from `Particles` settings.\n"
      "  \"moment\": \"Density\", -- Name of the moment getter, available "
      "                           getters are: Density, Vx, Vy, Vz, Vr, Vphi\n"
      "                           mVxVx, mVxVy, mVxVz, mVyVy, mVyVz, mVzVz\n"
      "                           mVrVr, mVrVphi, mVrVz, mVphiVphi, mVphiVz.\n"
      "  \"start\": [ox, oy, oz], -- Starting point of a diagnosed region, in\n"
      "                           global coordinates of c/w_pe units.\n"
      "                           Optional, zeros will be used if empty.\n"
      "  \"size\": [sx, sy, sz] -- Sizes of a diagnosed region along each\n"
      "                         direction, in global coordinates of\n"
      "                         c/w_pe units. Optional, \"Geometry\"\n"
      "                         settings will be used if empty.\n"
      "}";
      // "i.e. mVxVy, mVxVz, ...\n"
      // "\n"
    // "  - density, Vx_moment, mVxVy_moment, etc. collect moments onto (x, y, "
    // "z) coordinates in units of c/w_pe.\n";
    return help;
  }
};

#endif  // SRC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H
