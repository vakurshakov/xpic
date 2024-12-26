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
    // clang-format off
    return "\n"
      "Usage: The structure of the distribution_moment diagnostics description\n"
      "\"moment_name\": -- Name of the collected distribution moment, listed after description.\n"
      "{\n"
      "  \"sort\": \"electrons\", -- Particles sort name, must match names from particles section.\n"
      "  \"start\": [ox, oy, oz], -- Starting point of a diagnostic in _global_ coordinates.\n"
      "  \"size\":  [sx, sy, sz], -- Sizes of a diagnosed region along each coordinate in _global_ coordinates.\n"
      "}\n"
      "\n"
      "Available distribution moment names are:\n"
      "  - density (zeroth_moment for x, y, z);\n"
      "  - Vx_moment and moments of Vy, Vz, Vr, Vphi;\n"
      "  - mVxVx_moment and right-first combinations with Vy, Vz, Vr, Vphi, i.e. mVxVy, mVxVz, ...\n"
      "\n"
      "The region, described by \"start\", \"size\" and \"dp\", should use units coherent with distribution moment:\n"
      "  - density, Vx_moment, mVxVy_moment, etc. collect moments onto (x, y, z) coordinates in units of c/w_pe.\n";
    // clang-format on
  }
};

#endif  // SRC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H
