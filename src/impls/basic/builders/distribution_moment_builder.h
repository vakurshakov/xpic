#ifndef SRC_BASIC_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H
#define SRC_BASIC_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H

#include "src/impls/basic/builders/diagnostic_builder.h"
#include "src/diagnostics/distribution_moment.h"

namespace basic {

class Distribution_moment_builder : public Diagnostic_builder {
public:
  Distribution_moment_builder(
    const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics,
    const std::string& moment_name, const std::string& proj_name);

  PetscErrorCode build(const Configuration::json_t& diag_info) override;

private:
  const char* usage_message() const override {
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
  }

  std::string moment_name;
  std::string proj_name;

  struct Moment_description {
    std::string particles_name;
    Field_view::Region region;
    MPI_Comm comm;
  };

  using Moments_description = std::vector<Moment_description>;
  Moments_description moments_desc_;

  PetscErrorCode parse_moment_info(const Configuration::json_t& json, Moment_description& desc);
};

}

#endif  // SRC_BASIC_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H
