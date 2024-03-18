#ifndef SRC_BASIC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H
#define SRC_BASIC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H

#include "src/impls/basic/diagnostics/builders/diagnostic_builder.h"
#include "src/impls/basic/diagnostics/distribution_moment.h"

namespace basic {

class Distribution_moment_builder : public Diagnostic_builder {
public:
  Distribution_moment_builder(
    const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics,
    const std::string& moment_name, const std::string& proj_name);

  PetscErrorCode build(const Configuration::json_t& diag_info) override;

private:
  constexpr std::string usage_message() const override {
    return "\n" "Usage is unspecified yet.";
  }

  std::string moment_name;
  std::string proj_name;

  struct Moment_description {
    std::string particles_name;
    Distribution_moment::Region region;
    MPI_Comm comm;
  };

  using Moments_description = std::vector<Moment_description>;
  Moments_description moments_desc_;

  PetscErrorCode parse_moment_info(const Configuration::json_t& json, Moment_description& desc);
  // PetscErrorCode check_moment_description(const Moment_description& desc);
  // PetscErrorCode attach_moment_description(const DM& da, Moment_description&& desc);
  const Particles& find_sort(const std::string& sort_name) const;
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_BUILDERS_DISTRIBUTION_MOMENT_BUILDER_H
