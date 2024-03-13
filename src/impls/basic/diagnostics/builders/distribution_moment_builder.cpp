#include "distribution_moment_builder.h"

namespace basic {

Distribution_moment_builder::Distribution_moment_builder(
  const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics,
  const std::string& moment_name, const std::string& proj_name)
  : Diagnostic_builder(simulation, diagnostics),
    moment_name(moment_name), proj_name(proj_name) {}

PetscErrorCode Distribution_moment_builder::build(const Configuration::json_t& diag_info) {
  PetscFunctionBeginUser;
  const Particles& particles = simulation_.particles_[0];

  LOG_INFO("Add {}_of_{} diagnostic for {}", moment_name, proj_name, particles.parameters().sort_name);

  std::string res_dir = CONFIG().out_dir + "/" + moment_name + "_of_" + proj_name;

  diagnostics_.emplace_back(std::make_unique<Distribution_moment>(
    res_dir, simulation_.da_, particles,
    std::make_unique<Moment>(particles, moment_name),
    std::make_unique<Projector>(particles, proj_name)));

  PetscFunctionReturn(PETSC_SUCCESS);
}

}
