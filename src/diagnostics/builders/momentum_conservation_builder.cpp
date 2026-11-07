#include "momentum_conservation_builder.h"

#include "src/diagnostics/momentum_conservation.h"

MomentumConservationBuilder::MomentumConservationBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : DiagnosticBuilder(simulation, diagnostics)
{
}

PetscErrorCode MomentumConservationBuilder::build(const Configuration::json_t& /* info */)
{
  PetscFunctionBeginUser;
  std::vector<const interfaces::Particles*> particles;
  for (const auto& sort : simulation_.particles_)
    particles.emplace_back(sort.get());

  diagnostics_.emplace_back(std::make_unique<MomentumConservation>(
    simulation_.da, simulation_.E, simulation_.B, particles));

  LOG("  Momentum conservation diagnostic is added");
  PetscFunctionReturn(PETSC_SUCCESS);
}
