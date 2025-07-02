#include "energy_conservation.h"

#include "src/utils/configuration.h"

namespace ecsimcorr {

EnergyConservation::EnergyConservation(const Simulation& simulation)
  : ::EnergyConservation(simulation)

{
  B = simulation.B;
  B0 = simulation.B0;

  std::vector<const interfaces::Particles*> storage;
  for (const auto& particles : simulation.particles_)
    storage.push_back(particles.get());

  energy = std::make_unique<Energy>(simulation.E, simulation.B, storage);
}

PetscErrorCode EnergyConservation::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(VecAXPY(B, -1.0, B0));
  PetscCall(::EnergyConservation::add_columns(t));

  PetscInt off = 3;
  for (const auto& sort : simulation.particles_) {
    auto* particles = dynamic_cast<ecsimcorr::Particles*>(sort.get());
    auto&& name = sort->parameters.sort_name;
    auto&& cwd = particles->lambda_dK;
    auto&& pwd = particles->pred_dK - dt * particles->pred_w;
    auto&& ldk = particles->corr_dK - dt * particles->corr_w;
    add(13, "CWD_" + name, "{: .6e}", cwd, ++off);
    add(13, "PWD_" + name, "{: .6e}", pwd, ++off);
    add(13, "LdK_" + name, "{: .6e}", ldk, ++off);
    ++off;
  }

  /// @note Esirkepov current finally created electric field, so its work should be used
  PetscReal corr_w = 0.0;
  for (const auto& sort : simulation.particles_) {
    corr_w += dynamic_cast<ecsimcorr::Particles*>(sort.get())->corr_w;
  }

  add(13, "WD", "{: .6e}", dK - dt * corr_w);
  PetscCall(VecAXPY(B, +1.0, B0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ecsimcorr
