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

PetscErrorCode EnergyConservation::add_titles()
{
  PetscFunctionBeginUser;
  PetscCall(::EnergyConservation::add_titles());

  PetscInt off = 3;
  for (const auto& sort : simulation.particles_) {
    const auto& name = sort->parameters.sort_name;
    add_title("λδK_" + name, ++off);
    add_title("PWΔ_" + name, ++off);
    add_title("CWΔ_" + name, ++off);
    ++off;
  }

  add_title("WΔ");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_args(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(VecAXPY(B, -1.0, B0));
  PetscCall(::EnergyConservation::add_args(t));

  PetscInt off = 3;
  for (const auto& sort : simulation.particles_) {
    auto* particles = dynamic_cast<ecsimcorr::Particles*>(sort.get());
    add_arg(particles->lambda_dK, ++off);
    add_arg(particles->pred_dK - dt * particles->pred_w, ++off);
    add_arg(particles->corr_dK - dt * particles->corr_w, ++off);
    ++off;
  }

  /// @note Esirkepov current finally created electric field, so its work should be used
  PetscReal corr_w = 0.0;
  for (const auto& sort : simulation.particles_) {
    corr_w += dynamic_cast<ecsimcorr::Particles*>(sort.get())->corr_w;
  }

  add_arg(dK - dt * corr_w);
  PetscCall(VecAXPY(B, +1.0, B0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ecsimcorr
