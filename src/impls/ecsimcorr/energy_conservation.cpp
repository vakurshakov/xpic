#include "energy_conservation.h"

#include "src/utils/configuration.h"

namespace ecsimcorr {

EnergyConservation::EnergyConservation(const Simulation& simulation)
  : ::EnergyConservation(simulation)

{
  B = simulation.B;
  B0 = simulation.B0;

  fields_energy = std::make_unique<FieldsEnergy>(
    simulation.world.da, simulation.E, simulation.B);

  std::vector<const interfaces::Particles*> storage;
  for (const auto& particles : simulation.particles_)
    storage.push_back(particles.get());
  particles_energy = std::make_unique<ParticlesEnergy>(storage);
}

EnergyConservation::EnergyConservation( //
  const interfaces::Simulation& simulation,
  std::shared_ptr<FieldsEnergy> fields_energy,
  std::shared_ptr<ParticlesEnergy> particles_energy)
  : ::EnergyConservation(simulation, fields_energy, particles_energy)
{
  auto&& _simulation = dynamic_cast<const ecsimcorr::Simulation&>(simulation);
  B = _simulation.B;
  B0 = _simulation.B0;
}

PetscErrorCode EnergyConservation::add_titles()
{
  PetscFunctionBeginUser;
  PetscCall(::EnergyConservation::add_titles());

  PetscInt off = 2;
  for (const auto& particles : particles_energy->particles_) {
    off++;
    add_title("λδK_" + particles->parameters.sort_name, off);
  }

  add_title("δE+δB+ΔtJE");
  add_title("|δK-ΔtJE|");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_args()
{
  PetscFunctionBeginUser;
  PetscCall(VecAXPY(B, -1.0, B0));
  PetscCall(::EnergyConservation::add_args());

  PetscInt off = 2;
  for (const auto& particles : particles_energy->particles_) {
    off++;
    add_arg(dynamic_cast<const ecsimcorr::Particles*>(particles)->lambda_dK, off);
  }

  /// @note Esirkepov current finally created electric field, so its work should be used
  PetscReal corr_w = 0.0;
  for (const auto& particles : particles_energy->particles_) {
    corr_w += dt * dynamic_cast<const ecsimcorr::Particles*>(particles)->corr_w;
  }

  add_arg(dF + corr_w);
  add_arg(std::abs(dK - corr_w));
  PetscCall(VecAXPY(B, +1.0, B0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ecsimcorr
