#include "energy_conservation.h"

#include "src/commands/fields_damping.h"
#include "src/commands/inject_particles.h"
#include "src/commands/remove_particles.h"
#include "src/utils/configuration.h"

EnergyConservation::EnergyConservation( //
  const interfaces::Simulation& simulation,
  std::shared_ptr<FieldsEnergy> fields_energy,
  std::shared_ptr<ParticlesEnergy> particles_energy)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/energy_conservation.txt"),
    simulation(simulation),
    fields_energy(fields_energy),
    particles_energy(particles_energy)
{
}

EnergyConservation::EnergyConservation(const interfaces::Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/energy_conservation.txt"), simulation(simulation)
{
}

PetscErrorCode EnergyConservation::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(fields_energy->calculate_energies());
  PetscCall(particles_energy->calculate_energies());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_titles()
{
  PetscFunctionBeginUser;
  add_title("δE");
  add_title("δB");

  for (const auto& particles : particles_energy->particles_) {
    add_title("δK_" + particles->parameters.sort_name);
  }

  for (const auto& command : simulation.step_presets_) {
    if (dynamic_cast<FieldsDamping*>(command.get())) {
      add_title("Damped(δE+δB)");
    }
    if (auto&& injection = dynamic_cast<InjectParticles*>(command.get())) {
      add_title("Inj_" + injection->get_ionized_name());
      add_title("Inj_" + injection->get_ejected_name());
    }
    if (auto&& remove = dynamic_cast<RemoveParticles*>(command.get())) {
      add_title("Rm_" + remove->get_particles_name());
    }
  }

  add_title("δE+δB+δK");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_args(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscReal prev_E = fields_energy->get_electric_energy();
  PetscReal prev_B = fields_energy->get_magnetic_energy();
  fields_energy->calculate_energies();
  PetscReal dE = fields_energy->get_electric_energy() - prev_E;
  PetscReal dB = fields_energy->get_magnetic_energy() - prev_B;

  dF = dE + dB;
  add_arg(dE);
  add_arg(dB);

  dK = 0.0;
  auto&& K0 = particles_energy->get_energies();
  particles_energy->calculate_energies();
  auto&& K = particles_energy->get_energies();

  for (PetscInt i = 0; i < (PetscInt)K.size(); ++i) {
    add_arg(K[i] - K0[i]);
    dK += K[i] - K0[i];
  }

  for (const auto& command : simulation.step_presets_) {
    if (auto&& damp = dynamic_cast<FieldsDamping*>(command.get())) {
      add_arg(damp->get_damped_energy());
      dF += damp->get_damped_energy();
    }
    if (auto&& injection = dynamic_cast<InjectParticles*>(command.get())) {
      add_arg(injection->get_ionized_energy());
      add_arg(injection->get_ejected_energy());
      dK -= injection->get_ionized_energy() + injection->get_ejected_energy();
    }
    if (auto&& remove = dynamic_cast<RemoveParticles*>(command.get())) {
      add_arg(remove->get_removed_energy());
      dK += remove->get_removed_energy();
    }
  }

  add_arg(dF + dK);
  PetscFunctionReturn(PETSC_SUCCESS);
}
