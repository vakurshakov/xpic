#include "energy_conservation.h"

#include "src/commands/fields_damping.h"
#include "src/commands/inject_particles.h"
#include "src/commands/remove_particles.h"
#include "src/utils/configuration.h"

EnergyConservation::EnergyConservation( //
  const interfaces::Simulation& simulation,
  std::shared_ptr<FieldsEnergy> fields_energy,
  std::shared_ptr<ParticlesEnergy> particles_energy)
  : file_(SyncFile(CONFIG().out_dir + "/" + filename_)),
    simulation(simulation),
    fields_energy(fields_energy),
    particles_energy(particles_energy)
{
}

EnergyConservation::EnergyConservation(const interfaces::Simulation& simulation)
  : simulation(simulation)
{
}

PetscErrorCode EnergyConservation::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t == 0) {
    PetscCall(add_titles());
    PetscCall(write_formatted("{:^13s}  ", titles_));

    fields_energy->calculate_energies();
    particles_energy->calculate_energies();
  }

  PetscCall(add_args());
  PetscCall(write_formatted("{: .6e}  ", args_));

  if (t % diagnose_period_ == 0)
    PetscCall(file_.flush());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_titles()
{
  PetscFunctionBeginUser;
  add_title("dE");
  add_title("dB");

  for (const auto& particles : particles_energy->particles_) {
    add_title("dK_" + particles->parameters.sort_name);
  }

  for (const auto& command : simulation.step_presets_) {
    if (dynamic_cast<FieldsDamping*>(command.get())) {
      add_title("Damped(dE+dB)");
    }
    if (auto&& injection = dynamic_cast<InjectParticles*>(command.get())) {
      add_title("Inj_" + injection->get_ionized_name());
      add_title("Inj_" + injection->get_ejected_name());
    }
    if (auto&& remove = dynamic_cast<RemoveParticles*>(command.get())) {
      add_title("Rm_" + remove->get_particles_name());
    }
  }

  add_title("dE+dB+dK");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_args()
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

void EnergyConservation::add_arg(PetscReal arg, PetscInt pos)
{
  add(arg, args_, pos);
}

void EnergyConservation::add_title(std::string title, PetscInt pos)
{
  add(title, titles_, pos);
}
