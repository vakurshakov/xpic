#include "energy_conservation.h"

#include "src/commands/fields_damping.h"
#include "src/commands/inject_particles.h"
#include "src/commands/remove_particles.h"
#include "src/utils/configuration.h"

EnergyConservation::EnergyConservation( //
  const interfaces::Simulation& simulation, std::shared_ptr<Energy> energy)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/energy_conservation.txt"),
    simulation(simulation),
    energy(energy)
{
  E0 = B0 = dE = dB = dF = dK = 0.0;
  std::fill_n(std::back_inserter(K0), energy->particles.size(), 0.0);
  std::fill_n(std::back_inserter(K), energy->particles.size(), 0.0);
}

EnergyConservation::EnergyConservation(const interfaces::Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/energy_conservation.txt"),
    simulation(simulation)
{
  E0 = B0 = dE = dB = dF = dK = 0.0;
  std::fill_n(std::back_inserter(K0), energy->particles.size(), 0.0);
  std::fill_n(std::back_inserter(K), energy->particles.size(), 0.0);
}

PetscErrorCode EnergyConservation::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t == 0) {
    PetscCall(initialize());
    E0 = energy->get_electric_energy();
    B0 = energy->get_magnetic_energy();
    K0 = energy->get_kinetic_energies();
  }

  PetscCall(energy->diagnose(t));
  PetscCall(TableDiagnostic::diagnose(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(energy->initialize());
  PetscCall(TableDiagnostic::initialize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_titles()
{
  PetscFunctionBeginUser;
  add_title("time");
  add_title("δE");
  add_title("δB");

  for (const auto& particles : energy->particles) {
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

PetscErrorCode EnergyConservation::add_args(PetscInt t)
{
  PetscFunctionBeginUser;
  dE = energy->get_electric_energy() - E0;
  dB = energy->get_magnetic_energy() - B0;
  K = energy->get_kinetic_energies();

  dF = dE + dB;
  add_arg(t);
  add_arg(dE);
  add_arg(dB);

  dK = 0.0;
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
