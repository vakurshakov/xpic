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
}

EnergyConservation::EnergyConservation(const interfaces::Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/energy_conservation.txt"),
    simulation(simulation)
{
}

PetscErrorCode EnergyConservation::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t == 0) {
    E0 = B0 = dE = dB = dF = dK = 0.0;
    std::fill_n(std::back_inserter(K0), energy->particles.size(), 0.0);
    std::fill_n(std::back_inserter(K), energy->particles.size(), 0.0);
    PetscCall(initialize());
  }

  E0 = energy->get_electric_energy();
  B0 = energy->get_magnetic_energy();
  K0 = energy->get_kinetic_energies();

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

PetscErrorCode EnergyConservation::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  add(6, "Time", "{:d}", t);

  dE = energy->get_electric_energy() - E0;
  dB = energy->get_magnetic_energy() - B0;
  K = energy->get_kinetic_energies();

  dF = dE + dB;
  add(13, "dE", "{: .6e}", dE);
  add(13, "dB", "{: .6e}", dB);

  dK = 0.0;
  for (PetscInt i = 0; i < (PetscInt)K.size(); ++i) {
    auto&& n = energy->particles[i]->parameters.sort_name;
    add(13, "dK_" + n, "{: .6e}", K[i] - K0[i]);
    dK += K[i] - K0[i];
  }

  for (const auto& command : simulation.step_presets_) {
    if (auto&& damp = dynamic_cast<FieldsDamping*>(command.get())) {
      add(13, "Damped(E+B)", "{: .6e}", damp->get_damped_energy());
      dF += damp->get_damped_energy();
    }
    if (auto&& injection = dynamic_cast<InjectParticles*>(command.get())) {
      auto&& ni = injection->get_ionized_name();
      auto&& ne = injection->get_ejected_name();
      auto&& wi = injection->get_ionized_energy();
      auto&& we = injection->get_ejected_energy();
      add(13, "Inj_" + ni, "{: .6e}", wi);
      add(13, "Inj_" + ne, "{: .6e}", we);
      dK -= wi + we;
    }
    if (auto&& remove = dynamic_cast<RemoveParticles*>(command.get())) {
      auto&& n = remove->get_particles_name();
      auto&& w = remove->get_removed_energy();
      add(13, "Rm_" + n, "{: .6e}", w);
      dK += remove->get_removed_energy();
    }
  }

  add(13, "dE+dB+dK", "{: .6e}", dF + dK);
  PetscFunctionReturn(PETSC_SUCCESS);
}
