#include "energy_conservation.h"

#include <iomanip>

#include "src/commands/fields_damping.h"
#include "src/commands/inject_particles.h"
#include "src/commands/remove_particles.h"
#include "src/utils/configuration.h"

EnergyConservation::EnergyConservation( //
  const interfaces::Simulation& simulation,
  std::shared_ptr<FieldsEnergy> fields_energy,
  std::shared_ptr<ParticlesEnergy> particles_energy)
  : file_(SyncFile(CONFIG().out_dir + "/temporal/energy_conservation.txt")),
    simulation(simulation),
    fields_energy(fields_energy),
    particles_energy(particles_energy)
{
}

PetscErrorCode EnergyConservation::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  // PetscCall(VecAXPY(B, -1.0, B0));

  if (t == simulation.start) {
    PetscCall(write_header());
    fields_energy->calculate_energies();
    particles_energy->calculate_energies();
  }

  auto output = [&](PetscReal x) {
    file_() << std::setw(14) << x;
  };

  PetscReal prev_E = fields_energy->get_electric_energy();
  PetscReal prev_B = fields_energy->get_magnetic_energy();
  fields_energy->calculate_energies();
  PetscReal dE = fields_energy->get_electric_energy() - prev_E;
  PetscReal dB = fields_energy->get_magnetic_energy() - prev_B;
  PetscReal dF = dE + dB;
  output(dE);
  output(dB);

  PetscReal dK = 0.0;
  auto&& K0 = particles_energy->get_energies();
  particles_energy->calculate_energies();
  auto&& K = particles_energy->get_energies();

  for (PetscInt i = 0; i < (PetscInt)K.size(); ++i) {
    output(K[i] - K0[i]);
    dK += K[i] - K0[i];
  }

  for (const auto& command : simulation.step_presets_) {
    if (auto&& damp = dynamic_cast<FieldsDamping*>(command.get())) {
      output(damp->get_damped_energy());
      dF += damp->get_damped_energy();
    }
    if (auto&& injection = dynamic_cast<InjectParticles*>(command.get())) {
      output(injection->get_ionized_energy());
      output(injection->get_ejected_energy());
      dK -= injection->get_ionized_energy() + injection->get_ejected_energy();
    }
    if (auto&& remove = dynamic_cast<RemoveParticles*>(command.get())) {
      output(remove->get_removed_energy());
      dK += remove->get_removed_energy();
    }
  }

  output(dF + dK);
  file_() << "\n";

  if (t % diagnose_period_ == 0)
    file_.flush();

  // PetscCall(VecAXPY(B, +1.0, B0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::write_header()
{
  PetscFunctionBeginUser;
  file_() << "Delta(E)\tDelta(B)\t";

  for (const auto& particles : particles_energy->particles_) {
    auto&& name = particles->parameters.sort_name;
    file_() << "Delta(K_" << name << ")\t";
  }

  for (const auto& command : simulation.step_presets_) {
    if (dynamic_cast<FieldsDamping*>(command.get()))
      file_() << "Damped(dE+dB)\t";
    if (auto&& injection = dynamic_cast<InjectParticles*>(command.get())) {
      file_() << "Inj_" << injection->get_ionized_name() << "\t";
      file_() << "Inj_" << injection->get_ejected_name() << "\t";
    }
    if (auto&& remove = dynamic_cast<RemoveParticles*>(command.get()))
      file_() << "Rm_" << remove->get_particles_name() << "\t";
  }

  file_() << "Total(dE+dB+dK)\t";
  file_() << "\n";
  PetscFunctionReturn(PETSC_SUCCESS);
}
