#include "src/impls/ecsimcorr/energy.h"

#include <iomanip>

#include "src/commands/fields_damping.h"
#include "src/commands/inject_particles.h"
#include "src/commands/remove_particles.h"
#include "src/utils/configuration.h"

namespace ecsimcorr {

EnergyDiagnostic::EnergyDiagnostic(const Simulation& simulation)
  : simulation(simulation)
{
  file_ = SyncFile(CONFIG().out_dir + "/energy.dat");

  E = simulation.E;
  B = simulation.B;
  B0 = simulation.B0;

  fields_energy = std::make_unique<FieldsEnergy>(simulation.world_.da, E, B);

  ParticlesEnergy::ParticlesPointersVector storage;
  storage.reserve(simulation.particles_.size());

  for (const auto& particles : simulation.particles_)
    storage.emplace_back(particles.get());

  particles_energy = std::make_unique<ParticlesEnergy>(storage);
}

PetscErrorCode EnergyDiagnostic::diagnose(timestep_t t)
{
  PetscFunctionBeginUser;
  if (t == 0)
    PetscCall(write_header());

  auto output = [&](PetscReal x) {
    file_() << std::setw(14) << x;
  };

  PetscCall(VecAXPY(B, -1.0, B0));
  PetscReal prev_we = fields_energy->get_electric_energy();
  PetscReal prev_wb = fields_energy->get_magnetic_energy();
  fields_energy->calculate_energies();
  PetscReal dE = fields_energy->get_electric_energy() - prev_we;
  PetscReal dB = fields_energy->get_magnetic_energy() - prev_wb;
  output(dE);
  output(dB);
  PetscCall(VecAXPY(B, +1.0, B0));

  std::vector<PetscReal> prev_energies = particles_energy->get_energies();
  particles_energy->calculate_energies();
  std::vector<PetscReal> new_energies = particles_energy->get_energies();

  PetscReal dK = 0.0;
  for (std::size_t i = 0; i < new_energies.size(); ++i) {
    dK += new_energies[i] - prev_energies[i];
    output(new_energies[i] - prev_energies[i]);
  }

  for (const auto& particles : simulation.particles_)
    output(particles->lambda_energy);

  if (simulation.damping)
    output(simulation.damping->get_damped_energy());

  for (const auto& command : simulation.step_presets_) {
    if (auto&& injection = dynamic_cast<InjectParticles*>(command.get())) {
      output(injection->get_ionized_energy());
      output(injection->get_ejected_energy());
    }
    if (auto&& remove = dynamic_cast<RemoveParticles*>(command.get()))
      output(remove->get_removed_energy());
  }

  /// @note Esirkepov current finally created electric field, so its work should be used
  PetscReal w2 = 0.0;

  for (const auto& particles : simulation.particles_)
    w2 += dt * particles->w2;

  output((dE + dB) + dK);
  output((dE + dB) + w2);
  output(dK - w2);

  file_() << "\n";

  if (t % diagnose_period == 0)
    file_.flush();

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyDiagnostic::write_header()
{
  PetscFunctionBeginUser;
  file_() << "Delta(E)\tDelta(B)\t";

  for (const auto& particles : simulation.particles_)
    file_() << "Delta(K_" << particles->parameters().sort_name << ")\t";

  for (const auto& particles : simulation.particles_)
    file_() << "Lambda(K_" << particles->parameters().sort_name << ")\t";

  if (simulation.damping)
    file_() << "Damped(E+B)\t";

  for (const auto& command : simulation.step_presets_) {
    if (auto&& injection = dynamic_cast<InjectParticles*>(command.get())) {
      file_() << "Inj_" << injection->get_ionized_name() << "\t";
      file_() << "Inj_" << injection->get_ejected_name() << "\t";
    }
    if (auto&& remove = dynamic_cast<RemoveParticles*>(command.get()))
      file_() << "Rm_" << remove->get_particles_name() << "\t";
  }

  file_() << "Total(dE+dB+dK)\t";
  file_() << "Total(dE+dB+dt*JE)\t";
  file_() << "Work(dK-dt*JE)\t";

  file_() << "\n";
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ecsimcorr
