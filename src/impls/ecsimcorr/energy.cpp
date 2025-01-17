#include "src/impls/ecsimcorr/energy.h"

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

  PetscCall(VecAXPY(B, -1.0, B0));
  PetscReal prev_we = fields_energy->get_electric_energy();
  PetscReal prev_wb = fields_energy->get_magnetic_energy();
  fields_energy->calculate_energies();
  PetscReal dE = fields_energy->get_electric_energy() - prev_we;
  PetscReal dB = fields_energy->get_magnetic_energy() - prev_wb;
  file_() << dE << "\t";
  file_() << dB << "\t";
  PetscCall(VecAXPY(B, +1.0, B0));

  std::vector<PetscReal> prev_energies = particles_energy->get_energies();
  particles_energy->calculate_energies();
  std::vector<PetscReal> new_energies = particles_energy->get_energies();

  PetscReal dK = 0.0;
  for (std::size_t i = 0; i < new_energies.size(); ++i) {
    dK += new_energies[i] - prev_energies[i];
    file_() << new_energies[i] - prev_energies[i] << "\t";
  }

  if (simulation.damping)
    file_() << simulation.damping->get_damped_energy() << "\t";

  for (const auto& command : simulation.step_presets_) {
    if (auto&& injection = dynamic_cast<InjectParticles*>(command.get())) {
      file_() << injection->get_ionized_energy() << "\t";
      file_() << injection->get_ejected_energy() << "\t";
    }
    if (auto&& remove = dynamic_cast<RemoveParticles*>(command.get()))
      file_() << remove->get_removed_energy() << "\t";
  }

  PetscReal dtJE = dt * simulation.w1;
  file_() << (dE + dB) - dK << "\t";
  file_() << (dE + dB) - dtJE << "\t";
  file_() << dK - dtJE << "\t";

  file_() << "\n";

  // if (t % diagnose_period)
  file_.flush();

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyDiagnostic::write_header()
{
  PetscFunctionBeginUser;
  file_() << "delta(E)\tdelta(dB)\t";

  for (const auto& particles : simulation.particles_)
    file_() << "delta(K_" << particles->parameters().sort_name << ")\t";

  if (simulation.damping)
    file_() << "Damped\t";

  for (const auto& command : simulation.step_presets_) {
    if (auto&& injection = dynamic_cast<InjectParticles*>(command.get())) {
      file_() << "Inj_" << injection->get_ionized_name() << "\t";
      file_() << "Inj_" << injection->get_ejected_name() << "\t";
    }
    if (auto&& remove = dynamic_cast<RemoveParticles*>(command.get()))
      file_() << "Rm_" << remove->get_particles_name() << "\t";
  }

  file_() << "Total[dE+dB-dK]\t";
  file_() << "Total[dE+dB-dt*JE]\t";
  file_() << "Work[dK-dt*JE]\t";

  file_() << "\n";
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace ecsimcorr
