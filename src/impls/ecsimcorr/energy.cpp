#include "src/impls/ecsimcorr/energy.h"

#include "src/commands/fields_damping.h"
#include "src/commands/inject_particles.h"
#include "src/commands/remove_particles.h"
#include "src/utils/configuration.h"

namespace ecsimcorr {

EnergyDiagnostic::EnergyDiagnostic(const Simulation& simulation)
  : simulation(simulation)
{
  file_ = SyncFile(CONFIG().out_dir + "/energies.dat");

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

  PetscReal work = 0.0;
  PetscReal total = 0.0;

  PetscCall(VecAXPY(B, -1.0, B0));
  PetscReal prev_we = fields_energy->get_electric_energy();
  PetscReal prev_wb = fields_energy->get_magnetic_energy();
  fields_energy->calculate_energies();
  PetscReal diff_we = fields_energy->get_electric_energy() - prev_we;
  PetscReal diff_wb = fields_energy->get_magnetic_energy() - prev_wb;
  file_() << diff_we << "\t";
  file_() << diff_wb << "\t";
  total += (diff_we + diff_wb);
  PetscCall(VecAXPY(B, +1.0, B0));

  std::vector<PetscReal> prev_energies = particles_energy->get_energies();
  particles_energy->calculate_energies();
  std::vector<PetscReal> new_energies = particles_energy->get_energies();

  for (std::size_t i = 0; i < new_energies.size(); ++i) {
    work += new_energies[i] - prev_energies[i];
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

  work -= dt * simulation.w1;
  total -= dt * simulation.w1;
  file_() << dt * simulation.w1 << "\t";
  file_() << work << "\t";
  file_() << total << "\t";

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

  file_() << "Work[dt*JE]\t";
  file_() << "Work[delta(K)-dt*JE]\t";
  file_() << "Total[E+B-dt*JE]\t";

  file_() << "\n";
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace ecsimcorr
