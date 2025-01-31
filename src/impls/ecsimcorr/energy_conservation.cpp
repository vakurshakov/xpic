#include "src/impls/ecsimcorr/energy_conservation.h"

#include <iomanip>

#include "src/commands/fields_damping.h"
#include "src/commands/inject_particles.h"
#include "src/commands/remove_particles.h"
#include "src/utils/configuration.h"

namespace ecsimcorr {

EnergyConservation::EnergyConservation(const Simulation& simulation)
  : simulation(simulation)
{
  file_ = SyncFile(CONFIG().out_dir + "/energy_conservation.dat");

  E = simulation.E;
  B = simulation.B;
  B0 = simulation.B0;

  fields_energy = std::make_unique<FieldsEnergy>(simulation.world.da, E, B);

  ParticlesEnergy::ParticlesPointersVector storage;
  for (const auto& particles : simulation.particles_)
    storage.push_back(particles.get());

  particles_energy = std::make_unique<ParticlesEnergy>(storage);
}

PetscErrorCode EnergyConservation::diagnose(timestep_t t)
{
  PetscFunctionBeginUser;
  PetscCall(VecAXPY(B, -1.0, B0));

  if (t == 0) {
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

    const auto& particles = simulation.particles_[i];
    output(particles->lambda_dK);
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

  /// @note Esirkepov current finally created electric field, so its work should be used
  PetscReal corr_w = 0.0;
  for (const auto& particles : simulation.particles_)
    corr_w += dt * particles->corr_w;

  output(dF + dK);
  output(dF + corr_w);
  output(std::abs(dK - corr_w));

  file_() << "\n";

  if (t % diagnose_period == 0)
    file_.flush();

  PetscCall(VecAXPY(B, +1.0, B0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::write_header()
{
  PetscFunctionBeginUser;
  file_() << "Delta(E)\tDelta(B)\t";

  for (const auto& particles : simulation.particles_) {
    auto&& name = particles->parameters.sort_name;
    file_() << "Delta(K_" << name << ")\t";
    file_() << "Lambda(dK_" << name << ")\t";
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
  file_() << "Total(dE+dB+dt*JE)\t";
  file_() << "Work(|dK-dt*JE|)\t";

  file_() << "\n";
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ecsimcorr
