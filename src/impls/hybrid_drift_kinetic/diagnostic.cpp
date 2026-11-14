#include "diagnostic.h"

#include "src/impls/hybrid_drift_kinetic/simulation.h"

namespace hybrid_drift_kinetic {

EnergyConservation::EnergyConservation(const Simulation& simulation)
  : drift_kinetic::EnergyConservation(simulation), simulation(simulation)
{
}

PetscErrorCode EnergyConservation::fill_energy_cons(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(drift_kinetic::EnergyConservation::fill_energy_cons(t));
  const auto& sorts = simulation.interfaces::Simulation::particles_;

  // The work of magnetization uses the full magnetic-field increment.
  Vec field_change;
  PetscCall(DMGetGlobalVector(simulation.da, &field_change));
  PetscCall(VecWAXPY(field_change, -1, simulation.B_hk, simulation.B));
  PetscCall(VecScale(field_change, 2));

  // Before the first step M, J and E_hk are zero, so all balances vanish.
  PetscReal MdB;
  PetscCall(VecDot(simulation.M, field_change, &MdB));
  const PetscReal field_balance = dE + dB + dEJ - MdB;
  const std::string field_title = "dW-tEJ_field";
  energy_cons.add(std::max<PetscInt>(13, field_title.size()), field_title, "{: .6e}", field_balance);

  for (std::size_t i = 0; i < sorts.size(); i++) {
    const auto& sort = sorts[i];
    PetscReal EJ, MdB = 0;
    PetscCall(VecDot(simulation.E_hk, sort->J, &EJ));
    const auto dk = std::find_if(simulation.particles_.begin(),
      simulation.particles_.end(), [&](const auto& ptr)
      {
        return ptr.get() == sort.get();
      });
    if (dk != simulation.particles_.end())
      PetscCall(VecDot((*dk)->M, field_change, &MdB));
    const PetscReal balance = K[i] - K0[i] - dt * EJ + MdB;
    const std::string title = "dK-tEJ_" + sort->parameters.sort_name;
    energy_cons.add(std::max<PetscInt>(13, title.size()), title, "{: .6e}", balance);
  }
  PetscCall(DMRestoreGlobalVector(simulation.da, &field_change));
  PetscFunctionReturn(PETSC_SUCCESS);
}

ConvergenceHistory::ConvergenceHistory(const Simulation& simulation)
  : drift_kinetic::ConvergenceHistory(simulation), hybrid_simulation(simulation)
{
}

PetscErrorCode ConvergenceHistory::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(drift_kinetic::ConvergenceHistory::add_columns(t));
  for (const auto& sort : hybrid_simulation.kinetic_particles_) {
    const std::string title = "AvgCN_" + sort->parameters.sort_name;
    add(std::max<PetscInt>(13, title.size()), title, "{: .6e}",
      t == hybrid_simulation.start ? 0 : sort->get_average_iteration_number());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace hybrid_drift_kinetic
