#include "diagnostic.h"

#include "src/impls/hybrid_drift_kinetic/simulation.h"
#include "src/utils/configuration.h"

namespace hybrid_drift_kinetic {

namespace {

PetscInt width(const std::string& title)
{
  return std::max<PetscInt>(13, (PetscInt)title.size());
}

}  // namespace

EnergyConservation::EnergyConservation(const Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/hybrid_diagnostic.txt"),
    simulation(simulation)
{
}

// Called at the first diagnosed step, also when the run starts from a backup.
PetscErrorCode EnergyConservation::initialize()
{
  PetscFunctionBeginUser;
  if (initialized)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(calculate_kinetic_energies(K0));
  PetscCall(VecDot(simulation.E, simulation.E, &w_E0));
  PetscCall(VecDot(simulation.B, simulation.B, &w_B0));
  w_E0 *= 0.5;
  w_B0 *= 0.5;

  initialized = true;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(initialize());
  const auto& sorts = simulation.interfaces::Simulation::particles_;

  add(6, "Time", "{:d}", t);
  add(13, "AvgFieldIt", "{:d}", simulation.snes_iteration_number);

  std::vector<PetscReal> K;
  PetscCall(calculate_kinetic_energies(K));

  PetscReal w_E, w_B;
  PetscCall(VecDot(simulation.E, simulation.E, &w_E));
  PetscCall(VecDot(simulation.B, simulation.B, &w_B));
  w_E *= 0.5;
  w_B *= 0.5;

  PetscReal w_K = 0;
  PetscReal dK = 0;
  for (std::size_t i = 0; i < sorts.size(); i++) {
    w_K += K[i];
    dK += K[i] - K0[i];
  }

  add(13, "dK", "{: .6e}", dK);
  for (std::size_t i = 0; i < sorts.size(); i++) {
    const std::string title = "dK_" + sorts[i]->parameters.sort_name;
    add(width(title), title, "{: .6e}", K[i] - K0[i]);
  }
  add(13, "dE", "{: .6e}", w_E - w_E0);
  add(13, "dB", "{: .6e}", w_B - w_B0);
  add(13, "dE+dB+dK", "{: .6e}", (w_E - w_E0) + (w_B - w_B0) + dK);

  add(13, "wK", "{: .6e}", w_K);
  for (std::size_t i = 0; i < sorts.size(); i++) {
    const std::string title = "wK_" + sorts[i]->parameters.sort_name;
    add(width(title), title, "{: .6e}", K[i]);
  }
  add(13, "wE", "{: .6e}", w_E);
  add(13, "wB", "{: .6e}", w_B);
  add(13, "wEB + wK", "{: .6e}", w_E + w_B + w_K);

  // Split of the balance: the field part dW_EB + dt E.J - M.dB and, for each
  // sort, dK - dt E.J_s + M_s.dB (M_s = 0 for kinetic sorts). The total error
  // dE+dB+dK is their sum, so these columns localize it.
  Vec dB;
  PetscReal EJ, MdB;
  PetscCall(DMGetGlobalVector(simulation.da, &dB));
  PetscCall(VecWAXPY(dB, -1, simulation.B_hk, simulation.B));
  PetscCall(VecScale(dB, 2));

  PetscCall(VecDot(simulation.E_hk, simulation.J, &EJ));
  PetscCall(VecDot(simulation.M, dB, &MdB));
  add(width("dW-tEJ_field"), "dW-tEJ_field", "{: .6e}",
    (w_E - w_E0) + (w_B - w_B0) + dt * EJ - MdB);

  for (std::size_t i = 0; i < sorts.size(); i++) {
    const auto& sort = sorts[i];
    PetscCall(VecDot(simulation.E_hk, sort->J, &EJ));
    MdB = 0;
    const auto dk = std::find_if(simulation.particles_.begin(),
      simulation.particles_.end(), [&](const auto& ptr)
      {
        return ptr.get() == sort.get();
      });
    if (dk != simulation.particles_.end())
      PetscCall(VecDot((*dk)->M, dB, &MdB));

    const std::string title = "dK-tEJ_" + sort->parameters.sort_name;
    add(width(title), title, "{: .6e}", K[i] - K0[i] - dt * EJ + MdB);
  }
  PetscCall(DMRestoreGlobalVector(simulation.da, &dB));

  K0 = K;
  w_E0 = w_E;
  w_B0 = w_B;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::calculate_kinetic_energies(
  std::vector<PetscReal>& K) const
{
  PetscFunctionBeginUser;
  const auto& sorts = simulation.interfaces::Simulation::particles_;
  K.resize(sorts.size());
  for (std::size_t i = 0; i < sorts.size(); i++)
    K[i] = sorts[i]->kinetic_energy_local();

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, K.data(), (PetscMPIInt)K.size(), MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace hybrid_drift_kinetic
