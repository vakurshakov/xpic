#include "charge_conservation.h"

#include "src/utils/configuration.h"
#include "src/utils/profiler.h"

ChargeConservation::ChargeConservation(const interfaces::Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/charge_conservation.txt"),
    simulation(simulation)
{
}

PetscErrorCode ChargeConservation::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(TableDiagnostic::initialize());
  PetscCall(Profiler::register_event("charge_conservation", &event_diagnose));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ChargeConservation::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(Profiler::push_event(event_diagnose));

  add(6, "Time", "{:d}", t);

  // It is important to get `da_rho` as it has reduced dof
  DM da = simulation.da_rho;
  Mat divE = simulation.divE;

  Vec sum, diff;
  PetscReal norm[2];
  PetscCall(DMGetGlobalVector(da, &sum));
  PetscCall(DMGetGlobalVector(da, &diff));
  PetscCall(VecSet(sum, 0));

  for (PetscInt i = 0; i < (PetscInt)simulation.particles_.size(); ++i) {
    const auto& sort = *simulation.particles_[i];
    const auto& name = sort.parameters.sort_name;

    PetscCall(VecSet(diff, 0));

    // Computing partial derivative in time of charge density
    PetscCall(VecAYPX(diff, +1, sort.rho[1]));
    PetscCall(VecAYPX(diff, -1, sort.rho[0]));
    PetscCall(VecScale(diff, -1 / dt));

    PetscCall(VecAXPY(sum, 1, diff));

    // Evaluating continuity equation
    PetscCall(MatMultAdd(divE, sort.J, diff, diff));
    PetscCall(VecNorm(diff, NORM_1_AND_2, norm));

    add(13, "N1dQ_" + name, "{: .6e}", norm[0]);
    add(13, "N2dQ_" + name, "{: .6e}", norm[1]);
  }

  PetscCall(DMRestoreGlobalVector(da, &diff));

  PetscCall(MatMultAdd(divE, simulation.J, sum, sum));
  PetscCall(VecNorm(sum, NORM_1_AND_2, norm));

  add(13, "N1dQ_tot", "{: .6e}", norm[0]);
  add(13, "N2dQ_tot", "{: .6e}", norm[1]);

  PetscCall(DMRestoreGlobalVector(da, &sum));

  PetscCall(Profiler::pop_event(event_diagnose));
  PetscFunctionReturn(PETSC_SUCCESS);
}
