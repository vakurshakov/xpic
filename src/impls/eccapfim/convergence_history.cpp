#include "convergence_history.h"

namespace eccapfim {

ConvergenceHistory::ConvergenceHistory(const Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/convergence_history.txt"),
    simulation(simulation)
{
}

PetscErrorCode ConvergenceHistory::add_titles()
{
  PetscFunctionBeginUser;
  add_title("time");

  for (const auto& sort : simulation.particles_) {
    const auto& name = sort->parameters.sort_name;
    add_title("AvgCN_" + name);
    add_title("AvgCL_" + name);
  }

  add_title("SnesFuncEvals");
  add_title("SnesIterNum");
  add_title("SnesConvHist");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @todo Here, the natural need in different formatting is visible
PetscErrorCode ConvergenceHistory::add_args(PetscInt t)
{
  PetscFunctionBeginUser;
  add_arg(t);

  for (const auto& sort : simulation.particles_) {
    add_arg(sort->get_average_iteration_number());
    add_arg(sort->get_average_number_of_traversed_cells());
  }

  SNES snes = simulation.snes;

  PetscInt it, fev, len;
  PetscReal* hist;

  PetscCall(SNESGetIterationNumber(snes, &it));
  PetscCall(SNESGetNumberFunctionEvals(snes, &fev));
  PetscCall(SNESGetConvergenceHistory(snes, &hist, nullptr, &len));

  add_arg(fev);
  add_arg(it);

  for (PetscInt i = 0; i < len; ++i)
    add_arg(hist[i]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace eccapfim
