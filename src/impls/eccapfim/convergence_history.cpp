#include "convergence_history.h"

namespace eccapfim {

ConvergenceHistory::ConvergenceHistory(const Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/convergence_history.txt"),
    simulation(simulation)
{
}

PetscErrorCode ConvergenceHistory::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  add(6, "Time", "{:d}", t);

  for (const auto& sort : simulation.particles_) {
    const auto& name = sort->parameters.sort_name;
    auto cn = sort->get_average_iteration_number();
    auto cl = sort->get_average_number_of_traversed_cells();
    add(8, "AvgCN_" + name, "{:.3f}", cn);
    add(8, "AvgCL_" + name, "{:.3f}", cl);
  }

  SNES snes = simulation.snes;

  PetscInt it, fev, len;
  PetscReal* hist;

  PetscCall(SNESGetIterationNumber(snes, &it));
  PetscCall(SNESGetNumberFunctionEvals(snes, &fev));
  PetscCall(SNESGetConvergenceHistory(snes, &hist, nullptr, &len));

  add(6, "FEvals", "{:d}", fev);
  add(6, "ItNum", "{:d}", it);

  if (len == 0) {
    add(12, "ConvHist", "{}", "");
  }
  else {
    for (PetscInt i = 0; i < len; ++i)
      add(12, "ConvHist", "{:8.6e}", hist[i]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace eccapfim
