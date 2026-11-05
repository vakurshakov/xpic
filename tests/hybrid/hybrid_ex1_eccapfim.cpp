#include "src/impls/eccapfim/simulation.h"
#include "tests/hybrid/hybrid_ex1_comparison.h"

static constexpr char help[] =
  "Energy and charge conservation of the \"eccapfim\" model.                \n"
  "                                                                          \n"
  "The same uniform Maxwellian plasma as in hybrid_ex1_hybrid_kinetic, \n"
  "both sorts are fully kinetic. The step dt = 0.2 / Omega_e resolves the    \n"
  "electron gyration, the total time is the same. The balance is written to \n"
  "temporal/energy*.txt and charge_conservation.txt and checked by           \n"
  "tools/hybrid_ex1_energy.py.                                       \n";

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscCall(ex1::overwrite_config(__FILE__, "eccapfim", "kinetic", "kinetic", 0.2));
  Configuration::save(ex1::out_dir(__FILE__));
  PetscCall(ex1::run_timed<eccapfim::Simulation>(__FILE__));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
