#include "src/impls/drift_kinetic/simulation.h"
#include "tests/hybrid/hybrid_ex1_comparison.h"

static constexpr char help[] =
  "Energy and charge conservation of the \"drift_kinetic\" model.           \n"
  "                                                                          \n"
  "The same uniform Maxwellian plasma as in hybrid_ex1_hybrid_kinetic, \n"
  "both sorts are drift-kinetic guiding centers, dt = 10 / Omega_e. The      \n"
  "balance is written to temporal/dk_diagnostic.txt and checked by           \n"
  "tools/hybrid_ex1_energy.py.                                       \n";

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscCall(ex1::overwrite_config(  //
    __FILE__, "drift_kinetic", "drift_kinetic", "drift_kinetic", 10.0));
  Configuration::save(ex1::out_dir(__FILE__));
  PetscCall(ex1::run_timed<drift_kinetic::Simulation>(__FILE__));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
