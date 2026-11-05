#include "src/impls/hybrid_kinetic/simulation.h"
#include "tests/hybrid/hybrid_ex1_comparison.h"

static constexpr char help[] =
  "Energy and charge conservation of the \"hybrid_kinetic\" model.          \n"
  "                                                                          \n"
  "A uniform Maxwellian plasma (T = 0.1 keV, mi/me = 100, 50 markers per    \n"
  "cell for each sort) in a uniform field B = 1 is modeled in a periodic    \n"
  "3x3x3 box with dx = 10 >> r_D. Ions are fully kinetic (Crank-Nicolson),  \n"
  "electrons are drift-kinetic guiding centers, dt = 10 / Omega_e. The       \n"
  "balance is written to temporal/hybrid_diagnostic.txt and checked by       \n"
  "tools/hybrid_ex1_energy.py.                                       \n";

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscCall(ex1::overwrite_config(  //
    __FILE__, "hybrid_kinetic", "kinetic", "drift_kinetic", 10.0));
  Configuration::save(ex1::out_dir(__FILE__));
  PetscCall(ex1::run_timed<hybrid_kinetic::Simulation>(__FILE__));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
