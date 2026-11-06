#include "src/impls/drift_kinetic/simulation.h"
#include "src/impls/eccapfim/simulation.h"
#include "src/impls/ecsimcorr/simulation.h"
#include "src/impls/hybrid_kinetic/simulation.h"
#include "tests/hybrid/hybrid_ex1_comparison.h"

static constexpr char help[] =
  "Energy and charge conservation in a uniform Maxwellian plasma.\n"
  "Select the model with -ex1_model: drift_kinetic, eccapfim, ecsimcorr,\n"
  "or hybrid_kinetic (default).\n";

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  char model[PETSC_MAX_PATH_LEN] = "hybrid_kinetic";
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-ex1_model", model, sizeof(model), nullptr));
  const std::string simulation(model);
  PetscCheck(simulation == "drift_kinetic" || simulation == "eccapfim" ||
      simulation == "ecsimcorr" || simulation == "hybrid_kinetic",
    PETSC_COMM_WORLD, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown hybrid ex1 model: %s", model);

  const std::string file = (std::filesystem::path(__FILE__).parent_path() /
    ("hybrid_ex1_" + simulation + ".cpp")).string();
  const bool drift = simulation == "drift_kinetic";
  const bool hybrid = simulation == "hybrid_kinetic";
  PetscCall(ex1::overwrite_config(file, simulation, drift ? "drift_kinetic" : "kinetic",
    (drift || hybrid) ? "drift_kinetic" : "kinetic", (drift || hybrid) ? 10 : 0.2));
  Configuration::save(ex1::out_dir(file));

  if (drift)
    PetscCall(ex1::run_timed<drift_kinetic::Simulation>(file));
  else if (simulation == "eccapfim")
    PetscCall(ex1::run_timed<eccapfim::Simulation>(file));
  else if (simulation == "ecsimcorr")
    PetscCall(ex1::run_timed<ecsimcorr::Simulation>(file));
  else
    PetscCall(ex1::run_timed<hybrid_kinetic::Simulation>(file));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
