#include "src/pch.h"
#include "src/interfaces/simulation.h"
#include "src/utils/configuration.h"
#include "src/utils/utils.h"

static constexpr std::string_view help =
  "Usage: [mpiexec] simulation.out <config.json>\n";

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help.data()));

#if PERF_LEVEL > 0
  PetscCall(PetscLogDefaultBegin());
#endif

  if (argc < 2) {
    LOG(help);
    return EXIT_FAILURE;
  }

  try {
    Configuration::init(argv[1]);
    Configuration::save();

    Simulation_up simulation = build_simulation();
    PetscCall(simulation->initialize());
    PetscCall(simulation->calculate());
  }
  catch (const std::exception& e) {
    LOG("what(): {}", e.what());
    PetscCallMPI(MPI_Abort(PETSC_COMM_WORLD, EXIT_FAILURE));
  }
  catch (...) {
    LOG("Unknown exception handled!");
    PetscCallMPI(MPI_Abort(PETSC_COMM_WORLD, EXIT_FAILURE));
  }

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
