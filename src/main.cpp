#include "src/pch.h"
#include "src/interfaces/simulation.h"
#include "src/utils/configuration.h"

static char help[] = "Usage: simulation.out <config.json>\n";

int main(int argc, char** argv) {
  if (argc != 2) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, help));
    return EXIT_FAILURE;
  }

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  try {
    const Configuration& config = CONFIG();
    config.init(argv[1]);
    config.save();

    LOG_INIT(config.out_dir + "/simulation.log");

    std::unique_ptr<interfaces::Simulation> simulation = interfaces::build_simulation();

    PetscCall(simulation->initialize());
    PetscCall(simulation->calculate());
  }
  catch (const std::exception& e) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "what(): %s\n", e.what()));
    PetscCallMPI(MPI_Abort(PETSC_COMM_WORLD, EXIT_FAILURE));
  }
  catch (...) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Unknown exception handled!\n"));
    PetscCallMPI(MPI_Abort(PETSC_COMM_WORLD, EXIT_FAILURE));
  }

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
