#include "src/pch.h"
#include "src/interfaces/simulation.h"
#include "src/utils/configuration.h"
#include "src/utils/utils.h"

static constexpr char help[] =
  "Usage: [mpiexec] simulation.out <config.json>\n";

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  if (argc < 2) {
    LOG(help);
    return EXIT_FAILURE;
  }

  try {
    Configuration::init(argv[1]);
    Configuration::save(CONFIG().out_dir);

    std::unique_ptr<interfaces::Simulation> simulation = build_simulation();
    PetscCall(simulation->initialize());
    PetscCall(simulation->calculate());
    PetscCall(simulation->finalize());
  }
  catch (const std::exception& e) {
    LOG("what(): {}", e.what());
    PetscCallMPI(MPI_Abort(PETSC_COMM_WORLD, EXIT_FAILURE));
  }
  catch (...) {
    LOG("Unknown exception handled!");
    PetscCallMPI(MPI_Abort(PETSC_COMM_WORLD, EXIT_FAILURE));
  }

  LOG_FLUSH();
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
