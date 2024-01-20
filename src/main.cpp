#include <iostream>

#include "src/pch.h"
#include "src/interfaces/simulation.h"
#include "src/utils/configuration.h"

static char help[] = "Usage: simulation.out <config.json>\n";

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << help << std::endl;
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
    std::cerr << "what(): " << e.what() << "\n" << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown exception handled!\n" << std::endl;
  }

  return EXIT_SUCCESS;
}
