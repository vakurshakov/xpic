#include <iostream>

#include "src/pch.h"
#include "src/interfaces/simulation.h"
#include "src/interfaces/simulation_factory.h"
#include "src/utils/configuration.h"

int main(int argc, const char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: simulation.out <config.json>\n" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    const Configuration& config = CONFIG();
    config.init(argv[1]);
    config.save();

    LOG_INIT(config.out_dir + "/simulation.log");

    Simulation_factory factory;
    std::unique_ptr<Simulation> simulation = factory.build();

    simulation->initialize();
    simulation->calculate();
  }
  catch (const std::exception& e) {
    std::cerr << "what(): " << e.what() << "\n" << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown exception handled!\n" << std::endl;
  }

  return EXIT_SUCCESS;
}
