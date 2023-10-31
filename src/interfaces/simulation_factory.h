#ifndef SRC_INTERFACES_SIMULATION_FACTORY_H
#define SRC_INTERFACES_SIMULATION_FACTORY_H

#include "src/pch.h"
#include "src/interfaces/simulation.h"
#include "src/utils/configuration.h"

class Simulation_factory {
 public:
  Simulation_factory() = default;

  std::unique_ptr<Simulation> build();
};

#endif  // SRC_INTERFACES_SIMULATION_FACTORY_H
