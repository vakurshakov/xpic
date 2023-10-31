#include "simulation_factory.h"

using Simulation_up = std::unique_ptr<Simulation>;

Simulation_up Simulation_factory::build() {
  Simulation_up simulation = std::make_unique<Simulation>();

  return simulation;
}
