#include "simulation_factory.h"

using Simulation_up = std::unique_ptr<Simulation>;

Simulation_up Simulation_factory::build() {
  Simulation_up simulation = nullptr;

  std::string simulation_str = CONFIG().get("Simulation");
  throw std::runtime_error("Unkown simulation is used: " + simulation_str);
}
