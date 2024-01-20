#include "simulation.h"

#include "src/implementations/basic/simulation.h"

namespace interfaces {

using Simulation_up = std::unique_ptr<Simulation>;

Simulation_up build_simulation() {
  Simulation_up simulation = nullptr;

  std::string simulation_str = CONFIG().get("Simulation");
  if (simulation_str == "basic") {
    return std::make_unique<basic::Simulation>();
  }

  throw std::runtime_error("Unkown simulation is used: " + simulation_str);
}

}
