#include "particles.h"

namespace hybrid_drift_kinetic {

Particles::Particles(
  interfaces::Simulation& simulation, const SortParameters& parameters)
  : eccapfim::Particles(simulation, parameters)
{
}

}  // namespace hybrid_drift_kinetic
