#ifndef SRC_IMPLS_HYBRID_DRIFT_KINETIC_PARTICLES_H
#define SRC_IMPLS_HYBRID_DRIFT_KINETIC_PARTICLES_H

#include "src/impls/eccapfim/particles.h"

namespace hybrid_drift_kinetic {

class Particles : public eccapfim::Particles {
public:
  Particles(interfaces::Simulation& simulation, const SortParameters& parameters);
};

}  // namespace hybrid_drift_kinetic

#endif  // SRC_IMPLS_HYBRID_DRIFT_KINETIC_PARTICLES_H
