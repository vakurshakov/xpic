#ifndef SRC_CONSTANTS_H
#define SRC_CONSTANTS_H

#define THERE_ARE_FIELDS              true
#define FIELDS_ARE_DIAGNOSED          true

#define THERE_ARE_PARTICLES           true
#define PARTICLES_ARE_DIAGNOSED       true

#define LOGGING                       true
#define TIME_PROFILING                true

// BOUNDARY_CONDITIONS:
#define NONE        -1
#define PEC         +0
#define PMC         +1
#define PERIODIC    +2
#define CONTINUOUS  +3

#define X_BOUNDARY_CONDITION          CONTINUOUS
#define Y_BOUNDARY_CONDITION          CONTINUOUS

namespace physical_constants {
  constexpr double e   = 1.0;
  constexpr double me  = 1.0;
  constexpr double Mp  = 1836.0;
  constexpr double mec2 = 511.0;
}

#include <cstdint>
using timestep_t = std::size_t;

#endif  // SRC_CONSTANTS_H
