#ifndef SRC_CONSTANTS_H
#define SRC_CONSTANTS_H

#define there_are_fields              true
#define fields_are_diagnosed          true

#define there_are_particles           true
#define particles_are_diagnosed       true

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

extern double dx;
extern double dy;
extern double dz;

extern int SIZE_NX;
extern int SIZE_NY;
extern int SIZE_NZ;

extern double SIZE_LX;
extern double SIZE_LY;
extern double SIZE_LZ;

extern double dt;
extern timestep_t TIME;
extern timestep_t DIAGNOSE_PERIOD;

#endif  // SRC_CONSTANTS_H
