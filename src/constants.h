#ifndef SRC_CONSTANTS_H
#define SRC_CONSTANTS_H

#define THERE_ARE_FIELDS              true
#define FIELDS_ARE_DIAGNOSED          true

#define THERE_ARE_PARTICLES           true
#define PARTICLES_ARE_DIAGNOSED       true

#define LOGGING                       true
#define TIME_PROFILING                true

#include <petscsystypes.h>
using timestep_t = PetscInt;

namespace physical_constants {

constexpr PetscReal e = 1.0;       // units of e
constexpr PetscReal me = 1.0;      // units of me
constexpr PetscReal Mp = 1836.0;   // units of me
constexpr PetscReal mec2 = 511.0;  // KeV

}

// To avoid multiple definitions, we put useful global variables into unnamed namespace
namespace {

// Geometry constants, to be set in `Configuration::init()`
PetscReal dx = 0.0;  // c / w_pe
PetscReal dy = 0.0;  // c / w_pe
PetscReal dz = 0.0;  // c / w_pe
PetscReal dt = 0.0;  // 1 / w_pe

PetscReal geom_x = 0.0;  // c / w_pe
PetscReal geom_y = 0.0;  // c / w_pe
PetscReal geom_z = 0.0;  // c / w_pe
PetscReal geom_t = 0.0;  // 1 / w_pe

PetscInt geom_nx = 0;  // units of dx
PetscInt geom_ny = 0;  // units of dy
PetscInt geom_nz = 0;  // units of dz
PetscInt geom_nt = 0;  // units of dt

PetscInt diagnose_period = 0;  // units of dt

}

#endif  // SRC_CONSTANTS_H
