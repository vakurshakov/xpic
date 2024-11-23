#ifndef SRC_INTERFACES_SORT_PARAMETERS_H
#define SRC_INTERFACES_SORT_PARAMETERS_H

#include "src/pch.h"
#include "src/utils/utils.h"

struct Sort_parameters {
  PetscInt Np;         // Number of particles in a cell.
  PetscReal n;         // Reference density of the particles.
  PetscReal q;         // Reference charge of the particles.
  PetscReal m;         // Mass of the particles in a sort.
  PetscReal px = 0.0;  // Inital impulse in x direction.
  PetscReal py = 0.0;  // Inital impulse in y direction.
  PetscReal pz = 0.0;  // Inital impulse in z direction.
  PetscReal Tx = 0.0;  // Temperature in x direction.
  PetscReal Ty = 0.0;  // Temperature in y direction.
  PetscReal Tz = 0.0;  // Temperature in z direction.
  std::string sort_name = {};
};


#pragma omp declare simd notinbranch
PetscReal __0th_order_spline(PetscReal s);

#pragma omp declare simd notinbranch
PetscReal __1st_order_spline(PetscReal s);

#pragma omp declare simd notinbranch
PetscReal __2nd_order_spline(PetscReal s);

#pragma omp declare simd notinbranch
PetscReal __3rd_order_spline(PetscReal s);

#pragma omp declare simd notinbranch
PetscReal __4th_order_spline(PetscReal s);

#pragma omp declare simd notinbranch
PetscReal __5th_order_spline(PetscReal s);

#if (PARTICLES_FORM_FACTOR == 0)
static constexpr PetscReal shape_radius = 0.5;
static constexpr const auto& shape_function = __0th_order_spline;
#elif (PARTICLES_FORM_FACTOR == 1)
static constexpr PetscReal shape_radius = 1.0;
static constexpr const auto& shape_function = __1st_order_spline;
#elif (PARTICLES_FORM_FACTOR == 2)
static constexpr PetscReal shape_radius = 1.5;
static constexpr const auto& shape_function = __2nd_order_spline;
#elif (PARTICLES_FORM_FACTOR == 3)
static constexpr PetscReal shape_radius = 2.0;
static constexpr const auto& shape_function = __3rd_order_spline;
#elif (PARTICLES_FORM_FACTOR == 4)
static constexpr PetscReal shape_radius = 2.5;
static constexpr const auto& shape_function = __4th_order_spline;
#elif (PARTICLES_FORM_FACTOR == 5)
static constexpr PetscReal shape_radius = 3.0;
static constexpr const auto& shape_function = __5th_order_spline;
#else
  #error "Unknown PARTICLES_FORM_FACTOR is specified!"
#endif

static constexpr PetscInt shape_width =
  static_cast<PetscInt>(2.0 * shape_radius) + 1;

#endif  // SRC_INTERFACES_SORT_PARAMETERS_H
