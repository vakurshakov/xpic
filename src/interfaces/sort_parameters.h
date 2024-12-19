#ifndef SRC_INTERFACES_SORT_PARAMETERS_H
#define SRC_INTERFACES_SORT_PARAMETERS_H

#include "src/pch.h"
#include "src/utils/utils.h"

struct SortParameters {
  PetscInt Np;         // [dimensionless], Number of particles in a cell.
  PetscReal n;         // [n0], Reference density of the particles.
  PetscReal q;         // [e],  Reference charge of the particles.
  PetscReal m;         // [me], Mass of the particles in a sort.
  PetscReal px = 0.0;  // [me c], Inital impulse in x direction.
  PetscReal py = 0.0;  // [me c], Inital impulse in y direction.
  PetscReal pz = 0.0;  // [me c], Inital impulse in z direction.
  PetscReal Tx = 0.0;  // [keV], Temperature in x direction.
  PetscReal Ty = 0.0;  // [keV], Temperature in y direction.
  PetscReal Tz = 0.0;  // [keV], Temperature in z direction.
  std::string sort_name;
};


#pragma omp declare simd notinbranch
PetscReal spline_of_0th_order(PetscReal s);

#pragma omp declare simd notinbranch
PetscReal spline_of_1st_order(PetscReal s);

#pragma omp declare simd notinbranch
PetscReal spline_of_2nd_order(PetscReal s);

#pragma omp declare simd notinbranch
PetscReal spline_of_3rd_order(PetscReal s);

#pragma omp declare simd notinbranch
PetscReal spline_of_4th_order(PetscReal s);

#pragma omp declare simd notinbranch
PetscReal spline_of_5th_order(PetscReal s);

#if (PARTICLES_FORM_FACTOR == 0)
static constexpr PetscReal shape_radius = 0.5;
static constexpr const auto& shape_function = spline_of_0th_order;
#elif (PARTICLES_FORM_FACTOR == 1)
static constexpr PetscReal shape_radius = 1.0;
static constexpr const auto& shape_function = spline_of_1st_order;
#elif (PARTICLES_FORM_FACTOR == 2)
static constexpr PetscReal shape_radius = 1.5;
static constexpr const auto& shape_function = spline_of_2nd_order;
#elif (PARTICLES_FORM_FACTOR == 3)
static constexpr PetscReal shape_radius = 2.0;
static constexpr const auto& shape_function = spline_of_3rd_order;
#elif (PARTICLES_FORM_FACTOR == 4)
static constexpr PetscReal shape_radius = 2.5;
static constexpr const auto& shape_function = spline_of_4th_order;
#elif (PARTICLES_FORM_FACTOR == 5)
static constexpr PetscReal shape_radius = 3.0;
static constexpr const auto& shape_function = spline_of_5th_order;
#else
  #error "Unknown PARTICLES_FORM_FACTOR is specified!"
#endif

static constexpr PetscInt shape_width =
  static_cast<PetscInt>(2.0 * shape_radius) + 1;

#endif  // SRC_INTERFACES_SORT_PARAMETERS_H
