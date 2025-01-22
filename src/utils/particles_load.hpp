#ifndef SRC_UTILS_PARTICLES_LOAD_H
#define SRC_UTILS_PARTICLES_LOAD_H

#include "src/pch.h"
#include "src/interfaces/sort_parameters.h"
#include "src/utils/geometries.h"
#include "src/utils/vector3.h"

using CoordinateGenerator = std::function<Vector3R()>;

using MomentumGenerator =
  std::function<Vector3R(const Vector3R& /* reference */)>;

struct PreciseCoordinate {
  PreciseCoordinate(const DotGeometry& dot);
  Vector3R operator()();
  DotGeometry dot;
};


struct CoordinateInBox {
  CoordinateInBox(const BoxGeometry& box);
  Vector3R operator()();
  BoxGeometry box;
};


struct CoordinateInCircle {
  CoordinateInCircle(PetscReal radius, const Vector3R& center);
  Vector3R operator()();
  PetscReal radius_;
  Vector3R center_;
};

struct CoordinateInCylinder {
  CoordinateInCylinder(
    PetscReal radius, PetscReal height, const Vector3R& center);
  Vector3R operator()();
  CoordinateInCircle gen_;
  PetscReal height_;
};


struct CoordinateOnAnnulus {
  CoordinateOnAnnulus(
    PetscReal inner_r, PetscReal outer_r, const Vector3R& center);
  Vector3R operator()();
  PetscReal inner_r2_;
  PetscReal outer_r2_;
  Vector3R center_;
};


struct PreciseMomentum {
  PreciseMomentum(const Vector3R& value);
  Vector3R operator()(const Vector3R& coordinate);
  Vector3R value_;
};


struct MaxwellianMomentum {
  MaxwellianMomentum(const SortParameters& params, bool tov = false);
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params_;
  bool tov_;
};


struct AngularMomentum {
  AngularMomentum(const SortParameters& params, const Vector3R& center);
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params_;
  Vector3R center_;
};

#endif  // SRC_UTILS_PARTICLES_LOAD_H
