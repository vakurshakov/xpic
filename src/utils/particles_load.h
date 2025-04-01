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
  Vector3R operator()();
  Vector3R dot;
};


struct CoordinateInBox {
  Vector3R operator()();
  BoxGeometry box;
};


struct CoordinateInCylinder {
  Vector3R operator()();
  CylinderGeometry cyl;
};


struct CoordinateOnAnnulus {
  CoordinateOnAnnulus(const Vector3R& center, //
    PetscReal inner_r, PetscReal outer_r, PetscReal height);
  Vector3R operator()();
  Vector3R center;
  PetscReal inner_r2;
  PetscReal outer_r2;
  PetscReal height;
};


struct PreciseMomentum {
  Vector3R operator()(const Vector3R& coordinate);
  Vector3R value;
};


struct MaxwellianMomentum {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  bool tov = false;
};


struct AngularMomentum {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  Vector3R center;
};

#endif  // SRC_UTILS_PARTICLES_LOAD_H
