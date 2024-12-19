#ifndef SRC_UTILS_PARTICLES_LOAD_H
#define SRC_UTILS_PARTICLES_LOAD_H

#include "src/pch.h"
#include "src/interfaces/sort_parameters.h"
#include "src/utils/vector3.h"


class CoordinateInBox {
public:
  CoordinateInBox(const Vector3R& min, const Vector3R& max);
  Vector3R operator()();

private:
  Vector3R min_;
  Vector3R max_;
};


class CoordinateInCircle {
public:
  CoordinateInCircle(PetscReal radius, const Vector3R& center);
  Vector3R operator()();

private:
  friend class CoordinateInCylinder;

  PetscReal radius_;
  Vector3R center_;
};

class CoordinateInCylinder {
public:
  CoordinateInCylinder(PetscReal radius, PetscReal height, const Vector3R& center);
  Vector3R operator()();

private:
  CoordinateInCircle gen_;
  PetscReal height_;
};


class CoordinateOnAnnulus {
public:
  CoordinateOnAnnulus(
    PetscReal inner_r, PetscReal outer_r, const Vector3R& center);

  Vector3R operator()();

private:
  PetscReal inner_r2_;
  PetscReal outer_r2_;
  Vector3R center_;
};


PetscReal temperature_momentum(PetscReal temperature, PetscReal mass);


class MaxwellianMomentum {
public:
  MaxwellianMomentum(const SortParameters& params);
  Vector3R operator()(const Vector3R& coordinate);

private:
  SortParameters params_;
};


class AngularMomentum {
public:
  AngularMomentum(const SortParameters& params, const Vector3R& center);
  Vector3R operator()(const Vector3R& coordinate);

private:
  SortParameters params_;
  Vector3R center_;
};

#endif  // SRC_UTILS_PARTICLES_LOAD_H
