#ifndef SRC_UTILS_PARTICLES_LOAD_H
#define SRC_UTILS_PARTICLES_LOAD_H

#include "src/interfaces/sort_parameters.h"
#include "src/utils/geometries.h"
#include "src/utils/halton_generator.h"
#include "src/utils/vector3.h"

using CoordinateGenerator = std::function<Vector3R()>;

using MomentumGenerator =
  std::function<Vector3R(const Vector3R& /* reference */)>;

/// @brief Returns the dots one by one, cycling after the last one.
struct PreciseCoordinate {
  PreciseCoordinate(std::vector<Vector3R> dots);
  PreciseCoordinate(const Vector3R& dot);
  Vector3R operator()();
  std::vector<Vector3R> dots;
  std::size_t next = 0;
};

struct CoordinateInBox {
  Vector3R operator()();
  BoxGeometry box;
};

struct CoordinateInCylinder {
  Vector3R operator()();
  CylinderGeometry cyl;
};

/// @brief Randomly samples a cosine density profile along one box axis.
struct CoordinateInBoxCosineHump {
  Vector3R operator()();
  BoxGeometry box;
  Axis axis = Z;
};

struct CoordinateInBoxQS {
  Vector3R operator()();
  BoxGeometry box;
  HaltonGenerator sequence;
};

struct CoordinateInCylinderQS {
  Vector3R operator()();
  CylinderGeometry cyl;
  HaltonGenerator sequence;
};

struct CoordinateInBoxCosineHumpQS {
  Vector3R operator()();
  BoxGeometry box;
  Axis axis = Z;
  HaltonGenerator sequence;
};

struct CoordinateOnAnnulus {
  Vector3R operator()();
  AnnulusGeometry ann;
};

/// @brief Returns the values one by one, cycling after the last one.
struct PreciseMomentum {
  PreciseMomentum(std::vector<Vector3R> values);
  PreciseMomentum(const Vector3R& value);
  Vector3R operator()(const Vector3R& coordinate);
  std::vector<Vector3R> values;
  std::size_t next = 0;
};

struct MaxwellianMomentum {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  bool tov = false;
};

struct MaxwellShiftedCosine {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  BoxGeometry box;
  Vector3R velocity;
  Vector3R wave_number;
  Vector3R phase;
};

struct MaxwellShiftedSine {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  BoxGeometry box;
  Vector3R velocity;
  Vector3R wave_number;
  Vector3R phase;
};

struct AngularMomentum {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  Vector3R center;
};

#endif  // SRC_UTILS_PARTICLES_LOAD_H
