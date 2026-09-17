#ifndef SRC_UTILS_PARTICLES_LOAD_H
#define SRC_UTILS_PARTICLES_LOAD_H

#include <complex>

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

/// @brief Generates paired coordinates with an exact ion-sound density profile.
struct CoordinateIonSoundPaired {
  CoordinateIonSoundPaired(BoxGeometry box,
    const Vector3R& amplitude, const Vector3R& wave_number,
    const Vector3R& phase, std::size_t active_axis_lattice_pairs = 0);

  Vector3R operator()();

  BoxGeometry box;
  Vector3R amplitude;
  Vector3R wave_number;
  Vector3R phase{0.0, 0.0, 0.0};
  Axis axis = Z;
  std::size_t active_axis_lattice_pairs = 0;
  std::size_t pair_counter = 0;
  bool return_second = false;
  Vector3R paired_coordinate;
};

/// @brief Randomly samples a cosine density profile along one box axis.
struct CoordinateInBoxCosineHump {
  Vector3R operator()();
  BoxGeometry box;
  Axis axis = Z;
};

struct CoordinateOnAnnulus {
  Vector3R operator()();
  AnnulusGeometry ann;
};

struct PreciseMomentum {
  Vector3R operator()(const Vector3R& coordinate);
  Vector3R value;
};

PetscReal temperature_momentum(PetscReal temperature, PetscReal mass);

struct MaxwellianMomentum {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  bool tov = false;
};

struct MaxwellCosinePerturbation {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  BoxGeometry box;
  Vector3R a;
  Vector3R m;
};

/// @brief Generates a Maxwellian velocity distribution with a sinusoidal shift.
struct MaxwellShiftedSine {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  BoxGeometry box;
  Vector3R velocity;
  Vector3R wave_number;
  Vector3R phase{0.0, 0.0, 0.0};
};

/// @brief Generates paired velocities matching ion-sound density moments.
struct KineticIonSoundMoments {
  KineticIonSoundMoments(const SortParameters& params, BoxGeometry box,
    PetscReal force_electric_amplitude, PetscReal omega_real, PetscReal gamma,
    const Vector3R& wave_number, const Vector3R& field_phase,
    const Vector3R& density_amplitude, const Vector3R& density_phase);

  Vector3R operator()(const Vector3R& coordinate);

  SortParameters params;
  BoxGeometry box;
  PetscReal force_electric_amplitude;
  PetscReal omega_real;
  PetscReal gamma;
  Vector3R wave_number;
  Vector3R field_phase;
  Vector3R density_amplitude;
  Vector3R density_phase;
  Axis axis = Z;
  PetscReal k_parallel = 0.0;
  PetscReal equilibrium_variance = 0.0;
  std::complex<PetscReal> density_harmonic;
  std::complex<PetscReal> flux_harmonic;
  std::complex<PetscReal> second_moment_harmonic;
  bool return_antithetic = false;
  Vector3R second_velocity;
  Vector3R first_coordinate;
  std::size_t candidate_counter = 0;

private:
  void local_parallel_parameters(const Vector3R& coordinate,
    PetscReal& bulk_velocity, PetscReal& variance) const;
};

struct AngularMomentum {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  Vector3R center;
};

#endif  // SRC_UTILS_PARTICLES_LOAD_H
