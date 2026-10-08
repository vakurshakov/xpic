#ifndef SRC_UTILS_PARTICLES_LOAD_H
#define SRC_UTILS_PARTICLES_LOAD_H

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
struct KineticIonSoundCoordinates {
  KineticIonSoundCoordinates(BoxGeometry box, const Vector3R& amplitude,
    const Vector3R& wave_number, const Vector3R& phase,
    std::size_t active_axis_lattice_pairs);

  Vector3R operator()();

  BoxGeometry box;
  Vector3R n_pa;  // density perturbation amplitude
  Vector3R n_pm;  // density perturbation wave number
  Vector3R n_pp;  // density perturbation phase
  std::size_t pairs_total_number;

private:
  Axis axis = Z;
  Vector3R paired_coordinate;
  std::size_t pair_number = 0;
  bool return_second_coordinate = false;
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

/// @brief Generates paired velocities matching ion-sound density moments.
struct KineticIonSoundMoments {
  KineticIonSoundMoments(const SortParameters& params, BoxGeometry box,
    PetscReal force_electric_amplitude, PetscReal omega_real, PetscReal gamma,
    const Vector3R& wave_number, const Vector3R& field_phase,
    const Vector3R& density_amplitude, const Vector3R& density_phase);

  Vector3R operator()(const Vector3R& coordinate);

  SortParameters params;
  BoxGeometry box;
  PetscReal E_pa;        // electric field perturbation amplitude
  Vector3R E_pp;         // electric field perturbation phase
  Vector3R n_pa;         // density perturbation amplitude
  Vector3R n_pp;         // density perturbation phase
  PetscReal omega_real;  // real part of the dispersion relation frequency
  PetscReal gamma;       // decrement (negative imag. part of the frequency)

private:
  Axis axis = Z;
  PetscReal k_par;     // wave number (parallel to `axis`)
  PetscReal v2_0;      // initial value of second moment (v^2)
  PetscComplex n_p;    // initial density perturbation
  PetscComplex v_p;    // initial velocity perturbation
  PetscComplex nv2_p;  // initial temperature perturbation

  Vector3R first_coordinate;
  Vector3R second_velocity;
  std::size_t pair_number = 1;
  bool return_second_velocity = false;

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
