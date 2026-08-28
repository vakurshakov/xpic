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

// Fully deterministic sine-density loader. The undisplaced base coordinate
// uses the 3-D Halton sequence (bases 2, 3, 5) on every axis. Separate
// generator instances with the same box and call count therefore produce the
// same electron/ion base coordinates while still applying their
// species-specific sine displacement afterwards. The displacement reproduces
// the requested density perturbation with an O(amplitude^2) systematic error.
//
// Every displaced coordinate is returned twice. Use together with
// `MaxwellShiftedSineQuiet`, which assigns opposite thermal velocities to the
// two collocated particles while preserving the same bulk sine shift.
struct CoordinateInBoxQuietSinePaired {
  Vector3R operator()();
  BoxGeometry box;
  Vector3R amplitude;
  Vector3R wave_number;
  Vector3R phase{0.0, 0.0, 0.0};
  mutable std::size_t pair_counter = 0;
  mutable bool return_second = false;
  mutable Vector3R paired_coordinate;
};

// Exact-CDF quiet loader for a single sinusoidally perturbed axis. Unlike the
// approximate displacement loader above, this samples
//   p(s) = 1 + amplitude * sin(2*pi*mode*s/L + phase)
// without an O(amplitude^2) coordinate-map error. Every coordinate is returned
// twice so a paired momentum loader can cancel perpendicular thermal current.
// With `active_axis_lattice_pairs > 0`, the perturbed axis uses all midpoint
// quantiles (j+1/2)/N instead of a van-der-Corput sequence. This produces an
// especially accurate low-mode charge harmonic; the other axes remain Halton.
struct CoordinateInBoxQuietSineExactPaired {
  CoordinateInBoxQuietSineExactPaired(BoxGeometry box,
    const Vector3R& amplitude, const Vector3R& wave_number,
    const Vector3R& phase, std::size_t active_axis_lattice_pairs = 0);

  Vector3R operator()();

  BoxGeometry box;
  Vector3R amplitude;
  Vector3R wave_number;
  Vector3R phase{0.0, 0.0, 0.0};
  Axis axis = Z;
  std::size_t active_axis_lattice_pairs = 0;
  mutable std::size_t pair_counter = 0;
  mutable bool return_second = false;
  mutable Vector3R paired_coordinate;
};

// Regular (fully deterministic) uniform-box loader, taking the quiet-start
// idea to the limit: instead of scattering particles at random
// (`CoordinateInBox`) or along a
// low-discrepancy sequence, the particles are placed on a regular lattice so
// that EVERY grid cell contains exactly `Np` particles at exactly the target
// uniform density — zero loading noise.
//
// Per cell the `Np` particles form a sub-lattice np[X] * np[Y] * np[Z] = Np,
// factored as balanced as possible (near the cube root). Globally this is a
// single regular grid with G[a] = Ncell[a] * np[a] nodes along each axis,
// node `g` sitting at the cell-centred position (g + 0.5) / G[a] of the box,
// so no particle lands on a box face and each cell owns exactly np[a] nodes
// per axis. Cell sizes are read from the global `Dx`.
//
// Stateful via `counter`; MPI-safe because the node at a given global index
// is a pure function of the index — every rank walks the same lattice and
// keeps the nodes that fall inside its sub-domain.
struct CoordinateInBoxQuiet {
  Vector3R operator()();
  BoxGeometry box;
  PetscInt Np = 1;  // particles per grid cell
  mutable std::size_t counter = 0;
};

struct CoordinateInCylinderCosineHump {
  Vector3R operator()();
  CylinderGeometry cyl;
};

// Quiet cosine-hump loader inside a box: along the chosen Cartesian `axis`
// the density is the compact cosine hump
//     n(s) = (1/L) * (1 + cos(pi s / L)),    |s| <= L,
// where s = r[axis] - center[axis] and L = (box.max[axis] - box.min[axis]) / 2
// is the half-extent on that axis. The density vanishes (with zero slope) at
// the box faces s = +-L, so there are no edge discontinuities. The other two
// axes are sampled uniformly across the full box extent — no geometric
// reshaping, just plain rectangular fill.
//
// Quiet start: all three Halton draws use Van der Corput on prime bases
// (2, 3, 5) instead of `random_01`, matching the paired sine loader.
// Base 2 drives the hump axis (inverse CDF), bases 3 and 5 drive the two
// uniform axes. Stateful via `counter`; MPI-safe because the sequence is
// deterministic per index.
struct CoordinateInBoxCosineHump {
  Vector3R operator()();
  BoxGeometry box;
  Axis axis = Z;
  mutable std::size_t counter = 0;
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

// Maxwell-Juttner sampling + bulk velocity shifted by a sine profile,
// V(r) = velocity * sin(2*pi*m*r/L + phi), where `velocity` is given directly
// in units of c (no implicit sqrt(T/m) scaling, unlike CosinePerturbation).
struct MaxwellShiftedSine {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  BoxGeometry box;
  Vector3R velocity;
  Vector3R wave_number;
  Vector3R phase{0.0, 0.0, 0.0};
};

// Antithetic quiet-start form of `MaxwellShiftedSine`. Thermal momentum is
// sampled with a deterministic 6-D Halton Box-Muller sequence instead of the
// global RNG, stratifying both Gaussian radii and phases and reducing noise in
// second moments (in particular v_perp^2 and the drift-kinetic magnetic
// moment). Every sample is followed by its exact opposite; the prescribed
// bulk sine shift is added to both. Together with
// `CoordinateInBoxQuietSinePaired`, each collocated pair therefore has the
// requested mean velocity and exactly zero thermal current.
struct MaxwellShiftedSineQuiet {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  BoxGeometry box;
  Vector3R velocity;
  Vector3R wave_number;
  Vector3R phase{0.0, 0.0, 0.0};
  bool return_antithetic = false;
  Vector3R thermal_velocity;
  std::size_t pair_counter = 0;
};

// Non-relativistic Maxwellian velocity loader for drift-kinetic quiet starts.
// Each accepted sample is followed by its exact opposite. Components are
// sampled directly with variance T/(m*mec2); unlike the momentum loaders, no
// relativistic p -> v conversion is applied. Samples with |v| >= c are
// rejected because a Gaussian has unbounded support while particle velocities
// must remain subluminal.
struct MaxwellianVelocityQuiet {
  explicit MaxwellianVelocityQuiet(const SortParameters& params);

  Vector3R operator()(const Vector3R& coordinate);

  SortParameters params;
  bool return_antithetic = false;
  Vector3R thermal_velocity;
  std::size_t candidate_counter = 0;
};

// Positive ion-sound quasimode for a regular real-velocity particle PDF.
// Given the complex Landau frequency omega = omega_real - i*gamma, the
// configured density harmonic and particle-interpolated electric-force
// harmonic determine the first two parallel velocity moments through the
// linear Vlasov hierarchy,
//   M1 = omega*M0/k,
//   M2 = (omega*M1 - i*(q/m)*n0*E)/k.
// At each position this loader samples the positive local Gaussian whose
// untruncated density, particle flux, and second parallel moment are M0,M1,M2.
// Matching all three moments suppresses the early ballistic/plasma transient
// and leaves the forward Landau pole dominant.  This is a finite-time
// quasimode: a regular positive all-time damped Vlasov eigenfunction does not
// exist on the real velocity axis.
//
// Consecutive calls must receive the same coordinate.  The parallel thermal
// velocities and both perpendicular components are antithetic; both particles
// carry the same local bulk velocity.  Subluminal rejection preserves the
// pair mean but very slightly truncates the variance; it is negligible for the
// non-relativistic temperatures for which the construction is intended.
// The active Cartesian wave axis must be parallel to the background magnetic
// field, because that component is treated as v_parallel.
struct KineticIonSoundMomentsQuiet {
  KineticIonSoundMomentsQuiet(const SortParameters& params, BoxGeometry box,
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
