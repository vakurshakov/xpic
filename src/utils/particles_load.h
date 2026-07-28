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

struct CoordinateInBoxSineDensity {
  Vector3R operator()();
  BoxGeometry box;
  Vector3R amplitude;
  Vector3R wave_number;
};

// Linear-perturbation loader: places particles by uniform sampling and then
// applies the small-amplitude displacement that reproduces the target density
//   n(r) = 1 + sum_alpha amplitude[alpha] * sin(2 pi k[alpha] (r[alpha] - r0) / L[alpha])
// to leading order in `amplitude`. Beats rejection sampling because the
// perturbation is encoded by deterministic displacement, not by reweighting
// the random stream — shot noise stays at the uniform-loading level
// (~1/sqrt(N_per_cell)), and the perturbation itself is noise-free up to
// O(amplitude^2). Suitable for linear-mode tests with |amplitude| << 1.
struct CoordinateInBoxDisplacedSine {
  Vector3R operator()();
  BoxGeometry box;
  Vector3R amplitude;
  Vector3R wave_number;
};

// Quiet-start variant of `CoordinateInBoxDisplacedSine`: along every
// perturbed axis (non-zero amplitude AND non-zero wave_number) the uniform
// sample is replaced by the Halton low-discrepancy sequence (bases 2, 3, 5
// in order of perturbed axes). Non-perturbed axes still use the global RNG.
//
// Halton sequence has discrepancy O(log N / N), so the per-cell loading
// noise scales as ~1/N instead of the Monte-Carlo ~1/sqrt(N). For typical
// PIC loads this drops residual density wobble from O(0.1–1%) into the
// numerical-zero range, leaving only the O(amplitude^2) systematic error
// of the displacement formula itself.
//
// Stateful: the call counter is shared between species only if the same
// generator instance is reused (e.g. by `SetPairedParticles`). MPI-safe
// because the sequence is deterministic — every rank advances through the
// same indices and keeps the particles that land inside its sub-domain.
struct CoordinateInBoxQuietSine {
  Vector3R operator()();
  BoxGeometry box;
  Vector3R amplitude;
  Vector3R wave_number;
  Vector3R phase{0.0, 0.0, 0.0};
  mutable std::size_t counter = 0;
};

// Antithetic-pair variant of `CoordinateInBoxQuietSine`: every generated
// coordinate is returned twice. Use together with `MaxwellShiftedSineQuiet`,
// which assigns opposite thermal velocities to the two particles while
// preserving the same bulk sine shift. The pair then has exactly zero thermal
// current at one position.
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

// Regular (fully deterministic) uniform-box loader, following the "quiet
// start" idea of `CoordinateInBoxQuietSine` but taken to the limit: instead
// of scattering particles at random (`CoordinateInBox`) or along a
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
// (2, 3, 5) instead of `random_01`, matching `CoordinateInBoxQuietSine`.
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

// Antithetic quiet-start form of `MaxwellShiftedSine`. A sampled thermal
// velocity is followed by its exact opposite; the same prescribed bulk sine
// shift is added to both. Together with `CoordinateInBoxQuietSinePaired`, each
// collocated pair therefore has mean velocity equal to the requested shift
// and zero thermal current.
struct MaxwellShiftedSineQuiet {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  BoxGeometry box;
  Vector3R velocity;
  Vector3R wave_number;
  Vector3R phase{0.0, 0.0, 0.0};
  bool return_antithetic = false;
  Vector3R thermal_velocity;
};

struct AngularMomentum {
  Vector3R operator()(const Vector3R& coordinate);
  SortParameters params;
  Vector3R center;
};

#endif  // SRC_UTILS_PARTICLES_LOAD_H
