#ifndef SRC_BASIC_PARTICLES_PARTICLES_H
#define SRC_BASIC_PARTICLES_PARTICLES_H

#include "src/pch.h"

#include <petscdm.h>

#include "src/vectors/vector3.h"
#include "src/interfaces/particles/particles.h"

namespace basic {

class Simulation;
struct Shape;

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const Particles_parameters& parameters);
  ~Particles() override;

  PetscErrorCode add_particle(const Point& point);
  const std::vector<Point>& get_points() const { return points_; }

  PetscErrorCode push();
  PetscErrorCode communicate();

private:
  static constexpr int OMP_CHUNK_SIZE  = 16;

  void fill_shape(const Vector3I& p_g, const Vector3R& p_r, Shape& shape, bool shift);
  void interpolate(const Vector3I& p_g, Shape& no, Shape& sh, Vector3R& point_E, Vector3R& point_B) const;
  void push(const Vector3R& point_E, const Vector3R& point_B, Point& point) const;

  void decompose(const Vector3I& p_g, Shape& old_shape, Shape& new_shape, const Point& point);

  using Compute_j = std::function<PetscReal(PetscInt, PetscInt, PetscInt, PetscReal*)>;
  void decompose_dir(const Vector3I& p_g, const Compute_j& compute_j, Axis dir);

  PetscInt to_contiguous_index(PetscInt x, PetscInt y, PetscInt z) {
    constexpr PetscInt dim = 3;
    return (z * dim + y) * dim + x;
  }

  void from_contiguous_index(PetscInt index, PetscInt& x, PetscInt& y, PetscInt& z) {
    constexpr PetscInt dim = 3;
    x = (index) % dim;
    y = (index / dim) % dim;
    z = (index / dim) / dim;
  }

  std::vector<Point> points_;

  Simulation& simulation_;
  Vec local_E, local_B, local_J;
  Vector3R ***E, ***B, ***J;

  const PetscMPIInt* neighbours;
  Vector3R l_start;
  Vector3R l_end;
  Vector3I  l_width;
};


/**
 * @brief Storage for particle's coordinate - `r` (global, in PetscReal units of dx, dy, dz),
 * and a nearest grid point to particle - `g` (rounded, shifted by `shape_radius`).
 */
struct Node {
  Vector3R r;
  Vector3I g;

  Node(const Vector3R& __r) {
    r = {
      __r.x() / dx,
      __r.y() / dy,
      __r.z() / dz,
    };

    g = {
      (geom_nx > 1) ? ROUND(r.x()) - shape_radius : 0,
      (geom_ny > 1) ? ROUND(r.y()) - shape_radius : 0,
      (geom_nz > 1) ? ROUND(r.z()) - shape_radius : 0,
    };
  }
};

struct Shape {
  /// @note `Vector3I::dim` is used as a coordinate space dimensionality
  PetscReal shape[shape_width * shape_width * shape_width * Vector3I::dim];

  #pragma omp declare simd linear(x, y, z: 1), notinbranch
  static constexpr PetscInt index(PetscInt x, PetscInt y, PetscInt z) {
    return ((z * shape_width + y) * shape_width + x);
  }

  #pragma omp declare simd linear(i: 1), notinbranch
  constexpr PetscReal& operator()(PetscInt index, PetscInt comp) {
    return shape[index * Vector3I::dim + comp];
  }
};

}

#endif  // SRC_BASIC_PARTICLES_PARTICLES_H
