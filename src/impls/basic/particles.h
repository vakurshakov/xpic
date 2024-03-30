#ifndef SRC_BASIC_PARTICLES_PARTICLES_H
#define SRC_BASIC_PARTICLES_PARTICLES_H

#include "src/pch.h"

#include <petscdm.h>

#include "src/vectors/vector_classes.h"
#include "src/interfaces/particles/particles.h"

namespace basic {

class Simulation;

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

  struct Shape;
  void fill_shape(const Vector3<PetscInt>& p_g, const Vector3<PetscReal>& p_r, Shape& shape, bool shift);
  void interpolate(const Vector3<PetscInt>& p_g, Shape& no, Shape& sh, Vector3<PetscReal>& point_E, Vector3<PetscReal>& point_B) const;
  void push(const Vector3<PetscReal>& point_E, const Vector3<PetscReal>& point_B, Point& point) const;

  void decompose(const Vector3<PetscInt>& p_g, Shape& old_shape, Shape& new_shape, const Point& point);

  using Compute_j = std::function<PetscReal(PetscInt, PetscInt, PetscInt, PetscReal*)>;
  void decompose_dir(const Vector3<PetscInt>& p_g, const Compute_j& compute_j, Axis dir);

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
  Vector3<PetscReal> ***E, ***B, ***J;

  const PetscMPIInt* neighbours;
  Vector3<PetscReal> l_start;
  Vector3<PetscReal> l_end;
  Vector3<PetscInt>  l_width;
};


/**
 * @brief Storage for particle's coordinate - `r` (global, in PetscReal units of dx, dy, dz),
 * and a nearest grid point to particle - `g` (rounded, shifted by `shape_radius`).
 */
struct Node {
  Vector3<PetscReal> r;
  Vector3<PetscInt> g;

  Node(const Vector3<PetscReal>& __r) {
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

}

#endif  // SRC_BASIC_PARTICLES_PARTICLES_H
