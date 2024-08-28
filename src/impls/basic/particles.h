#ifndef SRC_BASIC_PARTICLES_PARTICLES_H
#define SRC_BASIC_PARTICLES_PARTICLES_H

#include "src/pch.h"

#include <petscdm.h>

#include "src/utils/vector3.h"
#include "src/interfaces/particles/particles.h"

#include "src/impls/particle_shape.h"

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
  Vector3I l_width;
};

}

#endif  // SRC_BASIC_PARTICLES_PARTICLES_H
