#ifndef SRC_IMPLS_PARTICLE_SHAPE_H
#define SRC_IMPLS_PARTICLE_SHAPE_H

#include "src/interfaces/sort_parameters.h"

/**
 * @brief Storage for particle's coordinate - `r` (global, in PetscReal units of
 * dx, dy, dz), and a nearest grid point to particle - `g` (rounded, shifted by
 * `shape_radius`).
 */
struct Node {
  Node() = default;
  Node(const Vector3R& __r);

  Vector3R r;
  Vector3I g;
};

struct Shape {
  /// @note `Vector3I::dim` is used as a coordinate space dimensionality
  PetscReal shape[shape_width * shape_width * shape_width * Vector3I::dim];

#pragma omp declare simd linear(x, y, z : 1), notinbranch
  static constexpr PetscInt index(PetscInt x, PetscInt y, PetscInt z)
  {
    return ((z * shape_width + y) * shape_width + x);
  }

#pragma omp declare simd linear(index : 1), notinbranch
  constexpr PetscReal& operator()(PetscInt index, PetscInt comp)
  {
    return shape[index * Vector3I::dim + comp];
  }

#pragma omp declare simd linear(index : 1), notinbranch
  constexpr const PetscReal& operator()(PetscInt index, PetscInt comp) const
  {
    return shape[index * Vector3I::dim + comp];
  }
};

/// @note If shift is false, fills shape[x - i], otherwise fills shape[x - (i + 0.5)]
PetscErrorCode fill_shape(const Vector3I& p_g, const Vector3R& p_r,
  const Vector3I& l_width, bool shift, Shape& shape);

#endif  // SRC_IMPLS_PARTICLE_SHAPE_H
