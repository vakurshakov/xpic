#ifndef SRC_IMPLS_PARTICLE_SHAPE_H
#define SRC_IMPLS_PARTICLE_SHAPE_H

#include "src/interfaces/sort_parameters.h"
#include "src/utils/vector3.h"

/**
 * @brief Storage for particle's coordinate - `r` (global, in PetscReal units of
 * dx, dy, dz), and a nearest grid point to particle - `g` (rounded, shifted by
 * `shape_radius`).
 */
struct Node {
  Node() = default;
  Node(const Vector3R& pr);

  static Vector3R make_r(const Vector3R& pr);
  static Vector3I make_g(const Vector3R& nr);

  Vector3R r;
  Vector3I g;
};

class Shape {
public:
  Shape() = default;

  /// @note If `shift` is false, fills `shape[x - i]`, otherwise fills `shape[x - (i + 0.5)]`
  inline void fill(const Vector3I& p_g, const Vector3R& p_r, bool shift)
  {
    fill(p_g, p_r, shift, shape_function, shape_width);
  }

  void fill(const Vector3I& p_g, const Vector3R& p_r, bool shift,
    PetscReal (&sfunc)(PetscReal), PetscInt width);

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

private:
  /// @note `Vector3I::dim` is used as a coordinate space dimensionality
  PetscReal shape[POW3(shape_width) * Vector3I::dim];
};


#endif  // SRC_IMPLS_PARTICLE_SHAPE_H
