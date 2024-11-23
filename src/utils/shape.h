#ifndef SRC_UTILS_SHAPE_H
#define SRC_UTILS_SHAPE_H

#include "src/interfaces/sort_parameters.h"
#include "src/utils/vector3.h"

enum ShapeType : PetscInt {
  No = 0,
  Sh = 1,
  Old = 2,
  New = 3,
};

/**
 * @brief Structure containing pre-computed values ​​of shape function
 * calls. Demo desmos program: https://www.desmos.com/calculator/vo3qf9q0h8.
 *
 * @note This class will work _only_ with `size`(s) less than `shape_width`.
 * @note Any version of `setup()` must be called before accessing shape by indices.
 */
class Shape {
public:
  Shape() = default;

  Vector3I start, size;

  static Vector3R make_r(const Vector3R& pr);
  static Vector3I make_g(const Vector3R& nr);

  static Vector3I make_start(const Vector3R& p_r, PetscReal radius);
  static Vector3I make_end(const Vector3R& p_r, PetscReal radius);

  inline PetscInt s_p(PetscInt z, PetscInt y, PetscInt x) const
  {
    return indexing::petsc_index(z, y, x, 0, size[Z], size[Y], size[X], 1);
  }

  constexpr PetscInt i_p(PetscInt i, ShapeType t, PetscInt c) const
  {
    return i * shape_geom + ((t % 2) * Vector3I::dim + c);
  }

  inline void setup(const Vector3R& old_r, const Vector3R& new_r)
  {
    setup(old_r, new_r, shape_width, shape_function);
  }

  inline void setup(const Vector3R& p_r)
  {
    setup(p_r, shape_width, shape_function);
  }

  void setup(const Vector3R& old_r, const Vector3R& new_r, PetscReal radius,
    PetscReal (&sfunc)(PetscReal));

  void setup(const Vector3R& r, PetscReal radius, PetscReal (&sfunc)(PetscReal));

  /// @returns Vector of shape products corresponding to electric fields.
  /// @note No check is performed to see if the `ShapeType::No/Sh` type pair is set.
  constexpr Vector3R electric(PetscInt i) const
  {
    return {
      shape[i_p(i, No, Z)] * shape[i_p(i, No, Y)] * shape[i_p(i, Sh, X)],
      shape[i_p(i, No, Z)] * shape[i_p(i, Sh, Y)] * shape[i_p(i, No, X)],
      shape[i_p(i, Sh, Z)] * shape[i_p(i, No, Y)] * shape[i_p(i, No, X)],
    };
  }

  /// @returns Vector of shape products corresponding to magnetic fields.
  /// @note No check is performed to see if the `ShapeType::No/Sh` type pair is set.
  constexpr Vector3R magnetic(PetscInt i) const
  {
    return {
      shape[i_p(i, Sh, Z)] * shape[i_p(i, Sh, Y)] * shape[i_p(i, No, X)],
      shape[i_p(i, Sh, Z)] * shape[i_p(i, No, Y)] * shape[i_p(i, Sh, X)],
      shape[i_p(i, No, Z)] * shape[i_p(i, Sh, Y)] * shape[i_p(i, Sh, X)],
    };
  }

#pragma omp declare simd linear(i : 1), notinbranch
  /// @note Only the constant access is provided, to prepare the shape use `setup()` methods.
  constexpr PetscReal operator()(PetscInt i, ShapeType t, PetscInt c) const
  {
    return shape[i_p(i, t, c)];
  }

private:
  void fill(const Vector3R& p_r1, const Vector3R& p_r2, ShapeType t1,
    ShapeType t2, PetscReal (&sfunc)(PetscReal));

  /// @note `Vector3I::dim` is used as a coordinate space dimensionality.
  static constexpr PetscInt shape_geom = POW3(shape_width);
  static constexpr PetscInt shape_comp = Vector3I::dim * 2;
  PetscReal shape[shape_geom * shape_comp];
};

#endif  // SRC_UTILS_SHAPE_H