#ifndef SRC_UTILS_SHAPE_H
#define SRC_UTILS_SHAPE_H

#include "src/interfaces/sort_parameters.h"
#include "src/utils/vector3.h"

enum ShapeType : std::uint8_t {
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

  constexpr PetscInt s_p(PetscInt x, PetscInt y, PetscInt z) const
  {
    return indexing::petsc_index(x, y, z, 0, size[X], size[Y], size[Z], 1);
  }

  static constexpr PetscInt i_p(PetscInt i, ShapeType t, PetscInt c)
  {
    return i * shape_comp + ((t % 2) * Vector3I::dim + c);
  }

  void setup(const Vector3R& p_r)
  {
    setup(p_r, shape_radius, shape_function);
  }

  void setup(const Vector3R& old_r, const Vector3R& new_r)
  {
    setup(old_r, new_r, shape_radius, shape_function);
  }

  void setup(const Vector3R& r, PetscReal radius, PetscReal (&sfunc)(PetscReal));

  void setup(const Vector3R& old_r, const Vector3R& new_r, PetscReal radius,
    PetscReal (&sfunc)(PetscReal));

  /// @returns Vector of shape products corresponding to electric fields.
  /// @note No check is performed to see if the `ShapeType::No/Sh` type pair is set.
  constexpr Vector3R electric(PetscInt i) const
  {
    return Vector3R{
      shape[i_p(i, No, Z)] * shape[i_p(i, No, Y)] * shape[i_p(i, Sh, X)],
      shape[i_p(i, No, Z)] * shape[i_p(i, Sh, Y)] * shape[i_p(i, No, X)],
      shape[i_p(i, Sh, Z)] * shape[i_p(i, No, Y)] * shape[i_p(i, No, X)],
    };
  }

  /// @returns Vector of shape products corresponding to magnetic fields.
  /// @note No check is performed to see if the `ShapeType::No/Sh` type pair is set.
  constexpr Vector3R magnetic(PetscInt i) const
  {
    return Vector3R{
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
  static Vector3R make_r(const Vector3R& r);
  static Vector3I make_g(const Vector3R& p_r);

  static Vector3I make_start(const Vector3R& p_r, PetscReal radius);
  static Vector3I make_end(const Vector3R& p_r, PetscReal radius);

  void fill(const Vector3R& p_r1, const Vector3R& p_r2, ShapeType t1,
    ShapeType t2, PetscReal (&sfunc)(PetscReal));

  /// @note `Vector3I::dim` is used as a coordinate space dimensionality.
  static constexpr PetscInt shape_geom = POW3(shape_width);
  static constexpr PetscInt shape_comp = Vector3I::dim * 2;
  static constexpr std::size_t shape_size =
    static_cast<std::size_t>(shape_geom) * shape_comp;

  std::array<PetscReal, shape_size> shape;
};

#endif  // SRC_UTILS_SHAPE_H
