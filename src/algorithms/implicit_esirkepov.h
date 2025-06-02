#ifndef SRC_ALGORITHMS_IMPLICIT_ESIRKEPOV_H
#define SRC_ALGORITHMS_IMPLICIT_ESIRKEPOV_H

#include "src/interfaces/point.h"
#include "src/utils/shape.h"

class ImplicitEsirkepov {
public:
  ImplicitEsirkepov(Vector3R*** E_g, Vector3R*** B_g, Vector3R*** J_g);

  void interpolate(Vector3R& E_p, Vector3R& B_p, //
    const Vector3R& rn, const Vector3R& r0);

  void decompose(PetscReal alpha, const Vector3R& v, //
    const Vector3R& rn, const Vector3R& r0);

private:
  enum Type {
    electric,
    magnetic,
  };

  struct Shape {
    static constexpr PetscReal shr1 = 1.0;
    static constexpr PetscReal shr2 = 1.5;
    static constexpr PetscInt shw1 = (PetscInt)(2 * shr1) + 1;
    static constexpr PetscInt shw2 = (PetscInt)(2 * shr2) + 1;
    static constexpr PetscInt shc = Vector3R::dim;
    static constexpr PetscInt shm = POW3(shw2) * shc;
    static constexpr PetscReal (&sfunc1)(PetscReal) = spline_of_1st_order;
    static constexpr PetscReal (&sfunc2)(PetscReal) = spline_of_2nd_order;

    Vector3I start, size;
    Vector3R cache[shm];

    void setup(const Vector3R& rn, const Vector3R& r0, Type t);
  };

  Vector3R*** E_g;
  Vector3R*** B_g;
  Vector3R*** J_g;

  Shape shape[2];
};

#endif  // SRC_ALGORITHMS_IMPLICIT_ESIRKEPOV_H

