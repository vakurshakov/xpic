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

//private:
protected:
  enum Type {
    electric,
    magnetic,
  };

  struct Shape {
    static constexpr PetscInt shw1 = 2;
    static constexpr PetscInt shw2 = 3;
    static constexpr PetscInt shc = Vector3R::dim;
    static constexpr PetscInt shm = POW2(shw2) * shw1 * shc;

    static constexpr PetscReal sfunc_1(PetscReal s)
    {
      return 1.0 - std::abs(s);
    }

    static constexpr PetscReal sfunc_21(PetscReal s)
    {
      s = std::abs(s);
      return (0.75 - s * s);
    }

    static constexpr PetscReal sfunc_22(PetscReal s)
    {
      s = std::abs(s);
      return 0.5 * POW2(1.5 - s);
    }

    static constexpr PetscReal (*sfunc_2[3])(PetscReal) = {
      sfunc_22,
      sfunc_21,
      sfunc_22,
    };

    Vector3I start;
    PetscReal cache[shm];

    virtual void setup(const Vector3R& rn, const Vector3R& r0, Type t);
  };

  Vector3R*** E_g;
  Vector3R*** B_g;
  Vector3R*** J_g;

  Shape shape[2];
};

#endif  // SRC_ALGORITHMS_IMPLICIT_ESIRKEPOV_H

