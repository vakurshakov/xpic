#ifndef SRC_UTILS_UTILS_H
#define SRC_UTILS_UTILS_H

#include "src/pch.h"

struct Triplet {
  PetscInt row;
  PetscInt col;
  PetscReal value;
};

#define R3C(expr) (expr), (expr), (expr)
#define R3DX(expr) (expr.x), (expr.y), (expr.z)
#define R3CX(expr) (expr##x), (expr##y), (expr##z)

#endif // SRC_UTILS_UTILS_H
