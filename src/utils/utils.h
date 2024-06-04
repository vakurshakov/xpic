#ifndef SRC_UTILS_UTILS_H
#define SRC_UTILS_UTILS_H

#include "src/pch.h"

struct Triplet {
  PetscInt row;
  PetscInt col;
  PetscReal value;
};

enum Axis : PetscInt {
  X = 0,
  Y = 1,
  Z = 2,
  C = 3,
};

#define REP2(A) A, A
#define REP3(A) A, A, A
#define REP4(A) A, A, A, A

#define REP2_N(A) A##1, A##2
#define REP3_N(A) A##1, A##2, A##3
#define REP4_N(A) A##1, A##2, A##3, A##4

#define REP2_X(A) A##x, A##y
#define REP3_X(A) A##x, A##y, A##z
#define REP4_X(A) A##x, A##y, A##z, A##c

#define REP2_A(A) A[0], A[1]
#define REP3_A(A) A[0], A[1], A[2]
#define REP4_A(A) A[0], A[1], A[2], A[3]

#define REP2_AP(A) A[1], A[0]
#define REP3_AP(A) A[2], A[1], A[0]
#define REP4_AP(A) A[2], A[1], A[0], A[3]

#define POW2(A) A * A
#define POW3(A) A * A * A
#define POW4(A) A * A * A * A
#define POW5(A) A * A * A * A * A

#define ROUND(s) static_cast<PetscInt>(std::round(s))
#define TO_STEP(s, ds)  ROUND((s) / (ds))

#endif // SRC_UTILS_UTILS_H
