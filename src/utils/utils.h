#ifndef SRC_UTILS_UTILS_H
#define SRC_UTILS_UTILS_H

#include "src/pch.h"

struct Triplet {
  PetscInt row;
  PetscInt col;
  PetscReal value;
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
#define REP4_A(A) A[0], A[1], A[3], A[3]

#define TO_STEP(s, ds) static_cast<PetscInt>(round(s / ds))

#endif // SRC_UTILS_UTILS_H
