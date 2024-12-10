#ifndef SRC_UTILS_UTILS_H
#define SRC_UTILS_UTILS_H

#include "src/pch.h"

enum Axis : std::uint8_t {
  X = 0,
  Y = 1,
  Z = 2,
  C = 3,
};

/// @todo #pragma omp declare simd linear(z, y, x : 1), notinbranch
/// @todo Additional motivation to make geom_n{x,y,z} constexpr is {s,v}_g.
namespace indexing {

/// @brief Standard notation inside PETSc, it's then reused to create aliases
constexpr PetscInt petsc_index(PetscInt z, PetscInt y, PetscInt x, PetscInt c,
  PetscInt /* size_z */, PetscInt size_y, PetscInt size_x, PetscInt size_c)
{
  return ((z * size_y + y) * size_x + x) * size_c + c;
}

/// @brief Grid indices for scalar fields in PETSc natural ordering
inline PetscInt s_g(PetscInt z, PetscInt y, PetscInt x)
{
  return petsc_index(z, y, x, 0, geom_nz, geom_ny, geom_nz, 1);
}

/// @brief Grid indices for vector fields in PETSc natural ordering
inline PetscInt v_g(PetscInt z, PetscInt y, PetscInt x, PetscInt c)
{
  return petsc_index(z, y, x, c, geom_nz, geom_ny, geom_nz, 3);
}

}  // namespace indexing


#define REP2(A) A, A
#define REP3(A) A, A, A
#define REP4(A) A, A, A, A

#define REP2_N(A) A##1, A##2
#define REP3_N(A) A##1, A##2, A##3
#define REP4_N(A) A##1, A##2, A##3, A##4

#define REP2_X(A) A##x, A##y
#define REP3_X(A) A##x, A##y, A##z
#define REP4_X(A) A##x, A##y, A##z, A##c

#define REP2_A(A) (A)[0], (A)[1]
#define REP3_A(A) (A)[0], (A)[1], (A)[2]
#define REP4_A(A) (A)[0], (A)[1], (A)[2], (A)[3]

#define REP2_AP(A) (A)[1], (A)[0]
#define REP3_AP(A) (A)[2], (A)[1], (A)[0]
#define REP4_AP(A) (A)[2], (A)[1], (A)[0], (A)[3]

#define POW2(A) ((A) * (A))
#define POW3(A) ((A) * (A) * (A))
#define POW4(A) ((A) * (A) * (A) * (A))
#define POW5(A) ((A) * (A) * (A) * (A) * (A))

#define TO_STEP(s, ds) static_cast<PetscInt>(std::round((s) / (ds)))

#define STR_IMPL(x)    #x
#define CAT_IMPL(a, b) a##b

#define STR(x)    STR_IMPL(x)
#define CAT(a, b) CAT_IMPL(a, b)


// NOLINTBEGIN
#define PetscCallThrow(...)                                      \
  do {                                                           \
    PetscStackUpdateLine;                                        \
    PetscErrorCode ierr_petsc_call_ = __VA_ARGS__;               \
    if (PetscUnlikely(ierr_petsc_call_ != PETSC_SUCCESS)) {      \
      std::stringstream msg;                                     \
      msg << "PETSC ERROR: " << PETSC_FUNCTION_NAME_CXX << "() " \
          << "at " << __FILE__ << ":" << __LINE__ << "\n";       \
      throw std::runtime_error(msg.str());                       \
    }                                                            \
  }                                                              \
  while (0)

#define DEFAULT_COPYABLE(Class)                   \
  Class(const Class& other) = default;            \
  Class& operator=(const Class& other) = default; \
  Class(Class&& other) = delete;                  \
  Class& operator=(Class&& other) = delete;

#define DEFAULT_MOVABLE(Class)                   \
  Class(const Class& other) = delete;            \
  Class& operator=(const Class& other) = delete; \
  Class(Class&& other) = default;                \
  Class& operator=(Class&& other) = default;
// NOLINTEND

#if LOGGING
  #define LOG_FLUSH()   std::cout.flush()
  #define LOG_IMPL(...) std::cout << std::format(__VA_ARGS__) << "\n";
  #define LOG(...)                            \
    do {                                      \
      PetscMPIInt rank;                       \
      MPI_Comm_rank(PETSC_COMM_WORLD, &rank); \
      if (rank == 0)                          \
        LOG_IMPL(__VA_ARGS__);                \
    }                                         \
    while (0)
#else
  #define LOG_FLUSH()
  #define LOG_IMPL(...)
  #define LOG(...)
#endif

#endif  // SRC_UTILS_UTILS_H
