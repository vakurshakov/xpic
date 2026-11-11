# Dependency versions, the single source of truth. Sourced by `setenv.sh`.
# MIN: hard-known minimum (empty if unverified), TESTED: verified versions, PINNED: installed by scripts.

# C++20 compiler (gcc, g++): minimum unverified. `GCC_DIR` is required for Clang-based stacks.
COMPILER_MIN_VERSION=
COMPILER_TESTED="GCC 15.1, GCC 16.2, AOCC 5.2.0 (with GCC_DIR)"

# gfortran: needed locally only, because PETSc's OpenBLAS install always passes `FC=`.
GFORTRAN_TESTED="GCC 15.1"

CMAKE_MIN_VERSION=3.10
CMAKE_TESTED=4.4

MAKE_MIN_VERSION=
GIT_MIN_VERSION=

# MPI is never built from source. Minimum unverified.
MPI_MIN_VERSION=
MPI_TESTED="MPICH 5.0.2 (stark), Open MPI 5.0.11 (local)"

# BLAS/LAPACK is installed through PETSc.
BLAS_TESTED="OpenBLAS 0.3.34 (local), f2cblaslapack and AOCL 5.3.0 (stark)"

PETSC_MIN_VERSION=3.26
PETSC_PINNED_TAG=v3.26.1

JSON_MIN_VERSION=3.11
JSON_PINNED_TAG=v3.11.2

PYTHON_MIN_VERSION=3.6
PYTHON_TESTED=3.14

# Optional post-processing in `tools/`; build and tests do not depend on them.
PYTHON_OPTIONAL_PACKAGES="numpy matplotlib scipy"

# Prints an error and returns 1 unless `found` >= `required` (compared with `sort -V`).
# Usage: check_min_version <name> <found> <required>
check_min_version() {
  local name="$1" found="$2" required="$3"
  if [[ -z "$found" ]]; then
    echo "Cannot determine the version of $name (required >= $required)." >&2
    return 1
  fi
  if [[ "$(printf '%s\n%s\n' "$required" "$found" | sort -V | head -n1)" != "$required" ]]; then
    echo "$name $found is too old, required >= $required." >&2
    return 1
  fi
}

# Prints an error and returns 1 if the PETSc or json found by the environment is below its minimum.
check_dependency_versions() {
  local header="$PETSC_DIR/include/petscversion.h" major minor
  if [[ -f "$header" ]]; then
    major=$(sed -n 's/^#define PETSC_VERSION_MAJOR *//p' "$header")
    minor=$(sed -n 's/^#define PETSC_VERSION_MINOR *//p' "$header")
    check_min_version PETSc "$major.$minor" "$PETSC_MIN_VERSION" || return 1
  fi

  header="$JSON_DIR/include/nlohmann/detail/abi_macros.hpp"
  if [[ -f "$header" ]]; then
    major=$(sed -n 's/^#define NLOHMANN_JSON_VERSION_MAJOR *\([0-9]*\).*/\1/p' "$header")
    minor=$(sed -n 's/^#define NLOHMANN_JSON_VERSION_MINOR *\([0-9]*\).*/\1/p' "$header")
    check_min_version nlohmann::json "$major.$minor" "$JSON_MIN_VERSION" || return 1
  fi
}
