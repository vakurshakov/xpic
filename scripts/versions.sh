# Dependency versions, the single source of truth.
# Usage: source setenv.sh.
#
# MIN: hard-known minimum (empty if unverified),
# TESTED: verified versions,
# PINNED: installed by scripts.

# Required at a minimum so that the install-deps.sh can fetch dependencies
GIT_MIN_VERSION=1.8.5

# C++20 compiler (gcc, g++). `GCC_DIR` is required for Clang-based stacks.
# On stark GCC is built by scripts/cluster/install-gcc.sh.
COMPILER_MIN_VERSION=
GCC_PINNED_VERSION=15.1.0
COMPILER_TESTED="GCC 15.1, GCC 16.2, AOCC 5.2.0 (with GCC_DIR)"

# gfortran: needed locally only, because PETSc's OpenBLAS install always passes `FC=`.
GFORTRAN_TESTED="GCC 15.1"

# MPI is never built locally; built by the cluster scripts on stark.
MPI_MIN_VERSION=
MPICH_PINNED_VERSION=5.0.2
MPI_TESTED="Open MPI 5.0.11 (local), MPICH 5.0.2 (stark)"

MAKE_MIN_VERSION=

CMAKE_MIN_VERSION=3.10
CMAKE_TESTED=4.4

JSON_MIN_VERSION=3.11
JSON_PINNED_TAG=v3.12.0

PETSC_MIN_VERSION=3.26
PETSC_PINNED_TAG=v3.26.1

# BLAS/LAPACK is installed through PETSc locally.
# OpenBLAS is built by the cluster scripts on stark.
# f2cblaslapack is taken through PETSc `--download-f2cblaslapack`.
OPENBLAS_PINNED_VERSION=0.3.34
BLAS_TESTED="OpenBLAS 0.3.34 (local), f2cblaslapack and AOCL 5.3.0 (stark)"

PYTHON_MIN_VERSION=3.6
PYTHON_TESTED=3.14

# Not installed by install-deps.sh. Needed by the Python post-processing in
# `tools/` and by the `*_plot` smoke tests; the build does not depend on them.
PYTHON_OPTIONAL_PACKAGES="numpy matplotlib scipy"

# PETSc `configure` fails without it, and minimal images (e.g. Arch) do not ship it.
DIFF_MIN_VERSION=


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

# Prints an error and returns 1 unless the nlohmann::json in `dir` exists and is at least JSON_MIN_VERSION.
# Usage: verify_json <dir>
verify_json() {
  local header="$1/include/nlohmann/detail/abi_macros.hpp" major minor
  [[ -f "$1/include/nlohmann/json.hpp" && -f "$header" ]] || {
    echo "JSON_DIR=$1 has no include/nlohmann/json.hpp or include/nlohmann/detail/abi_macros.hpp" >&2; return 1;
  }
  major=$(sed -n 's/^#define NLOHMANN_JSON_VERSION_MAJOR *\([0-9]*\).*/\1/p' "$header")
  minor=$(sed -n 's/^#define NLOHMANN_JSON_VERSION_MINOR *\([0-9]*\).*/\1/p' "$header")
  check_min_version nlohmann::json "$major.$minor" "$JSON_MIN_VERSION"
}

# Same for PETSc and PETSC_MIN_VERSION.
# Usage: verify_petsc <dir>
verify_petsc() {
  local header="$1/include/petscversion.h" major minor
  [[ -f "$header" ]] || {
    echo "PETSC_DIR=$1 has no include/petscversion.h" >&2; return 1;
  }
  major=$(sed -n 's/^#define PETSC_VERSION_MAJOR *//p' "$header")
  minor=$(sed -n 's/^#define PETSC_VERSION_MINOR *//p' "$header")
  check_min_version PETSc "$major.$minor" "$PETSC_MIN_VERSION"
}

# Verifies the dependencies found by the environment.
check_dependency_versions() {
  verify_json "$JSON_DIR" && verify_petsc "$PETSC_DIR"
}
