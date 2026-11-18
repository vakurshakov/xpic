# Stack on stark: GCC + MPICH + OpenBLAS (C-translated LAPACK), built by scripts/cluster/. Requires STACK_ROOT.
# Layout: <component>/<name>-<version> holds sources, <component>/<compiler> the prefix.

if [[ -z "$STACK_ROOT" ]]; then
  echo "STACK_ROOT is required (e.g. STACK_ROOT=/path/to/stack/root)." >&2
  return 1
fi

compiler="GCC-${GCC_PINNED_VERSION%.*}"

export GCC_DIR="${GCC_DIR:-$STACK_ROOT/$compiler}"
export MPI_DIR="${MPI_DIR:-$STACK_ROOT/MPICH-$MPICH_PINNED_VERSION/$compiler}"
export JSON_DIR="${JSON_DIR:-$STACK_ROOT/json-${JSON_PINNED_TAG#v}}"
export PETSC_DIR="${PETSC_DIR:-$STACK_ROOT/petsc-${PETSC_PINNED_TAG#v}/$compiler-OpenBLAS}"
OPENBLAS_DIR="${OPENBLAS_DIR:-$STACK_ROOT/OpenBLAS-$OPENBLAS_PINNED_VERSION/$compiler}"

# Prefix installs have no arch.
export PETSC_ARCH=


# Used by the install scripts. The CPU target belongs to the stack, not to a script.
STACK_CC="$GCC_DIR/bin/gcc"
STACK_CXX="$GCC_DIR/bin/g++"
STACK_FLAGS="-O3 -march=native -mtune=native"
OPENBLAS_TARGET=ZEN

# PETSc: no Fortran, OpenBLAS (full BLAS, C-translated LAPACK) as BLAS/LAPACK.
# Rpaths make the installed libraries independent of LD_LIBRARY_PATH.
STACK_PETSC_BUILD_ARCH=linux-mpich-gcc-openblas-real
STACK_PETSC_BUILD_OPTIONS=(
  --with-blaslapack-dir="$OPENBLAS_DIR"
  COPTFLAGS="$STACK_FLAGS"
  CXXOPTFLAGS="$STACK_FLAGS"
  LDFLAGS="                     \
    -L$MPI_DIR/lib              \
    -L$OPENBLAS_DIR/lib         \
    -Wl,-rpath,$MPI_DIR/lib     \
    -Wl,-rpath,$GCC_DIR/lib64   \
    -Wl,-rpath,$OPENBLAS_DIR/lib"
)

export PATH="$MPI_DIR/bin:$GCC_DIR/bin:$PATH"

# Set, not prepended: the inherited value (e.g. Anaconda's old libstdc++ from .bashrc) must not
# shadow the stack's libraries. Anaconda's Python finds its own libraries through rpath.
export LD_LIBRARY_PATH="$MPI_DIR/lib:$GCC_DIR/lib64:$OPENBLAS_DIR/lib"

unset compiler
