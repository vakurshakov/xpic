# Stack on stark: GCC + MPICH + OpenBLAS BLAS + f2cblaslapack LAPACK, built by scripts/cluster/.
# Requires STACK_ROOT. Same GCC, MPICH and OpenBLAS as stark-gcc-openblas, only PETSc differs.
# Layout: <component>/<name>-<version> holds sources, <component>/<compiler> the prefix.

if [[ -z "$STACK_ROOT" ]]; then
  echo "STACK_ROOT is required (e.g. STACK_ROOT=/path/to/stack/root)." >&2
  return 1
fi

compiler="GCC-${GCC_PINNED_VERSION%.*}"

export GCC_DIR="${GCC_DIR:-$STACK_ROOT/$compiler}"
export MPI_DIR="${MPI_DIR:-$STACK_ROOT/MPICH-$MPICH_PINNED_VERSION/$compiler}"
export JSON_DIR="${JSON_DIR:-$STACK_ROOT/json-${JSON_PINNED_TAG#v}}"
export PETSC_DIR="${PETSC_DIR:-$STACK_ROOT/petsc-${PETSC_PINNED_TAG#v}/$compiler-F2C}"

# Prefix installs have no arch.
export PETSC_ARCH=

# Used by the install scripts. The CPU target belongs to the stack, not to a script.
STACK_CC="$GCC_DIR/bin/gcc"
STACK_CXX="$GCC_DIR/bin/g++"
STACK_FLAGS="-O3 -march=native -mtune=native"
OPENBLAS_TARGET=ZEN

# PETSc: no Fortran (--with-fc=0), so LAPACK cannot be built from the reference Fortran sources.
# BLAS/LAPACK  is f2cblaslapack: the reference LAPACK translated to C, built by PETSc
# from the local tarball in TARBALL_DIR (the cluster has no network).
# Rpaths make the installed libraries independent of LD_LIBRARY_PATH.
STACK_PETSC_BUILD_ARCH=linux-mpich-gcc-f2c-real
STACK_PETSC_BUILD_OPTIONS=(
  --download-f2cblaslapack="$TARBALL_DIR/$F2CBLASLAPACK_TARBALL"
  COPTFLAGS="$STACK_FLAGS"
  CXXOPTFLAGS="$STACK_FLAGS"
  LDFLAGS="                  \
    -L$MPI_DIR/lib           \
    -Wl,-rpath,$MPI_DIR/lib  \
    -Wl,-rpath,$GCC_DIR/lib64"
)

export PATH="$MPI_DIR/bin:$GCC_DIR/bin:$PATH"

# Set, not prepended: the inherited value (e.g. Anaconda's old libstdc++ from .bashrc) must not
# shadow the stack's libraries. Anaconda's Python finds its own libraries through rpath.
export LD_LIBRARY_PATH="$MPI_DIR/lib:$GCC_DIR/lib64"

unset compiler
