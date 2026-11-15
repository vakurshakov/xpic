# Stack on stark: AOCC (Clang) + MPICH + AOCL, built by scripts/cluster/. Requires STACK_ROOT.
# AOCC and AOCL are manual vendor installs (see their EULA), the scripts expect them to exist.
# Layout: <component>/<name>-<version> holds sources, <component>/<compiler> the prefix.

if [[ -z "$STACK_ROOT" ]]; then
  echo "STACK_ROOT is required (e.g. STACK_ROOT=/path/to/stack/root)." >&2
  return 1
fi

AOCC_VERSION=5.2.0
AOCL_VERSION=5.3.0
compiler="AOCC-$AOCC_VERSION"

# `MT` holds the multithreaded (OpenMP) AOCL libraries, `lib` of it links the LP64 interface
# (4-byte integers, as PETSc is configured), see set_aocl_interface_symlink.sh in the AOCL tree.
AOCC_DIR="${AOCC_DIR:-$STACK_ROOT/aocc-compiler-$AOCC_VERSION}"
AOCL_DIR="${AOCL_DIR:-$STACK_ROOT/AOCL-$AOCL_VERSION/MT}"

# GCC is not a compiler here: only its libstdc++ and linker objects are used (see --gcc-toolchain).
export GCC_DIR="${GCC_DIR:-$STACK_ROOT/GCC-${GCC_PINNED_VERSION%.*}}"
export MPI_DIR="${MPI_DIR:-$STACK_ROOT/MPICH-$MPICH_PINNED_VERSION/$compiler}"
export JSON_DIR="${JSON_DIR:-$STACK_ROOT/json-${JSON_PINNED_TAG#v}}"
export PETSC_DIR="${PETSC_DIR:-$STACK_ROOT/petsc-${PETSC_PINNED_TAG#v}/$compiler-AOCL}"

# Prefix installs have no arch.
export PETSC_ARCH=

# Clang takes the legacy system GCC (and its linker objects, crtbegin.o etc.) by default, which
# fails to build C++20 code and to link against the newer libstdc++. `--gcc-toolchain` points both
# the compile and the link step to the stack's GCC instead.
gcc_toolchain="--gcc-toolchain=$GCC_DIR"

# Used by the install scripts. The CPU target belongs to the stack, not to a script.
STACK_CC="$AOCC_DIR/bin/clang"
STACK_CXX="$AOCC_DIR/bin/clang++"
STACK_FLAGS="-O3 -march=native -mtune=native $gcc_toolchain"

# MPICH is linked into shared libraries: libfabric needs PIC explicitly.
MPICH_FLAGS="$STACK_FLAGS -fPIC"
MPICH_CONFIGURE_ARGS=(--with-libfabric-args="--with-pic")

# PETSc: no Fortran (AOCC has no gfortran-compatible runtime in the stack, and 
# the code is C++), AOCL (BLIS + libFLAME) as BLAS/LAPACK, `amdlibm` as libm.
# Rpaths make the installed libraries independent of LD_LIBRARY_PATH.
STACK_PETSC_BUILD_ARCH=linux-mpich-aocc-aocl-real
STACK_PETSC_BUILD_OPTIONS=(
  --with-blaslapack-dir="$AOCL_DIR"
  LIBS=-lamdlibm
  COPTFLAGS="$STACK_FLAGS"
  CXXOPTFLAGS="$STACK_FLAGS"
  LDFLAGS="                   \
    $gcc_toolchain            \
    -L$MPI_DIR/lib            \
    -L$AOCL_DIR/lib           \
    -Wl,-rpath,$MPI_DIR/lib   \
    -Wl,-rpath,$GCC_DIR/lib64 \
    -Wl,-rpath,$AOCL_DIR/lib"
)

export PATH="$MPI_DIR/bin:$AOCC_DIR/bin:$PATH"

# Set, not prepended: the inherited value (e.g. Anaconda's old libstdc++ from .bashrc) must not
# shadow the stack's libraries. Anaconda's Python finds its own libraries through rpath.
export LD_LIBRARY_PATH="$MPI_DIR/lib:$GCC_DIR/lib64:$AOCC_DIR/lib:$AOCL_DIR/lib"

unset compiler gcc_toolchain
