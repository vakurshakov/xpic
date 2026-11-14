# Stack on stark: GCC + MPICH + OpenBLAS, built by scripts/cluster/. Requires STACK_ROOT.
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

export PATH="$MPI_DIR/bin:$GCC_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_DIR/lib:$GCC_DIR/lib64:$OPENBLAS_DIR/lib:$LD_LIBRARY_PATH"

# Anaconda ships its own libmpifort.so, which the dynamic linker picks up and breaks runs
# with `undefined symbol: MPII_F_TRUE`. Drop it from the search paths.
drop_anaconda() { echo "$1" | tr ':' '\n' | grep -v "anaconda3" | paste -sd:; }
export PATH="$(drop_anaconda "$PATH")"
export LD_LIBRARY_PATH="$(drop_anaconda "$LD_LIBRARY_PATH")"
unset -f drop_anaconda
unset compiler
