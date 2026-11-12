#!/usr/bin/env bash

# Clones PETSc at the pinned tag into PETSC_DIR and builds the requested arches. Sourced environment
# decides the location. A PETSC_DIR outside ./external is reused with only version and library checks.
# Usage: ./scripts/install-petsc.sh [debug|opt] [--force]
#   debug|opt  build only this arch (default: both)
#   --force    rebuild even if the pinned version is built

source ./scripts/setenv.sh || exit 1

arches=(debug opt)
for arg in "$@"; do
  case "$arg" in
    debug | opt) arches=("$arg") ;;
    --force) force=1 ;;
    *) echo "Unknown argument '$arg'. Usage: $0 [debug|opt] [--force]" >&2; exit 1 ;;
  esac
done

# Whether an arch holds a built library (a prefix-style install has it directly in PETSC_DIR).
is_built() {
  compgen -G "$PETSC_DIR/linux-mpi-$1/lib/libpetsc.*" > /dev/null
}

# Step 1: a PETSC_DIR outside ./external is reused, only verified.
if [[ "$PETSC_DIR" != ./external/* && "$PETSC_DIR" != "$PWD/external/"* ]]; then
  verify_petsc "$PETSC_DIR" || exit 1
  for arch in "${arches[@]}"; do
    if [[ -n "$PETSC_ARCH" ]]; then
      compgen -G "$PETSC_DIR/$PETSC_ARCH/lib/libpetsc.*" > /dev/null || {
        echo "PETSC_DIR=$PETSC_DIR has no built library for PETSC_ARCH=$PETSC_ARCH" >&2; exit 1;
      }
    else
      is_built "$arch" || compgen -G "$PETSC_DIR/lib/libpetsc.*" > /dev/null || {
        echo "PETSC_DIR=$PETSC_DIR has no built library for linux-mpi-$arch" >&2; exit 1;
      }
    fi
  done
  echo "Reusing PETSc from PETSC_DIR=$PETSC_DIR"
  exit 0
fi

# Step 2: fetch the pinned tag unless it is already there.
PETSC_TAG="$(git -C "$PETSC_DIR" describe --tags --exact-match 2> /dev/null)"
PETSC_REPO="https://gitlab.com/petsc/petsc.git"

if [[ "$PETSC_TAG" != "$PETSC_PINNED_TAG" ]]; then
  echo "Cloning PETSc $PETSC_PINNED_TAG into $PETSC_DIR"

  rm -rf "$PETSC_DIR"
  (
    # Not `git clone --branch`: for an annotated tag it warns "is not a commit!".
    GIT_ADVICE=0 git init "$PETSC_DIR" || exit 1
    GIT_ADVICE=0 git -C "$PETSC_DIR" fetch --depth 1 "$PETSC_REPO" tag "$PETSC_PINNED_TAG" || exit 1
    GIT_ADVICE=0 git -C "$PETSC_DIR" checkout "$PETSC_PINNED_TAG" || exit 1
  ) || {
    rm -rf "$PETSC_DIR"
    echo "Cloning failed (no network?). Copy PETSc to a directory and set PETSC_DIR." >&2
    exit 1
  }
fi

# Step 3: build every requested arch that is not built yet (or all with --force).
common_options=(
  --with-mpi-dir="$MPI_DIR"
  --with-openmp=1
  --with-openmp-kernels=1
  --with-threadsafety=1
  --with-cuda=0
  --with-cudac=0
  --download-openblas
)

debug_options=()

opt_options=(
  --with-debugging=0
  COPTFLAGS='-O3 -march=native -mtune=native'
  FOPTFLAGS='-O3 -march=native -mtune=native'
  CXXOPTFLAGS='-O3 -march=native -mtune=native'
)

for arch in "${arches[@]}"; do
  if [[ -z "$force" ]] && is_built "$arch"; then
    echo "PETSc $PETSC_PINNED_TAG (linux-mpi-$arch) is already built in $PETSC_DIR"
    continue
  fi

  echo "Building PETSc $PETSC_PINNED_TAG (linux-mpi-$arch)"

  options="${arch}_options[@]"
  (
    cd "$PETSC_DIR" || exit 1
    export PETSC_DIR="$PWD"
    unset PETSC_ARCH MPI_DIR # configure ignores (and warns about) variables named like its options
    ./configure PETSC_ARCH="linux-mpi-$arch" "${common_options[@]}" "${!options}" || exit 1
    make PETSC_ARCH="linux-mpi-$arch" all || exit 1
    make PETSC_ARCH="linux-mpi-$arch" check || exit 1
  ) || {
    echo "Building PETSc (linux-mpi-$arch) failed, see $PETSC_DIR/configure.log." >&2
    exit 1
  }
done

# Step 4: verify the result.
verify_petsc "$PETSC_DIR" || exit 1
