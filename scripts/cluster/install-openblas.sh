#!/usr/bin/env bash

# Builds OpenBLAS (full BLAS, C-translated LAPACK, no Fortran, OpenMP threads) from TARBALL_DIR
# into OPENBLAS_DIR of the stack. The CPU target (OPENBLAS_TARGET) comes from the stack file.
# It has no out-of-tree mode: it builds in the shared source tree and cleans it first.
# Run it on a node of the target architecture (`-march=native`), directly or with
# `qsub -q <queue> -pe smp 128 -l h_rt=4:00:00 -v STACK=...,STACK_ROOT=...,TARBALL_DIR=...`.
#
# Usage: ./scripts/cluster/install-openblas.sh [-h|--help] [--force]
#   --force     rebuild an existing prefix from a clean source tree
#   -h, --help  print this help and exit
#
# Environment:
#   STACK        name of scripts/stacks/<name>.sh (required)
#   STACK_ROOT   root of the stack installs (required)
#   TARBALL_DIR  directory with the source tarballs (required)
#   JOBS         make parallelism (default: scheduler slots, then nproc)

#$ -S /bin/bash
#$ -N install-openblas
#$ -V
#$ -cwd
#$ -j y

source ./scripts/usage.sh

for arg in "$@"; do
  case "$arg" in
    -h | --help) usage; exit 0 ;;
    --force) force=1 ;;
    *) echo "Unknown argument '$arg'." >&2; usage >&2; exit 1 ;;
  esac
done

source ./scripts/setenv.sh || exit 1
source ./scripts/cluster/common.sh

# Every install script declares the same variables and goes through the same steps (source,
# build tree, configure, build, install, check); only the commands inside the steps differ.
#   NAME     component directory in STACK_ROOT, holds the source (and the build tree)
#   TARBALL  file in TARBALL_DIR
#   SOURCE   directory the tarball unpacks to, inside the component directory
#   PREFIX   install prefix from the stack file
NAME="OpenBLAS-$OPENBLAS_PINNED_VERSION"
TARBALL="OpenBLAS-$OPENBLAS_PINNED_VERSION.tar.gz"
SOURCE="OpenBLAS-$OPENBLAS_PINNED_VERSION"
PREFIX="$OPENBLAS_DIR"

COMPONENT_DIR="$STACK_ROOT/$NAME"
SOURCE_DIR="$COMPONENT_DIR/$SOURCE"

# Step 1: guard the prefix and extract the source.
guard_prefix "$PREFIX" "$force"
extract_source "$TARBALL" "$COMPONENT_DIR" "$SOURCE" "$force"

# Step 2: in-tree build (no out-of-tree mode), so the shared source is cleaned first.
# Objects of an interrupted build (or of other flags) must not be reused.
cd "$SOURCE_DIR" || exit 1
[[ -f Makefile ]] && { make clean || exit 1; }

# Step 3: build, the options go to make.
# No gfortran: LAPACK is taken from its C translation. Threads come from libgomp.
make -j "$JOBS"             \
  CC="$STACK_CC"            \
  CFLAGS="$STACK_FLAGS"     \
  NOFORTRAN=1               \
  C_LAPACK=1                \
  TARGET="$OPENBLAS_TARGET" \
  USE_OPENMP=1 || exit 1

# Step 4: install.
make PREFIX="$PREFIX" install || exit 1

# Step 5: check the exported symbols.
for symbol in dgemm_ dgetrf_; do
  nm -D "$PREFIX/lib/libopenblas.so" | grep -q " T $symbol$" || {
    echo "libopenblas.so does not export $symbol." >&2
    exit 1
  }
done
