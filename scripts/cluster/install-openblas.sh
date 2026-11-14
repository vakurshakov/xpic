#!/usr/bin/env bash

# Builds OpenBLAS (full BLAS, C-translated LAPACK, no Fortran, OpenMP threads) from TARBALL_DIR
# into OPENBLAS_DIR of the stack. The CPU target (OPENBLAS_TARGET) comes from the stack file.
# Run it on a node of the target architecture (`-march=native`), directly or with
# `qsub`, e.g. `qsub -q <queue> -pe smp 128 -l h_rt=4:00:00 -v STACK=...,STACK_ROOT=...,TARBALL_DIR=...`.
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

guard_prefix "$OPENBLAS_DIR" "$force"
extract_source "$OPENBLAS_TARBALL" "$(dirname "$OPENBLAS_DIR")" "OpenBLAS-$OPENBLAS_PINNED_VERSION" "$force"

cd "$(dirname "$OPENBLAS_DIR")/OpenBLAS-$OPENBLAS_PINNED_VERSION" || exit 1

# No gfortran: LAPACK is taken from its C translation. Threads come from libgomp.
make -j "$JOBS" \
  CC="$STACK_CC" \
  CFLAGS="$STACK_FLAGS" \
  NOFORTRAN=1 \
  C_LAPACK=1 \
  TARGET="$OPENBLAS_TARGET" \
  USE_OPENMP=1 || exit 1

make PREFIX="$OPENBLAS_DIR" install || exit 1

for symbol in dgemm_ dgetrf_; do
  nm -D "$OPENBLAS_DIR/lib/libopenblas.so" | grep -q " T $symbol$" || {
    echo "libopenblas.so does not export $symbol." >&2
    exit 1
  }
done
