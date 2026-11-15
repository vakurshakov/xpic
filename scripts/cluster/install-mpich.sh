#!/usr/bin/env bash

# Builds MPICH (without Fortran) from TARBALL_DIR into MPI_DIR of the stack.
# Run it on a node of the target architecture (`-march=native`), directly or with
# `qsub`, e.g. `qsub -q <queue> -pe smp 128 -l h_rt=12:00:00 -v STACK=...,STACK_ROOT=...,TARBALL_DIR=...`.
#
# Usage: ./scripts/cluster/install-mpich.sh [-h|--help] [--force]
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

guard_prefix "$MPI_DIR" "$force"
extract_source "$MPICH_TARBALL" "$(dirname "$MPI_DIR")" "mpich-$MPICH_PINNED_VERSION" "$force"

cd "$(dirname "$MPI_DIR")/mpich-$MPICH_PINNED_VERSION" || exit 1

# The source tree is shared by the stacks (only the prefix differs), and objects of another
# compiler or flags (e.g. non-PIC ones) must not leak into this build.
[[ -f Makefile ]] && { make distclean || exit 1; }

./configure                                        \
  --prefix="$MPI_DIR"                              \
  --enable-fast=all,O3                             \
  --enable-threads=runtime                         \
  --disable-fortran                                \
  CC="$STACK_CC"                                   \
  CXX="$STACK_CXX"                                 \
  MPICHLIB_CFLAGS="${MPICH_FLAGS:-$STACK_FLAGS}"   \
  MPICHLIB_CXXFLAGS="${MPICH_FLAGS:-$STACK_FLAGS}" \
  "${MPICH_CONFIGURE_ARGS[@]}" || exit 1

make -j "$JOBS" || exit 1
make install || exit 1
