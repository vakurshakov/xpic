#!/usr/bin/env bash

# Builds MPICH (without Fortran) from TARBALL_DIR into MPI_DIR of the stack, out of tree.
# Run it on a node of the target architecture (`-march=native`), directly or with
# `qsub -q <queue> -pe smp 128 -l h_rt=12:00:00 -v STACK=...,STACK_ROOT=...,TARBALL_DIR=...`.
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
#$ -N install-mpich
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
NAME="MPICH-$MPICH_PINNED_VERSION"
TARBALL="mpich-$MPICH_PINNED_VERSION.tar.gz"
SOURCE="mpich-$MPICH_PINNED_VERSION"
PREFIX="$MPI_DIR"

COMPONENT_DIR="$STACK_ROOT/$NAME"
SOURCE_DIR="$COMPONENT_DIR/$SOURCE"
BUILD_DIR="$COMPONENT_DIR/build-$STACK"

# Step 1: guard the prefix and extract the source.
guard_prefix "$PREFIX" "$force"
extract_source "$TARBALL" "$COMPONENT_DIR" "$SOURCE" "$force"

# Step 2: out-of-tree build, always from an empty directory.
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR" || exit 1

# Step 3: configure.
"$SOURCE_DIR/configure"                                  \
  --prefix="$PREFIX"                                     \
  --enable-fast=all,O3                                   \
  --enable-threads=runtime                               \
  --disable-fortran                                      \
  CC="$STACK_CC"                                         \
  CXX="$STACK_CXX"                                       \
  MPICHLIB_CFLAGS="${STACK_MPICH_FLAGS:-$STACK_FLAGS}"   \
  MPICHLIB_CXXFLAGS="${STACK_MPICH_FLAGS:-$STACK_FLAGS}" \
  "${STACK_MPICH_BUILD_OPTIONS[@]}" || exit 1

# Step 4: build.
make -j "$JOBS" || exit 1

# Step 5: install.
make install || exit 1
