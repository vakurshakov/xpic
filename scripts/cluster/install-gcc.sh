#!/usr/bin/env bash

# Builds GCC (C, C++, LTO; no multilib) from TARBALL_DIR into GCC_DIR of the stack, with the host
# compiler, out of tree. GMP, MPFR and MPC must be on the host or unpacked into the sources
# beforehand (`contrib/download_prerequisites`): the script never downloads.
# Run it directly or with `qsub -q <queue> -pe smp 128 -l h_rt=12:00:00 -v STACK=...,STACK_ROOT=...,TARBALL_DIR=...`.
#
# Usage: ./scripts/cluster/install-gcc.sh [-h|--help] [--force]
#   --force     rebuild an existing prefix from a clean source tree
#   -h, --help  print this help and exit
#
# Environment:
#   STACK        name of scripts/stacks/<name>.sh (required)
#   STACK_ROOT   root of the stack installs (required)
#   TARBALL_DIR  directory with the source tarballs (required)
#   JOBS         make parallelism (default: scheduler slots, then nproc)

#$ -S /bin/bash
#$ -N install-gcc
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
NAME="GCC-${GCC_PINNED_VERSION%.*}"
TARBALL="gcc-$GCC_PINNED_VERSION.tar.gz"
SOURCE="gcc-$GCC_PINNED_VERSION"
PREFIX="$GCC_DIR"

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
"$SOURCE_DIR/configure"                 \
  --prefix="$PREFIX"                    \
  --disable-multilib                    \
  --with-dwarf2                         \
  --enable-valgrind-annotations         \
  --enable-lto                          \
  --enable-languages=c,c++,lto || exit 1

# Step 4: build.
make -j "$JOBS" || exit 1

# Step 5: install.
make install || exit 1

# Step 6: check the installed compiler.
[[ "$("$PREFIX/bin/gcc" -dumpfullversion)" == "$GCC_PINNED_VERSION" ]] || {
  echo "$PREFIX/bin/gcc is not GCC $GCC_PINNED_VERSION." >&2
  exit 1
}
