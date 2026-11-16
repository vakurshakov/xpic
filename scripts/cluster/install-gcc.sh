#!/usr/bin/env bash

# Builds GCC (C, C++, LTO; no multilib) from TARBALL_DIR into GCC_DIR of the stack, with the host
# compiler. The sources are extracted into GCC_DIR itself (as <prefix>/gcc-<version>), and the
# build happens out of tree. GMP, MPFR and MPC must be on the host or unpacked into the sources
# beforehand (`contrib/download_prerequisites`): the script never downloads.
# Run it directly or with `qsub`, e.g.
# `qsub -q <queue> -pe smp 128 -l h_rt=12:00:00 -v STACK=...,STACK_ROOT=...,TARBALL_DIR=...`.
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

guard_prefix "$GCC_DIR" "$force"
extract_source "$GCC_TARBALL" "$GCC_DIR" "gcc-$GCC_PINNED_VERSION" "$force"

# The prefix is the source directory's parent, so the build tree is next to the sources.
build_dir="$GCC_DIR/gcc-$GCC_PINNED_VERSION-build"
rm -rf "$build_dir"
mkdir -p "$build_dir" && cd "$build_dir" || exit 1

"$GCC_DIR/gcc-$GCC_PINNED_VERSION/configure" \
  --prefix="$GCC_DIR"                        \
  --disable-multilib                         \
  --with-dwarf2                              \
  --enable-valgrind-annotations              \
  --enable-lto                               \
  --enable-languages=c,c++,lto || exit 1

make -j "$JOBS" || exit 1
make install || exit 1

[[ "$("$GCC_DIR/bin/gcc" -dumpfullversion)" == "$GCC_PINNED_VERSION" ]] || {
  echo "$GCC_DIR/bin/gcc is not GCC $GCC_PINNED_VERSION." >&2
  exit 1
}
