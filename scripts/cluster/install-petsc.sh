#!/usr/bin/env bash

# Builds PETSc from TARBALL_DIR into PETSC_DIR of the stack (a prefix install, so PETSC_ARCH is
# empty for xpic) and runs `make check` from that prefix. Needs MPICH and the BLAS/LAPACK of the stack.
# Run it on a node of the target architecture (`-march=native`), directly or with
# `qsub`, e.g. `qsub -q <queue> -pe smp 128 -l h_rt=12:00:00 -v STACK=...,STACK_ROOT=...,TARBALL_DIR=...`.
#
# Usage: ./scripts/cluster/install-petsc.sh [-h|--help] [--force]
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

prefix="$PETSC_DIR"
petsc_dir="$(dirname "$prefix")"
petsc_src="$petsc_dir/petsc-${PETSC_PINNED_TAG#v}"

guard_prefix "$prefix" "$force"
extract_source "$PETSC_TARBALL" "$petsc_dir" "$(basename "$petsc_src")" "$force"

verify_petsc "$petsc_src" || exit 1

cd "$petsc_src" || exit 1

# configure wants PETSC_DIR to be the source tree and ignores (and warns about) variables named
# like its options: PETSC_ARCH and MPI_DIR (unset for configure only, its arguments use it).
export PETSC_DIR="$PWD"
unset PETSC_ARCH
arch="$STACK_PETSC_BUILD_ARCH"
rm -rf "$arch"

# KNOWN ISSUE: the three-dash `---with-openmp-kernels` is what produced the working stacks, and
# configure does not recognize it, so the option has no effect. Kept for a faithful rebuild;
# fixing it is a separate step (stack comparison follow-up).
env -u MPI_DIR ./configure PETSC_ARCH="$arch" \
  --prefix="$prefix"                          \
  CC="$MPI_DIR/bin/mpicc"                     \
  CXX="$MPI_DIR/bin/mpicxx"                   \
  --with-fc=0                                 \
  --with-mpiexec="$MPI_DIR/bin/mpiexec"       \
  --with-openmp=1                             \
  ---with-openmp-kernels=1                    \
  --with-threadsafety=1                       \
  --with-debugging=0                          \
  --with-x=0                                  \
  --with-ssl=0                                \
  "${STACK_PETSC_BUILD_OPTIONS[@]}" || exit 1

make PETSC_ARCH="$arch" -j "$JOBS" all || exit 1
make PETSC_ARCH="$arch" install || exit 1

# From the installed prefix, where xpic will use it.
make PETSC_DIR="$prefix" PETSC_ARCH="" check || exit 1
