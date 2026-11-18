#!/usr/bin/env bash

# Builds PETSc from TARBALL_DIR into PETSC_DIR of the stack (a prefix install, so PETSC_ARCH is
# empty for xpic) and runs `make check` from that prefix. Needs MPICH and the BLAS/LAPACK of the stack.
# Its build tree is the PETSC_ARCH directory inside the shared source tree, so it is not out of tree.
# Run it on a node of the target architecture (`-march=native`), directly or with
# `qsub -q <queue> -pe smp 128 -l h_rt=12:00:00 -v STACK=...,STACK_ROOT=...,TARBALL_DIR=...`.
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
#$ -N install-petsc
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
NAME="petsc-${PETSC_PINNED_TAG#v}"
TARBALL="petsc-${PETSC_PINNED_TAG#v}.tar.gz"
SOURCE="petsc-${PETSC_PINNED_TAG#v}"
PREFIX="$PETSC_DIR"

COMPONENT_DIR="$STACK_ROOT/$NAME"
SOURCE_DIR="$COMPONENT_DIR/$SOURCE"

# Step 1: guard the prefix and extract the source.
guard_prefix "$PREFIX" "$force"
extract_source "$TARBALL" "$COMPONENT_DIR" "$SOURCE" "$force"

verify_petsc "$SOURCE_DIR" || exit 1

# Step 2: the build tree is the PETSC_ARCH directory in the source, always started empty.
# configure wants PETSC_DIR to be the source tree and ignores (and warns about) variables named
# like its options: PETSC_ARCH and MPI_DIR (unset for configure only, its arguments use it).
cd "$SOURCE_DIR" || exit 1
export PETSC_DIR="$PWD"
unset PETSC_ARCH
arch="$STACK_PETSC_BUILD_ARCH"
rm -rf "$arch"

# Step 3: configure.
# KNOWN ISSUE: the three-dash `---with-openmp-kernels` is what produced the working stacks, and
# configure does not recognize it, so the option has no effect. Kept for a faithful rebuild;
# fixing it is a separate step (stack comparison follow-up).
env -u MPI_DIR ./configure PETSC_ARCH="$arch" \
  --prefix="$PREFIX"                          \
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

# Step 4: build.
make PETSC_ARCH="$arch" -j "$JOBS" all || exit 1

# Step 5: install.
make PETSC_ARCH="$arch" install || exit 1

# Step 6: check from the installed prefix, where xpic will use it.
make PETSC_DIR="$PREFIX" PETSC_ARCH="" check || exit 1
