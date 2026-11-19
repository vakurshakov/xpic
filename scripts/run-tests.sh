#!/bin/bash

# Rebuilds Release, then runs ctest on it.
#
# Usage: ./scripts/run-tests.sh [-h|--help] [ctest options, e.g. -L smoke --output-on-failure]
#   -h, --help  print this help and exit (`ctest --help` lists the ctest options)
#
# Environment (also see scripts/setenv.sh):
#   CTEST_PARALLEL_LEVEL  parallel tests (default: 4, `-j <n>` overrides)
#   OMP_NUM_THREADS       OpenMP threads per test (default: 1)
#   STACK       name of scripts/stacks/<name>.sh sourced first (default: local)
#   MPI_DIR, JSON_DIR, PETSC_DIR, PETSC_ARCH, GCC_DIR  dependency locations

source "$(dirname "${BASH_SOURCE[0]}")/usage.sh"
if [[ $1 == "-h" || $1 == "--help" ]]; then
  usage
  exit 0
fi

source ./scripts/setenv.sh || exit 1

build_type=Release

./scripts/build.sh $build_type || exit 1

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# Four cores (overridden by calling with `-j <n>`), so the machine stays responsive.
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-4}

ctest --test-dir build/$build_type "$@"
