#!/bin/bash

if [[ $1 == "--help" ]]; then
  ctest --help
  exit 0
fi

source ./scripts/setenv.sh || exit 1

build_type=Release

source ./scripts/build.sh $build_type

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# Four cores (overridden by calling with `-j <n>`),
# and low priority, so the machine stays responsive.
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-4}

nice -n 10 ctest --test-dir build/$build_type "$@"
