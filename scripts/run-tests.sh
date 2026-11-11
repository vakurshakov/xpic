#!/bin/bash

if [[ $1 == "--help" ]]; then
  ctest --help
  exit 0
fi

source ./scripts/setenv.sh || exit 1

build_type=Release

source ./scripts/build.sh $build_type

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CTEST_PARALLEL_LEVEL=${CTEST_PARALLEL_LEVEL:-$(nproc)} # overridden by `-j <n>`

ctest --test-dir build/$build_type "$@"
