#!/bin/bash

if [[ $1 == "--help" ]]; then
  ctest --help
  exit 0
fi

source ./scripts/setenv.sh || exit 1

build_type=Release

source ./scripts/build.sh $build_type

export OMP_NUM_THREADS=16
export CTEST_PARALLEL_LEVEL=16

ctest --test-dir build/$build_type $@
