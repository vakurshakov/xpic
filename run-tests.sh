#!/bin/bash

if [[ $1 == "--help" ]]; then
  ctest --help
  exit 0
fi

source ./header.sh

source ./build.sh

build_type=Release

export OMP_NUM_THREADS=4
export CTEST_PARALLEL_LEVEL=4

ctest --test-dir build/$build_type $@
