#!/bin/bash

source ./header.sh

source ./build.sh RELEASE

if [[ $? != 0 ]]; then
  echo "Build was unsuccessful, exiting the $0"
  exit 1
fi

export OMP_NUM_THREADS=4

ctest --test-dir build/ $@
