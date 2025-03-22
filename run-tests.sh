#!/bin/bash

source ./header.sh

source ./build.sh RELEASE

export OMP_NUM_THREADS=4

ctest --test-dir build/ $@
