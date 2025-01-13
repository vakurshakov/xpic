#!/bin/sh

export MPI_DIR="/opt/mpich/"
export JSON_DIR="external/json"
export PETSC_DIR="external/petsc"

cmake -S . -B build/ -D CMAKE_BUILD_TYPE=DEBUG

cmake --build build/ --parallel 8
