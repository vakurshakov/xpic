#!/bin/bash

source ./header.sh

check_build() {
  if [[ $? != 0 ]]; then
    echo "Build was unsuccessful, exiting the $0"
    exit 1
  fi
}

cmake -S . -B build/Debug -D CMAKE_BUILD_TYPE=Debug
cmake --build build/Debug --parallel 4
check_build

cmake -S . -B build/Release -D CMAKE_BUILD_TYPE=Release
cmake --build build/Release --parallel 4
check_build
