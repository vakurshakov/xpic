#!/bin/bash

source ./header.sh

check_build() {
  if [[ $? != 0 ]]; then
    echo "Build was unsuccessful, exiting the $0"
    exit 1
  fi
}

if [[ $1 == "Debug" || $# == 0 ]]; then
    cmake -S . -B build/Debug -D CMAKE_BUILD_TYPE=Debug
    cmake --build build/Debug --parallel 4
    check_build
fi

if [[ $1 == "Release" || $# == 0 ]]; then
    cmake -S . -B build/Release -D CMAKE_BUILD_TYPE=Release
    cmake --build build/Release --parallel 4
    check_build
fi
