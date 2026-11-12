#!/bin/bash

# Usage: build.sh [Debug|Release] [cmake --build options, e.g. -j 16]
# Without a build type, both are built.

source ./scripts/setenv.sh || exit 1
check_dependency_versions || exit 1

check_build() {
  if [[ $? != 0 ]]; then
    echo "Build was unsuccessful, exiting the $0"
    exit 1
  fi
}

# Four cores (overridden by calling with `-j <n>`), so the machine stays responsive.
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-4}

build_types=(Debug Release)
if [[ $1 == "Debug" || $1 == "Release" ]]; then
  build_types=("$1")
  shift
fi

for build_type in "${build_types[@]}"; do
  cmake -S . -B build/$build_type -D CMAKE_BUILD_TYPE=$build_type
  cmake --build build/$build_type "$@"
  check_build
done
