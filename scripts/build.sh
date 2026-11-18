#!/bin/bash

# Configures and builds xpic with CMake into build/<type>.
# 
# Usage: ./scripts/build.sh [-h|--help] [Debug|Release] [cmake --build options, e.g. -j 16]
#   Debug|Release  build only this type (default: both)
#   -h, --help     print this help and exit
# 
# Environment (also see scripts/setenv.sh):
#   CMAKE_BUILD_PARALLEL_LEVEL  build jobs (default: 4, `-j <n>` overrides)
#   STACK       name of scripts/stacks/<name>.sh sourced first (default: local)
#   MPI_DIR, JSON_DIR, PETSC_DIR, PETSC_ARCH, GCC_DIR  dependency locations

source "$(dirname "${BASH_SOURCE[0]}")/usage.sh"
if [[ $1 == "-h" || $1 == "--help" ]]; then
  usage
  exit 0
fi

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
