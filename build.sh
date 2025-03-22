#!/bin/bash

source ./header.sh

type=${1:-RELEASE}

cmake -S . -B build/ -D CMAKE_BUILD_TYPE=$type

cmake --build build/ --parallel 8
