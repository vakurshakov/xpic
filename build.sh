#!/bin/bash

source ./header.sh

cmake -S . -B build/ -D CMAKE_BUILD_TYPE=RELEASE

cmake --build build/ --parallel 8
