#!/bin/sh

cmake -S . -B build/ -D CMAKE_BUILD_TYPE=DEBUG

cmake --build build/ --parallel 8
