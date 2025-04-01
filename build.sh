#!/bin/bash

source ./header.sh

cmake -S . -B build/Debug -D CMAKE_BUILD_TYPE=Debug
cmake --build build/Debug --parallel 4

cmake -S . -B build/Release -D CMAKE_BUILD_TYPE=Release
cmake --build build/Release --parallel 4
