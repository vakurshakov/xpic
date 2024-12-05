#!/bin/bash

build=./build
commands=$build/compile_commands.json

sed -i 's/-fdeps-format=p1689r5//g' $commands
sed -i 's/-fmodules-ts//g' $commands
sed -i -e 's/-fmodule-mapper=[^ ]*.o.modmap//g' $commands

run-clang-tidy $@ -j 8 -p $build -extra-arg="-I/opt/mpich/include" 2>&1 > clang-tidy.log