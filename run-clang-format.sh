#!/bin/bash

find src/ -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i;