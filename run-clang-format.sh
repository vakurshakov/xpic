#!/bin/bash

find src/ tests/ -type f \( -name '*.h' -o -name '*.cpp' \) -exec clang-format -i "{}" \;