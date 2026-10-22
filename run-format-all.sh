#!/usr/bin/env bash

# Formats the whole project. Local edits are formatted by the PostToolUse hook
# (.claude/hooks/format.sh), run this only for project-wide style fixes.
# Usage: ./run-format-all.sh [--install]

install_command() {
  if command -v pacman &> /dev/null; then
    echo "sudo pacman -S clang"
  elif command -v apt-get &> /dev/null; then
    echo "sudo apt-get install clang-format"
  elif command -v brew &> /dev/null; then
    echo "brew install clang-format"
  fi
}

if ! command -v clang-format &> /dev/null; then
  cmd=$(install_command)
  if [ -z "$cmd" ]; then
    echo "'clang-format' is not installed, please install LLVM/clang-format." >&2
    exit 1
  fi

  if [ "$1" != "--install" ]; then
    echo "'clang-format' is not installed. Run: $cmd" >&2
    echo "Or rerun with --install to do it automatically." >&2
    exit 1
  fi

  $cmd || exit 1
fi

find src/ tests/ -type f \( -name '*.h' -o -name '*.cpp' \) -exec clang-format -i "{}" \;
