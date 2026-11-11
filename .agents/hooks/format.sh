#!/usr/bin/env bash

# PostToolUse hook: formats the C/C++ file just written or edited by Claude.

if ! command -v clang-format &> /dev/null; then
  echo "[Claude Hook Warning]: 'clang-format' is not installed, skipping formatting." >&2
  echo "Install it with: ./run-format-all.sh --install" >&2
  # Exit 0 so that Claude continues working without formatting.
  exit 0
fi

file_path=$(sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')

if [[ "$file_path" =~ \.(c|cpp|cc|h|hpp)$ ]] && [ -f "$file_path" ]; then
  clang-format -i "$file_path" 2>/dev/null
fi

exit 0
